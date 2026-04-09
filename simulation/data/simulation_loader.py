
"""Load split data and build shuffled timelines for the lockstep simulation.

Reads from the directory structure produced by split_data.py:
    simulation/data/splits/{machine_type}/{machine_id}/warmup/*.wav
    simulation/data/splits/{machine_type}/{machine_id}/test_normal/*.wav
    simulation/data/splits/{machine_type}/{machine_id}/test_abnormal/*.wav

Builds a timeline per node — an ordered list of (wav_path, label) pairs
that the lockstep trainer iterates through one timestep at a time.
"""

from dataclasses import dataclass 
from pathlib import Path 
from itertools import repeat, zip_longest
import random 


@dataclass(frozen=True)
class NodeTimeline:
    """All data for one node, ready for the lockstep simulation."""
    node_id: str 
    machine_type: str 
    machine_id: str 
    warmup_paths: list[str]
    test_paths: list[str]
    test_labels: list[int] # 0 = normal, 1 = abnormal 

def _build_node_id(machine_type: str, machine_id: str) -> str:
    return f"{machine_type}_{machine_id}"

def _load_sorted_wavs(directory: Path) -> list[str]:
    """Load all .wav paths from a directory, sorted by name."""
    if not directory.exists():
        raise FileNotFoundError(f"Split directory not found: {directory}")
    paths = sorted(directory.glob("*.wav"))
    if not paths:
        raise ValueError(f"No .wav files found in {directory}")
    return [str(p) for p in paths]

# ── Shuffle strategies  ────────────────────────────────────────────────────────

def _shuffle_random(
    normal_paths: list[str],
    abnormal_paths: list[str],
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    """Uniform shuffle of normal + abnormal clips."""
    combined = [*zip(normal_paths, repeat(0)), *zip(abnormal_paths, repeat(1))]
    rng.shuffle(combined) 
    paths, labels = zip(*combined)
    return list(paths), list(labels) 


def _shuffle_block_random(
    normal_paths: list[str],
    abnormal_paths: list[str],
    block_size: int,
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    """Group anomalies into contiguous blocks, place at random positions.

    Anomaly clips are divided into blocks of block_size. Each block is
    inserted at a random position within the normal stream. Simulates
    real-world anomalies that persist across multiple consecutive clips.
    """
    normal = normal_paths.copy()
    rng.shuffle(normal)

    abnormal = abnormal_paths.copy()
    rng.shuffle(abnormal)

    # Split abnormals into blocks
    blocks = [
        abnormal[i:i + block_size]
        for i in range(0, len(abnormal), block_size)
    ]

    # Insert each block at a random position in the normal stream
    result_paths: list[str] = normal.copy()
    result_labels: list[int] = [0] * len(normal)

    for block in blocks:
        insert_pos = rng.randint(0, len(result_paths))
        for j, path in enumerate(block):
            result_paths.insert(insert_pos + j, path)
            result_labels.insert(insert_pos + j, 1)

    return result_paths, result_labels


def _shuffle_block_fixed(
    normal_paths: list[str],
    abnormal_paths: list[str],
    block_size: int,
    block_interval: int,
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    """Anomaly blocks at regular intervals in the normal stream.

    Every block_interval normal clips, insert a block of block_size
    anomaly clips. Clips within each block are shuffled but the
    placement pattern is deterministic.
    """
    normal   = rng.sample(normal_paths,   len(normal_paths))
    abnormal = rng.sample(abnormal_paths, len(abnormal_paths))

    normal_chunks   = [normal[i:i + block_interval]   for i in range(0, len(normal),   block_interval)]
    abnormal_blocks = [abnormal[i:i + block_size] for i in range(0, len(abnormal), block_size)]

    labelled = [
        (p, label)
        for normal_chunk, abnormal_block in zip_longest(normal_chunks, abnormal_blocks, fillvalue=[])
        for p, label in [*zip(normal_chunk, repeat(0)), *zip(abnormal_block, repeat(1))]
    ]

    paths, labels = zip(*labelled)
    return list(paths), list(labels)

# ── Public Functions ────────────────────────────────────────────────────────

def load_node_timeline(
    splits_dir: Path,
    machine_type: str,
    machine_id: str,
    warmup_count: int,
    shuffle_mode: str,
    rng: random.Random,
    block_size: int = 5,
    block_interval: int = 20,
) -> NodeTimeline:
    """Load one node's split data and build its timeline.

    Args:
        splits_dir: Root of the split output (e.g. simulation/data/splits).
        machine_type: e.g. "fan".
        machine_id: e.g. "id_00".
        warmup_count: Number of warmup clips to use (truncates the split's
            max available warmup to this count).
        shuffle_mode: One of "random", "block_random", "block_fixed".
        rng: Seeded random.Random instance for reproducibility.
        block_size: Anomaly block size (for block modes).
        block_interval: Normal clips between blocks (for block_fixed).

    Returns:
        NodeTimeline with warmup paths and shuffled test timeline.
    """
    node_dir = splits_dir / machine_type / machine_id

    warmup_all = _load_sorted_wavs(node_dir / "warmup")
    test_normal = _load_sorted_wavs(node_dir / "test_normal")
    test_abnormal = _load_sorted_wavs(node_dir / "test_abnormal")

    # Truncate warmup to configured count
    if warmup_count > len(warmup_all):
        raise ValueError(
            f"{machine_type}/{machine_id}: requested warmup_count={warmup_count} "
            f"but only {len(warmup_all)} warmup clips available"
        )
    warmup_paths = warmup_all[:warmup_count]

    # Build shuffled test timeline
    if shuffle_mode == "random":
        test_paths, test_labels = _shuffle_random(
            test_normal, test_abnormal, rng,
        )
    elif shuffle_mode == "block_random":
        test_paths, test_labels = _shuffle_block_random(
            test_normal, test_abnormal, block_size, rng,
        )
    elif shuffle_mode == "block_fixed":
        test_paths, test_labels = _shuffle_block_fixed(
            test_normal, test_abnormal, block_size, block_interval, rng,
        )
    else:
        raise ValueError(
            f"Unknown shuffle_mode '{shuffle_mode}'. "
            f"Expected: random, block_random, block_fixed"
        )

    return NodeTimeline(
        node_id=_build_node_id(machine_type, machine_id),
        machine_type=machine_type,
        machine_id=machine_id,
        warmup_paths=warmup_paths,
        test_paths=test_paths,
        test_labels=test_labels,
    )


def load_all_timelines(
    splits_dir: Path,
    machine_types: list[str],
    machine_ids: list[str],
    warmup_count: int,
    shuffle_mode: str,
    seed: int,
    block_size: int = 5,
    block_interval: int = 20,
) -> dict[str, list[NodeTimeline]]:
    """Load timelines for all nodes, grouped by machine type.

    Args:
        splits_dir: Root of split output.
        machine_types: e.g. ["fan", "pump", "slider", "valve"].
        machine_ids: e.g. ["id_00", "id_02", "id_04", "id_06"].
        warmup_count: Warmup clips per node.
        shuffle_mode: Shuffle strategy name.
        seed: RNG seed for reproducible shuffling.
        block_size: For block shuffle modes.
        block_interval: For block_fixed mode.

    Returns:
        Dict mapping machine_type to list of NodeTimelines.
    """
    rng = random.Random(seed)

    timelines_by_type: dict[str, list[NodeTimeline]] = {}

    for machine_type in machine_types:
        timelines = []
        for machine_id in machine_ids:
            timeline = load_node_timeline(
                splits_dir=splits_dir,
                machine_type=machine_type,
                machine_id=machine_id,
                warmup_count=warmup_count,
                shuffle_mode=shuffle_mode,
                rng=rng,
                block_size=block_size,
                block_interval=block_interval,
            )
            timelines.append(timeline)
        timelines_by_type[machine_type] = timelines

    return timelines_by_type