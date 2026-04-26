
"""Load split data and build shuffled timelines for the lockstep simulation.

Reads from the directory structure produced by split_data.py:
    simulation/data/splits/{snr}/{machine_type}/{machine_id}/warmup/*.wav
    simulation/data/splits/{snr}/{machine_type}/{machine_id}/test_normal/*.wav
    simulation/data/splits/{snr}/{machine_type}/{machine_id}/test_abnormal/*.wav

Builds a timeline per machine: an ordered list of (wav_path, label) pairs that
the lockstep trainer iterates through one timestep at a time. The same
timeline is shared by every channel of the same (machine_type, machine_id).
"""

from dataclasses import dataclass, field
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
    # Clip indices where a new segment starts and CUSUM should be reset.
    # Populated only by the "rounds" shuffle mode; empty for all other modes.
    round_boundaries: list[int] = field(default_factory=list)

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
    """Spread anomaly blocks evenly across the normal pool.

    The configured block_interval is used as a hint, not a hard constraint.
    With n_blocks anomaly blocks and len(normal) normals, the function
    splits the normals into (n_blocks + 1) segments of roughly equal size
    and inserts one block between each pair. This guarantees:
        - Every anomaly clip appears in the timeline (no data dropped)
        - Blocks are evenly distributed end-to-end (no trailing pile-up)

    A warning is printed if the auto-scaled interval differs significantly
    from the requested block_interval, so the user knows their hint was
    overridden.
    """
    normal = normal_paths.copy()
    rng.shuffle(normal)

    abnormal = abnormal_paths.copy()
    rng.shuffle(abnormal)

    blocks = [
        abnormal[i:i + block_size]
        for i in range(0, len(abnormal), block_size)
    ]
    n_blocks = len(blocks)

    if n_blocks == 0:
        return normal, [0] * len(normal)

    # Split normals into (n_blocks + 1) segments. Place one block between
    # each pair, so the layout is:
    # [normals_seg_0][block_0][normals_seg_1][block_1]...[block_N-1][normals_seg_N]
    # Leftover normals get distributed one extra to the early segments.
    base_size, leftover = divmod(len(normal), n_blocks + 1)
    segment_sizes = [
        base_size + (1 if i < leftover else 0)
        for i in range(n_blocks + 1)
    ]

    # Warn if the actual interval drifts far from the user's hint
    actual_interval = base_size
    if abs(actual_interval - block_interval) > 2:
        print(
            f"  Note: block_interval={block_interval} requested but only "
            f"{actual_interval} normals fit between blocks "
            f"({n_blocks} blocks across {len(normal)} normals)."
        )

    result_paths: list[str] = []
    result_labels: list[int] = []
    normal_idx = 0

    for i, seg_size in enumerate(segment_sizes):
        # Normal segment
        for _ in range(seg_size):
            result_paths.append(normal[normal_idx])
            result_labels.append(0)
            normal_idx += 1

        # Anomaly block (after every segment except the last)
        if i < n_blocks:
            for path in blocks[i]:
                result_paths.append(path)
                result_labels.append(1)

    return result_paths, result_labels

def _shuffle_rounds(
    normal_paths: list[str],
    abnormal_paths: list[str],
    n_rounds: int,
    normal_per_round: int,
    anomaly_per_round: int,
    rng: random.Random,
) -> tuple[list[str], list[int], list[int]]:
    """Build the layout n_rounds × (normal_per_round, anomaly_per_round).

    Each round is one normal segment followed by one anomaly segment. Returns
    the timeline plus the clip indices at every segment boundary so lockstep
    can call cusum_reset() there. Reproduces the OLD gmm/evaluate.py round
    structure where CUSUM was zeroed between every segment.

    Returns:
        (paths, labels, boundaries). boundaries has 2*n_rounds - 1 entries.
    """
    needed_normal  = n_rounds * normal_per_round
    needed_anomaly = n_rounds * anomaly_per_round
    if len(normal_paths) < needed_normal:
        raise ValueError(
            f"rounds shuffle: need {needed_normal} normal clips "
            f"({n_rounds} × {normal_per_round}) but only "
            f"{len(normal_paths)} available"
        )
    if len(abnormal_paths) < needed_anomaly:
        raise ValueError(
            f"rounds shuffle: need {needed_anomaly} abnormal clips "
            f"({n_rounds} × {anomaly_per_round}) but only "
            f"{len(abnormal_paths)} available"
        )

    normal = normal_paths.copy()
    rng.shuffle(normal)
    abnormal = abnormal_paths.copy()
    rng.shuffle(abnormal)

    paths: list[str] = []
    labels: list[int] = []
    boundaries: list[int] = []

    n_idx = 0
    a_idx = 0
    for r in range(n_rounds):
        if r > 0:
            boundaries.append(len(paths))
        for _ in range(normal_per_round):
            paths.append(normal[n_idx])
            labels.append(0)
            n_idx += 1
        boundaries.append(len(paths))
        for _ in range(anomaly_per_round):
            paths.append(abnormal[a_idx])
            labels.append(1)
            a_idx += 1

    return paths, labels, boundaries


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
    n_rounds: int = 2,
    normal_per_round: int = 30,
    anomaly_per_round: int = 30,
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

    # Sample warmup_count clips at random from the available warmup pool.
    # Sequential slicing would re-introduce a chronological bias because the
    # symlinks created by split_data.py keep their original (time-ordered)
    # filenames, so _load_sorted_wavs returns them sorted by recording time.
    # rng.sample preserves the stationarity assumption: every warmup clip is
    # an IID draw, and any sub-slice of the result (fit, val) is itself a
    # valid random subsample.
    if warmup_count > len(warmup_all):
        raise ValueError(
            f"{machine_type}/{machine_id}: requested warmup_count={warmup_count} "
            f"but only {len(warmup_all)} warmup clips available"
        )
    warmup_paths = rng.sample(warmup_all, warmup_count)

    # Build shuffled test timeline
    boundaries: list[int] = []
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
    elif shuffle_mode == "rounds":
        test_paths, test_labels, boundaries = _shuffle_rounds(
            test_normal, test_abnormal,
            n_rounds, normal_per_round, anomaly_per_round, rng,
        )
    else:
        raise ValueError(
            f"Unknown shuffle_mode '{shuffle_mode}'. "
            f"Expected: random, block_random, block_fixed, rounds"
        )

    return NodeTimeline(
        node_id=_build_node_id(machine_type, machine_id),
        machine_type=machine_type,
        machine_id=machine_id,
        warmup_paths=warmup_paths,
        test_paths=test_paths,
        test_labels=test_labels,
        round_boundaries=boundaries,
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
    n_rounds: int = 2,
    normal_per_round: int = 30,
    anomaly_per_round: int = 30,
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
                n_rounds=n_rounds,
                normal_per_round=normal_per_round,
                anomaly_per_round=anomaly_per_round,
            )
            timelines.append(timeline)
        timelines_by_type[machine_type] = timelines

    return timelines_by_type