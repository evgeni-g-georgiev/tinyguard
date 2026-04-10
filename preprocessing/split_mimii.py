#!/usr/bin/env python3
"""
split_mimii.py — Create a fixed, reproducible train/test split for MIMII.

Saves a JSON manifest assigning every WAV clip to either the separator
training set or the inference evaluation set. Running this once means
separator/train.py and inference/run.py always use the same pre-determined
clips — no re-sampling at runtime.

Split logic (mirrors the original simulate_deployment.py sampling):
  - Normal clips: shuffle with fixed seed, take first TRAIN_CLIPS (60) for
    training, the remainder for evaluation.
  - Abnormal clips: shuffle with fixed seed, take up to N_ROUNDS × MONITOR_CLIPS
    (90) for evaluation.
  - n_rounds is capped by available clips.

Paths in the manifest are stored relative to MIMII_ROOT for portability.
Resolve them in downstream scripts with:  str(MIMII_ROOT / path)

Input
-----
  data/mimii/   fully populated (run data/download_mimii.py first)

Output
------
  outputs/mimii_splits/splits.json

Usage
-----
    python preprocessing/split_mimii.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MIMII_ROOT, MIMII_SPLITS,
    MACHINE_TYPES, MACHINE_IDS,
    TRAIN_CLIPS, MONITOR_CLIPS, N_ROUNDS, SEED,
)



def _check_mimii_root():
    """Check that the MIMII dataset directory exists."""
    if not MIMII_ROOT.is_dir():
        print(f"ERROR: MIMII data not found at {MIMII_ROOT}/")
        print("Run:  python data/download_mimii.py  first.")
        sys.exit(1)


def _list_machine_audio(machine_dir):
    """List normal and abnormal WAV files for one machine directory."""
    normal_paths = sorted((machine_dir / "normal").glob("*.wav"))
    abnormal_paths = sorted((machine_dir / "abnormal").glob("*.wav"))
    return normal_paths, abnormal_paths


def _compute_n_rounds(norm_perm, anom_perm):
    """Compute the maximum number of monitoring rounds supported by the data."""
    return min(
        N_ROUNDS,
        (len(norm_perm) - TRAIN_CLIPS) // MONITOR_CLIPS,
        len(anom_perm) // MONITOR_CLIPS,
    )


def _rel_path(path):
    """Convert an absolute path to a path relative to MIMII_ROOT."""
    # Store paths relative to MIMII_ROOT so the manifest is portable
    # across machines with different absolute mount points.
    return str(Path(path).relative_to(MIMII_ROOT))


def _build_machine_split(mtype, mid):
    """Build the split manifest entry for one machine."""
    machine_dir = MIMII_ROOT / mtype / mid
    key = f"{mtype}/{mid}"

    if not machine_dir.is_dir():
        print(f"  WARNING: {key} — directory not found, skipping.")
        return key, None

    normal_paths, abnormal_paths = _list_machine_audio(machine_dir)

    if not normal_paths:
        print(f"  WARNING: {key} — no normal clips found, skipping.")
        return key, None

    if not abnormal_paths:
        print(f"  WARNING: {key} — no abnormal clips found, skipping.")
        return key, None

    rng = np.random.default_rng(SEED)
    norm_perm = list(rng.permutation([str(p) for p in normal_paths]))
    anom_perm = list(rng.permutation([str(p) for p in abnormal_paths]))

    # Cap n_rounds by what the data can actually support:
    #   - N_ROUNDS is the configured maximum
    #   - second term: how many full monitoring windows fit in the remaining normal clips
    #   - third term: how many full windows fit in the abnormal clips
    n_rounds = _compute_n_rounds(norm_perm, anom_perm)

    if n_rounds <= 0 or len(norm_perm) < TRAIN_CLIPS:
        print(
            f"  WARNING: {key} — insufficient clips "
            f"(norm={len(norm_perm)}, anom={len(anom_perm)}), skipping."
        )
        return key, None

    split = {
        "train_normal": [_rel_path(p) for p in norm_perm[:TRAIN_CLIPS]],
        "test_normal": [
            _rel_path(p)
            for p in norm_perm[TRAIN_CLIPS:TRAIN_CLIPS + n_rounds * MONITOR_CLIPS]
        ],
        "test_abnormal": [
            _rel_path(p)
            for p in anom_perm[:n_rounds * MONITOR_CLIPS]
        ],
        "n_rounds": n_rounds,
    }

    return key, split



def split_mimii():
    """Create and save the fixed MIMII train/test split manifest."""
    _check_mimii_root()
    MIMII_SPLITS.parent.mkdir(parents=True, exist_ok=True)

    splits = {}
    missing = []

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key, split = _build_machine_split(mtype, mid)

            if split is None:
                missing.append(key)
                continue

            splits[key] = split
            print(
                f"  {key}: {len(split['train_normal'])} train, "
                f"{len(split['test_normal'])} test_normal, "
                f"{len(split['test_abnormal'])} test_abnormal, "
                f"{split['n_rounds']} rounds"
            )

    with open(MIMII_SPLITS, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved splits for {len(splits)} machines → {MIMII_SPLITS}")
    if missing:
        print(f"Skipped: {missing}")

    return {
        "splits_path": MIMII_SPLITS,
        "n_machines": len(splits),
        "skipped": missing,
    }



if __name__ == "__main__":
    split_mimii()