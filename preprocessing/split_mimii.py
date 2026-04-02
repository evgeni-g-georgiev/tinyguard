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


def main():
    if not MIMII_ROOT.is_dir():
        print(f"ERROR: MIMII data not found at {MIMII_ROOT}/")
        print("Run:  python data/download_mimii.py  first.")
        import sys; sys.exit(1)

    MIMII_SPLITS.parent.mkdir(parents=True, exist_ok=True)

    splits = {}
    missing = []

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            machine_dir = MIMII_ROOT / mtype / mid
            key = f"{mtype}/{mid}"

            if not machine_dir.is_dir():
                print(f"  WARNING: {key} — directory not found, skipping.")
                missing.append(key)
                continue

            normal_paths   = sorted((machine_dir / "normal").glob("*.wav"))
            abnormal_paths = sorted((machine_dir / "abnormal").glob("*.wav"))

            if not normal_paths:
                print(f"  WARNING: {key} — no normal clips found, skipping.")
                missing.append(key)
                continue
            if not abnormal_paths:
                print(f"  WARNING: {key} — no abnormal clips found, skipping.")
                missing.append(key)
                continue

            rng = np.random.default_rng(SEED)
            norm_perm = list(rng.permutation([str(p) for p in normal_paths]))
            anom_perm = list(rng.permutation([str(p) for p in abnormal_paths]))

            # Cap n_rounds by what the data can actually support:
            #   - N_ROUNDS is the configured maximum
            #   - second term: how many full monitoring windows fit in the remaining normal clips
            #   - third term: how many full windows fit in the abnormal clips
            n_rounds = min(
                N_ROUNDS,
                (len(norm_perm) - TRAIN_CLIPS) // MONITOR_CLIPS,
                len(anom_perm) // MONITOR_CLIPS,
            )

            if n_rounds <= 0 or len(norm_perm) < TRAIN_CLIPS:
                print(f"  WARNING: {key} — insufficient clips (norm={len(norm_perm)}, "
                      f"anom={len(anom_perm)}), skipping.")
                missing.append(key)
                continue

            # Store paths relative to MIMII_ROOT so the manifest is portable
            # across machines with different absolute mount points.
            def rel(p):
                return str(Path(p).relative_to(MIMII_ROOT))

            splits[key] = {
                "train_normal":  [rel(p) for p in norm_perm[:TRAIN_CLIPS]],
                "test_normal":   [rel(p) for p in norm_perm[TRAIN_CLIPS:TRAIN_CLIPS + n_rounds * MONITOR_CLIPS]],
                "test_abnormal": [rel(p) for p in anom_perm[:n_rounds * MONITOR_CLIPS]],
                "n_rounds":      n_rounds,
            }
            print(f"  {key}: {len(splits[key]['train_normal'])} train, "
                  f"{len(splits[key]['test_normal'])} test_normal, "
                  f"{len(splits[key]['test_abnormal'])} test_abnormal, "
                  f"{n_rounds} rounds")

    with open(MIMII_SPLITS, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved splits for {len(splits)} machines → {MIMII_SPLITS}")
    if missing:
        print(f"Skipped: {missing}")


if __name__ == "__main__":
    main()
