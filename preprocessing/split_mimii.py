#!/usr/bin/env python3
"""Create a fixed train/test split manifest for one MIMII SNR variant.

Per machine: the first ``TRAIN_CLIPS`` shuffled normal clips go to training,
the next ``N_ROUNDS * MONITOR_CLIPS`` go to the monitoring normal set, and the
first ``N_ROUNDS * MONITOR_CLIPS`` shuffled abnormal clips go to the monitoring
abnormal set. ``n_rounds`` is capped by available clip counts.

Example:

    python preprocessing/split_mimii.py                 # -6 dB (default)
    python preprocessing/split_mimii.py \\
        --data-root data/mimii_0db \\
        --out preprocessing/outputs/mimii_splits/splits_0db.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MACHINE_IDS,
    MACHINE_TYPES,
    MIMII_NEG6DB_ROOT,
    MIMII_NEG6DB_SPLITS,
    MONITOR_CLIPS,
    N_ROUNDS,
    SEED,
    TRAIN_CLIPS,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a fixed train/test split manifest for a MIMII dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default=str(MIMII_NEG6DB_ROOT),
                        help="Root directory of the MIMII dataset to split.")
    parser.add_argument("--out", type=str, default=str(MIMII_NEG6DB_SPLITS),
                        help="Output path for the splits JSON manifest.")
    return parser


def _list_machine_audio(machine_dir: Path):
    normal_paths   = sorted((machine_dir / "normal").glob("*.wav"))
    abnormal_paths = sorted((machine_dir / "abnormal").glob("*.wav"))
    return normal_paths, abnormal_paths


def _compute_n_rounds(norm_perm: list, anom_perm: list) -> int:
    return min(
        N_ROUNDS,
        (len(norm_perm) - TRAIN_CLIPS) // MONITOR_CLIPS,
        len(anom_perm) // MONITOR_CLIPS,
    )


def _build_machine_split(mtype: str, mid: str, data_root: Path) -> tuple[str, dict | None]:
    """Return ``(key, split_dict)``; split_dict is ``None`` when the machine is skipped."""
    machine_dir = data_root / mtype / mid
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

    rng       = np.random.default_rng(SEED)
    norm_perm = list(rng.permutation([str(p) for p in normal_paths]))
    anom_perm = list(rng.permutation([str(p) for p in abnormal_paths]))

    n_rounds = _compute_n_rounds(norm_perm, anom_perm)
    if n_rounds <= 0 or len(norm_perm) < TRAIN_CLIPS:
        print(
            f"  WARNING: {key} — insufficient clips "
            f"(norm={len(norm_perm)}, anom={len(anom_perm)}), skipping."
        )
        return key, None

    def _rel(p: str) -> str:
        return str(Path(p).relative_to(data_root))

    split = {
        "train_normal": [_rel(p) for p in norm_perm[:TRAIN_CLIPS]],
        "test_normal": [
            _rel(p)
            for p in norm_perm[TRAIN_CLIPS:TRAIN_CLIPS + n_rounds * MONITOR_CLIPS]
        ],
        "test_abnormal": [_rel(p) for p in anom_perm[:n_rounds * MONITOR_CLIPS]],
        "n_rounds": n_rounds,
    }
    return key, split


def split_mimii(data_root: Path, out_path: Path) -> dict:
    """Write the split manifest for every machine under ``data_root`` to ``out_path``."""
    if not data_root.is_dir():
        print(f"ERROR: MIMII data not found at {data_root}/")
        print("Run:  python data/download_mimii.py --snr {-6dB|0dB|6dB}")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits:  dict = {}
    missing: list = []

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key, split = _build_machine_split(mtype, mid, data_root)
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

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved splits for {len(splits)} machines → {out_path}")
    if missing:
        print(f"Skipped: {missing}")

    return {"splits_path": out_path, "n_machines": len(splits), "skipped": missing}


if __name__ == "__main__":
    args = _build_parser().parse_args()
    split_mimii(Path(args.data_root), Path(args.out))
