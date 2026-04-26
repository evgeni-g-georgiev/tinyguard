#!/usr/bin/env python3
"""Download and extract one SNR variant of the MIMII dataset.

MIMII is hosted as Zenodo record 3384388 with one zip per machine type and SNR
level. This script fetches the zips for the requested SNR, unpacks each into
``data/mimii_{snr}/{machine_type}/{machine_id}/`` and deletes the zip.

Requires ``wget`` and ``unzip`` on ``PATH``. Each SNR variant is roughly 30 GB
on disk after extraction.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MACHINE_IDS,
    MACHINE_TYPES,
    MIMII_0DB_ROOT,
    MIMII_6DB_ROOT,
    MIMII_NEG6DB_ROOT,
)

MIMII_ZENODO_RECORD = "3384388"

# --snr value -> (Zenodo filename prefix, extraction root).
SNR_TO_ROOT = {
    "-6dB": ("-6_dB", MIMII_NEG6DB_ROOT),
    "0dB":  ("0_dB",  MIMII_0DB_ROOT),
    "6dB":  ("6_dB",  MIMII_6DB_ROOT),
}

APPROX_ZIP_SIZES = {
    "fan":    "~9.5 GB",
    "pump":   "~7.1 GB",
    "slider": "~6.6 GB",
    "valve":  "~6.4 GB",
}


def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        print(f"ERROR: {name} not found on PATH. Install it and retry.")
        sys.exit(1)


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading → {dest}")
    subprocess.run(
        ["wget", "--continue", "--progress=bar:force", "-O", str(dest), url],
        check=True,
    )


def _extract_and_flatten(zip_path: Path, machine_type: str, output_dir: Path) -> None:
    """Unpack a machine-type zip, stripping its single top-level directory."""
    print(f"  Extracting {zip_path.name}")
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        subprocess.run(["unzip", "-q", str(zip_path), "-d", tmp], check=True)

        top_dirs = [
            d for d in os.listdir(tmp)
            if (Path(tmp) / d).is_dir() and d != "__MACOSX"
        ]
        if len(top_dirs) != 1:
            raise RuntimeError(
                f"Expected one top-level directory in {zip_path.name}, got {top_dirs}"
            )
        src_root = Path(tmp) / top_dirs[0]

        for mid_dir in sorted(src_root.iterdir()):
            if not mid_dir.is_dir():
                continue
            dst = output_dir / machine_type / mid_dir.name
            if dst.exists():
                print(f"    {machine_type}/{mid_dir.name} already present — skipping")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mid_dir), str(dst))
            n_wav = sum(1 for _ in dst.rglob("*.wav"))
            print(f"    → {machine_type}/{mid_dir.name}  ({n_wav} WAVs)")


def _machine_type_complete(output_dir: Path, machine_type: str) -> bool:
    return all((output_dir / machine_type / mid).is_dir() for mid in MACHINE_IDS)


def _print_summary(output_dir: Path) -> None:
    print(f"\n{'-' * 60}")
    print(f"  Summary: {output_dir}/")
    total = 0
    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            path = output_dir / mtype / mid
            if path.is_dir():
                n = sum(1 for _ in path.rglob("*.wav"))
                total += n
                print(f"    {mtype}/{mid}: {n} WAVs")
            else:
                print(f"    {mtype}/{mid}: MISSING")
    print(f"  Total: {total} WAV files")
    print(f"{'-' * 60}\n")


def download_mimii(snr: str) -> Path:
    """Download and extract the given SNR variant. Returns the extraction root."""
    if snr not in SNR_TO_ROOT:
        raise ValueError(f"snr must be one of {list(SNR_TO_ROOT)}, got {snr!r}")

    _check_tool("wget")
    _check_tool("unzip")

    snr_prefix, output_dir = SNR_TO_ROOT[snr]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 60}")
    print(f"  MIMII dataset — SNR {snr}, 4 machine types")
    print(f"  Destination: {output_dir}/")
    print(f"{'-' * 60}")

    for mtype in MACHINE_TYPES:
        if _machine_type_complete(output_dir, mtype):
            print(f"\n  {mtype}: all {len(MACHINE_IDS)} machine IDs already present — skipping.")
            continue

        zip_name = f"{snr_prefix}_{mtype}.zip"
        url      = f"https://zenodo.org/api/records/{MIMII_ZENODO_RECORD}/files/{zip_name}/content"
        dest     = output_dir / zip_name

        print(f"\n  {zip_name} ({APPROX_ZIP_SIZES[mtype]})")
        if dest.exists():
            print("  Zip already present — skipping download.")
        else:
            _download(url, dest)

        _extract_and_flatten(dest, mtype, output_dir)
        dest.unlink()
        print(f"  Removed {zip_name}")

    _print_summary(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract one SNR variant of the MIMII dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snr", choices=list(SNR_TO_ROOT), default="-6dB",
        help="Which MIMII SNR variant to download.",
    )
    args = parser.parse_args()
    download_mimii(args.snr)


if __name__ == "__main__":
    main()
