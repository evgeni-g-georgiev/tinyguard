#!/usr/bin/env python3
"""
download_mimii.py — Download and extract the MIMII dataset (6 dB SNR, all 4 machine types).

Downloads from Zenodo record 3384388. Each machine type is a separate zip (~6–10 GB).
Total download: ~30 GB.

Usage:
    python data/download_mimii.py

Requirements:
    unzip must be installed: sudo apt install unzip
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIMII_ROOT, MACHINE_TYPES, MACHINE_IDS

# ── config ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = str(MIMII_ROOT)

# Zenodo record 3384388 — MIMII 6 dB files.
# MD5 checksums can be verified at: https://zenodo.org/records/3384388
FILES = [
    {
        "name": "6_dB_fan.zip",
        "url":  "https://zenodo.org/api/records/3384388/files/6_dB_fan.zip/content",
        "md5":  None,   # fill in from Zenodo record page if strict verification is needed
        "size": "~9.5 GB",
        "machine_type": "fan",
    },
    {
        "name": "6_dB_pump.zip",
        "url":  "https://zenodo.org/api/records/3384388/files/6_dB_pump.zip/content",
        "md5":  None,
        "size": "~7.1 GB",
        "machine_type": "pump",
    },
    {
        "name": "6_dB_slider.zip",
        "url":  "https://zenodo.org/api/records/3384388/files/6_dB_slider.zip/content",
        "md5":  None,
        "size": "~6.6 GB",
        "machine_type": "slider",
    },
    {
        "name": "6_dB_valve.zip",
        "url":  "https://zenodo.org/api/records/3384388/files/6_dB_valve.zip/content",
        "md5":  None,
        "size": "~6.4 GB",
        "machine_type": "valve",
    },
]


# ── helpers ─────────────────────────────────────────────────────────────────

def check_unzip():
    result = subprocess.run(["which", "unzip"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: unzip not found. Install it with: sudo apt install unzip")
        sys.exit(1)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str):
    """Download with wget — resumes partial downloads automatically."""
    print(f"  Downloading to {dest} …")
    subprocess.run(
        ["wget", "--continue", "--progress=bar:force", "-O", dest, url],
        check=True,
    )


def verify(path: str, expected_md5: str | None) -> bool:
    if expected_md5 is None:
        return True   # skip verification if checksum not provided
    print(f"  Verifying {os.path.basename(path)} …", end=" ", flush=True)
    actual = md5_file(path)
    if actual == expected_md5:
        print("OK")
        return True
    print(f"FAILED\n  Expected: {expected_md5}\n  Got:      {actual}")
    return False


def extract_and_flatten(zip_path: str, machine_type: str):
    """
    Extract zip to a temp dir then move files into the expected layout:
      data/mimii/{machine_type}/{machine_id}/{normal,abnormal}/*.wav

    The MIMII zips contain an internal top-level folder (e.g. dev_data_fan_6dB/)
    which is stripped during extraction.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(dir=OUTPUT_DIR)
    try:
        print(f"  Extracting {os.path.basename(zip_path)} …")
        subprocess.run(["unzip", "-q", zip_path, "-d", tmp], check=True)

        # Find the top-level folder inside the zip (varies by machine type)
        top_dirs = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
        if len(top_dirs) != 1:
            raise RuntimeError(f"Expected one top-level dir in zip, found: {top_dirs}")
        src_root = os.path.join(tmp, top_dirs[0])

        # Move each machine_id folder into place
        for mid in os.listdir(src_root):
            src = os.path.join(src_root, mid)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(OUTPUT_DIR, machine_type, mid)
            if os.path.exists(dst):
                print(f"    {machine_type}/{mid} already exists — skipping move.")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            n_wav = sum(1 for _, _, fs in os.walk(dst) for f in fs if f.endswith(".wav"))
            print(f"    → {machine_type}/{mid}  ({n_wav} WAVs)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    check_unzip()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  MIMII dataset — 6 dB SNR, 4 machine types (~30 GB total)")
    print(f"  Destination: {OUTPUT_DIR}/")
    print(f"{'─' * 60}\n")

    for f in FILES:
        mtype   = f["machine_type"]
        dest    = os.path.join(OUTPUT_DIR, f["name"])

        # Check if this machine type is already fully extracted
        extracted_ids = [
            mid for mid in MACHINE_IDS
            if os.path.isdir(os.path.join(OUTPUT_DIR, mtype, mid))
        ]
        if len(extracted_ids) == len(MACHINE_IDS):
            print(f"  {mtype}: all {len(MACHINE_IDS)} machine IDs already present — skipping.")
            continue

        print(f"\n  {f['name']} ({f['size']})")

        if os.path.exists(dest):
            print(f"  Zip already present — skipping download.")
        else:
            download(f["url"], dest)

        if not verify(dest, f["md5"]):
            print(f"  ERROR: {f['name']} checksum mismatch. Delete and retry.")
            continue

        extract_and_flatten(dest, mtype)

        # Remove zip to save disk space after successful extraction
        os.remove(dest)
        print(f"  Removed {f['name']} (extraction complete)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Summary: {OUTPUT_DIR}/")
    total_wavs = 0
    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            path = os.path.join(OUTPUT_DIR, mtype, mid)
            if os.path.isdir(path):
                n = sum(1 for _, _, fs in os.walk(path) for f in fs if f.endswith(".wav"))
                total_wavs += n
                print(f"    {mtype}/{mid}: {n} WAVs")
            else:
                print(f"    {mtype}/{mid}: MISSING")
    print(f"  Total: {total_wavs} WAV files")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
