#!/usr/bin/env python3
"""
download_fsd50k.py — Download and extract the FSD50K evaluation set (~6.2 GB).

Downloads to data/fsd50k/ alongside the MIMII data.
The eval set is a 2-part split zip; 7z is used to extract it.

Usage:
    python scripts/download_fsd50k.py

Requirements:
    7z must be installed: sudo apt install p7zip-full
"""

import hashlib
import os
import subprocess
import sys

# ── config ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = "data/fsd50k"

FILES = [
    {
        "name": "FSD50K.eval_audio.z01",
        "url":  "https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.z01/content",
        "md5":  "3090670eaeecc013ca1ff84fe4442aeb",
        "size": "3.2 GB",
    },
    {
        "name": "FSD50K.eval_audio.zip",
        "url":  "https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.zip/content",
        "md5":  "6fa47636c3a3ad5c7dfeba99f2637982",
        "size": "3.0 GB",
    },
]

EXTRACT_ENTRY = "FSD50K.eval_audio.zip"   # 7z uses the .zip as the entry point
EXTRACTED_DIR = "FSD50K.eval_audio"


# ── helpers ─────────────────────────────────────────────────────────────────

def check_7z():
    result = subprocess.run(["which", "7z"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: 7z not found. Install it with: sudo apt install p7zip-full")
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


def verify(path: str, expected_md5: str) -> bool:
    print(f"  Verifying {os.path.basename(path)} …", end=" ", flush=True)
    actual = md5_file(path)
    if actual == expected_md5:
        print("OK")
        return True
    print(f"FAILED\n  Expected: {expected_md5}\n  Got:      {actual}")
    return False


def extract(zip_path: str, output_dir: str):
    """Extract split zip using 7z. Automatically finds .z01 part."""
    print(f"  Extracting {os.path.basename(zip_path)} …")
    subprocess.run(
        ["7z", "x", zip_path, f"-o{output_dir}", "-y"],
        check=True,
    )


# ── main ────────────────────────────────────────────────────────────────────

def main():
    check_7z()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  FSD50K eval set  (~6.2 GB total, 2 parts)")
    print(f"  Destination: {OUTPUT_DIR}/")
    print(f"{'─' * 60}\n")

    all_ok = True
    for f in FILES:
        dest = os.path.join(OUTPUT_DIR, f["name"])

        # Skip download if file already exists with correct checksum
        if os.path.exists(dest):
            print(f"  {f['name']} ({f['size']}) — already present, verifying …")
            if verify(dest, f["md5"]):
                continue
            print("  Checksum mismatch — re-downloading.")

        print(f"\n  {f['name']} ({f['size']})")
        download(f["url"], dest)
        if not verify(dest, f["md5"]):
            all_ok = False
            print(f"  ERROR: {f['name']} is corrupt. Delete it and retry.")

    if not all_ok:
        print("\nDownload failed — fix errors above before extracting.")
        sys.exit(1)

    # ── Extract ───────────────────────────────────────────────────────────
    extracted_path = os.path.join(OUTPUT_DIR, EXTRACTED_DIR)
    if os.path.isdir(extracted_path):
        n = sum(1 for _, _, files in os.walk(extracted_path) for _ in files)
        print(f"\n  {EXTRACTED_DIR}/ already exists ({n} files) — skipping extraction.")
    else:
        print(f"\n  Extracting split zip …")
        extract(os.path.join(OUTPUT_DIR, EXTRACT_ENTRY), OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    extracted_path = os.path.join(OUTPUT_DIR, EXTRACTED_DIR)
    wav_count = sum(
        1 for _, _, files in os.walk(extracted_path)
        for f in files if f.endswith(".wav")
    )
    print(f"\n{'─' * 60}")
    print(f"  Done.")
    print(f"  Audio files: {wav_count} WAVs in {extracted_path}/")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
