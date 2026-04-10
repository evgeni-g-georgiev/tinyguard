#!/usr/bin/env python3
"""
download_fsd50k.py — Download the FSD50K eval zip files (approx. 6.2 GB total).
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FSD50K_ARCHIVE_DIR


def _fsd50k_files():
    """Return metadata for the FSD50K files that should be downloaded."""
    return [
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

def _file_path(fsd50k_file):
    """Return the output path for a single FSD50K file."""
    return FSD50K_ARCHIVE_DIR / fsd50k_file["name"]


def _print_file_summary(fsd50k_file):
    """Print a short summary for one file."""
    print(f"{fsd50k_file['name']} | {fsd50k_file['size']}")


def _check_wget():
    """Fail fast if wget is not installed."""
    if shutil.which("wget") is None:
        print("ERROR: wget not found. Install it and retry.")
        sys.exit(1)


def _md5_file(path):
    """Compute the MD5 checksum for a file."""
    digest = hashlib.md5()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path, expected_md5):
    """Verify one downloaded file."""
    print(f"  Verifying {path.name} ...", end=" ", flush=True)
    actual_md5 = _md5_file(path)

    if actual_md5 == expected_md5:
        print("OK")
        return True

    print(f"FAILED\n  Expected: {expected_md5}\n  Got:      {actual_md5}")
    return False


def _download_file(url, dest):
    """Download one file with resume support and a progress bar."""
    print(f"  Downloading to {dest} ...")
    subprocess.run(
        ["wget", "--continue", "--progress=bar:force", "-O", str(dest), url],
        check=True,
    )


def download_fsd50k():
    """Download all FSD50K eval zip files and return their local paths."""

    # Step 1: create the output directory.
    _check_wget()
    FSD50K_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: print a summary of what will be downloaded.
    print(f"\n{'-' * 60}")
    print("Downloading the FSD50K eval set (approx. 6.2 GB total)")
    print(f"Destination: {FSD50K_ARCHIVE_DIR}/")
    print(f"{'-' * 60}\n")

    file_paths = []
    all_ok = True

    # Step 3: loop over each file and handle download or skip logic.
    for fsd50k_file in _fsd50k_files():
        dest = _file_path(fsd50k_file)
        _print_file_summary(fsd50k_file)

        # Step 4: if the file already exists, verify it before deciding to skip.
        if dest.exists():
            print("File already exists. Verifying the file.")
            if _verify_file(dest, fsd50k_file["md5"]):
                file_paths.append(dest)
                continue
            print("Checksum mismatch, re-downloading the file.")

        # Step 5: download the file if it is missing or failed verification.
        _download_file(fsd50k_file["url"], dest)

        # Step 6: verify the downloaded file before recording it as complete.
        if not _verify_file(dest, fsd50k_file["md5"]):
            print(f"ERROR: {dest.name} is corrupt. Delete the file and retry.")
            all_ok = False
            continue

        file_paths.append(dest)

    # Step 7: stop early if any file failed verification.
    if not all_ok:
        print("\nDownload failed. Fix the errors above before extraction.")
        sys.exit(1)

    print(f"\nAll files downloaded and saved to '{FSD50K_ARCHIVE_DIR}/'.")
    return file_paths


if __name__ == "__main__":
    download_fsd50k()
