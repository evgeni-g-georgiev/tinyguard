"""
download_mimii.py — Download the MIMII 6 dB zip files (approx 30 GB total).
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIMII_ARCHIVE_DIR


def _mimii_6db_files():
    """Return metadata for the MIMII 6 dB files that should be downloaded."""
    return [
        {
            "name": "6_dB_fan.zip",
            "url": "https://zenodo.org/api/records/3384388/files/6_dB_fan.zip/content",
            "md5": None,
            "size": "9.5 GB",
            "machine_type": "fan",
            "signal_to_noise_ratio": "6 dB",
        },
        {
            "name": "6_dB_pump.zip",
            "url": "https://zenodo.org/api/records/3384388/files/6_dB_pump.zip/content",
            "md5": None,
            "size": "7.1 GB",
            "machine_type": "pump",
            "signal_to_noise_ratio": "6 dB",
        },
        {
            "name": "6_dB_slider.zip",
            "url": "https://zenodo.org/api/records/3384388/files/6_dB_slider.zip/content",
            "md5": None,
            "size": "6.6 GB",
            "machine_type": "slider",
            "signal_to_noise_ratio": "6 dB",
        },
        {
            "name": "6_dB_valve.zip",
            "url": "https://zenodo.org/api/records/3384388/files/6_dB_valve.zip/content",
            "md5": None,
            "size": "6.4 GB",
            "machine_type": "valve",
            "signal_to_noise_ratio": "6 dB",
        },
    ]


def _archive_path(mimii_file):
    """Return the output path for a single archive."""
    return MIMII_ARCHIVE_DIR / mimii_file["name"]


def _print_file_summary(mimii_file):
    """Print a short summary for one file."""
    print(
        f"{mimii_file['name']} | "
        f"{mimii_file['size']} | "
        f"{mimii_file['machine_type']} | "
        f"{mimii_file['signal_to_noise_ratio']}"
    )


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
    """Verify one downloaded file if a checksum is available."""
    if expected_md5 is None:
        return True

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

def download_mimii():
    """Download all MIMII 6 dB files and return their local paths."""

    # Step 1: Create the output directory.
    _check_wget()
    MIMII_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: print a summary of what will be downloaded.
    print(f"\n{'-' * 60}")
    print("Downloading the MIMII dataset — 6 dB signal-to-noise ratio, 4 machine types (approx. 30 GB total)")
    print(f"Destination: {MIMII_ARCHIVE_DIR}/")
    print(f"{'-' * 60}\n")

    file_paths = []

    # Step 3: loop over each MIMII zip file and handle download or skip logic.
    for mimii_file in _mimii_6db_files():
        dest = _archive_path(mimii_file)
        _print_file_summary(mimii_file)

        # Step 4: if the file already exists, verify it before deciding to skip.
        if dest.exists():
            print("File already exists. Verifying the file.")
            if _verify_file(dest, mimii_file["md5"]):
                file_paths.append(dest)
                continue
            print("Checksum mismatch, re-downloading the file.")

        # Step 5: download the file if it is missing or failed verification.
        _download_file(mimii_file["url"], dest)

        # Step 6: verify the downloaded file before recording it as complete.
        if not _verify_file(dest, mimii_file["md5"]):
            print(f"ERROR: {dest.name} is corrupt. Delete the file and retry.")
            continue

        file_paths.append(dest)

    # Step 7: print the final status and return the downloaded file paths.
    print(f"\nAll files downloaded and saved to '{MIMII_ARCHIVE_DIR}/'.")
    return file_paths


if __name__ == "__main__":
    download_mimii()
