"""
extract_fsd50k.py — Extract the downloaded FSD50K eval zip files.
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FSD50K_ARCHIVE_DIR, FSD50K_AUDIO


def _extract_entry():
    """Return the zip file used as the 7z entry point."""
    return FSD50K_ARCHIVE_DIR / "FSD50K.eval_audio.zip"


def _split_zip_files():
    """Return the split zip files required for extraction."""
    return [
        FSD50K_ARCHIVE_DIR / "FSD50K.eval_audio.z01",
        FSD50K_ARCHIVE_DIR / "FSD50K.eval_audio.zip",
    ]


def _check_7z():
    """Fail fast if 7z is not installed."""
    if shutil.which("7z") is None:
        print("ERROR: 7z not found. Install it and retry.")
        sys.exit(1)


def _check_downloaded_files():
    """Fail if the required split zip files are missing."""
    missing_files = [path for path in _split_zip_files() if not path.exists()]

    if missing_files:
        print("ERROR: missing FSD50K download files:")
        for path in missing_files:
            print(f"  {path}")
        print("Run: python -m data.download_fsd50k")
        sys.exit(1)


def _extract_zip(zip_path, output_dir):
    """Extract the split zip using 7z."""
    print(f"Extracting {zip_path.name} ...")
    subprocess.run(
        ["7z", "x", str(zip_path), f"-o{output_dir}", "-y"],
        check=True,
    )


def _count_wavs(path):
    """Count WAV files under a directory."""
    return sum(1 for _ in path.rglob("*.wav"))


def _print_summary():
    """Print a short summary of the extracted FSD50K audio."""
    wav_count = _count_wavs(FSD50K_AUDIO)

    print(f"\n{'-' * 60}")
    print("FSD50K extraction complete.")
    print(f"Audio files: {wav_count} WAVs in {FSD50K_AUDIO}/")
    print(f"{'-' * 60}\n")


def extract_fsd50k():
    """Extract the downloaded FSD50K zip files."""

    # Step 1: check dependencies and required input files.
    _check_7z()
    _check_downloaded_files()
    FSD50K_AUDIO.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 60}")
    print("FSD50K eval set — extraction")
    print(f"Source: {FSD50K_ARCHIVE_DIR}/")
    print(f"Destination: {FSD50K_AUDIO.parent}/")
    print(f"{'-' * 60}\n")

    # Step 2: skip extraction if the final audio directory already exists.
    if FSD50K_AUDIO.is_dir():
        file_count = sum(1 for path in FSD50K_AUDIO.rglob("*") if path.is_file())
        print(f"{FSD50K_AUDIO.name}/ already exists ({file_count} files). Skipping extraction.")
        _print_summary()
        return FSD50K_AUDIO

    # Step 3: extract the split zip into the output directory.
    _extract_zip(_extract_entry(), FSD50K_AUDIO.parent)

    # Step 4: print a final summary and return the extracted directory.
    _print_summary()
    return FSD50K_AUDIO


if __name__ == "__main__":
    extract_fsd50k()
