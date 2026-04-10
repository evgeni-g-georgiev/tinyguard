""" 
Extract the downloaded MIMII 6 dB zip files.
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIMII_ARCHIVE_DIR, MIMII_ROOT, MIMII_TMP_DIR, MACHINE_TYPES, MACHINE_IDS


def _list_zip_files():
    """Return all downloaded MIMII zip files."""
    return sorted(MIMII_ARCHIVE_DIR.glob("*.zip"))


def _check_unzip():
    """Fail fast if unzip is not installed."""
    if shutil.which("unzip") is None:
        print("ERROR: unzip not found. Install it and retry.")
        sys.exit(1)


def _machine_type_from_zip(zip_path):
    """Infer machine type from a file like 6_dB_fan.zip."""
    return zip_path.stem.split("_")[-1]


def _all_machine_ids_present(machine_type):
    """Return True if all expected machine IDs already exist for one machine type."""
    for machine_id in MACHINE_IDS:
        path = MIMII_ROOT / machine_type / machine_id
        if not path.is_dir():
            return False
    return True


def _temp_dir_for_zip(zip_path):
    """Return the temporary extraction directory for one zip file."""
    return MIMII_TMP_DIR / zip_path.stem


def _extract_zip(zip_path, temp_dir):
    """Extract one zip file into a clean temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {zip_path.name} ...")
    subprocess.run(
        ["unzip", "-q", str(zip_path), "-d", str(temp_dir)],
        check=True,
    )


def _find_extracted_root(temp_dir):
    """Find the single top-level directory created by extraction."""
    top_dirs = [
        path for path in temp_dir.iterdir()
        if path.is_dir() and path.name != "__MACOSX"
    ]

    if len(top_dirs) != 1:
        raise RuntimeError(
            f"Expected one top-level directory in {temp_dir}, found: {top_dirs}"
        )

    return top_dirs[0]


def _move_machine_ids(src_root, machine_type):
    """Move extracted machine-id folders into the final MIMII layout."""
    extracted_paths = []

    for src in sorted(src_root.iterdir()):
        if not src.is_dir():
            continue

        dest = MIMII_ROOT / machine_type / src.name

        if dest.exists():
            print(f"    {machine_type}/{src.name} already exists — skipping move.")
            extracted_paths.append(dest)
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

        wav_count = _count_wavs(dest)
        print(f"    -> {machine_type}/{src.name} ({wav_count} WAVs)")
        extracted_paths.append(dest)

    return extracted_paths


def _clean_temp_dir(temp_dir):
    """Remove one temporary extraction directory."""
    shutil.rmtree(temp_dir, ignore_errors=True)


def _remove_zip_file(zip_path):
    """Delete a zip file after successful extraction."""
    zip_path.unlink()
    print(f"  Removed {zip_path.name} (extraction complete)")


def _count_wavs(path):
    """Count WAV files under a directory."""
    return sum(1 for _ in path.rglob("*.wav"))


def _print_summary():
    """Print a summary of the extracted MIMII dataset."""
    print(f"\n{'-' * 60}")
    print(f"Summary: {MIMII_ROOT}/")

    total_wavs = 0

    for machine_type in MACHINE_TYPES:
        for machine_id in MACHINE_IDS:
            path = MIMII_ROOT / machine_type / machine_id

            if path.is_dir():
                wav_count = _count_wavs(path)
                total_wavs += wav_count
                print(f"    {machine_type}/{machine_id}: {wav_count} WAVs")
            else:
                print(f"    {machine_type}/{machine_id}: MISSING")

    print(f"Total: {total_wavs} WAV files")
    print(f"{'-' * 60}\n")


def extract_mimii():
    """Extract all downloaded MIMII zip files into the final directory layout."""

    # Step 1: create the output directories.
    _check_unzip()
    MIMII_ROOT.mkdir(parents=True, exist_ok=True)
    MIMII_TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: discover the downloaded zip files.
    zip_files = _list_zip_files()
    if not zip_files:
        print(f"ERROR: no zip files found in {MIMII_ARCHIVE_DIR}/")
        print("Run: python -m data.download_mimii")
        sys.exit(1)

    print(f"\n{'-' * 60}")
    print("MIMII dataset — extraction")
    print(f"Source: {MIMII_ARCHIVE_DIR}/")
    print(f"Destination: {MIMII_ROOT}/")
    print(f"{'-' * 60}\n")

    extracted_paths = []

    # Step 3: extract each zip file unless that machine type is already complete.
    for zip_path in zip_files:
        machine_type = _machine_type_from_zip(zip_path)

        if _all_machine_ids_present(machine_type):
            print(
                f"{machine_type}: all {len(MACHINE_IDS)} machine IDs already present — skipping."
            )
            continue

        temp_dir = _temp_dir_for_zip(zip_path)
        print(f"\n  {zip_path.name}")

        _extract_zip(zip_path, temp_dir)

        try:
            src_root = _find_extracted_root(temp_dir)
            extracted_paths.extend(_move_machine_ids(src_root, machine_type))
        finally:
            _clean_temp_dir(temp_dir)

        # Step 4: remove the zip after successful extraction to save disk space.
        _remove_zip_file(zip_path)

    # Step 5: print a final summary and return the extracted directories.
    _print_summary()
    return extracted_paths


if __name__ == "__main__":
    extract_mimii()
