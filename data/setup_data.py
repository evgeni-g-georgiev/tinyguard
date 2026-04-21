from config import MIMII_ROOT, MACHINE_TYPES, MACHINE_IDS
from data.download_mimii import download_mimii
from data.extract_mimii import extract_mimii


def _mimii_ready():
    """Return True if all expected MIMII machine folders already exist."""
    return all(
        (MIMII_ROOT / machine_type / machine_id).is_dir()
        for machine_type in MACHINE_TYPES
        for machine_id in MACHINE_IDS
    )


def setup_data():
    """Download and extract MIMII data if not already present."""
    if _mimii_ready():
        print("MIMII already extracted. Skipping download and extraction.")
    else:
        download_mimii()
        extract_mimii()


if __name__ == "__main__":
    setup_data()
