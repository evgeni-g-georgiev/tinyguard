from config import YAMNET_PATH, FSD50K_AUDIO, MIMII_ROOT, MACHINE_TYPES, MACHINE_IDS
from data.download_yamnet import download_yamnet
from data.download_fsd50k import download_fsd50k
from data.extract_fsd50k import extract_fsd50k
from data.download_mimii import download_mimii
from data.extract_mimii import extract_mimii


def _yamnet_ready():
    """Return True if the YAMNet file is already present."""
    return YAMNET_PATH.is_file()


def _fsd50k_ready():
    """Return True if FSD50K has already been extracted."""
    return FSD50K_AUDIO.is_dir() and any(FSD50K_AUDIO.rglob("*.wav"))


def _mimii_ready():
    """Return True if all expected MIMII machine folders already exist."""
    return all(
        (MIMII_ROOT / machine_type / machine_id).is_dir()
        for machine_type in MACHINE_TYPES
        for machine_id in MACHINE_IDS
    )


def setup_data():
    """Download and extract only the data assets that are still missing."""

    # Step 1: ensure YAMNet is present.
    if _yamnet_ready():
        print("YAMNet already present. Skipping.")
    else:
        download_yamnet()

    # Step 2: ensure FSD50K is fully prepared.
    if _fsd50k_ready():
        print("FSD50K already extracted. Skipping download and extraction.")
    else:
        download_fsd50k()
        extract_fsd50k()

    # Step 3: ensure MIMII is fully prepared.
    if _mimii_ready():
        print("MIMII already extracted. Skipping download and extraction.")
    else:
        download_mimii()
        extract_mimii()


if __name__ == "__main__":
    setup_data()
