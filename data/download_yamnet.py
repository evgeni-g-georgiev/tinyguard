"""
download_yamnet.py — Download the YAMNet TFLite model.
"""

import kagglehub
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import YAMNET_PATH

MODEL_HANDLE = "google/yamnet/tfLite/classification-tflite/1"


def _find_tflite_file(model_dir):
    """Find the single .tflite file inside the downloaded model directory."""
    tflite_files = list(model_dir.rglob("*.tflite"))

    if len(tflite_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .tflite file in {model_dir}, found: {tflite_files}"
        )

    return tflite_files[0]

def download_yamnet():
    """Download the YAMNet TFLite model and save it to YAMNET_PATH."""

    # Step 1: create the output directory.
    YAMNET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 2: download the model package and locate the TFLite file.
    model_dir = Path(kagglehub.model_download(MODEL_HANDLE))
    source_tflite = _find_tflite_file(model_dir)

    # Step 4: copy the TFLite file to the project path.
    shutil.copy2(source_tflite, YAMNET_PATH)

    print(f"Saved TFLite file to: {YAMNET_PATH}")
    return YAMNET_PATH


if __name__ == "__main__":
    download_yamnet()

