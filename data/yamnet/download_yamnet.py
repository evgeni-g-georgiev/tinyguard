from pathlib import Path
import shutil
import sys

import kagglehub

for root_dir in (Path.cwd(), Path.cwd().parent):
    if (root_dir / "config.py").exists():
        sys.path.insert(0, str(root_dir))
        break
else:
    raise FileNotFoundError("config.py not found in current or parent directory")

from config import YAMNET_PATH

MODEL_HANDLE = "google/yamnet/tfLite/classification-tflite/1"


def download_yamnet():
    """Download the YAMNet TFLite model and save it to YAMNET_PATH."""

    YAMNET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Download the model and get the local cached directory path.
    model_dir = Path(kagglehub.model_download(MODEL_HANDLE))

    # Find all .tflite files inside the downloaded model directory.
    tflite_files = list(model_dir.rglob("*.tflite"))

    # Validate that a single .tflite file was found.
    if len(tflite_files) != 1:
        raise RuntimeError(f"Expected exactly one .tflite file in {model_dir}, found: {tflite_files}")

    # Copy the .tflite file to the target path used by the project.
    shutil.copy2(tflite_files[0], YAMNET_PATH)

    print(f"Saved TFLite file to: {YAMNET_PATH}")


if __name__ == "__main__":
    download_yamnet()