"""Download pretrained YAMNet model and class map to models/yamnet/.

Downloads:
  - yamnet.tflite  (~3.7 MB) — the frozen model for embedding extraction
  - yamnet_class_map.csv      — AudioSet 521 class labels (useful for sanity checks)

Usage:
    python scripts/download_yamnet.py
"""

import os
import sys
import urllib.request

DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "yamnet")

FILES = {
    "yamnet.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "audio_classifier/yamnet/float32/latest/yamnet.tflite"
    ),
    "yamnet_class_map.csv": (
        "https://raw.githubusercontent.com/tensorflow/models/"
        "master/research/audioset/yamnet/yamnet_class_map.csv"
    ),
}


def download_with_progress(url: str, dest: str):
    """Download a file with a progress bar."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Already exists ({size_mb:.1f} MB), skipping.")
        return

    print(f"  URL: {url}")

    # Get file size
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))

    # Download with progress
    downloaded = 0
    chunk_size = 64 * 1024  # 64 KB

    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total > 0:
                pct = downloaded / total * 100
                bar_len = 40
                filled = int(bar_len * downloaded / total)
                bar = "=" * filled + "-" * (bar_len - filled)
                sys.stdout.write(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)"
                )
            else:
                sys.stdout.write(f"\r  Downloaded {downloaded / 1e6:.1f} MB")
            sys.stdout.flush()

    print()  # newline after progress bar


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading YAMNet to {os.path.abspath(DEST_DIR)}/\n")

    for filename, url in FILES.items():
        dest = os.path.join(DEST_DIR, filename)
        print(f"[{filename}]")
        try:
            download_with_progress(url, dest)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            sys.exit(1)

    # Verify
    tflite_path = os.path.join(DEST_DIR, "yamnet.tflite")
    size_mb = os.path.getsize(tflite_path) / 1e6
    print(f"\nDone. Model size: {size_mb:.1f} MB")
    print(f"Files saved to: {os.path.abspath(DEST_DIR)}/")


if __name__ == "__main__":
    main()
