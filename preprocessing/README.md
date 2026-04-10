
# preprocessing/

This folder contains architecture files which implement reusable the preprocessing components: audio loading, mel-spectrogram construction, and YAMNet loading.


## Folder Structure

```text
preprocessing/
|- loader.py
|- mel_spectrogram.py
|- yamnet_loading.py
|- split_mimii.py
|- outputs/
|  |- mimii_splits/
|- README.md
````` 

## File Roles

| File | Role |
|---|---|
| `loader.py` | Shared audio-loading and chunking helpers. |
| `mel_spectrogram.py` | Shared mel-spectrogram construction helper used by the preprocessing pipeline. |
| `yamnet_loading.py` | Shared helper for loading the local YAMNet TFLite model. |

