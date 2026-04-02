# data/

Data acquisition. Downloads raw audio datasets and houses the YAMNet model used for preprocessing.

## Contents

| File / Folder | Description |
|---|---|
| `download_fsd50k.py` | Downloads the FSD50K eval set (~6.2 GB, 2-part zip) from Zenodo |
| `download_mimii.py` | Downloads the MIMII dataset at 6 dB SNR (~30 GB, 4 machine types) from Zenodo |
| `fsd50k/` | FSD50K eval audio — created by `download_fsd50k.py` |
| `mimii/` | MIMII audio — created by `download_mimii.py` |
| `yamnet/` | YAMNet TFLite model (pre-loaded) — used by `preprocessing/extract_embeddings.py` |

## Usage

```bash
python data/download_fsd50k.py   # → data/fsd50k/FSD50K.eval_audio/*.wav
python data/download_mimii.py    # → data/mimii/{fan,pump,slider,valve}/{id_*}/{normal,abnormal}/*.wav
```

Downloads resume automatically if interrupted. MIMII zips are deleted after extraction to save space.

## Output layout

```
data/
├── fsd50k/FSD50K.eval_audio/     ~10K WAV clips, 16 kHz mono
├── mimii/
│   └── {fan,pump,slider,valve}/
│       └── {id_00,id_02,id_04,id_06}/
│           ├── normal/*.wav
│           └── abnormal/*.wav
└── yamnet/
    ├── yamnet.tflite             4 MB float32 TFLite model
    └── yamnet_class_map.csv
```

## Notes

- FSD50K is used **only** to train the AcousticEncoder. MIMII is **never** used in training — it is reserved for evaluation.
- `data/yamnet/yamnet.tflite` is used offline during preprocessing and is **not** deployed to the Arduino.
