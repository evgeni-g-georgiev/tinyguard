# data/

This purppose of this folder is to download the data used by the rest of the repository. 

## Execution

Execute the data downloading pipline from the repository root using

```bash
python -m data.setup_data
```

This downloads the raw FSD50K and MIMII datasets, and extract the datasets into the locations expected by preprocessing, distillation, training, and evaluation. It also downloads the YAMNet model. If specific data is already present it is not re-downloaded.

## File Tree

```text
data/
|-- README.md
|-- setup_data.py
|-- download_yamnet.py
|-- download_fsd50k.py
|-- extract_fsd50k.py
|-- download_mimii.py
|-- extract_mimii.py
├── archives/
│   ├── fsd50k_archives/
│   └── mimii_archives/
├── fsd50k/
│   └── FSD50K.eval_audio/
├── mimii/
│   └── {fan,pump,slider,valve}/
└── yamnet/
    └── yamnet.tflite
```

## Script Roles

- `setup_data.py`: top-level entrypoint for the full data setup pipeline.
- `download_yamnet.py`: downloads the YAMNet TFLite model used during preprocessing.
- `download_fsd50k.py`: downloads the raw FSD50K split zip files into `data/archives/fsd50k_archives/`.
- `extract_fsd50k.py`: extracts the downloaded FSD50K archive into `data/fsd50k/FSD50K.eval_audio/`.
- `download_mimii.py`: downloads the raw MIMII zip files into `data/archives/mimii_archives/`.
- `extract_mimii.py`: extracts the downloaded MIMII archives into `data/mimii/`.

## Data Layout

After setup, the folder structure is expected to look like this:

```text
data/
├── archives/
│   ├── fsd50k_archives/
│   │   ├── FSD50K.eval_audio.z01
│   │   └── FSD50K.eval_audio.zip
│   └── mimii_archives/
│       ├── 6_dB_fan.zip
│       ├── 6_dB_pump.zip
│       ├── 6_dB_slider.zip
│       └── 6_dB_valve.zip
├── fsd50k/
│   └── FSD50K.eval_audio/
│       └── *.wav
├── mimii/
│   ├── fan/
│   │   └── {id_00,id_02,id_04,id_06}/
│   │       ├── normal/*.wav
│   │       └── abnormal/*.wav
│   ├── pump/
│   ├── slider/
│   └── valve/
└── yamnet/
    └── yamnet.tflite
```


## Dataset Summary

### YAMNet

`data/yamnet/yamnet.tflite` is the pretrained YAMNet model used during offline preprocessing. It is loaded by the embedding extraction pipeline and is not part of the downstream anomaly-detection dataset itself.

Used by:
- `preprocessing/extract_embeddings.py`

### FSD50K

FSD50K provides the external audio corpus used to build the teacher-side preprocessing outputs. The extracted audio is used to create cached YAMNet embeddings and log-mel features that support the representation-learning and distillation stages.

Used by:
- `preprocessing/extract_embeddings.py`
- `preprocessing/compute_mels.py`
- `distillation/train.py`

### MIMII

MIMII is the machine-condition dataset used for the anomaly-detection experiments. After extraction, it is stored in a machine-type and machine-id directory layout and is used for dataset splitting, separator training, inference simulation, and GMM-based evaluation.

Used by:
- `preprocessing/split_mimii.py`
- `separator/train.py`
- `inference/run.py`
- `gmm/train.py`
- `gmm/evaluate.py`
