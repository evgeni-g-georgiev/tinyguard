# inference/

Simulates the on-device monitoring lifecycle. For each machine, runs 3 rounds of 5-minute normal monitoring followed by 5-minute anomaly injection, using the pre-trained separator artefacts from `separator/train.py`.

The threshold is fixed from training — no future data is used. This matches real deployment, where the node has no access to future clips when deciding where to draw the line.

## Files

| File | Description |
|---|---|
| `run.py` | Loads separator artefacts, scores test clips, detects anomalies, plots timelines, aggregates results. |

## Usage

```bash
python inference/run.py
python inference/run.py --checkpoint distillation/outputs/student/acoustic_encoder.pt
```

## Inputs

| File | Description |
|---|---|
| `distillation/outputs/student/acoustic_encoder.pt` | Frozen AcousticEncoder |
| `separator/outputs/separator/{mtype}_{mid}.pt` | Trained SVDD artefacts (16 files, from `separator/train.py`) |
| `preprocessing/outputs/mimii_splits/splits.json` | Clip manifest — provides `test_normal` and `test_abnormal` paths |
| `data/mimii/` | MIMII WAV files |

## Outputs

| File | Description |
|---|---|
| `inference/outputs/inference/results.yaml` | Per-machine, per-round metrics (detection, false alarms, delay) |
| `inference/outputs/inference/{mtype}_{mid}.png` | Timeline plot per machine — anomaly scores over time with threshold, detection events, and false positives annotated |

## Scoring

Each 10-second clip yields ~10 frames. The clip score is:

```
score(clip) = max over frames of ||f_s(frame) − c||²
```

A clip is flagged as anomalous if `score > threshold`. Max-frame scoring catches brief anomalous events (e.g. a single knocky frame) that mean-pooling would dilute.

## Expected results (AcousticEncoder + SVDD, 16 machines × 3 rounds)

| Metric | Value |
|---|---|
| Detection rate | 97.9% |
| False alarm rate | 5.5% |
| Median detection delay | 20 s |
