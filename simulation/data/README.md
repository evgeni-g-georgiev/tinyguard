# Data pipeline

Two steps: download the raw MIMII data, then split it into per-machine
warmup and test sets. Step 2 runs automatically when you start a simulation
for an SNR for the first time.

## Manual split

```bash
python -m simulation.data.split_data --snr 6dB \
    --mimii-root data/mimii_6db --splits-dir simulation/data/splits/6dB
```

This produces `simulation/data/splits/<snr>/` with symlinks (not copies)
into the raw data:

```
simulation/data/splits/6dB/
  fan/id_00/
    warmup/         calibration data; truncated at runtime by warmup_count
    test_normal/    balanced count across all 16 nodes
    test_abnormal/  balanced count across all 16 nodes
  ...
  surplus_abnormal/ leftover anomaly clips not used in the test set
```

A fresh split is cheap because it is symlinks only.

## Files

- `split_data.py` plans and executes the warmup / test split.
- `simulation_loader.py` loads splits into `NodeTimeline` objects, with
  configurable shuffle modes (`random`, `block_random`, `block_fixed`).
