# Data pipeline

## Download

```bash
python data/download_mimii.py --snr 6dB   # ~30 GB, one SNR at a time
python data/download_mimii.py --snr 0dB
python data/download_mimii.py --snr -6dB
```

Downloads from Zenodo record 3384388. Each SNR variant extracts to
`data/mimii_<dir>/{fan,pump,slider,valve}/{id_00,id_02,id_04,id_06}/`,
where `<dir>` is `neg6db`, `0db`, or `6db`.

Split

The split step is run automatically by `simulation/run_simulation.py`
when `simulation/data/splits/<snr>/` is missing. To run it manually:

```bash
python -m simulation.data.split_data --snr 6dB \
    --mimii-root data/mimii_6db --splits-dir simulation/data/splits/6dB
```

Produces `simulation/data/splits/<snr>/` with symlinks into the raw data:
simulation/data/splits/6dB/
fan/id_00/{warmup, test_normal, test_abnormal}/*.wav
...
surplus_abnormal/fan/id_00/*.wav

- warmup: calibration data (size truncated at runtime by simulation.warmup_count)
- test_normal / test_abnormal: balanced counts across all 16 nodes
- surplus_abnormal: leftover anomaly clips not used in test

Symlinks (not copies) are used so a fresh split is cheap.

Files

- split_data.py: planning + execution of the warmup/test split
- simulation_loader.py: loads splits into NodeTimeline objects with
  configurable shuffle modes (random, block_random, block_fixed)

---
