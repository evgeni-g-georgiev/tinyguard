# Simulation

A lockstep evaluation of tinyGUARD across the 16 MIMII machines (4 types,
4 IDs each), at one SNR variant at a time. Configurable for 1 to 8
microphone channels per machine.

## Running

```bash
# 1. Download MIMII for one SNR (~30 GB extracted).
python data/download_mimii.py --snr 6dB     # also: 0dB, -6dB

# 2. Run the simulation. The first run for each SNR builds the splits.
python -m simulation.run_simulation
```

The split step is skipped on subsequent runs for the same SNR. To re-split,
delete `simulation/data/splits/<snr>/` and re-run, or invoke
`simulation.data.split_data` manually.

## Configuration

`simulation/configs/default.yaml` is the single source of truth for a run.
Two knobs cover most experiments:

```yaml
snr:      "6dB"           # 6dB | 0dB | -6dB
channels: [0, 4]          # one entry per node, mic indices 0..7
                          # 1 entry  -> single-node baseline
                          # >1 entry -> node learning
```

Each SNR lives in its own directory tree (`data/mimii_<snr>/`,
`simulation/data/splits/<snr>/`), so different SNRs coexist without
interfering. For the full set of options (GMM hyperparameters, plot toggles,
shuffle modes, t-SNE plots), see [configs/README.md](configs/README.md).

To run an experiment without overwriting `default.yaml`:

```bash
cp simulation/configs/default.yaml simulation/configs/my_experiment.yaml
python -m simulation.run_simulation --config simulation/configs/my_experiment.yaml
```

## Output

Each run creates a timestamped directory under `simulation/outputs/runs/`:

```
2026-04-26_10-22-15/
  config.yaml       verbatim copy of the config used
  results.json      per-node and per-group traces (scores, S_t, labels, alarms)
  summary.txt       metrics table (clip-level + block-level, per machine type)
  plots/
    grid.png        score timeline grid, one cell per (machine type, ID)
    grid_compare_single.png   same grid for the single-channel baseline
    ch<N>/          full-size per-node timeline plots
    fused/          per-machine fused-score plots (when channels > 1)
    compare_single/ baseline single-channel plots for NL-vs-independent comparison
    latent/         t-SNE and score-distribution grids (off by default)
```

Both clip-level metrics (AUC, precision, recall, F1) and block-level metrics
(detection rate, false-alarm rate, detection delay) are reported. The
block-level metrics treat each contiguous run of anomalous clips as a single
event, which matches the intended deployment behaviour.

## Module layout

```
simulation/
  run_simulation.py     entry point: reads YAML, builds nodes/groups, runs lockstep
  lockstep.py           calibration + evaluation phases (yields one TimestepResult per clip)
  metrics.py            ClipMetrics and BlockMetrics
  formatters.py         live per-timestep line + final results tables
  configs/              experiment YAMLs (default.yaml + variants)
  data/                 split_data.py + simulation_loader.py
  node/
    node.py             one mic channel + one fitted GMMDetector
    group.py            N nodes of one machine + sigma-weighted fusion
  reporting/            JSON results, summary.txt, timeline + latent plots
  outputs/              run artefacts (one timestamped subdir per run)
```

Latent t-SNE plots are slow and disabled by default. Enable them in the
config when needed:

```yaml
latent_plot:
  enabled:     true
  node_subset: [0, 3]    # which channels to plot; [] means all
```
