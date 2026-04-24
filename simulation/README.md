# Simulation

Lockstep simulation of N-node TWFR-GMM anomaly detection across the 16
MIMII machines (4 types × 4 IDs). Configurable for 1–8 microphone channels
per machine, with greedy r-diversity, σ-weighted fusion, and rich
clip-level + block-level metrics.

## Quickstart

```bash
# 1. Download MIMII data for one SNR (~30 GB per SNR)
python data/download_mimii.py --snr 6dB      # also: 0dB, -6dB

# 2. Run the simulation (auto-splits on first run)
python -m simulation.run_simulation
```

Step 1 is required; raw MIMII data must be on disk. The split step is
automatic — `simulation/data/splits/<snr>/` is created from `data/mimii_<snr>/`
on the first run for that SNR.

## Switching SNR or channels

Edit `simulation/configs/default.yaml`:

```yaml
channels: [1, 4, 6]   # 1..8 entries, mic indices 0..7
snr: "0dB"            # 6dB | 0dB | -6dB
```

Each SNR lives in its own directory (`data/mimii_<snr>/`,
`simulation/data/splits/<snr>/`), so SNRs coexist on disk.

## Output

Each run creates a timestamped directory under `simulation/outputs/runs/`:

```
simulation/outputs/runs/2026-04-25_00-14-23/
  config.yaml       — verbatim copy of the YAML used
  results.json      — per-node + per-group full traces (scores, S_t, labels, alarms)
  summary.txt       — human-readable metrics table (clip + block, per-type + overall)
  plots/
    grid.png        — N×4 score timeline grid (fused if len(channels) > 1)
    grid_compare_single.png — same grid for the baseline single channel
    ch<N>/          — per-node full-size timeline plots, one folder per channel
    fused/          — per-machine fused-z-score plots (when len(channels) > 1)
    compare_single/ — baseline single-channel plots for NL-vs-independent comparison
    latent/         — t-SNE + score-distribution grids (when latent_plot.enabled)
```

## Module layout

```
simulation/
  run_simulation.py     — entry point: reads YAML, builds nodes/groups, runs lockstep
  lockstep.py           — calibration + evaluation phases (yields TimestepResult)
  metrics.py            — pure metric computation (ClipMetrics, BlockMetrics)
  formatters.py         — terminal output (per-timestep live line + result tables)
  configs/              — YAML experiment configs (see configs/README.md)
  data/                 — split_data.py + simulation_loader.py (see data/README.md)
  node/
    node.py             — Node: one mic channel + one fitted GMMDetector
    group.py            — Group: N nodes of one machine + σ-weighted fusion
  reporting/            — JSON results, summary.txt, timeline + latent plots
  outputs/              — run artefacts (one timestamped subdir per run)
```

## Disabling slow steps

Latent t-SNE plots are off by default. To enable:

```yaml
latent_plot:
  enabled: true
  node_subset: [0, 3]   # which channels; [] = all
```
