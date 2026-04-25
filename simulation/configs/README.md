# Config reference

`default.yaml` is the single source of truth for a simulation run.
Copy it to create experiment variants:

```bash
cp simulation/configs/default.yaml simulation/configs/my_experiment.yaml
python -m simulation.run_simulation --config simulation/configs/my_experiment.yaml
```

## Key sections

Channel selection. One node per channel; greedy r-diversity within each machine:

```yaml
channels:    [0, 1, 2, 3, 4, 5, 6, 7]   # 1..8 entries, mic indices 0..7
                                         # len==1: single-node, no fusion
                                         # len>1:  one Group per machine
temperature: 100.0                       # softmax temp for fusion weights
```

GMM params. Passed straight to `gmm.detector.GMMDetector`:

```yaml
gmm:
  n_mels:        64           # 32 / 64 / 128
  n_components:  2            # K
  threshold_pct: 0.95         # k = 95th-percentile val NLL
  cusum_h_sigma: 5.0          # h = max(sigma_val * 5, cusum_h_floor)
  cusum_h_floor: 1.0
  seed:          42
```

Simulation runtime:

```yaml
simulation:
  warmup_count:    60           # 50 fit + 10 val (matches gmm/ defaults)
  shuffle_mode:    block_fixed  # random | block_random | block_fixed
  block_size:      10           # anomaly clips per block (block modes)
  block_interval:  10           # normal clips between blocks (block_fixed hint)
  seed:            42
```

Plot toggles:

```yaml
plot:
  show_per_node:           true   # per-node z-score overlays on group plots
  show_fused:              true   # fused trace on group plots
  show_cusum_accumulator:  true   # S_t line on per-node + group plots
  show_k_and_h_lines:      true   # horizontal k and h dashed lines
  compare_node_idx:        0      # which channel index to render in compare_single/
```

Latent t-SNE plots (off by default; slow):

```yaml
latent_plot:
  enabled:      false
  node_subset:  [0]              # which channel indices; [] means all
  perplexity:   30
  random_state: 42
```

SNR variant:

```yaml
snr: "-6dB"                     # 6dB | 0dB | -6dB
data:
  mimii_root:    data/mimii_{snr}            # {snr} expanded via _MIMII_SNR_DIR
  splits_dir:    simulation/data/splits/{snr}
  machine_types: [fan, pump, slider, valve]
  machine_ids:   [id_00, id_02, id_04, id_06]
```

Note: `snr` is a display string ("-6dB", "0dB", "6dB"). The `mimii_root`
template expands via a mapping ("-6dB" -> "neg6db", etc.) so paths resolve to
the on-disk dirs `data/mimii_neg6db/`, `data/mimii_0db/`, `data/mimii_6db/`.
The splits dir keeps the display form.

## Backward compatibility

Configs that still use `n_nodes: N` (instead of `channels: [...]`) keep
working. The loader treats it as `channels: [0, 1, ..., N-1]`.

## Block-level metrics

The simulation always reports both clip-level (per-clip alarm bool) and
block-level (per-anomaly-block, fires-at-all-inside-block) metrics:

- AUC, P / R / F1 at clip level
- bP / bR / bF1 at block level, plus mean detection lag and mean unflag time

Block-level recall is the closest analogue to gmm/'s "detection rate".
