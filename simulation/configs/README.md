# Config reference

`default.yaml` is the source of truth for a simulation run. Copy it to
create experiment variants:

```bash
cp simulation/configs/default.yaml simulation/configs/my_experiment.yaml
python -m simulation.run_simulation --config simulation/configs/my_experiment.yaml
```

## Channels and fusion

One node per channel. With more than one node, the group uses
`softmax(-sigma_val / T)` to weight nodes by fit quality.

```yaml
channels:    [0, 1, 2, 3, 4, 5, 6, 7]   # 1..8 entries, mic indices 0..7
                                         # 1 entry  -> single-node, no fusion
                                         # >1 entry -> one Group per machine
temperature: 100.0                       # softmax temperature
                                         #   T=1   hard selection (collapse)
                                         #   T=100 near-equal, slight bias
                                         #   T->inf exactly equal weights
```

## GMM and detector

Passed straight to `gmm.detector.GMMDetector`:

```yaml
gmm:
  n_mels:        64           # 32 / 64 / 128
  n_components:  2
  cusum_h_sigma: 20.0         # h = max(sigma_val * 20, cusum_h_floor); k = max(val_nlls)
  cusum_h_floor: 1.0
  seed:          42
```

## Simulation loop

```yaml
simulation:
  warmup_count:    60           # 50 fit + 10 val (matches gmm/ defaults)
  shuffle_mode:    block_fixed  # random | block_random | block_fixed
  block_size:      10           # anomaly clips per block (block modes)
  block_interval:  10           # normal clips between blocks (block_fixed)
  manual_reset:    false        # if true, alarms latch until anomaly->normal boundary
  seed:            42
```

## Plots

```yaml
plot:
  show_per_node:           true   # per-node overlays on group plots
  show_fused:              true   # fused trace on group plots
  show_cusum_accumulator:  true   # S_t line on per-node and group plots
  show_k_and_h_lines:      true   # k and h reference lines
  compare_node_idx:        0      # which channel to render in compare_single/

latent_plot:
  enabled:      false             # t-SNE; slow
  node_subset:  [0]               # which channels; [] means all
  perplexity:   30
  random_state: 42
```

## SNR and data paths

`snr` is the display string. The path templates expand `{snr}` to the
on-disk directory name (`-6dB` -> `neg6db`, `0dB` -> `0db`, `6dB` -> `6db`),
which is why the raw MIMII directories use the suffix form.

```yaml
snr: "-6dB"                     # 6dB | 0dB | -6dB
data:
  mimii_root:    data/mimii_{snr}
  splits_dir:    simulation/data/splits/{snr}
  machine_types: [fan, pump, slider, valve]
  machine_ids:   [id_00, id_02, id_04, id_06]
```

## Metrics

Both clip-level and block-level metrics are reported every run. A "block" is
a contiguous run of anomalous clips, treated as a single event:

- Clip level: AUC, precision, recall, F1.
- Block level: detection rate, false-alarm rate, mean detection delay, mean
  unflag time.

## Backward compatibility

Configs with `n_nodes: N` (instead of `channels: [...]`) still work. The
loader treats it as `channels: [0, 1, ..., N-1]`.
