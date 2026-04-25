# deployment/: Two-Node On-Device TWFR-GMM Detector

Arduino port of the [gmm/](../gmm/) Python pipeline. Runs on two
Arduino Nano 33 BLE Sense Rev 2 boards that exchange confidence signals
over Bluetooth LE. No pretrained weights. No data leaves the devices.

## Hardware

| | |
|---|---|
| Board | Arduino Nano 33 BLE Sense Rev 2 (one per node) |
| Processor | Nordic nRF52840, Arm Cortex-M4F at 64 MHz |
| SRAM | 256 KB |
| Flash | 1 MB |
| Microphone | Built-in MP34DT06JTR PDM mic |
| Interface | USB-Serial at 115200 baud |
| Alarm indicator | Built-in LED (pin 13) |

## State machine

Each board runs `COLLECT -> TRAIN -> SYNC -> MONITOR`:

- **COLLECT** captures 60 ten-second clips from the PDM mic. For each
  clip the full `mel_buf[N_MELS][N_FRAMES]` spectrogram is built hop by
  hop, then `compute_all_r_features()` extracts features for every
  candidate in `R_CANDIDATES`. The LED toggles once per clip.
- **TRAIN** fits a 2-component diagonal GMM for every candidate `r`,
  calibrates the threshold on 10 held-out clips, and keeps the one with
  the lowest mean val NLL.
- **SYNC** exchanges val statistics between the two nodes over BLE. If
  Node B picked the same `r` as Node A, it switches to the next-best
  candidate (greedy diversity, matching `simulation/`), re-fitting from
  the already-captured spectrograms. Both nodes then call
  `nl_calibrate()` to set up a fused CUSUM. If SYNC times out (30 s),
  each node falls back to solo mode using only its own CUSUM.
- **MONITOR** scores each new clip, exchanges NLLs over BLE, and fires
  the alarm from the fused CUSUM. Solo mode uses the local CUSUM.

## File reference

| File | Purpose |
|---|---|
| `tinyml_gmm.ino` | Top-level sketch. COLLECT/TRAIN/SYNC/MONITOR state machine and Serial output |
| `config.h` | All tunable constants: sample rate, FFT size, N_MELS, r grid, GMM + CUSUM parameters, BLE flag, node identity |
| `audio.h` | PDM microphone setup and interrupt-driven hop capture |
| `spectrogram.h` | Per-hop FFT + mel filterbank + log, filling `mel_buf[N_MELS][N_FRAMES]` |
| `mel_filterbank.h` | Pre-computed `MEL_FB[N_MELS][513]`. Regenerate with `export_mel_filterbank.py` |
| `features.h` | GWRP feature extraction for any r (max, mean, or sort + geometric weights) |
| `gmm.h` | Diagonal GMM: init + E-step + M-step (with collapse guard) + score_clip |
| `detector.h` | Post-fit calibration (k = max val NLL, plus CUSUM h) and `cusum_update()` |
| `node_learning.h` | Fit-quality softmax weights, z-score fusion, fused CUSUM calibration |
| `ble.h` | BLE GATT service. Node A is peripheral, Node B is central. Four characteristics for val-data and NLL exchange |
| `export_mel_filterbank.py` | Offline regeneration of `mel_filterbank.h` using librosa |

## Memory budget

With `N_MELS = 64`, `N_FRAMES = 312`, `N_R_CANDIDATES = 4`,
`N_TRAIN_CLIPS = 60`:

```
features[4][60][64]  = 60 KB
mel_buf[64][312]     = 78 KB
FFT scratch          = 12 KB
GMM state + misc     = a few KB
```

Well within the 256 KB SRAM on the board.

## Flashing

1. Install the Arduino IDE 2.x, the `Arduino Mbed OS Nano Boards`
   package, and the `arduinoFFT` library (version 2.x).
2. Open `deployment/tinyml_gmm.ino` in the IDE.
3. In `config.h`, set `NODE_ID` to `NODE_A` before flashing the first
   board, then to `NODE_B` before flashing the second.
4. If you changed `N_FFT` or `N_MELS` in `config.h`, regenerate the
   filterbank:
   ```
   python deployment/export_mel_filterbank.py
   ```
5. Pick `Tools -> Board -> Arduino Mbed OS Nano Boards -> Arduino Nano
   33 BLE`, select the correct port, and upload.
6. Open the Serial monitor at 115200 baud to follow the state machine.

## Serial output

During MONITOR each clip prints one line per board:

```
[A] score=143.221  rolling=151.034  S=0.000  S_fused=0.000
[A] score=512.884  rolling=241.318  S=325.561  S_fused=1.84  *** FUSED ANOMALY ***
```

The LED mirrors the current alarm state (fused when SYNCed, local when
solo).
