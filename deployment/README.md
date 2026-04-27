# Deployment

The on-device port of tinyGUARD. Two Arduino Nano 33 BLE Sense Rev 2 boards
each capture audio from their built-in microphone, fit a GMM during a
warm-up phase, then exchange confidence signals over Bluetooth LE while
monitoring. Data and models stay on the boards.

## Hardware

| | |
|---|---|
| Board | Arduino Nano 33 BLE Sense Rev 2 (one per node) |
| Processor | Nordic nRF52840, Arm Cortex-M4F at 64 MHz |
| SRAM | 256 KB |
| Flash | 1 MB |
| Microphone | Built-in MP34DT06JTR PDM mic |
| Serial | USB at 115200 baud |
| Alarm indicator | Built-in LED (pin 13) |

## Flashing

1. Install the Arduino IDE 2.x, the `Arduino Mbed OS Nano Boards` package,
   and the `arduinoFFT` library (version 2.x).
2. Open `deployment/tinyml_gmm.ino` in the IDE.
3. In `config.h`, set `NODE_ID` to `NODE_A` for the first board, then to
   `NODE_B` for the second. Each board needs the right ID before it is
   flashed.
4. Pick `Tools -> Board -> Arduino Mbed OS Nano Boards -> Arduino Nano
   33 BLE`, select the correct port, and upload.
5. Open the Serial monitor at 115200 baud to follow the state machine.

If you change `N_FFT` or `N_MELS` in `config.h`, regenerate the precomputed
mel filterbank that ships with the sketch:

```bash
python deployment/export_mel_filterbank.py
```

## State machine

Each board runs through four states in order:

1. **COLLECT.** Captures 60 ten-second clips from the PDM mic. For each
   clip the full `mel_buf[N_MELS][N_FRAMES]` log-mel spectrogram is built
   hop by hop, then `compute_all_r_features()` extracts a feature vector
   for every `r` in the candidate grid. The LED toggles once per clip so
   you can see progress.
2. **TRAIN.** Fits a two-component diagonal GMM for every candidate `r`,
   calibrates the threshold on the last 10 clips, and keeps the `r` with
   the lowest mean validation NLL.
3. **SYNC.** The two boards exchange validation statistics over BLE. If
   Node B picked the same `r` as Node A, it switches to the next-best
   candidate (greedy diversity, matching the simulator) and re-fits from
   the spectrograms it already captured. Both boards then call
   `nl_calibrate()` to set up the fused CUSUM. If SYNC times out
   (`SYNC_TIMEOUT_MS` in `config.h`, currently 3 minutes), each board
   falls back to solo mode and uses only its local CUSUM.
4. **MONITOR.** Each new clip is scored, NLLs are exchanged over BLE, and
   the alarm fires from the fused CUSUM. The LED mirrors the alarm state.

## Serial output

During MONITOR each clip prints one line per board:

```
[A] score=143.221  rolling=151.034  S=0.000  S_fused=0.000
[A] score=512.884  rolling=241.318  S=325.561  S_fused=1.84  *** FUSED ANOMALY ***
```

## File reference

| File | Purpose |
|---|---|
| `tinyml_gmm.ino` | Top-level sketch: state machine and Serial output |
| `config.h` | All tunable constants. Keep in sync with `gmm/config.py` |
| `audio.h` | PDM microphone setup and interrupt-driven hop capture |
| `spectrogram.h` | Per-hop FFT, mel filterbank, log, fills `mel_buf` |
| `mel_filterbank.h` | Pre-computed `MEL_FB[N_MELS][513]` |
| `features.h` | GWRP feature extraction for any `r` |
| `gmm.h` | Diagonal GMM: init, EM, score |
| `detector.h` | Threshold calibration and `cusum_update()` |
| `node_learning.h` | Fit-quality fusion weights, z-scoring, fused CUSUM |
| `ble.h` | BLE GATT service. Node A is peripheral, Node B is central |
| `export_mel_filterbank.py` | Offline regeneration of `mel_filterbank.h` |
