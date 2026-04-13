# On-Device TWFR-GMM Anomaly Detector

A fully self-contained anomaly detector that runs on the **Arduino Nano 33 BLE Sense Rev 2**.
The device records 10 minutes of normal machine audio, trains a Gaussian Mixture Model entirely
on-chip, then continuously monitors for deviations. No pre-trained neural network weights are
required and no data ever leaves the device.

This is an Arduino port of the `gmm/` Python pipeline in this repository. It is **not** the
CNN encoder + Deep SVDD pipeline described in the top-level README — that pipeline requires
a pre-trained CNN stored in flash, which is a separate approach.

---

## Hardware

| | |
|---|---|
| Board | Arduino Nano 33 BLE Sense Rev 2 |
| Processor | Nordic nRF52840, Arm Cortex-M4F @ 64 MHz |
| SRAM | 256 KB |
| Flash | 1 MB |
| Microphone | MP34DT06JTR PDM microphone (on-board) |
| Interface | USB-Serial (Serial Monitor at 115200 baud) |
| Anomaly indicator | Built-in LED (pin 13) |

---

## What is deployed

The model is a **TWFR-GMM** (Time-Weighted Frequency Domain Representation + Gaussian Mixture
Model), from Guan et al., *arXiv:2305.03328 (2023)*. There are no neural network weights — the
entire model is derived from audio recorded by the device itself during the training phase.

**Feature extraction — TWFR**

Each 10-second audio clip is converted to a log-mel spectrogram (128 mel bins × 312 time frames).
This is compressed to a single 128-dimensional vector by pooling across the time axis using
a weighted ranking scheme (GWRP). The pooling weight `r ∈ [0, 1]` controls the blend between
max pooling (`r=0`, emphasises peak transients) and mean pooling (`r=1`, emphasises steady-state
energy). See [Current assumptions](#current-assumptions-and-limitations) for why `r` is fixed
at 1.0 in this baseline.

**Model — diagonal GMM**

A 2-component Gaussian Mixture Model with diagonal covariance is fitted on the training features
using the EM algorithm. Each new clip is scored as the negative log-likelihood under the
best-fitting component:

```
score = -max_k  log N(feature | mu_k, sigma2_k)
```

Higher scores indicate greater deviation from the learned normal distribution.

**Detection — CUSUM**

A Page-Hinkley CUSUM runs over the stream of clip scores:

```
S_t = max(0,  S_{t-1} + score_t - k)
alarm fires when S_t >= h
```

where `k` is the detection threshold (95th percentile of held-out validation clip scores) and
`h = 5 × std(validation scores)`. A single outlier clip raises `S_t` by at most a few sigma,
then decays; three or more consecutively anomalous clips drive `S_t` past `h`.

---

## On-device phases

The sketch runs a three-state machine:

### 1. COLLECT (~10 minutes)

The microphone captures audio continuously. Every 512 samples (one hop), the hop is appended
to an overlap buffer, a Hann-windowed FFT is computed, the power spectrum is projected through
the mel filterbank, and the log-mel value for each bin is added to a running accumulator.
After 312 hops (one complete 10-second clip), `compute_twfr()` reads the per-bin mean from
the accumulator and stores it as a 128-dimensional feature vector. This repeats for 60 clips.

The LED blinks with each completed clip to show progress.

### 2. TRAIN (seconds)

After all 60 clips are collected:

1. **r selection**: For each candidate `r` value (currently just `r=1.0`), the GMM is fit on
   the first 50 clips and the mean training log-likelihood is recorded. The `r` with the
   highest log-likelihood is kept. With one candidate this is a single fit.
2. **GMM fit**: EM runs for up to 100 iterations on the 50 fit clips (convergence tolerance 1e-4).
3. **Threshold calibration**: The remaining 10 clips (held out during fitting) are scored. The
   detection threshold is set at the 95th percentile of their NLL scores. The CUSUM alarm height
   `h` is set at 5 × std of those same scores. Using held-out clips is important: the GMM
   achieves artificially high likelihood on its own training clips, so calibrating the threshold
   on training clips would place it in a regime that real test clips never reach.

Threshold, `cusum_k`, and `cusum_h` are printed to Serial.

### 3. MONITOR (continuous)

Each new 10-second clip is scored. The CUSUM accumulator is updated. If it reaches `h`, the
LED lights and `*** ANOMALY ***` is printed. The accumulator resets after each alarm, allowing
re-detection if the anomaly continues.

---

## Current assumptions and limitations

The Python pipeline in `gmm/` supports a self-supervised search over 9 candidate `r` values and
can operate with any `r ∈ [0, 1]`. The following simplifications were made to fit within 256 KB
SRAM on the baseline deployment.

### r fixed at 1.0 (mean pooling only)

**What was cut**: The TWFR r-search and any `r < 1.0`.

**Why**: For `r < 1.0`, TWFR requires sorting the time-frame values for each mel bin, which
means the full spectrogram must be held in memory during extraction:

```
128 mel bins × 312 frames × 4 bytes = 159 744 bytes ≈ 156 KB
```

Adding this to the FFT scratch buffers (~12 KB) and the training feature matrix (~30 KB)
would exhaust 256 KB SRAM.

**Fix**: When `r = 1.0`, TWFR is mathematically identical to a per-bin time mean, which is
order-invariant. The mean is accumulated online frame-by-frame into `mel_accumulator[128]`
(512 bytes) with no full spectrogram buffer needed. The sort step in `features.h` is bypassed.

**Path to enabling**: Reduce `N_MELS` from 128 to 64 in `config.h`. The full spectrogram buffer
then costs 64 × 312 × 4 ≈ 78 KB, which fits alongside the other buffers. Update
`spectrogram.h` to store `float spectrogram[N_MELS][N_FRAMES]` and restore the write path
as documented in the comments there. Regenerate `mel_filterbank.h` accordingly.

### N_R_CANDIDATES = 1 (no r-search)

Direct consequence of locking `r = 1.0`. The r-search loop in `tinyml_gmm.ino` is present
and will iterate over `R_CANDIDATES[]` correctly once additional values are added, but with a
single candidate it degenerates to one GMM fit.

### BLE disabled (ENABLE_BLE 0)

The BLE stack consumes additional flash and SRAM. It is disabled in the baseline to leave
headroom. The `config.h` flag `ENABLE_BLE` should only be set to 1 after `N_MELS` has been
reduced to 64.

### Mel filterbank pre-computed offline

`mel_filterbank.h` contains the librosa mel filterbank as a hardcoded `const float` C array
(`MEL_FB[128][513]`, ~256 KB of flash). It is numerically identical to the Python pipeline.
This avoids runtime filter construction at the cost of a large flash constant. Regenerate it
with `export_mel_filterbank.py` any time `N_FFT` or `N_MELS` changes.

---

## Memory budget (r=1.0 baseline)

| Buffer | SRAM |
|---|---|
| `features[1][60][128]` — training feature matrix | ~30 KB |
| `audio_buf[1024]` + `vReal[1024]` + `vImag[1024]` — FFT scratch | ~12 KB |
| `mel_accumulator[128]` — online log-mel mean | 512 B |
| `resp[50][2]` — EM responsibilities (only during TRAIN) | 400 B |
| `gm_mu[2][128]`, `gm_sigma2[2][128]`, `gm_pi[2]`, `gm_lognorm[2]` | ~2 KB |
| `score_history[5]`, CUSUM state, misc | < 100 B |

| Constant | Flash |
|---|---|
| `MEL_FB[128][513]` — mel filterbank | ~256 KB |
| Sketch + Arduino runtime + libraries | ~varies |

---

## File reference

| File | Purpose |
|---|---|
| `config.h` | Single source of truth for all constants: sample rate, FFT size, N_MELS, training split sizes, r candidates, GMM parameters, CUSUM parameters, BLE flag |
| `audio.h` | PDM microphone setup and interrupt-driven hop capture. `audio_begin()` initialises the microphone; `audio_read_hop()` returns true when a new 512-sample hop is ready in `pdm_buf` |
| `spectrogram.h` | Frame-by-frame spectrogram processing. Each call to `spectrogram_process_hop()` runs the FFT, applies the mel filterbank, takes the log, and adds to the online accumulator. `spectrogram_get_feature()` returns the per-bin mean when the clip is complete |
| `mel_filterbank.h` | Pre-computed `const float MEL_FB[128][513]` array. Generated offline by `export_mel_filterbank.py` using librosa to ensure numerical parity with the Python pipeline |
| `features.h` | GWRP weight computation and `compute_twfr()`. For `r=1.0` delegates directly to `spectrogram_get_feature()` (no sort needed). For `r<1.0` the sort+weighted-dot-product path is stubbed out pending the N_MELS=64 upgrade |
| `gmm.h` | Complete GMM implementation: random initialisation, E-step (log-sum-exp stabilised responsibilities), M-step (weighted mean and variance with variance floor and collapsed-component guard), convergence check, and `score_clip()` (negative max-component log-likelihood per Eq. 3 of Guan et al.) |
| `detector.h` | Post-fit calibration (`calibrate()`: 95th-percentile threshold + CUSUM `k` and `h` from held-out validation clips) and online detection (`cusum_update()`: Page-Hinkley step, returns true on alarm) |
| `export_mel_filterbank.py` | Offline Python script. Calls `librosa.filters.mel` and writes the result as a C header. Re-run whenever `N_FFT` or `N_MELS` changes in `config.h` |
| `tinyml_gmm.ino` | Top-level Arduino sketch. Owns the COLLECT → TRAIN → MONITOR state machine, the rolling-mean display buffer, Serial output, and the LED alarm indicator |

---

## Quick-start

### Prerequisites

- **Arduino IDE 2.x** — download from [arduino.cc/en/software](https://www.arduino.cc/en/software)
- **Board package**: in Arduino IDE, go to *Tools → Board → Boards Manager*, search for
  `Arduino Mbed OS Nano Boards` and install it (this provides the Nano 33 BLE board definition
  and the bundled `PDM` library)
- **ArduinoFFT library** (version 2.x): in Arduino IDE, go to *Tools → Manage Libraries*,
  search for `arduinoFFT` by Enrique Condes and install version 2.x

### Upload and run

1. **Connect** the Arduino Nano 33 BLE Sense Rev 2 to your computer via USB.

2. **(Optional) Regenerate the mel filterbank** if you have changed `N_FFT` or `N_MELS` in
   `config.h`. From the repository root:
   ```
   python deployment/export_mel_filterbank.py
   ```
   This overwrites `deployment/mel_filterbank.h`. Skip this step if you have not changed those
   parameters — the pre-generated file in the repository is correct for the default settings.

3. **Open the sketch**: in Arduino IDE, *File → Open*, navigate to `deployment/tinyml_gmm.ino`.

4. **Select the board**: *Tools → Board → Arduino Mbed OS Nano Boards → Arduino Nano 33 BLE*.

5. **Select the port**: *Tools → Port*, choose the port that appears when the board is plugged in
   (e.g. `COM3` on Windows, `/dev/ttyACM0` on Linux, `/dev/cu.usbmodem...` on macOS).

6. **Upload**: click the upload button (right-arrow icon) or press `Ctrl+U`. Compilation takes
   approximately 30–60 seconds the first time.

7. **Open the Serial Monitor**: *Tools → Serial Monitor* (or `Ctrl+Shift+M`). Set the baud rate
   to **115200**. You should see:
   ```
   === TinyML GMM ===
   SRAM mel_accumulator : 512 bytes
   SRAM audio_buf (FFT) : 12 KB
   SRAM features        : 30 KB
   COLLECT phase — 10 min training window starting now.
   ```

8. **Place the board** near the machine you want to monitor (microphone facing the sound source).
   The LED blinks with each completed clip. After 60 clips (~10 minutes):
   ```
   TRAIN phase — fitting GMM...
     r=1.00 mean_ll=-42.1234
     EM converged at iter 23
     threshold=187.3421
     cusum_k=187.3421
     cusum_h=94.6710
   MONITOR phase — anomaly detection active.
   ```

9. **Monitor output**: each 10-second clip prints one line:
   ```
   score=143.221  rolling=151.034  S=0.000
   score=512.884  rolling=241.318  S=325.561  *** ANOMALY ***
   ```
   The LED (pin 13) lights when an anomaly alarm fires.

---

## Extending the baseline

### Enable r-search (r < 1.0)

1. In `config.h`, reduce `N_MELS` from 128 to 64.
2. In `spectrogram.h`, replace `mel_accumulator[N_MELS]` with
   `float spectrogram[N_MELS][N_FRAMES]` and rewrite `spectrogram_process_hop()` to write
   each hop's mel values into `spectrogram[m][frame_count]` rather than accumulating a sum.
   Implement `spectrogram_get_feature()` to copy a single row for the r=1 path.
3. In `features.h`, implement the `r < 1.0` path in `compute_twfr()`: sort each row of
   the spectrogram buffer, compute GWRP weights, and take the dot product.
4. In `config.h`, populate `R_CANDIDATES[]` with additional values (e.g. `{0.5f, 0.9f, 1.0f}`)
   and update `N_R_CANDIDATES` accordingly.
5. Update `features[N_R_CANDIDATES][N_TRAIN_CLIPS][N_MELS]` in `tinyml_gmm.ino`. With
   `N_MELS=64` and 3 r candidates: 3 × 60 × 64 × 4 = 46 KB — well within budget.
6. Regenerate `mel_filterbank.h` with `export_mel_filterbank.py` (N_MELS=64, produces
   `MEL_FB[64][513]`, ~128 KB flash).

### Enable BLE

Set `ENABLE_BLE 1` in `config.h` **only after completing the N_MELS=64 step above**. Add
BLE advertisement and notification code in `tinyml_gmm.ino` to broadcast the anomaly score
or alarm state to a central node.
