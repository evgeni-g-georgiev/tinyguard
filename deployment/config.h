// config.h
#pragma once

// ── Audio / spectrogram ───────────────────────────────────────────────
#define SAMPLE_RATE     16000
#define N_FFT           1024
#define HOP_LENGTH      512
#define N_MELS          128    // reduce to 64 when enabling BLE or r-search
#define LOG_OFFSET      1e-6f

// Frames per 10-second clip (streaming, no centre-pad):
//   floor((160000 - N_FFT) / HOP_LENGTH) + 1 = 311
//   +1 for the initial hop whose left half is silence = 312
#define N_FRAMES        312
#define CLIP_SECS       10

// ── Training split ────────────────────────────────────────────────────
#define N_TRAIN_CLIPS   60
#define N_FIT_CLIPS     50     // GMM is fitted on these
#define N_VAL_CLIPS     10     // threshold is calibrated on these

// ── r candidates (TWFR) ───────────────────────────────────────────────
// Baseline: single fixed r=1 (mean pooling — no sort needed).
// Future: add more values, ensure N_MELS=64 first.
static const float R_CANDIDATES[] = { 1.0f };
#define N_R_CANDIDATES  1

// ── GMM ───────────────────────────────────────────────────────────────
#define N_COMPONENTS    2
#define MAX_EM_ITER     100
#define EM_TOL          1e-4f
#define VARIANCE_FLOOR  1e-6f
#define MIN_NK_FRAC     0.01f  // reinitialise component if N_k < this × N

// ── Detection ─────────────────────────────────────────────────────────
#define THRESHOLD_PCT   0.95f  // 95th percentile of val NLLs
#define CUSUM_H_SIGMA   5.0f   // cusum_h = this × std(val_nlls)
#define CUSUM_H_FLOOR   1.0f   // minimum cusum_h (degenerate val set guard)

// ── BLE (disabled for baseline) ───────────────────────────────────────
#define ENABLE_BLE      0      // flip to 1 only after reducing N_MELS to 64
