// Shared constants for the on-device pipeline. Keep in sync with gmm/config.py.
#pragma once

// ── Node identity ────────────────────────────────────────────────────────────
// Set NODE_ID before flashing each board.
#define NODE_A  0
#define NODE_B  1
#define NODE_ID NODE_A

// Default r used before the TRAIN-phase r-search runs.
#define NODE_R_A 1.0f
#define NODE_R_B 0.0f

#if NODE_ID == NODE_A
  static const float R_NODE = NODE_R_A;
#else
  static const float R_NODE = NODE_R_B;
#endif

// ── Audio / spectrogram ──────────────────────────────────────────────────────
#define SAMPLE_RATE     16000
#define N_FFT           1024
#define HOP_LENGTH      512
#define N_MELS          64
#define LOG_OFFSET      1e-6f

// Frames per 10-second clip (streaming, no centre padding).
#define N_FRAMES        312
#define CLIP_SECS       10

// ── Training split ───────────────────────────────────────────────────────────
#define N_TRAIN_CLIPS   60
#define N_FIT_CLIPS     50
#define N_VAL_CLIPS     10

// ── GMM ──────────────────────────────────────────────────────────────────────
#define N_COMPONENTS    2
#define MAX_EM_ITER     100
#define EM_TOL          1e-4f
#define VARIANCE_FLOOR  1e-6f
#define MIN_NK_FRAC     0.01f  // reinit component when N_k < this * N

// ── Detection ────────────────────────────────────────────────────────────────
#define THRESHOLD_PCT   0.95f
#define CUSUM_H_SIGMA   5.0f
#define CUSUM_H_FLOOR   1.0f

// ── Node Learning ────────────────────────────────────────────────────────────
#define SIGMA_FLOOR     1e-8f
#define NL_TEMPERATURE  100.0f
#define N_R_CANDIDATES  4       // r grid: {0.5, 0.7, 0.9, 1.0}

// ── BLE ──────────────────────────────────────────────────────────────────────
#define ENABLE_BLE      1
