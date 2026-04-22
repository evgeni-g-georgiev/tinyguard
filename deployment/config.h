// config.h
#pragma once

// ── Node identity ─────────────────────────────────────────────────────
// Set NODE_ID before flashing each chip.
#define NODE_A  0
#define NODE_B  1
#define NODE_ID NODE_A   // ← change to NODE_B when flashing the second chip

// ── Node r values (TWFR pooling) ──────────────────────────────────────
#define NODE_R_A 1.0f    // mean pooling — streaming sum, no buffer needed
#define NODE_R_B 0.0f    // max pooling  — streaming max, no buffer needed

#if NODE_ID == NODE_A
  static const float R_NODE = NODE_R_A;
#else
  static const float R_NODE = NODE_R_B;
#endif

// ── Audio / spectrogram ───────────────────────────────────────────────
#define SAMPLE_RATE     16000
#define N_FFT           1024
#define HOP_LENGTH      512
#define N_MELS          64     // 64 bins: matches Python pipeline; fits BLE + streaming buffers
#define LOG_OFFSET      1e-6f

// Frames per 10-second clip (streaming, no centre-pad):
//   floor((160000 - N_FFT) / HOP_LENGTH) + 1 = 311
//   +1 for the initial hop whose left half is silence = 312
#define N_FRAMES        312
#define CLIP_SECS       10

// ── Training split ────────────────────────────────────────────────────
#define N_TRAIN_CLIPS   2
#define N_FIT_CLIPS     1      // GMM is fitted on these
#define N_VAL_CLIPS     1      // threshold is calibrated on these

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

// ── Node Learning ─────────────────────────────────────────────────────
#define SIGMA_FLOOR     1e-8f  // guards z-score division by sigma_val

// ── BLE ───────────────────────────────────────────────────────────────
#define ENABLE_BLE      1
