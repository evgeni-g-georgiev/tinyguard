// detector.h
#pragma once
#include <math.h>
#include <algorithm>
#include "config.h"
#include "gmm.h"

static float det_threshold;
static float det_cusum_k;
static float det_cusum_h;
static float det_cusum_S         = 0.0f;
static float det_cusum_S_display = 0.0f;   // ← pre-reset value for printing

// Call after fit_gmm(), passing the N_VAL_CLIPS held-out feature vectors.
inline void calibrate(float val_X[][N_MELS]) {
    float nlls[N_VAL_CLIPS];
    for (int i = 0; i < N_VAL_CLIPS; i++)
        nlls[i] = score_clip(val_X[i]);

    // 95th percentile of N_VAL_CLIPS=10 scores.
    std::sort(nlls, nlls + N_VAL_CLIPS);
    int pct_idx = (int)(N_VAL_CLIPS * THRESHOLD_PCT);
    if (pct_idx >= N_VAL_CLIPS) pct_idx = N_VAL_CLIPS - 1;
    det_threshold = nlls[pct_idx];
    det_cusum_k   = det_threshold;

    // std(val_nlls) for CUSUM alarm height.
    float mean = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) mean += nlls[i];
    mean /= N_VAL_CLIPS;
    float var = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) {
        float d = nlls[i] - mean; var += d * d;
    }
    float std_nll = sqrtf(var / N_VAL_CLIPS);
    det_cusum_h = fmaxf(CUSUM_H_SIGMA * std_nll, CUSUM_H_FLOOR);

    det_cusum_S         = 0.0f;
    det_cusum_S_display = 0.0f;

    Serial.print("  threshold=");   Serial.println(det_threshold, 4);
    Serial.print("  cusum_k=");     Serial.println(det_cusum_k, 4);
    Serial.print("  cusum_h=");     Serial.println(det_cusum_h, 4);
}

// Feed one clip's score into the CUSUM accumulator.
// Returns true if the alarm fires (anomaly detected).
inline bool cusum_update(float score) {
    det_cusum_S = fmaxf(0.0f, det_cusum_S + score - det_cusum_k);
    det_cusum_S_display = det_cusum_S;   // ← capture BEFORE potential reset
    if (det_cusum_S >= det_cusum_h) {
        det_cusum_S = 0.0f;              // reset accumulator for re-detection
        return true;
    }
    return false;
}
