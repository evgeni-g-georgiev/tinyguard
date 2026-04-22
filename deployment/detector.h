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
static float det_cusum_S_display = 0.0f;   // pre-reset value for printing

// Stored after calibrate() — used by node_learning.h for z-score fusion.
static float det_val_nlls[N_VAL_CLIPS];    // raw (unsorted) val NLL scores
static float det_mu_val    = 0.0f;         // mean(val_nlls)
static float det_sigma_val = 1.0f;         // std(val_nlls), floored at SIGMA_FLOOR

// Call after fit_gmm(), passing the N_VAL_CLIPS held-out feature vectors.
inline void calibrate(float val_X[][N_MELS]) {
    // Score all val clips.
    float nlls[N_VAL_CLIPS];
    for (int i = 0; i < N_VAL_CLIPS; i++)
        nlls[i] = score_clip(val_X[i]);

    // Store raw (unsorted) scores for node learning exchange.
    memcpy(det_val_nlls, nlls, sizeof(nlls));

    // mean and std of val NLLs.
    float mean = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) mean += nlls[i];
    mean /= N_VAL_CLIPS;
    float var = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) {
        float d = nlls[i] - mean; var += d * d;
    }
    float std_nll = sqrtf(var / N_VAL_CLIPS);

    det_mu_val    = mean;
    det_sigma_val = fmaxf(std_nll, SIGMA_FLOOR);

    // 95th percentile threshold (sort a copy).
    std::sort(nlls, nlls + N_VAL_CLIPS);
    int pct_idx = (int)(N_VAL_CLIPS * THRESHOLD_PCT);
    if (pct_idx >= N_VAL_CLIPS) pct_idx = N_VAL_CLIPS - 1;
    det_threshold = nlls[pct_idx];
    det_cusum_k   = det_threshold;
    det_cusum_h   = fmaxf(CUSUM_H_SIGMA * std_nll, CUSUM_H_FLOOR);

    det_cusum_S         = 0.0f;
    det_cusum_S_display = 0.0f;

    Serial.print("  threshold=");   Serial.println(det_threshold, 4);
    Serial.print("  cusum_k=");     Serial.println(det_cusum_k, 4);
    Serial.print("  cusum_h=");     Serial.println(det_cusum_h, 4);
    Serial.print("  mu_val=");      Serial.println(det_mu_val, 4);
    Serial.print("  sigma_val=");   Serial.println(det_sigma_val, 4);
}

// Feed one clip's score into the CUSUM accumulator.
// Returns true if the alarm fires (anomaly detected).
inline bool cusum_update(float score) {
    det_cusum_S = fmaxf(0.0f, det_cusum_S + score - det_cusum_k);
    det_cusum_S_display = det_cusum_S;
    if (det_cusum_S >= det_cusum_h) {
        det_cusum_S = 0.0f;
        return true;
    }
    return false;
}
