// node_learning.h — Fused CUSUM calibration and per-clip z-score fusion.
//
// Included by BOTH Node A and Node B. Each node receives the partner's val
// stats over BLE during SYNC, then calls nl_calibrate() to set up the fused
// CUSUM. During MONITOR each node sends its own NLL to the partner and calls
// nl_update() once both NLLs are available.
//
// Mirrors gmm/node_learning.py exactly:
//   weights     = softmax(-[mu_val_a, mu_val_b] / NL_TEMPERATURE)   (T=100)
//   z_i(x)     = (NLL_i(x) - mu_val_i) / sigma_val_i
//   fused_score = w_a * z_a + w_b * z_b
//   CUSUM       = max(0, S + fused_score - cusum_k); alarm if S >= cusum_h
//
// Argument ordering: A always first, B always second in both nl_calibrate and nl_update.
#pragma once
#include <math.h>
#include <string.h>
#include <algorithm>
#include "config.h"

// Set by nl_calibrate() — read each clip in MONITOR by nl_update().
static float nl_w_a             = 0.5f;
static float nl_w_b             = 0.5f;
static float nl_cusum_k         = 0.0f;
static float nl_cusum_h         = 1.0f;
static float nl_cusum_S         = 0.0f;
static float nl_cusum_S_display = 0.0f;

// Call once in SYNC after both nodes have calibrated their GMMs.
//   val_nlls_a[N_VAL_CLIPS], mu_a, sigma_a — Node A's val stats
//   val_nlls_b[N_VAL_CLIPS], mu_b, sigma_b — Node B's val stats
// Matches gmm/node_learning.py __init__() lines 183-205.
inline void nl_calibrate(
    const float* val_nlls_a, float mu_a, float sigma_a,
    const float* val_nlls_b, float mu_b, float sigma_b)
{
    // Fit-quality softmax with temperature NL_TEMPERATURE.
    // Lower val NLL std → more consistent node → higher weight.
    float neg_mu_a = -sigma_a / NL_TEMPERATURE;
    float neg_mu_b = -sigma_b / NL_TEMPERATURE;
    float mx = neg_mu_a > neg_mu_b ? neg_mu_a : neg_mu_b;
    float ea = expf(neg_mu_a - mx);
    float eb = expf(neg_mu_b - mx);
    nl_w_a = ea / (ea + eb);
    nl_w_b = eb / (ea + eb);

    // Fused val z-scores using both nodes' val NLLs.
    float fused_z[N_VAL_CLIPS];
    float fz_mean = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) {
        float z_a  = (val_nlls_a[i] - mu_a) / fmaxf(sigma_a, SIGMA_FLOOR);
        float z_b  = (val_nlls_b[i] - mu_b) / fmaxf(sigma_b, SIGMA_FLOOR);
        fused_z[i] = nl_w_a * z_a + nl_w_b * z_b;
        fz_mean   += fused_z[i];
    }
    fz_mean /= N_VAL_CLIPS;

    float fz_var = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) {
        float d = fused_z[i] - fz_mean; fz_var += d * d;
    }
    float fz_std = sqrtf(fz_var / N_VAL_CLIPS);

    // 95th-percentile threshold on fused val z-scores (mirrors GMMDetector._calibrate).
    float sorted_z[N_VAL_CLIPS];
    memcpy(sorted_z, fused_z, sizeof(fused_z));
    std::sort(sorted_z, sorted_z + N_VAL_CLIPS);
    int pct_idx = (int)(N_VAL_CLIPS * THRESHOLD_PCT);
    if (pct_idx >= N_VAL_CLIPS) pct_idx = N_VAL_CLIPS - 1;

    nl_cusum_k = sorted_z[pct_idx];
    nl_cusum_h = fmaxf(CUSUM_H_SIGMA * fz_std, CUSUM_H_FLOOR);
    nl_cusum_S = 0.0f;

    Serial.print("  [NL] w_a=");   Serial.print(nl_w_a, 3);
    Serial.print("  w_b=");        Serial.print(nl_w_b, 3);
    Serial.print("  cusum_k=");    Serial.print(nl_cusum_k, 4);
    Serial.print("  cusum_h=");    Serial.println(nl_cusum_h, 4);
}

// Call once per clip in MONITOR when both nodes' NLLs are available.
// Returns true when the fused CUSUM alarm fires.
// A always first, B always second — matches nl_calibrate ordering.
inline bool nl_update(float nll_a, float mu_a, float sigma_a,
                      float nll_b, float mu_b, float sigma_b)
{
    float z_a   = (nll_a - mu_a) / fmaxf(sigma_a, SIGMA_FLOOR);
    float z_b   = (nll_b - mu_b) / fmaxf(sigma_b, SIGMA_FLOOR);
    float fused = nl_w_a * z_a + nl_w_b * z_b;

    nl_cusum_S = fmaxf(0.0f, nl_cusum_S + fused - nl_cusum_k);
    nl_cusum_S_display = nl_cusum_S;
    if (nl_cusum_S >= nl_cusum_h) {
        nl_cusum_S = 0.0f;
        return true;
    }
    return false;
}
