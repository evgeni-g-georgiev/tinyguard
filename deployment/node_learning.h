// Two-node score fusion: fit-quality weights and CUSUM on fused val z-scores.
//
// nl_calibrate() is called once in SYNC after both nodes have finished their
// local TRAIN phase; nl_update() is called once per clip in MONITOR with both
// nodes' raw NLLs. Argument ordering is always (Node A, Node B).
#pragma once
#include <math.h>
#include <string.h>
#include <algorithm>
#include "config.h"

static float nl_w_a             = 0.5f;
static float nl_w_b             = 0.5f;
static float nl_cusum_k         = 0.0f;
static float nl_cusum_h         = 1.0f;
static float nl_cusum_S         = 0.0f;
static float nl_cusum_S_display = 0.0f;

inline void nl_calibrate(
    const float* val_nlls_a, float mu_a, float sigma_a,
    const float* val_nlls_b, float mu_b, float sigma_b)
{
    // Fit-quality softmax over -sigma_val / T; lower sigma_val → higher weight.
    float neg_sigma_a = -sigma_a / NL_TEMPERATURE;
    float neg_sigma_b = -sigma_b / NL_TEMPERATURE;
    float mx = neg_sigma_a > neg_sigma_b ? neg_sigma_a : neg_sigma_b;
    float ea = expf(neg_sigma_a - mx);
    float eb = expf(neg_sigma_b - mx);
    nl_w_a = ea / (ea + eb);
    nl_w_b = eb / (ea + eb);

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

// Feed one pair of NLLs into the fused CUSUM; returns true when the alarm fires.
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
