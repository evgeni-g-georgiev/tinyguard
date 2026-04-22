// node_learning.h — Fused CUSUM calibration and per-clip z-score fusion.
//
// Runs entirely on Node B (the coordinator). Node A's val stats and per-clip
// NLL scores arrive over BLE; Node B's come from detector.h globals.
//
// Mirrors gmm/node_learning.py exactly:
//   weights     = softmax(-[mu_val_a, mu_val_b])   (fit-quality weights)
//   z_i(x)     = (NLL_i(x) - mu_val_i) / sigma_val_i
//   fused_score = w_a * z_a + w_b * z_b
//   CUSUM       = max(0, S + fused_score - cusum_k); alarm if S >= cusum_h
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
//   mu_a, sigma_a            — received from Node A over BLE (8 bytes, fits in any MTU)
//   val_nlls_b[N_VAL_CLIPS], mu_b, sigma_b  — from det_val_nlls/det_mu_val/det_sigma_val
//
// Node A's val z-scores are ~N(0,1) on normal data. Their variance contribution
// (w_a²×1) is added theoretically so val_nlls_a never needs to cross BLE.
inline void nl_calibrate(
    float mu_a, float sigma_a,
    const float* val_nlls_b, float mu_b, float sigma_b)
{
    // Numerically stable softmax of (-mu_a, -mu_b) — better GMM fit → higher weight.
    float neg_mu_a = -mu_a;
    float neg_mu_b = -mu_b;
    float mx = neg_mu_a > neg_mu_b ? neg_mu_a : neg_mu_b;
    float ea = expf(neg_mu_a - mx);
    float eb = expf(neg_mu_b - mx);
    nl_w_a = ea / (ea + eb);
    nl_w_b = eb / (ea + eb);

    // Fused z-scores using Node B's val data only (Node A's z ~ N(0,1), mean contribution = 0).
    float fused_z[N_VAL_CLIPS];
    float fz_mean = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) {
        float z_b   = (val_nlls_b[i] - mu_b) / sigma_b;
        fused_z[i]  = nl_w_b * z_b;
        fz_mean    += fused_z[i];
    }
    fz_mean /= N_VAL_CLIPS;

    float fz_var = 0.0f;
    for (int i = 0; i < N_VAL_CLIPS; i++) {
        float d = fused_z[i] - fz_mean; fz_var += d * d;
    }
    // Add w_a² to account for Node A's theoretical N(0,1) z-score variance contribution.
    float fz_std = sqrtf(fz_var / N_VAL_CLIPS + nl_w_a * nl_w_a);

    // 95th-percentile threshold on fused val z-scores.
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

// Call once per clip in MONITOR. Returns true when fused alarm fires.
inline bool nl_update(float nll_a, float mu_a, float sigma_a,
                      float nll_b, float mu_b, float sigma_b)
{
    float z_a   = (nll_a - mu_a) / sigma_a;
    float z_b   = (nll_b - mu_b) / sigma_b;
    float fused = nl_w_a * z_a + nl_w_b * z_b;

    nl_cusum_S = fmaxf(0.0f, nl_cusum_S + fused - nl_cusum_k);
    nl_cusum_S_display = nl_cusum_S;
    if (nl_cusum_S >= nl_cusum_h) {
        nl_cusum_S = 0.0f;
        return true;
    }
    return false;
}
