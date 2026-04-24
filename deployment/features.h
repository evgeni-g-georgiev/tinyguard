// TWFR feature extraction via Global Weighted Ranking Pooling over mel_buf.
#pragma once
#include <algorithm>
#include <string.h>
#include "config.h"
#include "spectrogram.h"

static const float R_CANDIDATES[N_R_CANDIDATES] = {0.5f, 0.7f, 0.9f, 1.0f};

// Set to R_NODE at boot; overwritten after the TRAIN-phase r-search.
static float chosen_r = R_NODE;

// Compute the N_MELS-dimensional TWFR feature for one r value.
//   r <= 0    per-bin max over time
//   r >= 1    per-bin mean over time
//   0 < r < 1 per-bin sort-descending + weights w_t = r^t / sum(r^t)
inline void compute_feature_r(float r, float* out) {
    if (r <= 0.0f) {
        for (int m = 0; m < N_MELS; m++) {
            float mx = mel_buf[m][0];
            for (int t = 1; t < N_FRAMES; t++) mx = fmaxf(mx, mel_buf[m][t]);
            out[m] = mx;
        }
    } else if (r >= 1.0f) {
        for (int m = 0; m < N_MELS; m++) {
            float s = 0.0f;
            for (int t = 0; t < N_FRAMES; t++) s += mel_buf[m][t];
            out[m] = s / N_FRAMES;
        }
    } else {
        float sorted[N_FRAMES];
        for (int m = 0; m < N_MELS; m++) {
            memcpy(sorted, mel_buf[m], N_FRAMES * sizeof(float));
            std::sort(sorted, sorted + N_FRAMES, std::greater<float>());
            float Z = 0.0f, val = 0.0f, w = 1.0f;
            for (int t = 0; t < N_FRAMES; t++) {
                val += w * sorted[t];
                Z   += w;
                w   *= r;
            }
            out[m] = val / Z;
        }
    }
}

// Compute features for every candidate in a single pass over mel_buf.
inline void compute_all_r_features(float out[N_R_CANDIDATES][N_MELS]) {
    for (int i = 0; i < N_R_CANDIDATES; i++)
        compute_feature_r(R_CANDIDATES[i], out[i]);
}

// MONITOR-phase entry point: extract with chosen_r.
inline void compute_twfr(float* out) {
    compute_feature_r(chosen_r, out);
}
