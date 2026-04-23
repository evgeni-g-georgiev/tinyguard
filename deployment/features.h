// features.h — TWFR feature extraction for all r values.
//
// All r variants read from mel_buf[N_MELS][N_FRAMES] (spectrogram.h).
// During COLLECT: compute_all_r_features() fills features for every candidate.
// During MONITOR: compute_twfr() uses chosen_r selected by the r-search in TRAIN.
#pragma once
#include <algorithm>
#include <string.h>
#include "config.h"
#include "spectrogram.h"

// r-search grid — matches Python gmm/config.py R_CANDIDATES.
static const float R_CANDIDATES[N_R_CANDIDATES] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

// Set to R_NODE (config.h default) at boot; overwritten after TRAIN r-search.
static float chosen_r = R_NODE;

// Compute the N_MELS-dimensional TWFR feature from mel_buf for a given r.
//   r <= 0 : max pooling over time (max of each mel bin across all frames)
//   r >= 1 : mean pooling over time
//   0 < r < 1 : GWRP — sort each bin descending, apply geometric weights r^t / Z
// Matches gmm/features.py _gwrp() exactly.
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
        // GWRP: for each mel bin, sort frames descending then apply w_t = r^t / Z.
        float sorted[N_FRAMES];  // ~1.2 KB stack
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

// Compute features for all N_R_CANDIDATES r values in one pass over mel_buf.
// out[r_idx] receives the N_MELS feature vector for R_CANDIDATES[r_idx].
inline void compute_all_r_features(float out[N_R_CANDIDATES][N_MELS]) {
    for (int i = 0; i < N_R_CANDIDATES; i++)
        compute_feature_r(R_CANDIDATES[i], out[i]);
}

// Thin wrapper used by tinyml_gmm.ino MONITOR — uses chosen_r set after TRAIN.
inline void compute_twfr(float* out) {
    compute_feature_r(chosen_r, out);
}
