// features.h
#pragma once
#include <algorithm>
#include <string.h>
#include "config.h"
#include "spectrogram.h"

// These are only used when r < 1.0.  Kept here so the future r-search
// path compiles without structural changes.  At N_R_CANDIDATES=1, r=1.0f,
// neither array is ever accessed at runtime.
static float gwrp_w[N_FRAMES];
static float _sorted_row[N_FRAMES];

inline void compute_gwrp_weights(float r, int T) {
    if (r >= 1.0f) {
        float inv = 1.0f / T;
        for (int i = 0; i < T; i++) gwrp_w[i] = inv;
        return;
    }
    if (r <= 0.0f) {
        gwrp_w[0] = 1.0f;
        for (int i = 1; i < T; i++) gwrp_w[i] = 0.0f;
        return;
    }
    float sum = 0.0f, ri = 1.0f;
    for (int i = 0; i < T; i++) { gwrp_w[i] = ri; sum += ri; ri *= r; }
    for (int i = 0; i < T; i++) gwrp_w[i] /= sum;
}

inline void compute_twfr(float r, float* feature_out) {
    if (r >= 1.0f) {
        // r=1: mean pooling. Feature already accumulated online in spectrogram.h.
        // No sort needed — mean is order-invariant.
        spectrogram_get_feature(feature_out);
        return;
    }

    // r<1: full sort + GWRP required.
    // This path needs the full spectrogram buffer (not present in baseline).
    // To enable: reduce N_MELS to 64, restore spectrogram[N_MELS][N_FRAMES]
    // in spectrogram.h, and pass it here.
    // Left unimplemented in the baseline to keep SRAM within budget.
}
