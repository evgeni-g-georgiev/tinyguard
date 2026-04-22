// features.h
#pragma once
#include "config.h"
#include "spectrogram.h"

// Extract the TWFR feature for this node's r value.
// Both r=1.0 (Node A, mean) and r=0.0 (Node B, max) are accumulated online
// in spectrogram.h — spectrogram_get_feature() returns the correct value for
// each node. No sort or full spectrogram buffer is required for either node.
inline void compute_twfr(float* feature_out) {
    spectrogram_get_feature(feature_out);
}
