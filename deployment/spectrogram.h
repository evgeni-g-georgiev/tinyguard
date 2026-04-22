// spectrogram.h
#pragma once
#include <arduinoFFT.h>
#include <string.h>
#include "config.h"
#include "mel_filterbank.h"

// Overlap buffer and FFT scratch — global to avoid stack pressure.
static float audio_buf[N_FFT] = {0};
static float vReal[N_FFT];
static float vImag[N_FFT];

// Online accumulator — no full spectrogram buffer needed for either node.
//   Node A (r=1.0): running sum → divide by frame_count → mean pooling.
//   Node B (r=0.0): running max → copy directly          → max pooling.
// Both are streaming accumulators; neither requires N_MELS × N_FRAMES storage.
static float mel_accumulator[N_MELS];
static int   frame_count = 0;

static ArduinoFFT<float> FFT(vReal, vImag, N_FFT, (float)SAMPLE_RATE);

inline void spectrogram_reset() {
    memset(audio_buf, 0, sizeof(audio_buf));
#if NODE_ID == NODE_B
    for (int m = 0; m < N_MELS; m++) mel_accumulator[m] = -1e30f; // running max init
#else
    memset(mel_accumulator, 0, sizeof(mel_accumulator));           // running sum init
#endif
    frame_count = 0;
}

// Process one hop of audio. Returns true when the clip is complete.
inline bool spectrogram_process_hop(const int16_t* raw) {
    if (frame_count >= N_FRAMES) return true;

    // Slide overlap buffer left, append new samples.
    memmove(audio_buf, audio_buf + HOP_LENGTH, HOP_LENGTH * sizeof(float));
    for (int i = 0; i < HOP_LENGTH; i++)
        audio_buf[HOP_LENGTH + i] = (float)raw[i] / 32768.0f;

    // FFT.
    memcpy(vReal, audio_buf, N_FFT * sizeof(float));
    memset(vImag, 0,         N_FFT * sizeof(float));
    FFT.windowing(FFTWindow::Hann, FFTDirection::Forward);
    FFT.compute(FFTDirection::Forward);

    // Power spectrum in-place (bins 0..N_FFT/2).
    for (int b = 0; b <= N_FFT / 2; b++)
        vReal[b] = vReal[b] * vReal[b] + vImag[b] * vImag[b];

    // Mel filterbank + log → update accumulator.
    for (int m = 0; m < N_MELS; m++) {
        float mel_val = 0.0f;
        for (int b = 0; b <= N_FFT / 2; b++)
            mel_val += MEL_FB[m][b] * vReal[b];
        float log_mel = logf(mel_val + LOG_OFFSET);
#if NODE_ID == NODE_B
        mel_accumulator[m] = fmaxf(mel_accumulator[m], log_mel); // running max
#else
        mel_accumulator[m] += log_mel;                            // running sum
#endif
    }

    frame_count++;
    return (frame_count >= N_FRAMES);
}

// Call once after spectrogram_process_hop returns true.
// Node A: returns per-bin mean (sum / frame_count).
// Node B: returns per-bin max (already in accumulator).
inline void spectrogram_get_feature(float* feature_out) {
#if NODE_ID == NODE_B
    for (int m = 0; m < N_MELS; m++)
        feature_out[m] = mel_accumulator[m];
#else
    float inv = 1.0f / frame_count;
    for (int m = 0; m < N_MELS; m++)
        feature_out[m] = mel_accumulator[m] * inv;
#endif
}
