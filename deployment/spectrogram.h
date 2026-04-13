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

// Online mean accumulator (replaces spectrogram[N_MELS][N_FRAMES]).
// For r=1.0, TWFR = mean across time frames per mel bin.
// This is computed incrementally: no full spectrogram buffer needed.
// Savings vs full buffer: 128 × 312 × 4 = 156 KB.
//
// To restore full spectrogram storage (needed for r≠1.0):
//   1. Reduce N_MELS to 64 in config.h
//   2. Replace mel_accumulator with: float spectrogram[N_MELS][N_FRAMES]
//   3. In spectrogram_process_hop: write to spectrogram[:,frame_count] instead
static float mel_accumulator[N_MELS];
static int   frame_count = 0;

static ArduinoFFT<float> FFT(vReal, vImag, N_FFT, (float)SAMPLE_RATE);

inline void spectrogram_reset() {
    memset(audio_buf,       0, sizeof(audio_buf));
    memset(mel_accumulator, 0, sizeof(mel_accumulator));
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

    // Mel filterbank + log → add to accumulator.
    for (int m = 0; m < N_MELS; m++) {
        float mel_val = 0.0f;
        for (int b = 0; b <= N_FFT / 2; b++)
            mel_val += MEL_FB[m][b] * vReal[b];
        mel_accumulator[m] += logf(mel_val + LOG_OFFSET);
    }

    frame_count++;
    return (frame_count >= N_FRAMES);
}

// Call once after spectrogram_process_hop returns true.
// Writes the mean log-mel feature (mathematically identical to
// twfr_feature(spectrogram, r=1.0)) into feature_out[N_MELS].
inline void spectrogram_get_feature(float* feature_out) {
    float inv = 1.0f / frame_count;
    for (int m = 0; m < N_MELS; m++)
        feature_out[m] = mel_accumulator[m] * inv;
}
