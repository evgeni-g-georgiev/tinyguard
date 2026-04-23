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

// Full per-clip mel spectrogram buffer — written hop by hop, read by features.h.
// Layout [N_MELS][N_FRAMES]: mel_buf[m][t] = log-mel energy for bin m at frame t.
// 64 × 312 × 4 = 78 KB.  Supports all r values (0, 0.25, 0.5, 0.75, 1.0).
static float mel_buf[N_MELS][N_FRAMES];
static int   mel_frame_idx = 0;

static ArduinoFFT<float> FFT(vReal, vImag, N_FFT, (float)SAMPLE_RATE);

inline void spectrogram_reset() {
    memset(audio_buf, 0, sizeof(audio_buf));
    mel_frame_idx = 0;
}

// Process one hop of audio. Returns true when the clip is complete (mel_frame_idx == N_FRAMES).
inline bool spectrogram_process_hop(const int16_t* raw) {
    if (mel_frame_idx >= N_FRAMES) return true;

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

    // Mel filterbank + log → store frame in mel_buf.
    for (int m = 0; m < N_MELS; m++) {
        float mel_val = 0.0f;
        for (int b = 0; b <= N_FFT / 2; b++)
            mel_val += MEL_FB[m][b] * vReal[b];
        mel_buf[m][mel_frame_idx] = logf(mel_val + LOG_OFFSET);
    }

    mel_frame_idx++;
    return (mel_frame_idx >= N_FRAMES);
}
