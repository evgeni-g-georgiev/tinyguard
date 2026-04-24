// PDM microphone capture. One hop of HOP_LENGTH samples per ISR.
#pragma once
#include <PDM.h>
#include "config.h"

static int16_t       pdm_buf[HOP_LENGTH];
static volatile bool hop_ready = false;

static void _on_pdm_data() {
    PDM.read(pdm_buf, sizeof(pdm_buf));
    hop_ready = true;
}

inline void audio_begin() {
    PDM.onReceive(_on_pdm_data);
    PDM.setBufferSize(HOP_LENGTH * sizeof(int16_t));
    PDM.setGain(50);
    if (!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println("PDM init failed");
        while (true);
    }
}

// Consume the hop-ready flag. Returns true once per completed hop.
inline bool audio_read_hop() {
    if (!hop_ready) return false;
    hop_ready = false;
    return true;
}
