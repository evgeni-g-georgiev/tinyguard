// tinyml_gmm.ino — Two-node on-device Node Learning for anomaly detection.
//
// State machine:  COLLECT → TRAIN → SYNC → MONITOR
//
//   COLLECT  Both nodes record N_TRAIN_CLIPS audio clips and extract features.
//   TRAIN    Both nodes independently fit a GMM and calibrate their CUSUM.
//   SYNC     Node A publishes val stats over BLE; Node B connects, reads them,
//            and calibrates the fused CUSUM (node_learning.h).
//   MONITOR  Each clip: Node A sends its NLL to Node B over BLE; Node B fuses
//            both scores and makes the alarm decision.
//
// Node identity is controlled by NODE_ID in config.h.
// Flash Node A with NODE_ID=NODE_A, then Node B with NODE_ID=NODE_B.
#include "config.h"
#include "audio.h"
#include "spectrogram.h"
#include "features.h"
#include "gmm.h"
#include "detector.h"
#include "ble.h"
#if NODE_ID == NODE_B
#include "node_learning.h"
#endif

// ── Training feature matrix ───────────────────────────────────────────────────
// [N_TRAIN_CLIPS][N_MELS] — one fixed r per node, so no r-candidates dimension.
// 60 × 64 × 4 = 15.4 KB.
static float features[N_TRAIN_CLIPS][N_MELS];

// ── Monitoring score history for rolling-window display ───────────────────────
#define ROLLING_WINDOW  5
static float score_history[ROLLING_WINDOW] = {0};
static int   history_idx   = 0;
static int   history_count = 0;

static float rolling_mean(float score) {
    score_history[history_idx] = score;
    history_idx = (history_idx + 1) % ROLLING_WINDOW;
    if (history_count < ROLLING_WINDOW) history_count++;
    float sum = 0.0f;
    for (int i = 0; i < history_count; i++) sum += score_history[i];
    return sum / history_count;
}

// ── Node B: val-data packet received during SYNC ──────────────────────────────
// Lifted to file scope so MONITOR can reference pkt.mu_val / pkt.sigma_val.
#if NODE_ID == NODE_B
static ValDataPacket pkt;
#endif

// ── State machine ─────────────────────────────────────────────────────────────
enum State { COLLECT, TRAIN, SYNC, MONITOR };
static State state    = COLLECT;
static int   clip_idx = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

#if NODE_ID == NODE_A
    Serial.println("=== TinyML GMM — Node A (r=1.0, mean pooling) ===");
#else
    Serial.println("=== TinyML GMM — Node B (r=0.0, max pooling, coordinator) ===");
#endif
    Serial.print("SRAM features : "); Serial.print((int)sizeof(features) / 1024); Serial.println(" KB");

    ble_begin();
    audio_begin();
    spectrogram_reset();

    Serial.println("COLLECT phase — 10 min training window starting now.");
}

void loop() {
    switch (state) {

    // ── COLLECT ───────────────────────────────────────────────────────────────
    case COLLECT: {
        if (!audio_read_hop()) break;

        bool clip_done = spectrogram_process_hop(pdm_buf);

        if (clip_done) {
            compute_twfr(features[clip_idx]);
            clip_idx++;

            Serial.print("  clip "); Serial.print(clip_idx);
            Serial.print("/"); Serial.println(N_TRAIN_CLIPS);
            digitalWrite(LED_BUILTIN, clip_idx % 2);

            if (clip_idx >= N_TRAIN_CLIPS) {
                state = TRAIN;
            } else {
                spectrogram_reset();
            }
        }
        break;
    }

    // ── TRAIN ─────────────────────────────────────────────────────────────────
    case TRAIN: {
        Serial.println("TRAIN phase — fitting GMM...");
        digitalWrite(LED_BUILTIN, LOW);

        fit_gmm(features, N_FIT_CLIPS, /*seed=*/42);
        calibrate(features + N_FIT_CLIPS);   // val clips [50..59]

        spectrogram_reset();
        history_count = 0; history_idx = 0;
        state = SYNC;
        break;
    }

    // ── SYNC ──────────────────────────────────────────────────────────────────
    case SYNC: {
#if NODE_ID == NODE_A
        ble_publish_val_data();
        Serial.println("SYNC — waiting for Node B to connect...");
        while (!ble_poll_connected()) {}
        // Keep polling for 5 s so Node B can discoverAttributes() and readValue().
        // Without this, Node A stops processing BLE events right when Node B needs them.
        unsigned long t_grace = millis();
        while (millis() - t_grace < 5000UL) { BLE.poll(); }
        Serial.println("SYNC complete. Entering MONITOR.");

#else // NODE_B
        Serial.println("SYNC — connecting to Node A...");
        if (!ble_connect()) {
            Serial.println("HALT: could not reach Node A.");
            while (true) {}
        }
        if (!ble_read_val_data(&pkt)) {
            Serial.println("HALT: failed to read val data from Node A.");
            while (true) {}
        }
        nl_calibrate(pkt.mu_val, pkt.sigma_val,
                     det_val_nlls, det_mu_val, det_sigma_val);
        Serial.println("SYNC complete. Entering MONITOR.");
#endif

        spectrogram_reset();
        state = MONITOR;
        break;
    }

    // ── MONITOR ───────────────────────────────────────────────────────────────
    case MONITOR: {
        if (!audio_read_hop()) break;

        bool clip_done = spectrogram_process_hop(pdm_buf);

        if (clip_done) {
            float feature[N_MELS];
            compute_twfr(feature);
            float nll = score_clip(feature);

#if NODE_ID == NODE_A
            bool  alarm = cusum_update(nll);
            float rmean = rolling_mean(nll);
            ble_send_nll(nll);

            Serial.print("[A] score="); Serial.print(nll, 3);
            Serial.print("  rolling="); Serial.print(rmean, 3);
            Serial.print("  S=");       Serial.print(det_cusum_S_display, 3);
            if (alarm) Serial.print("  [A-ALARM]");
            Serial.println();
            digitalWrite(LED_BUILTIN, alarm ? HIGH : LOW);

#else // NODE_B
            ble_poll();
            float rmean = rolling_mean(nll);

            bool local_alarm = cusum_update(nll);

            bool fused_alarm = false;
            if (ble_nll_fresh) {
                fused_alarm   = nl_update(ble_last_nll, pkt.mu_val, pkt.sigma_val,
                                          nll,          det_mu_val, det_sigma_val);
                ble_nll_fresh = false;
            }

            Serial.print("[B] score=");  Serial.print(nll, 3);
            Serial.print("  rolling=");  Serial.print(rmean, 3);
            Serial.print("  S_b=");      Serial.print(det_cusum_S_display, 3);
            Serial.print("  S_fused=");  Serial.print(nl_cusum_S_display, 3);
            if (fused_alarm) Serial.print("  *** FUSED ANOMALY ***");
            if (local_alarm) Serial.print("  [B-local]");
            Serial.println();
            digitalWrite(LED_BUILTIN, fused_alarm ? HIGH : LOW);
#endif

            spectrogram_reset();
        }
        break;
    }
    } // switch
}
