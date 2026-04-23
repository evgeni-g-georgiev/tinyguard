// tinyml_gmm.ino — Two-node on-device Node Learning for anomaly detection.
//
// State machine:  COLLECT → TRAIN → SYNC → MONITOR
//
//   COLLECT  Both nodes record N_TRAIN_CLIPS audio clips and extract features
//            for all N_R_CANDIDATES r values simultaneously (full mel_buf used).
//   TRAIN    Both nodes independently run r-search over {0,0.25,0.5,0.75,1},
//            fit the winning GMM, and calibrate their CUSUM.
//   SYNC     Nodes exchange val stats bidirectionally over BLE (30 s timeout).
//            On success both calibrate the fused CUSUM (node_learning.h).
//            On timeout each continues in solo mode — no HALT.
//   MONITOR  Each clip: nodes exchange NLL scores over BLE and independently
//            compute the fused alarm.  Falls back to local CUSUM if partner
//            disconnects mid-session.
//
// Node identity is controlled by NODE_ID in config.h.
// Flash Node A with NODE_ID=NODE_A, then Node B with NODE_ID=NODE_B.
// Single-chip testing: flash either NODE_ID — SYNC times out after 30 s and
// the node enters solo monitoring using only its own CUSUM.
#include "config.h"
#include "audio.h"
#include "spectrogram.h"
#include "features.h"
#include "gmm.h"
#include "detector.h"
#include "node_learning.h"
#include "ble.h"

// ── Training feature matrix ───────────────────────────────────────────────────
// [N_R_CANDIDATES][N_TRAIN_CLIPS][N_MELS] — 5 × 60 × 64 × 4 = 75 KB.
static float features[N_R_CANDIDATES][N_TRAIN_CLIPS][N_MELS];

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

// ── Two-node state ────────────────────────────────────────────────────────────
static ValDataPacket partner_pkt;        // val stats received from partner in SYNC
static bool          ble_synced = false; // true → fused mode; false → solo mode

// Per-candidate val NLL means from the r-search loop — kept alive through SYNC so
// Node B can apply the diversity constraint without re-recording audio.
static float candidate_mu[N_R_CANDIDATES];

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
    Serial.println("=== TinyML GMM — Node A ===");
#else
    Serial.println("=== TinyML GMM — Node B ===");
#endif
    Serial.print("features : "); Serial.print((int)sizeof(features) / 1024); Serial.println(" KB");
    Serial.print("mel_buf  : "); Serial.print((int)sizeof(mel_buf)   / 1024); Serial.println(" KB");

    ble_begin();
    audio_begin();
    spectrogram_reset();

    Serial.println("COLLECT phase — recording 60 clips.");
}

void loop() {
    switch (state) {

    // ── COLLECT ───────────────────────────────────────────────────────────────
    case COLLECT: {
        if (!audio_read_hop()) break;

        bool clip_done = spectrogram_process_hop(pdm_buf);

        if (clip_done) {
            // Extract features for all r candidates from the completed mel_buf.
            float clip_all_r[N_R_CANDIDATES][N_MELS];
            compute_all_r_features(clip_all_r);
            for (int r_idx = 0; r_idx < N_R_CANDIDATES; r_idx++)
                memcpy(features[r_idx][clip_idx], clip_all_r[r_idx], N_MELS * sizeof(float));

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
        Serial.println("TRAIN phase — r-search...");
        digitalWrite(LED_BUILTIN, LOW);

        float best_mu    = 1e30f;
        int   best_r_idx = 0;

        for (int r_idx = 0; r_idx < N_R_CANDIDATES; r_idx++) {
            Serial.print("  r="); Serial.print(R_CANDIDATES[r_idx], 2);
            Serial.print("  fitting...");
            fit_gmm(features[r_idx], N_FIT_CLIPS, /*seed=*/42);
            calibrate(features[r_idx] + N_FIT_CLIPS);
            candidate_mu[r_idx] = det_mu_val;
            Serial.print("  mu_val="); Serial.println(det_mu_val, 4);
            if (det_mu_val < best_mu) { best_mu = det_mu_val; best_r_idx = r_idx; }
        }

        // Re-fit with the winning r to restore det_ globals for SYNC and MONITOR.
        chosen_r = R_CANDIDATES[best_r_idx];
        Serial.print("Best r="); Serial.print(chosen_r, 2);
        Serial.print("  mu_val="); Serial.println(best_mu, 4);
        fit_gmm(features[best_r_idx], N_FIT_CLIPS, /*seed=*/42);
        calibrate(features[best_r_idx] + N_FIT_CLIPS);

        spectrogram_reset();
        history_count = 0; history_idx = 0;
        state = SYNC;
        break;
    }

    // ── SYNC ──────────────────────────────────────────────────────────────────
    case SYNC: {
#if NODE_ID == NODE_A
        // Publish our val stats; wait up to 30 s for Node B to write back.
        ble_publish_val_data(det_mu_val, det_sigma_val, det_val_nlls, chosen_r);
        Serial.println("SYNC — advertising, waiting for Node B (30 s)...");

        {
            unsigned long t_sync = millis();
            bool got_b = false;
            while (millis() - t_sync < 30000UL) {
                if (ble_read_val_data_b(&partner_pkt)) { got_b = true; break; }
            }
            if (got_b) {
                // Short grace so Node B can subscribe to our NLL notifications.
                unsigned long t_grace = millis();
                while (millis() - t_grace < 2000UL) { BLE.poll(); }
                nl_calibrate(det_val_nlls,          det_mu_val,          det_sigma_val,
                             partner_pkt.val_nlls,  partner_pkt.mu_val,  partner_pkt.sigma_val);
                ble_synced = true;
                Serial.println("SYNC complete — fused mode.");
            } else {
                Serial.println("SYNC timeout — solo mode.");
            }
        }

#else // NODE_B
        Serial.println("SYNC — connecting to Node A (30 s)...");
        if (ble_connect()) {
            if (ble_read_val_data(&partner_pkt)) {
                // Diversity constraint: |r_B - r_A| >= 0.25 (mirrors Python train.py).
                // If violated, find the best r from the constrained candidate set and
                // re-train — features[5][60][64] is still in SRAM from TRAIN phase.
                {
                    float r_a = partner_pkt.chosen_r;
                    if (fabsf(chosen_r - r_a) < 0.25f) {
                        float best_div_mu  = 1e30f;
                        int   best_div_idx = -1;
                        for (int i = 0; i < N_R_CANDIDATES; i++) {
                            if (fabsf(R_CANDIDATES[i] - r_a) >= 0.25f &&
                                candidate_mu[i] < best_div_mu) {
                                best_div_mu  = candidate_mu[i];
                                best_div_idx = i;
                            }
                        }
                        if (best_div_idx >= 0 &&
                            fabsf(R_CANDIDATES[best_div_idx] - chosen_r) > 1e-3f) {
                            Serial.print("  [diversity] r_A="); Serial.print(r_a, 2);
                            Serial.print("  r_B "); Serial.print(chosen_r, 2);
                            chosen_r = R_CANDIDATES[best_div_idx];
                            Serial.print(" -> "); Serial.println(chosen_r, 2);
                            fit_gmm(features[best_div_idx], N_FIT_CLIPS, /*seed=*/42);
                            calibrate(features[best_div_idx] + N_FIT_CLIPS);
                        }
                    }
                }

                // Write our own val stats back to Node A (with final chosen_r).
                ValDataPacket my_pkt;
                my_pkt.mu_val    = det_mu_val;
                my_pkt.sigma_val = det_sigma_val;
                memcpy(my_pkt.val_nlls, det_val_nlls, N_VAL_CLIPS * sizeof(float));
                my_pkt.chosen_r  = chosen_r;
                ble_write_val_data(&my_pkt);

                nl_calibrate(partner_pkt.val_nlls, partner_pkt.mu_val, partner_pkt.sigma_val,
                             det_val_nlls,         det_mu_val,         det_sigma_val);
                ble_synced = true;
                Serial.println("SYNC complete — fused mode.");
            } else {
                Serial.println("SYNC: failed to read Node A val data — solo mode.");
            }
        } else {
            Serial.println("SYNC: Node A not found — solo mode.");
        }
#endif

        spectrogram_reset();
        state = MONITOR;
        break;
    }

    // ── MONITOR ───────────────────────────────────────────────────────────────
    case MONITOR: {
        // Poll BLE every iteration to keep the stack alive and process
        // incoming writes (Node A) or notifications (Node B).
#if NODE_ID == NODE_A
        ble_poll_nll_b();
#else
        ble_poll();
#endif

        if (!audio_read_hop()) break;

        bool clip_done = spectrogram_process_hop(pdm_buf);

        if (clip_done) {
            // Detect partner disconnect once per clip (avoids log spam).
            if (ble_synced && !ble_is_connected()) {
#if NODE_ID == NODE_A
                Serial.println("[A] Node B disconnected — solo mode.");
#else
                Serial.println("[B] Node A disconnected — solo mode.");
#endif
                ble_synced = false;
            }

            float feature[N_MELS];
            compute_twfr(feature);
            float nll         = score_clip(feature);
            bool  local_alarm = cusum_update(nll);
            float rmean       = rolling_mean(nll);

#if NODE_ID == NODE_A
            if (ble_synced) ble_send_nll(nll);

            bool fused_alarm = false;
            if (ble_synced && ble_nll_b_fresh) {
                fused_alarm     = nl_update(nll,            det_mu_val,         det_sigma_val,
                                            ble_last_nll_b, partner_pkt.mu_val, partner_pkt.sigma_val);
                ble_nll_b_fresh = false;
            }

            Serial.print("[A] score=");  Serial.print(nll, 3);
            Serial.print("  rolling=");  Serial.print(rmean, 3);
            Serial.print("  S=");        Serial.print(det_cusum_S_display, 3);
            if (ble_synced) {
                Serial.print("  S_fused="); Serial.print(nl_cusum_S_display, 3);
                if (fused_alarm)  Serial.print("  *** FUSED ANOMALY ***");
                if (local_alarm)  Serial.print("  [A-local]");
            } else {
                if (local_alarm)  Serial.print("  *** ANOMALY ***");
            }
            Serial.println();
            digitalWrite(LED_BUILTIN, (ble_synced ? fused_alarm : local_alarm) ? HIGH : LOW);

#else // NODE_B
            if (ble_synced) ble_send_nll_b(nll);

            bool fused_alarm = false;
            if (ble_synced && ble_nll_fresh) {
                fused_alarm   = nl_update(ble_last_nll, partner_pkt.mu_val, partner_pkt.sigma_val,
                                          nll,          det_mu_val,         det_sigma_val);
                ble_nll_fresh = false;
            }

            Serial.print("[B] score=");  Serial.print(nll, 3);
            Serial.print("  rolling=");  Serial.print(rmean, 3);
            Serial.print("  S=");        Serial.print(det_cusum_S_display, 3);
            if (ble_synced) {
                Serial.print("  S_fused="); Serial.print(nl_cusum_S_display, 3);
                if (fused_alarm)  Serial.print("  *** FUSED ANOMALY ***");
                if (local_alarm)  Serial.print("  [B-local]");
            } else {
                if (local_alarm)  Serial.print("  *** ANOMALY ***");
            }
            Serial.println();
            digitalWrite(LED_BUILTIN, (ble_synced ? fused_alarm : local_alarm) ? HIGH : LOW);
#endif

            spectrogram_reset();
        }
        break;
    }
    } // switch
}
