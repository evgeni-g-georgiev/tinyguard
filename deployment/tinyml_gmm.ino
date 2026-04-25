// Two-node on-device TWFR-GMM anomaly detector.
//
// State machine: COLLECT -> TRAIN -> SYNC -> MONITOR.
// Flash NODE_A on one board and NODE_B on the other (see config.h).
//
// Running a single board is supported: SYNC times out after SYNC_TIMEOUT_MS
// (set in config.h) and the board continues in solo mode using only its local
// CUSUM.
#include "config.h"
#include "audio.h"
#include "spectrogram.h"
#include "features.h"
#include "gmm.h"
#include "detector.h"
#include "node_learning.h"
#include "ble.h"

// [N_R_CANDIDATES][N_TRAIN_CLIPS][N_MELS] — ~60 KB at the defaults.
static float features[N_R_CANDIDATES][N_TRAIN_CLIPS][N_MELS];

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

static ValDataPacket partner_pkt;
static bool          ble_synced = false;   // true in fused mode, false in solo mode

// Per-candidate val NLL means, kept across SYNC so Node B can apply the
// diversity constraint without re-recording audio.
static float candidate_mu[N_R_CANDIDATES];

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

    // ── COLLECT ──────────────────────────────────────────────────────────────
    case COLLECT: {
        if (!audio_read_hop()) break;

        bool clip_done = spectrogram_process_hop(pdm_buf);

        if (clip_done) {
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

    // ── TRAIN ────────────────────────────────────────────────────────────────
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

        // Re-fit with the winning r so det_* globals are valid for SYNC and MONITOR.
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

    // ── SYNC ─────────────────────────────────────────────────────────────────
    case SYNC: {
#if NODE_ID == NODE_A
        ble_publish_val_data(det_mu_val, det_sigma_val, det_val_nlls, chosen_r);
        Serial.print("SYNC — advertising, waiting for Node B (up to ");
        Serial.print(SYNC_TIMEOUT_MS / 1000UL); Serial.println(" s)...");

        {
            unsigned long t_sync = millis();
            bool got_b = false;
            while (millis() - t_sync < SYNC_TIMEOUT_MS) {
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
        Serial.print("SYNC — connecting to Node A (up to ");
        Serial.print(SYNC_TIMEOUT_MS / 1000UL); Serial.println(" s)...");
        if (ble_connect()) {
            if (ble_read_val_data(&partner_pkt)) {
                // Greedy diversity: if Node B picked the same r as Node A, switch
                // to the best other candidate. Matches simulation/'s rule. With
                // N_R_CANDIDATES=4 and 2 nodes, a valid alternative always exists.
                {
                    float r_a = partner_pkt.chosen_r;
                    if (chosen_r == r_a) {
                        float best_div_mu  = 1e30f;
                        int   best_div_idx = -1;
                        for (int i = 0; i < N_R_CANDIDATES; i++) {
                            if (R_CANDIDATES[i] != r_a && candidate_mu[i] < best_div_mu) {
                                best_div_mu  = candidate_mu[i];
                                best_div_idx = i;
                            }
                        }
                        if (best_div_idx >= 0) {
                            Serial.print("  [diversity] r_A="); Serial.print(r_a, 2);
                            Serial.print("  r_B "); Serial.print(chosen_r, 2);
                            chosen_r = R_CANDIDATES[best_div_idx];
                            Serial.print(" -> "); Serial.println(chosen_r, 2);
                            fit_gmm(features[best_div_idx], N_FIT_CLIPS, /*seed=*/42);
                            calibrate(features[best_div_idx] + N_FIT_CLIPS);
                        }
                    }
                }

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

    // ── MONITOR ──────────────────────────────────────────────────────────────
    case MONITOR: {
#if NODE_ID == NODE_A
        ble_poll_nll_b();
#else
        ble_poll();
#endif

        if (!audio_read_hop()) break;

        bool clip_done = spectrogram_process_hop(pdm_buf);

        if (clip_done) {
            // Drop to solo mode when the partner disconnects.
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
    }
}
