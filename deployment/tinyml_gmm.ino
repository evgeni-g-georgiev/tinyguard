// tinyml_gmm.ino
#include "config.h"
#include "audio.h"
#include "spectrogram.h"
#include "features.h"
#include "gmm.h"
#include "detector.h"

// ── Training feature matrix ───────────────────────────────────────────────────
// [N_R_CANDIDATES][N_TRAIN_CLIPS][N_MELS]
// Baseline: 1 × 60 × 128 × 4 = 30 KB.
// Future r-search at N_MELS=64: 5 × 60 × 64 × 4 = 75 KB.
static float features[N_R_CANDIDATES][N_TRAIN_CLIPS][N_MELS];

// ── Monitoring score history for rolling-window mean ─────────────────────────
// ROLLING_WINDOW = 5 (last 5 clip scores, 50 seconds of context).
#define ROLLING_WINDOW  5
static float score_history[ROLLING_WINDOW] = {0};
static int   history_idx = 0;
static int   history_count = 0;

static float rolling_mean(float score) {
    score_history[history_idx] = score;
    history_idx = (history_idx + 1) % ROLLING_WINDOW;
    if (history_count < ROLLING_WINDOW) history_count++;
    float sum = 0.0f;
    for (int i = 0; i < history_count; i++) sum += score_history[i];
    return sum / history_count;
}

// ── State machine ─────────────────────────────────────────────────────────────
enum State { COLLECT, TRAIN, MONITOR };
static State state     = COLLECT;
static int   clip_idx  = 0;   // which clip we're currently filling
static int   hops_seen = 0;   // hops received for the current clip

void setup() {
    Serial.begin(115200);
    while (!Serial);   // wait for serial monitor (remove for standalone use)

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);   // on = collecting

    Serial.println("=== TinyML GMM ===");
    Serial.print("SRAM mel_accumulator : ");
    Serial.print((int)sizeof(mel_accumulator)); Serial.println(" bytes");
    Serial.print("SRAM audio_buf (FFT) : ");
    Serial.print((int)(sizeof(audio_buf) + sizeof(vReal) + sizeof(vImag)) / 1024); Serial.println(" KB");
    Serial.print("SRAM features        : ");
    Serial.print((int)sizeof(features) / 1024); Serial.println(" KB");


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
        hops_seen++;

        if (clip_done) {
            // Extract TWFR feature for each r candidate and store.
            for (int ri = 0; ri < N_R_CANDIDATES; ri++) {
                compute_twfr(R_CANDIDATES[ri], features[ri][clip_idx]);
            }

            clip_idx++;
            Serial.print("  clip "); Serial.print(clip_idx);
            Serial.print("/"); Serial.println(N_TRAIN_CLIPS);

            // Blink LED to indicate progress.
            digitalWrite(LED_BUILTIN, clip_idx % 2);

            if (clip_idx >= N_TRAIN_CLIPS) {
                state = TRAIN;
            } else {
                spectrogram_reset();
                hops_seen = 0;
            }
        }
        break;
    }

    // ── TRAIN ─────────────────────────────────────────────────────────────────
    case TRAIN: {
        Serial.println("TRAIN phase — fitting GMM...");
        digitalWrite(LED_BUILTIN, LOW);

        // r-search: for each candidate, fit a GMM and record mean log-likelihood.
        // With N_R_CANDIDATES=1 this degenerates to a single fit.
        int best_ri   = 0;
        float best_ll = -1e30f;

        for (int ri = 0; ri < N_R_CANDIDATES; ri++) {
            // Fit on the first N_FIT_CLIPS features (last N_VAL_CLIPS held out).
            fit_gmm(features[ri], N_FIT_CLIPS, /*seed=*/42);

            // Score the fit clips to get mean log-likelihood for r selection.
            float ll = 0.0f;
            for (int n = 0; n < N_FIT_CLIPS; n++) {
                // score_clip returns NLL; negate for LL
                ll -= score_clip(features[ri][n]);
            }
            ll /= N_FIT_CLIPS;
            Serial.print("  r="); Serial.print(R_CANDIDATES[ri]);
            Serial.print(" mean_ll="); Serial.println(ll, 4);

            if (ll > best_ll) { best_ll = ll; best_ri = ri; }
        }

        // Refit on best r (if N_R_CANDIDATES=1 this is a no-op refit, could skip).
        if (N_R_CANDIDATES > 1) {
            Serial.print("  Best r="); Serial.println(R_CANDIDATES[best_ri]);
            fit_gmm(features[best_ri], N_FIT_CLIPS, 42);
        }

        // Calibrate threshold from held-out val clips.
        calibrate(features[best_ri] + N_FIT_CLIPS);

        // Switch to monitoring.
        spectrogram_reset();
        hops_seen = 0;
        history_count = 0; history_idx = 0;
        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("MONITOR phase — anomaly detection active.");
        state = MONITOR;
        break;
    }

    // ── MONITOR ───────────────────────────────────────────────────────────────
    case MONITOR: {
        if (!audio_read_hop()) break;

        bool clip_done = spectrogram_process_hop(pdm_buf);

        if (clip_done) {
            // Compute feature using best r (stored in gm_* globals after TRAIN).
            float feature[N_MELS];
            compute_twfr(R_CANDIDATES[0], feature);   // r fixed at baseline

            float nll  = score_clip(feature);
            float rmean = rolling_mean(nll);
            bool  alarm = cusum_update(nll);

            Serial.print("score="); Serial.print(nll, 3);
            Serial.print("  rolling="); Serial.print(rmean, 3);
            Serial.print("  S="); Serial.print(det_cusum_S_display, 3);
            if (alarm) Serial.print("  *** ANOMALY ***");
            Serial.println();

            digitalWrite(LED_BUILTIN, alarm ? HIGH : LOW);

            spectrogram_reset();
        }
        break;
    }
    } // switch
}
