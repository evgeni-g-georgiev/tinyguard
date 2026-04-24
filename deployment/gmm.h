// Diagonal-covariance GMM fit by EM. score_clip returns -max_k log N(x|mu_k, Sigma_k).
#pragma once
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "config.h"

static float gm_mu      [N_COMPONENTS][N_MELS];
static float gm_sigma2  [N_COMPONENTS][N_MELS];
static float gm_pi      [N_COMPONENTS];
static float gm_lognorm [N_COMPONENTS];

// Responsibilities reused across EM iterations.
static float resp[N_FIT_CLIPS][N_COMPONENTS];

static const float LOG2PI = 1.8378770664f;

static void update_lognorm() {
    for (int k = 0; k < N_COMPONENTS; k++) {
        float s = (float)N_MELS * LOG2PI;
        for (int d = 0; d < N_MELS; d++) s += logf(gm_sigma2[k][d]);
        gm_lognorm[k] = -0.5f * s;
    }
}

// -0.5 * sum_d (x[d] - mu_k[d])^2 / sigma2_k[d]
static float quad(const float* x, int k) {
    float q = 0.0f;
    for (int d = 0; d < N_MELS; d++) {
        float diff = x[d] - gm_mu[k][d];
        q -= 0.5f * diff * diff / gm_sigma2[k][d];
    }
    return q;
}

static void gmm_init(float X[][N_MELS], int N, uint32_t seed) {
    srand(seed);

    int i0 = rand() % N;
    int i1;
    do { i1 = rand() % N; } while (i1 == i0);
    memcpy(gm_mu[0], X[i0], N_MELS * sizeof(float));
    memcpy(gm_mu[1], X[i1], N_MELS * sizeof(float));

    float mean_d[N_MELS] = {0};
    for (int n = 0; n < N; n++)
        for (int d = 0; d < N_MELS; d++)
            mean_d[d] += X[n][d];
    for (int d = 0; d < N_MELS; d++) mean_d[d] /= N;

    for (int d = 0; d < N_MELS; d++) {
        float var = 0.0f;
        for (int n = 0; n < N; n++) {
            float diff = X[n][d] - mean_d[d];
            var += diff * diff;
        }
        float v = fmaxf(var / N, VARIANCE_FLOOR);
        gm_sigma2[0][d] = gm_sigma2[1][d] = v;
    }

    for (int k = 0; k < N_COMPONENTS; k++) gm_pi[k] = 1.0f / N_COMPONENTS;
    update_lognorm();
}

static void e_step(float X[][N_MELS], int N) {
    for (int n = 0; n < N; n++) {
        float lp[N_COMPONENTS];
        for (int k = 0; k < N_COMPONENTS; k++)
            lp[k] = logf(gm_pi[k]) + gm_lognorm[k] + quad(X[n], k);

        float mx = lp[0];
        for (int k = 1; k < N_COMPONENTS; k++) if (lp[k] > mx) mx = lp[k];
        float sum_exp = 0.0f;
        for (int k = 0; k < N_COMPONENTS; k++) sum_exp += expf(lp[k] - mx);
        float log_sum = mx + logf(sum_exp);

        for (int k = 0; k < N_COMPONENTS; k++)
            resp[n][k] = expf(lp[k] - log_sum);
    }
}

static void m_step(float X[][N_MELS], int N) {
    for (int k = 0; k < N_COMPONENTS; k++) {
        float Nk = 0.0f;
        for (int n = 0; n < N; n++) Nk += resp[n][k];

        // Reinitialise collapsed component from a random sample.
        if (Nk < MIN_NK_FRAC * N) {
            int idx = rand() % N;
            memcpy(gm_mu[k], X[idx], N_MELS * sizeof(float));
            for (int d = 0; d < N_MELS; d++) gm_sigma2[k][d] = 1.0f;
            gm_pi[k] = 1.0f / N_COMPONENTS;
            continue;
        }

        for (int d = 0; d < N_MELS; d++) {
            float s = 0.0f;
            for (int n = 0; n < N; n++) s += resp[n][k] * X[n][d];
            gm_mu[k][d] = s / Nk;
        }

        for (int d = 0; d < N_MELS; d++) {
            float s = 0.0f;
            for (int n = 0; n < N; n++) {
                float diff = X[n][d] - gm_mu[k][d];
                s += resp[n][k] * diff * diff;
            }
            gm_sigma2[k][d] = fmaxf(s / Nk, VARIANCE_FLOOR);
        }

        gm_pi[k] = Nk / N;
    }
}

inline void fit_gmm(float X[][N_MELS], int N, uint32_t seed) {
    gmm_init(X, N, seed);

    float prev_ll = -1e30f;
    for (int iter = 0; iter < MAX_EM_ITER; iter++) {
        e_step(X, N);
        m_step(X, N);
        update_lognorm();

        float ll = 0.0f;
        for (int n = 0; n < N; n++) {
            float lp[N_COMPONENTS];
            for (int k = 0; k < N_COMPONENTS; k++)
                lp[k] = logf(gm_pi[k]) + gm_lognorm[k] + quad(X[n], k);
            float mx = lp[0];
            for (int k = 1; k < N_COMPONENTS; k++) if (lp[k] > mx) mx = lp[k];
            float sum_exp = 0.0f;
            for (int k = 0; k < N_COMPONENTS; k++) sum_exp += expf(lp[k] - mx);
            ll += mx + logf(sum_exp);
        }
        ll /= N;

        if (fabsf(ll - prev_ll) < EM_TOL) {
            Serial.print("  EM converged at iter "); Serial.println(iter);
            break;
        }
        prev_ll = ll;
    }
}

// Anomaly score; higher = more anomalous.
inline float score_clip(const float* feature) {
    float best = -1e30f;
    for (int k = 0; k < N_COMPONENTS; k++) {
        float lp = gm_lognorm[k] + quad(feature, k);
        if (lp > best) best = lp;
    }
    return -best;
}
