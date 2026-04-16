# gmm/config.py — Single source of truth for the GMM simulation.
#
# This file serves two roles:
#
#   1. Deployment fidelity: every constant has a 1-to-1 counterpart in
#      deployment/config.h.  The comment after each line cites the matching
#      C++ #define so the two files can be audited side-by-side.
#
#   2. Node Learning experiment parameters: NODE_R_A and NODE_R_B define the
#      fixed GWRP decay values for each node when running without --r-search.
#      R_CANDIDATES lists the values searched when --r-search is active; each
#      node independently selects the r that minimises its mean validation NLL.
#      See gmm/node_learning.py and gmm/train.py for the full background.

# ── Audio / spectrogram ───────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000        # SAMPLE_RATE
N_FFT           = 1024          # N_FFT
HOP_LENGTH      = 512           # HOP_LENGTH
N_MELS          = 64            # N_MELS  (64 bins: better-regularised at N=50; deployment path)
LOG_OFFSET      = 1e-6          # LOG_OFFSET
N_FRAMES        = 312           # N_FRAMES
CLIP_SECS       = 10            # CLIP_SECS

# ── Training split ────────────────────────────────────────────────────────────
N_TRAIN_CLIPS   = 60            # N_TRAIN_CLIPS
N_FIT_CLIPS     = 50            # N_FIT_CLIPS   (GMM fitted on these)
N_VAL_CLIPS     = 10            # N_VAL_CLIPS   (threshold calibrated on these)

# ── TWFR pooling ─────────────────────────────────────────────────────────────
# Fixed r values used by the standard (non-r-search) run.
# R is the single-node default (r=1 mean pooling, hardware baseline).
R               = 1.0           # R_CANDIDATES[] = { 1.0f }

# Candidates evaluated per node when --r-search is active.
# Mirrors the R_CANDIDATES[] array in deployment/config.h.
# r=0 → max pooling, r=1 → mean pooling, intermediate → GWRP.
R_CANDIDATES    = [0.0, 0.25, 0.5, 0.75, 1.0]

# ── GMM ───────────────────────────────────────────────────────────────────────
N_COMPONENTS    = 2             # N_COMPONENTS
MAX_EM_ITER     = 100           # MAX_EM_ITER
EM_TOL          = 1e-4          # EM_TOL
VARIANCE_FLOOR  = 1e-6          # VARIANCE_FLOOR
MIN_NK_FRAC     = 0.01          # MIN_NK_FRAC
SEED            = 42            # seed passed to fit_gmm()

# ── Detection / CUSUM ─────────────────────────────────────────────────────────
THRESHOLD_PCT   = 0.95          # THRESHOLD_PCT  (fraction, not integer)
CUSUM_H_SIGMA   = 5.0           # CUSUM_H_SIGMA
CUSUM_H_FLOOR   = 1.0           # CUSUM_H_FLOOR

# ── Monitoring display ────────────────────────────────────────────────────────
# Rolling mean window used for Serial diagnostics in tinyml_gmm.ino.
# Stored in event dicts for plotting; does NOT affect CUSUM detection.
ROLLING_WINDOW  = 5             # ROLLING_WINDOW (tinyml_gmm.ino)

# ── Node Learning ─────────────────────────────────────────────────────────────
# Fixed r values used when running without --r-search.
# NODE_R_A = 1.0: mean pooling — the only r feasible on Chip A (no buffer).
# NODE_R_B = 0.0: max pooling — deployable on Chip B with only a running max.
# With --r-search both nodes instead search over R_CANDIDATES independently.
NODE_R_A        = 1.0           # Node A: mean pooling (r=1 hardware baseline)
NODE_R_B        = 0.0           # Node B: max pooling  (r=0 deployable baseline)
