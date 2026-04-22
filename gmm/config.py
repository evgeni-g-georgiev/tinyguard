# gmm/config.py — Single source of truth for all GMM pipeline parameters.
#
# To experiment with the Python simulation, this is the only file you need
# to edit. Parameters are split into two sections:
#
#   DEPLOYMENT MIRROR  — every constant mirrors a #define in deployment/config.h
#                        (cited in the comment after each line).  If you change
#                        these you must update the C++ too to keep the simulation
#                        and firmware in sync.
#
#   EXPERIMENT PARAMS  — Python-simulation-only knobs with no C++ equivalent.
#                        Change these freely to explore different configurations.


# ══ DEPLOYMENT MIRROR ════════════════════════════════════════════════════════
# Keep in sync with deployment/config.h

# ── Audio / spectrogram ───────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000        # SAMPLE_RATE
N_FFT           = 1024          # N_FFT
HOP_LENGTH      = 512           # HOP_LENGTH
N_MELS          = 64            # N_MELS
LOG_OFFSET      = 1e-6          # LOG_OFFSET
N_FRAMES        = 312           # N_FRAMES
CLIP_SECS       = 10            # CLIP_SECS

# ── Training split ────────────────────────────────────────────────────────────
N_TRAIN_CLIPS   = 60            # N_TRAIN_CLIPS
N_FIT_CLIPS     = 50            # N_FIT_CLIPS   (GMM fitted on these)
N_VAL_CLIPS     = 10            # N_VAL_CLIPS   (threshold calibrated on these)

# ── Single-node r baseline ────────────────────────────────────────────────────
R               = 1.0           # R_CANDIDATES[] = { 1.0f }

# ── GMM ───────────────────────────────────────────────────────────────────────
N_COMPONENTS    = 2             # N_COMPONENTS
MAX_EM_ITER     = 100           # MAX_EM_ITER
EM_TOL          = 1e-4          # EM_TOL
VARIANCE_FLOOR  = 1e-6          # VARIANCE_FLOOR
MIN_NK_FRAC     = 0.01          # MIN_NK_FRAC

# ── Detection / CUSUM ─────────────────────────────────────────────────────────
THRESHOLD_PCT   = 0.95          # THRESHOLD_PCT  (fraction, not integer)
CUSUM_H_SIGMA   = 5.0           # CUSUM_H_SIGMA
CUSUM_H_FLOOR   = 1.0           # CUSUM_H_FLOOR


# ══ EXPERIMENT PARAMS ════════════════════════════════════════════════════════
# Python simulation only — no C++ equivalent, change freely

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED            = 42

# ── GWRP r-search grid ────────────────────────────────────────────────────────
# Values tried per node when running with --r-search.
# r=0 → max pooling, r=1 → mean pooling, intermediate → energy-weighted GWRP.
R_CANDIDATES    = [0.0, 0.25, 0.5, 0.75, 1.0]

# ── Node Learning defaults ────────────────────────────────────────────────────
# Fixed r values used when running without --r-search.
# NODE_R_A = 1.0: mean pooling — hardware-feasible on Node A (no sort buffer).
# NODE_R_B = 0.0: max pooling — hardware-feasible on Node B (running max only).
NODE_R_A        = 1.0
NODE_R_B        = 0.0

# ── Display ───────────────────────────────────────────────────────────────────
# Rolling mean window for Serial diagnostics (tinyml_gmm.ino) and timeline plots.
# Does NOT affect CUSUM detection logic.
ROLLING_WINDOW  = 5
