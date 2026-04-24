"""Shared constants for the Python pipeline and its C++ deployment mirror.

Values in the DEPLOYMENT MIRROR section must stay in sync with
deployment/config.h. Values in EXPERIMENT PARAMS are host-only.
"""

# ── Deployment mirror ────────────────────────────────────────────────────────
# Keep in sync with deployment/config.h.

# Audio / spectrogram
SAMPLE_RATE     = 16_000
N_FFT           = 1024
HOP_LENGTH      = 512
N_MELS          = 64
LOG_OFFSET      = 1e-6
N_FRAMES        = 312
CLIP_SECS       = 10

# Training split
N_TRAIN_CLIPS   = 60
N_FIT_CLIPS     = 50   # GMM fit
N_VAL_CLIPS     = 10   # threshold calibration

# Single-node baseline r
R               = 1.0

# GMM
N_COMPONENTS    = 2
MAX_EM_ITER     = 100
EM_TOL          = 1e-4
VARIANCE_FLOOR  = 1e-6
MIN_NK_FRAC     = 0.01

# Detection / CUSUM
THRESHOLD_PCT   = 0.95
CUSUM_H_SIGMA   = 5.0
CUSUM_H_FLOOR   = 1.0


# ── Experiment params ────────────────────────────────────────────────────────
# Host-only; change freely.

SEED            = 42

# r-search grid for --r-search
R_CANDIDATES    = [0.0, 0.25, 0.5, 0.75, 1.0]

# Fixed r values for single-run mode (no r-search)
NODE_R_A        = 1.0
NODE_R_B        = 0.0

# Window size for the diagnostic rolling mean in plots and Serial output
ROLLING_WINDOW  = 5
