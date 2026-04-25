"""Repo-level paths and constants.

Audio and algorithm constants live in ``gmm/config.py``; this module re-exports
the ones used by repo-level scripts under the names they expect.
"""

from pathlib import Path
from gmm.config import (
    SAMPLE_RATE,
    N_FFT,
    LOG_OFFSET,
    HOP_LENGTH    as GMM_HOP_LENGTH,
    N_MELS        as GMM_N_MELS,
    CLIP_SECS     as MIMII_CLIP_SECS,
    SEED,
)

ROOT = Path(__file__).parent

# MIMII dataset roots (one per SNR level).
MIMII_NEG6DB_ROOT = ROOT / "data/mimii_neg6db"
MIMII_0DB_ROOT    = ROOT / "data/mimii_0db"
MIMII_6DB_ROOT    = ROOT / "data/mimii_6db"

# MIMII machine config.
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]
