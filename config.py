"""
config.py — Single source of truth for all paths and constants.

Every script in this repo imports from here. Paths are absolute (derived from
__file__) so scripts can be run from any working directory.
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_ARCHIVE_DIR = ROOT / "data/archives"
MIMII_ARCHIVE_DIR = DATA_ARCHIVE_DIR / "mimii_archives"
FSD50K_ARCHIVE_DIR = DATA_ARCHIVE_DIR / "fsd50k_archives"

FSD50K_AUDIO = ROOT / "data/fsd50k/FSD50K.eval_audio"
MIMII_ROOT = ROOT / "data/mimii"
MIMII_TMP_DIR = ROOT / "data/tmp/mimii"


# ── External models ───────────────────────────────────────────────────────────
YAMNET_PATH   = ROOT / "data/yamnet/yamnet.tflite"

# ── Outputs ───────────────────────────────────────────────────────────────────
FSDCACHE_DIR  = ROOT / "distillation/outputs/fsd50k_cache"
PCA_DIR       = ROOT / "distillation/outputs/pca"
MIMII_SPLITS  = ROOT / "preprocessing/outputs/mimii_splits/splits.json"
STUDENT_DIR   = ROOT / "distillation/outputs/student"
SEPARATOR_DIR = ROOT / "separator/outputs/separator"
INFERENCE_DIR = ROOT / "inference/outputs/inference"
GMM_DIR       = ROOT / "gmm/outputs/gmm"
GMM_N_MELS     = 128   # paper (Guan et al. 2023) uses 128 Mel-filter banks
GMM_HOP_LENGTH = 512   # 50 % overlap of N_FFT=1024, per paper implementation

# ── Audio constants ───────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
FRAME_LEN    = 15_600   # 0.975 s at 16 kHz — must stay identical across all stages
chunk_seconds = FRAME_LEN / SAMPLE_RATE
N_FFT        = 1024
HOP_LENGTH   = 256
N_MELS       = 64
LOG_OFFSET   = 1e-6

# ── YAMNet dequantisation ─────────────────────────────────────────────────────
EMB_IDX    = 115
EMB_SCALE  = 0.022328350692987442
EMB_ZP     = -128

# ── PCA / distillation ────────────────────────────────────────────────────────
PCA_DIMS   = 32

# ── MIMII machine config ──────────────────────────────────────────────────────
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

# ── SVDD / deployment simulation ──────────────────────────────────────────────
TRAIN_CLIPS     = 60     # 10 min ÷ 10 s per clip
MONITOR_CLIPS   = 30     # 5 min per monitoring window
N_ROUNDS        = 3
CLIP_SECS       = 10.0
THRESHOLD_PCT   = 95
SEED            = 42
FS_EPOCHS       = 50     # fixed epoch count for on-device SVDD training
