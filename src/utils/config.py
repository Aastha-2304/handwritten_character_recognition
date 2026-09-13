"""
Central configuration: paths, constants, and hyperparameters shared
across preprocessing, models, and the Streamlit app.
"""
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IAM_DIR = DATA_DIR / "iam"

MODELS_DIR = ROOT_DIR / "models"
CRNN_MODEL_DIR = MODELS_DIR / "crnn"
TROCR_MODEL_DIR = MODELS_DIR / "trocr"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

# --- TrOCR ---
TROCR_MODEL_NAME = "microsoft/trocr-small-handwritten"

# --- CRNN / CTC ---
IMG_HEIGHT = 32          # fixed input height for line images
IMG_MAX_WIDTH = 800      # max width after aspect-preserving resize
CHANNELS = 1             # grayscale

# Character set the CRNN can predict (CTC blank is index 0)
CHARSET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
BLANK_TOKEN = "<blank>"

# --- Training defaults (used later in notebooks/training script) ---
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50

for _dir in (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, IAM_DIR,
    CRNN_MODEL_DIR, TROCR_MODEL_DIR, CHECKPOINTS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)