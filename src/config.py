import os
from pathlib import Path

# Absolute path to the 'src' directory
SRC_DIR = Path(__file__).resolve().parent

# Absolute path to the project root (one level above src)
ROOT_DIR = SRC_DIR.parent

# Resource directory paths
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# Specific model paths (accessible from anywhere in the code)
FACE_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = MODELS_DIR / "haarcascade_eye.xml"
MOUTH_CASCADE_PATH = MODELS_DIR / "haarcascade_mcs_mouth.xml"

MODEL_XGB_LEFT = MODELS_DIR / "xgb_eye_left.pkl"
MODEL_XGB_RIGHT = MODELS_DIR / "xgb_eye_right.pkl"

FACE_CASCADE_PATH = ROOT_DIR / "models" / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = ROOT_DIR / "models" / "haarcascade_eye.xml"
MOUTH_CASCADE_PATH = ROOT_DIR / "models" / "haarcascade_mouth.xml"

def validate_paths():
    """Verify that critical models exist on startup."""
    critical_files = [FACE_CASCADE_PATH, MODEL_XGB_LEFT]
    for p in critical_files:
        if not p.exists():
            print(f"WARNING: Missing {p}")
