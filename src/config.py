import os
from pathlib import Path

# Ruta absoluta al directorio 'src'
SRC_DIR = Path(__file__).resolve().parent

# Ruta absoluta a la raíz del proyecto (un nivel arriba de src)
ROOT_DIR = SRC_DIR.parent

# Rutas a directorios de recursos
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# Rutas específicas a modelos (accesibles desde cualquier parte del código)
FACE_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = MODELS_DIR / "haarcascade_eye.xml"
MOUTH_CASCADE_PATH = MODELS_DIR / "haarcascade_mcs_mouth.xml"

MODEL_XGB_LEFT = MODELS_DIR / "xgb_eye_left.pkl"
MODEL_XGB_RIGHT = MODELS_DIR / "xgb_eye_right.pkl"

def validate_paths():
    """Verifica que los modelos críticos existan al arrancar."""
    critical_files = [FACE_CASCADE_PATH, MODEL_XGB_LEFT]
    for p in critical_files:
        if not p.exists():
            print(f"⚠️  ADVERTENCIA: No se encuentra {p}")