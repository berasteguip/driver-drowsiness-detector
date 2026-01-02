from __future__ import annotations
from pathlib import Path
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

# --- CONFIGURACIÓN DE GRIDS ---
GRIDS = {
    "left": {
        'n_estimators': [400, 600],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 2],
        'reg_lambda': [1.0, 10.0]
    },
    "right": {
        'n_estimators': [400, 600],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 2],
        'reg_lambda': [1.0]
    }
}

def project_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]

def load_side_dataset(side: str, oversample_rate: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    root = project_root_from_this_file()
    base = root / "Proyecto Final" / "data" / "features" / side

    # Carga de archivos
    a = np.load(base / "active" / "features_hog.npz", allow_pickle=True)
    d = np.load(base / "drowsy" / "features_hog.npz", allow_pickle=True)

    X_active, y_active = a["X"], a["y"]
    X_drowsy, y_drowsy = d["X"], d["y"]

    # --- LÓGICA DE DUPLICACIÓN (30%) ---
    num_drowsy = len(y_drowsy)
    num_to_duplicate = int(num_drowsy * oversample_rate)
    
    if num_to_duplicate > 0:
        # Seleccionamos índices aleatorios de la carpeta drowsy
        rng = np.random.default_rng(42)
        idx_to_dup = rng.choice(num_drowsy, size=num_to_duplicate, replace=False)
        
        # Duplicamos las muestras seleccionadas
        X_dup = X_drowsy[idx_to_dup]
        y_dup = y_drowsy[idx_to_dup]
        
        # Concatenamos a los datos drowsy originales
        X_drowsy = np.vstack([X_drowsy, X_dup])
        y_drowsy = np.concatenate([y_drowsy, y_dup])
        
        print(f"[{side.upper()}] Oversampling: {num_to_duplicate} muestras duplicadas en clase 'drowsy'.")

    # Unión final
    X = np.vstack([X_active, X_drowsy]).astype(np.float32)
    y = np.concatenate([y_active, y_drowsy]).astype(np.int64)

    # Mezcla (Shuffle) para que el entrenamiento sea estable
    idx = np.random.default_rng(42).permutation(len(y))
    return X[idx], y[idx]

def run_grid_search(side: str, param_grid: dict):
    X, y = load_side_dataset(side)
    
    # El scale_pos_weight se recalcula automáticamente con las nuevas cantidades
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"\n" + "="*60)
    print(f" INICIANDO GRID SEARCH: {side.upper()} ")
    print(f"="*60)
    
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=skf,
        verbose=1, 
        n_jobs=-1
    )

    grid_search.fit(X, y)
    print(f"\n[RESULTADOS {side.upper()}] -> Mejor ROC_AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_params_

def train_final_and_save(side: str, best_params: dict) -> None:
    X, y = load_side_dataset(side)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=42
    )

    with tqdm(total=1, desc=f"Final fit {side}") as pbar:
        model.fit(X, y, verbose=False)
        pbar.update(1)

    out_dir = project_root_from_this_file() / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"xgb_eye_{side}.pkl")

if __name__ == "__main__":
    best_configs = {}
    for side in ["left", "right"]:
        best_configs[side] = run_grid_search(side, GRIDS[side])

    print("\n" + "#"*60)
    print(" RESUMEN FINAL ")
    print("#"*60)
    for side, params in best_configs.items():
        print(f"{side.upper()}: {params}")

    for side in ["left", "right"]:
        train_final_and_save(side, best_configs[side])