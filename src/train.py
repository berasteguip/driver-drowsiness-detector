# train_cv.py  (con barra de progreso usando tqdm)
# Entrena 1 modelo XGBoost por lado (left/right) con validación cruzada.
# Muestra progreso por folds y por lado.
#
# Ejecutas desde: Proyecto Final/driver-drowsiness-detector/src/processing

from __future__ import annotations

from pathlib import Path
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm


def project_root_from_this_file() -> Path:
    # .../Proyecto Final/driver-drowsiness-detector/src/processing/train_cv.py
    return Path(__file__).resolve().parents[3]


def load_side_dataset(side: str) -> tuple[np.ndarray, np.ndarray]:
    root = project_root_from_this_file()
    base = root / "Proyecto Final" / "data" / "features" / side

    a = np.load(base / "active" / "features_hog.npz", allow_pickle=True)
    d = np.load(base / "drowsy" / "features_hog.npz", allow_pickle=True)

    X = np.vstack([a["X"], d["X"]]).astype(np.float32)
    y = np.concatenate([a["y"], d["y"]]).astype(np.int64)

    idx = np.random.default_rng(42).permutation(len(y))
    return X[idx], y[idx]


def make_model(scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )


def cross_validate_side(side: str, k: int = 5) -> None:
    X, y = load_side_dataset(side)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    aucs, f1s = [], []

    print(f"\n=== Cross-validation for side: {side.upper()} ===")

    for fold, (tr, va) in enumerate(
        tqdm(skf.split(X, y), total=k, desc=f"[{side}] folds"),
        start=1
    ):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        model = make_model(scale_pos_weight=spw)

        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
        )

        p = model.predict_proba(Xva)[:, 1]
        yhat = (p >= 0.5).astype(int)

        auc = roc_auc_score(yva, p)
        f1 = f1_score(yva, yhat)

        aucs.append(auc)
        f1s.append(f1)

        tqdm.write(
            f"[{side}] fold {fold}/{k} | AUC={auc:.4f} | F1={f1:.4f} | n_val={len(yva)}"
        )

    print(
        f"\n[{side}] CV mean±std | "
        f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f} | "
        f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} | "
        f"pos_rate={(y==1).mean():.3f}"
    )


def train_final_and_save(side: str) -> None:
    X, y = load_side_dataset(side)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"\n=== Training final model for side: {side.upper()} ===")

    model = make_model(scale_pos_weight=spw)

    # Barra de progreso a nivel de entrenamiento (estimadores)
    # Nota: XGBoost no expone progreso interno fácilmente;
    # tqdm aquí indica fase, no árboles individuales.
    with tqdm(total=1, desc=f"[{side}] training") as pbar:
        model.fit(X, y, verbose=False)
        pbar.update(1)

    root = project_root_from_this_file()
    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"xgb_eye_{side}.pkl"
    joblib.dump(model, out_path)
    print(f"[{side}] saved model -> {out_path}")


if __name__ == "__main__":
    # Validación cruzada
    for side in ["left", "right"]:
        cross_validate_side(side, k=5)

    # Entrenamiento final
    for side in ["left", "right"]:
        train_final_and_save(side)
