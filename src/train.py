# train.py
# Entrena XGBoost a partir de data/features_hog.npz
# (X: (N,D), y: (N,))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import xgboost as xgb
import joblib
from utils import *

def main():
    # 1) Cargar features
    data = np.load("../../data/features_hog.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    # 2) Split (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # 3) Modelo (baseline razonable)
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    # 4) Entrenar (con early stopping usando un set de validación)
    #    Nota: aquí usamos el test como validación SOLO para simplificar el esqueleto.
    #    En serio: crea un val set dentro de train.
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 5) Evaluación
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    plot_roc_curve(y_test, y_prob)

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    # 6) Guardar modelo
    joblib.dump(clf, "models/xgb_hog.pkl")
    print("Saved: models/xgb_hog.pkl")


if __name__ == "__main__":
    main()
