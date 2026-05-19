"""Entraîne un modèle de prédiction d'affluence horaire.

Architecture : feature engineering temporel + GradientBoosting.
Plus rapide et robuste que Prophet pour notre usage (pas de tendance long terme,
features explicites jour/heure/saison/week-end).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "attendance.csv"
ARTIFACTS = ROOT / "ml_artifacts"
ARTIFACTS.mkdir(exist_ok=True)


FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday_season",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def main() -> None:
    print(f"Chargement {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    print(f"Total : {len(df):,} lignes horaires")

    X = df[FEATURE_COLS]
    y = df["visitors"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )
    print("\nEntraînement GradientBoostingRegressor (n=300, depth=5)...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    print(f"\n[Forecaster] MAE = {mae:.0f} visiteurs/h | MAPE = {mape*100:.1f}%")
    print(f"  Moyenne réelle : {y_test.mean():.0f} visiteurs/h")
    print(f"  Variance expliquée : {model.score(X_test, y_test):.3f}")

    joblib.dump(model, ARTIFACTS / "forecaster_model.joblib")

    # Feature importance
    fi = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    metrics = {
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(model.score(X_test, y_test)),
        "mean_visitors_per_hour": float(y_test.mean()),
        "feature_importance": fi,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(ARTIFACTS / "metrics_forecaster.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    print(f"\nModèle sauvegardé : {ARTIFACTS / 'forecaster_model.joblib'}")


if __name__ == "__main__":
    main()
