"""Service de prédiction d'affluence."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "ml_artifacts"
DATA = ROOT / "data"

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


@lru_cache(maxsize=1)
def _load_model():
    model = joblib.load(ARTIFACTS / "forecaster_model.joblib")
    with open(ARTIFACTS / "metrics_forecaster.json", encoding="utf-8") as fp:
        metrics = json.load(fp)
    return model, metrics


def _build_row(ts: datetime) -> dict:
    return {
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "month": ts.month,
        "is_weekend": int(ts.weekday() >= 5),
        "is_holiday_season": int(ts.month in (7, 8, 12)),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * ts.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * ts.month / 12)),
    }


def predict_range(start: datetime, hours: int = 168) -> list[dict]:
    """Prédit l'affluence horaire sur N heures (par défaut 7 jours)."""
    model, _ = _load_model()
    rows = []
    points = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        # Ouverture 9h-23h
        if ts.hour < 9 or ts.hour >= 24:
            continue
        feats = _build_row(ts)
        rows.append(feats)
        points.append(ts)

    X = pd.DataFrame(rows)[FEATURE_COLS]
    preds = model.predict(X)
    preds = np.maximum(preds, 0).round().astype(int)

    return [
        {
            "datetime": ts.strftime("%Y-%m-%d %H:%M"),
            "date": ts.strftime("%Y-%m-%d"),
            "hour": ts.hour,
            "day": ts.strftime("%A"),
            "visitors": int(v),
        }
        for ts, v in zip(points, preds)
    ]


def predict_day(date_str: str) -> dict:
    """Prédit l'affluence d'une journée par tranche horaire."""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return {"error": "Format de date invalide (YYYY-MM-DD attendu)"}

    preds = predict_range(date.replace(hour=0, minute=0), hours=24)
    total = sum(p["visitors"] for p in preds)
    peak = max(preds, key=lambda x: x["visitors"]) if preds else None

    # Recommandation staffing : 1 agent pour 200 visiteurs/h
    staffing = [
        {"hour": p["hour"], "agents_needed": max(1, p["visitors"] // 200)} for p in preds
    ]

    return {
        "date": date_str,
        "total_visitors": total,
        "hourly": preds,
        "peak_hour": peak,
        "staffing_recommendation": staffing,
    }


@lru_cache(maxsize=1)
def get_historical_summary() -> dict:
    """Stats historiques pour le dashboard."""
    df = pd.read_csv(DATA / "attendance.csv", parse_dates=["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Affluence par jour de semaine (ordre Lundi -> Dimanche)
    by_dow = df.groupby("day_of_week")["visitors"].mean().round().astype(int).to_dict()
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    by_dow_named = {days[k]: int(by_dow.get(k, 0)) for k in range(7)}

    # Affluence par mois (ordre Jan -> Déc)
    by_month = df.groupby("month")["visitors"].mean().round().astype(int).to_dict()
    months = [
        "Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
        "Juil", "Août", "Sep", "Oct", "Nov", "Déc",
    ]
    by_month_named = {months[m - 1]: int(by_month.get(m, 0)) for m in range(1, 13)}

    # Affluence par heure
    by_hour = df.groupby("hour")["visitors"].mean().round().astype(int).to_dict()
    by_hour_named = {f"{h:02d}h": int(by_hour[h]) for h in sorted(by_hour.keys())}

    return {
        "total_visitors_2y": int(df["visitors"].sum()),
        "avg_per_hour": float(df["visitors"].mean()),
        "max_recorded": int(df["visitors"].max()),
        "by_day_of_week": by_dow_named,
        "by_month": by_month_named,
        "by_hour": by_hour_named,
    }


def get_metrics() -> dict:
    _, metrics = _load_model()
    return metrics
