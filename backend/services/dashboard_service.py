"""Agrégations BI pour le dashboard."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"


@lru_cache(maxsize=1)
def _load_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA / "reviews.csv", parse_dates=["date"])


@lru_cache(maxsize=1)
def _load_competitors() -> list[dict]:
    with open(DATA / "competitors.json", encoding="utf-8") as fp:
        return json.load(fp)


def reviews_overview() -> dict:
    df = _load_reviews()

    by_sentiment = df["sentiment"].value_counts(normalize=True).to_dict()
    by_topic = df["topic"].value_counts().to_dict()
    by_language = df["language"].value_counts().to_dict()
    by_source = df["source"].value_counts().to_dict()
    by_rating = df["rating"].value_counts().sort_index().to_dict()

    avg_rating = float(df["rating"].mean())

    # Evolution mensuelle
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)
    monthly_sent = (
        df_monthly.groupby(["month", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .to_dict("records")
    )

    # Sentiment par topic
    sent_by_topic = (
        df.groupby(["topic", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .div(df.groupby("topic").size(), axis=0)
        .round(3)
        .reset_index()
        .to_dict("records")
    )

    return {
        "total_reviews": int(len(df)),
        "avg_rating": round(avg_rating, 2),
        "by_sentiment_pct": {k: round(v * 100, 1) for k, v in by_sentiment.items()},
        "by_topic": {str(k): int(v) for k, v in by_topic.items()},
        "by_language": {str(k): int(v) for k, v in by_language.items()},
        "by_source": {str(k): int(v) for k, v in by_source.items()},
        "by_rating": {str(int(k)): int(v) for k, v in by_rating.items()},
        "monthly_evolution": monthly_sent,
        "sentiment_by_topic": sent_by_topic,
    }


def latest_reviews(limit: int = 20, sentiment: str | None = None) -> list[dict]:
    df = _load_reviews()
    if sentiment:
        df = df[df["sentiment"] == sentiment]
    df = df.sort_values("date", ascending=False).head(limit)
    return df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict("records")


def competitor_pricing() -> dict:
    competitors = _load_competitors()
    # Tour Eiffel comme référence
    eiffel = {
        "name": "Tour Eiffel (sommet ascenseur)",
        "price_adult": 35,
        "price_youth": 18,
        "category": "Monument",
        "is_self": True,
    }
    all_attractions = [eiffel] + competitors

    avg_adult = sum(c["price_adult"] for c in competitors) / len(competitors)
    avg_youth = sum(c["price_youth"] for c in competitors) / len(competitors)

    return {
        "attractions": all_attractions,
        "market_avg_adult": round(avg_adult, 1),
        "market_avg_youth": round(avg_youth, 1),
        "eiffel_premium_pct_adult": round((eiffel["price_adult"] - avg_adult) / avg_adult * 100, 1),
        "eiffel_premium_pct_youth": round((eiffel["price_youth"] - avg_youth) / avg_youth * 100, 1),
    }
