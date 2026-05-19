"""Service d'analyse de sentiment et de classification thématique."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "ml_artifacts"


@lru_cache(maxsize=1)
def _load_models():
    sentiment = joblib.load(ARTIFACTS / "sentiment_model.joblib")
    topic = joblib.load(ARTIFACTS / "topic_model.joblib")
    with open(ARTIFACTS / "metrics_sentiment.json", encoding="utf-8") as fp:
        metrics = json.load(fp)
    return sentiment, topic, metrics


def predict(text: str) -> dict:
    """Prédit sentiment + thème pour un avis."""
    if not text or not text.strip():
        return {"error": "Texte vide"}
    sentiment_model, topic_model, _ = _load_models()

    sentiment = sentiment_model.predict([text])[0]
    sentiment_proba = sentiment_model.predict_proba([text])[0]
    sentiment_classes = sentiment_model.classes_

    topic = topic_model.predict([text])[0]
    topic_proba = topic_model.predict_proba([text])[0]
    topic_classes = topic_model.classes_

    return {
        "sentiment": str(sentiment),
        "sentiment_confidence": float(max(sentiment_proba)),
        "sentiment_distribution": {
            str(c): float(p) for c, p in zip(sentiment_classes, sentiment_proba)
        },
        "topic": str(topic),
        "topic_confidence": float(max(topic_proba)),
        "topic_distribution": dict(
            sorted(
                {str(c): float(p) for c, p in zip(topic_classes, topic_proba)}.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]
        ),
    }


def get_metrics() -> dict:
    """Métriques d'entraînement."""
    _, _, metrics = _load_models()
    return {
        "sentiment_f1_macro": metrics["sentiment"]["f1_macro"],
        "sentiment_f1_weighted": metrics["sentiment"]["f1_weighted"],
        "topic_f1_macro": metrics["topic"]["f1_macro"],
        "topic_f1_weighted": metrics["topic"]["f1_weighted"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "classes_sentiment": metrics["classes_sentiment"],
        "classes_topic": metrics["classes_topic"],
    }
