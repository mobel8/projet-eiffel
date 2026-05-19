"""Service RAG pour l'assistant visiteur multilingue."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "ml_artifacts"


@lru_cache(maxsize=1)
def _load_index():
    return joblib.load(ARTIFACTS / "rag_index.joblib")


def ask(question: str, top_k: int = 3, threshold: float = 0.12) -> dict:
    """Récupère les FAQ les plus pertinentes et renvoie une réponse."""
    if not question or not question.strip():
        return {"error": "Question vide"}

    index = _load_index()
    vec = index["vectorizer"].transform([question])
    sims = cosine_similarity(vec, index["matrix"])[0]

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        score = float(sims[i])
        item = index["faq"][i]
        results.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "category": item["category"],
                "score": round(score, 3),
            }
        )

    if results and results[0]["score"] >= threshold:
        best = results[0]
        answer = best["answer"]
        confidence = "high" if best["score"] > 0.35 else "medium"
    else:
        answer = (
            "Je ne suis pas sûr de comprendre votre question. "
            "Pouvez-vous reformuler ? Vous pouvez me demander : horaires, tarifs, "
            "accessibilité, restaurants, billetterie, animations…"
        )
        confidence = "low"

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": results,
    }


def get_categories() -> list[str]:
    index = _load_index()
    return sorted({item["category"] for item in index["faq"]})


def get_all_faq() -> list[dict]:
    index = _load_index()
    return index["faq"]
