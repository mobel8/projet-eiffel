"""Construit l'index vectoriel pour le RAG chatbot.

TF-IDF sur la FAQ (FR + EN) + cosinus pour retrieval.
Légèreté maximale : pas de dépendance lourde, < 5MB de RAM.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent.parent
FAQ_PATH = ROOT / "data" / "faq.json"
ARTIFACTS = ROOT / "ml_artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def main() -> None:
    print(f"Chargement FAQ : {FAQ_PATH}")
    with open(FAQ_PATH, encoding="utf-8") as fp:
        faq = json.load(fp)

    print(f"  -> {len(faq)} entrées")

    # On indexe question + réponse pour un retrieval plus robuste
    corpus = [f"{item['question']} {item['answer']}" for item in faq]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)

    print(f"  -> matrice TF-IDF : {matrix.shape}")

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "faq": faq,
        },
        ARTIFACTS / "rag_index.joblib",
    )
    print(f"Index sauvegardé : {ARTIFACTS / 'rag_index.joblib'}")


if __name__ == "__main__":
    main()
