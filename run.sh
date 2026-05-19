#!/usr/bin/env bash
# Script de démarrage EiffelPulse
set -e

cd "$(dirname "$0")"

echo "==> Activation du virtualenv"
source .venv/bin/activate

# Vérifie que les données et modèles existent, sinon les régénère
if [ ! -f "data/reviews.csv" ]; then
  echo "==> Génération des données synthétiques"
  python backend/ml/generate_data.py
fi

if [ ! -f "ml_artifacts/sentiment_model.joblib" ]; then
  echo "==> Entraînement des modèles"
  python backend/ml/train_sentiment.py
  python backend/ml/train_forecaster.py
  python backend/ml/build_rag.py
fi

echo ""
echo "==> EiffelPulse démarre sur http://localhost:5050"
echo ""
python backend/app.py
