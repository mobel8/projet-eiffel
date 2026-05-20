"""EiffelPulse - Application Flask principale.

Plateforme d'aide à la décision pour la Société d'Exploitation de la Tour Eiffel :
- Prédiction d'affluence (forecast)
- Analyse multilingue des avis visiteurs (reviews)
- Assistant visiteur RAG (chat)
- Veille tarifaire concurrentielle (competitors)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Permet l'exécution directe `python backend/app.py` ET `python -m backend.app`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.services import (
    chat_service,
    dashboard_service,
    external_apis_service,
    forecast_service,
    scraper_service,
    sentiment_service,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("eiffelpulse")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(FRONTEND),
        static_url_path="",
    )
    # Préserve l'ordre des clés dans les réponses JSON (important pour les charts)
    app.json.sort_keys = False
    CORS(app)

    # --- HEALTH --------------------------------------------------------- #
    @app.route("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "EiffelPulse",
                "version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    # --- SENTIMENT ------------------------------------------------------ #
    @app.route("/api/sentiment/predict", methods=["POST"])
    def sentiment_predict():
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Champ 'text' requis"}), 400
        return jsonify(sentiment_service.predict(text))

    @app.route("/api/sentiment/metrics")
    def sentiment_metrics():
        return jsonify(sentiment_service.get_metrics())

    # --- FORECAST ------------------------------------------------------- #
    @app.route("/api/forecast/day")
    def forecast_day():
        date_str = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
        return jsonify(forecast_service.predict_day(date_str))

    @app.route("/api/forecast/week")
    def forecast_week():
        from datetime import datetime as dt

        start = dt.now().replace(hour=9, minute=0, second=0, microsecond=0)
        preds = forecast_service.predict_range(start, hours=24 * 7)
        return jsonify(
            {
                "start": start.strftime("%Y-%m-%d %H:%M"),
                "hours": len(preds),
                "predictions": preds,
            }
        )

    @app.route("/api/forecast/historical")
    def forecast_historical():
        return jsonify(forecast_service.get_historical_summary())

    @app.route("/api/forecast/metrics")
    def forecast_metrics():
        return jsonify(forecast_service.get_metrics())

    # --- CHAT (RAG) ----------------------------------------------------- #
    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Champ 'question' requis"}), 400
        return jsonify(chat_service.ask(question))

    @app.route("/api/chat/faq")
    def chat_faq():
        return jsonify({"faq": chat_service.get_all_faq(), "categories": chat_service.get_categories()})

    # --- DASHBOARD ------------------------------------------------------ #
    @app.route("/api/dashboard/reviews")
    def dashboard_reviews():
        return jsonify(dashboard_service.reviews_overview())

    @app.route("/api/dashboard/reviews/latest")
    def dashboard_latest():
        limit = int(request.args.get("limit", 20))
        sentiment = request.args.get("sentiment")
        return jsonify({"reviews": dashboard_service.latest_reviews(limit, sentiment)})

    @app.route("/api/dashboard/competitors")
    def dashboard_competitors():
        return jsonify(dashboard_service.competitor_pricing())

    # --- EXTERNAL APIs (météo, fériés, devises, soleil, wiki) ---------- #
    @app.route("/api/external/health")
    def external_health():
        return jsonify(external_apis_service.health_check_all())

    @app.route("/api/external/weather")
    def external_weather():
        return jsonify(external_apis_service.get_weather())

    @app.route("/api/external/holidays")
    def external_holidays():
        country = request.args.get("country", "FR")
        return jsonify(external_apis_service.get_holidays(country))

    @app.route("/api/external/holidays/multi")
    def external_holidays_multi():
        countries_arg = request.args.get("countries")
        countries = countries_arg.split(",") if countries_arg else None
        return jsonify(external_apis_service.get_holidays_multi(countries))

    @app.route("/api/external/currency")
    def external_currency():
        return jsonify(external_apis_service.get_currency_rates())

    @app.route("/api/external/currency/convert")
    def external_currency_convert():
        try:
            amount = float(request.args.get("amount", "35"))
        except ValueError:
            return jsonify({"error": "amount invalide"}), 400
        return jsonify(external_apis_service.convert_eur(amount))

    @app.route("/api/external/sun-times")
    def external_sun():
        target = request.args.get("date")
        return jsonify(external_apis_service.get_sun_times(target))

    @app.route("/api/external/wiki")
    def external_wiki():
        lang = request.args.get("lang", "fr")
        return jsonify(external_apis_service.get_wiki_fact(lang))

    # --- SCRAPER -------------------------------------------------------- #
    @app.route("/api/scraper/competitors")
    def scraper_competitors():
        return jsonify(
            {
                "cached_competitors": scraper_service.get_cached_competitors(),
                "selenium_available": scraper_service.selenium_available(),
            }
        )

    # --- FRONTEND ROUTES ----------------------------------------------- #
    @app.route("/")
    def index():
        return send_from_directory(FRONTEND, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        # Empêche route catch-all de manger /api/*
        if path.startswith("api/"):
            return jsonify({"error": "Not Found"}), 404
        return send_from_directory(FRONTEND, path)

    # --- ERROR HANDLER -------------------------------------------------- #
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Endpoint introuvable", "path": request.path}), 404
        return send_from_directory(FRONTEND, "index.html")

    @app.errorhandler(500)
    def server_error(e):
        log.exception("Erreur serveur")
        return jsonify({"error": "Erreur interne", "detail": str(e)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    port = 5050
    print(f"\n  EiffelPulse démarre sur http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
