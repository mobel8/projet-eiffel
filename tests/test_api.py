"""Tests d'intégration de l'API EiffelPulse."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "EiffelPulse"


def test_sentiment_positive(client):
    r = client.post(
        "/api/sentiment/predict",
        json={"text": "La vue depuis le sommet est absolument magnifique !"},
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["sentiment"] == "positive"
    assert 0 <= data["sentiment_confidence"] <= 1
    assert "topic" in data


def test_sentiment_negative(client):
    r = client.post(
        "/api/sentiment/predict",
        json={"text": "Trop cher, file d'attente interminable, déception totale."},
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["sentiment"] == "negative"


def test_sentiment_empty(client):
    r = client.post("/api/sentiment/predict", json={"text": ""})
    assert r.status_code == 400


def test_forecast_day(client):
    r = client.get("/api/forecast/day?date=2026-07-15")
    assert r.status_code == 200
    data = r.get_json()
    assert data["date"] == "2026-07-15"
    assert data["total_visitors"] > 0
    assert len(data["hourly"]) > 0
    assert "peak_hour" in data


def test_forecast_historical(client):
    r = client.get("/api/forecast/historical")
    assert r.status_code == 200
    data = r.get_json()
    # Vérifie l'ordre des jours (régression du bug "alphabetical sort")
    assert list(data["by_day_of_week"].keys()) == [
        "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"
    ]
    # Vérifie l'ordre des mois
    assert list(data["by_month"].keys())[:3] == ["Jan", "Fév", "Mar"]


def test_chat_horaires(client):
    r = client.post("/api/chat", json={"question": "Quels sont les horaires ?"})
    assert r.status_code == 200
    data = r.get_json()
    assert "answer" in data
    assert "9h30" in data["answer"] or "9:30" in data["answer"]
    assert len(data["sources"]) > 0


def test_chat_english(client):
    r = client.post("/api/chat", json={"question": "How much does a ticket cost?"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["confidence"] in ("high", "medium", "low")


def test_dashboard_reviews(client):
    r = client.get("/api/dashboard/reviews")
    assert r.status_code == 200
    data = r.get_json()
    assert data["total_reviews"] == 5000
    assert data["avg_rating"] > 0


def test_dashboard_competitors(client):
    r = client.get("/api/dashboard/competitors")
    assert r.status_code == 200
    data = r.get_json()
    assert "attractions" in data
    assert any(a.get("is_self") for a in data["attractions"])


def test_404_api(client):
    r = client.get("/api/nonexistent")
    assert r.status_code == 404


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"EiffelPulse" in r.data


# ============== EXTERNAL APIs ============== #


def test_external_health(client):
    r = client.get("/api/external/health")
    assert r.status_code == 200
    data = r.get_json()
    # Au moins une des 5 sources doit répondre
    statuses = [data.get(k) for k in ("weather", "holidays", "currency", "sun_times", "wikipedia")]
    assert any(s == "ok" for s in statuses)


def test_external_weather(client):
    r = client.get("/api/external/weather")
    assert r.status_code == 200
    data = r.get_json()
    if data.get("fallback"):
        return  # API externe down, on accepte le fallback
    assert "current" in data
    assert "forecast_7d" in data
    assert data["current"]["temperature"] is not None
    assert "impact_estimate" in data
    assert 0 < data["impact_estimate"]["factor"] < 2


def test_external_holidays(client):
    r = client.get("/api/external/holidays")
    assert r.status_code == 200
    data = r.get_json()
    if data.get("fallback"):
        return
    assert "upcoming" in data
    if data["upcoming"]:
        h = data["upcoming"][0]
        assert "date" in h
        assert "name_local" in h
        assert "days_until" in h
        assert h["days_until"] >= 0


def test_external_currency_convert(client):
    r = client.get("/api/external/currency/convert?amount=35.30")
    assert r.status_code == 200
    data = r.get_json()
    if data.get("fallback"):
        return
    assert "conversions" in data
    assert "USD" in data["conversions"]
    assert "JPY" in data["conversions"]
    # EUR doit être présent et égal au montant
    assert data["conversions"]["EUR"] == 35.30


def test_external_sun_times(client):
    r = client.get("/api/external/sun-times")
    assert r.status_code == 200
    data = r.get_json()
    if data.get("fallback"):
        return
    assert data["sunrise"] is not None
    assert data["sunset"] is not None


def test_external_wiki(client):
    r = client.get("/api/external/wiki?lang=fr")
    assert r.status_code == 200
    data = r.get_json()
    if data.get("fallback"):
        return
    assert "Tour Eiffel" in data["title"] or "Eiffel" in data["title"]
    assert len(data["extract"]) > 100


def test_external_currency_invalid(client):
    r = client.get("/api/external/currency/convert?amount=invalid")
    assert r.status_code == 400
