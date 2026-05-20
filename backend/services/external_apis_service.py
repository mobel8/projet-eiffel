"""Intégrations APIs publiques utiles à la SETE.

5 sources de données externes, toutes gratuites et sans clé API :
- Open-Meteo : météo Paris (impact #1 sur l'affluence touristique)
- date.nager.at : jours fériés français pour l'année courante et suivante
- Frankfurter : taux de change EUR vers grandes devises touristiques
- Sunrise-Sunset.org : heures solaires pour recommandation "golden hour"
- Wikipedia REST : extraits factuels sur la Tour Eiffel (FR + EN)

Chaque appel est mis en cache (TTL différent selon la volatilité de la donnée)
et tombe gracieusement en mode dégradé si l'API est inaccessible.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone
from functools import wraps
from urllib.parse import quote
from urllib.request import Request, urlopen

log = logging.getLogger("eiffelpulse.external")

# Coordonnées Tour Eiffel
EIFFEL_LAT = 48.8584
EIFFEL_LON = 2.2945

# Codes météo WMO -> labels FR + icônes (emoji simples)
WMO_CODES = {
    0:  ("Ciel clair", "☀"),
    1:  ("Globalement clair", "🌤"),
    2:  ("Partiellement nuageux", "⛅"),
    3:  ("Couvert", "☁"),
    45: ("Brouillard", "🌫"),
    48: ("Brouillard givrant", "🌫"),
    51: ("Bruine légère", "🌦"),
    53: ("Bruine modérée", "🌦"),
    55: ("Bruine dense", "🌧"),
    61: ("Pluie légère", "🌦"),
    63: ("Pluie modérée", "🌧"),
    65: ("Pluie forte", "🌧"),
    71: ("Neige légère", "🌨"),
    73: ("Neige modérée", "🌨"),
    75: ("Neige forte", "❄"),
    80: ("Averses légères", "🌦"),
    81: ("Averses modérées", "🌧"),
    82: ("Averses violentes", "⛈"),
    95: ("Orage", "⛈"),
    96: ("Orage avec grêle", "⛈"),
    99: ("Orage violent avec grêle", "⛈"),
}

# Cache mémoire simple
_cache: dict[str, tuple[float, object]] = {}


def cached(ttl_seconds: int):
    """Décorateur de cache mémoire à TTL."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = fn.__name__ + repr(args) + repr(sorted(kwargs.items()))
            now = time.time()
            if key in _cache:
                ts, val = _cache[key]
                if now - ts < ttl_seconds:
                    return val
            try:
                val = fn(*args, **kwargs)
                _cache[key] = (now, val)
                return val
            except Exception as e:
                log.warning("Échec %s : %s — fallback", fn.__name__, e)
                # On garde l'ancienne valeur en cache si dispo
                if key in _cache:
                    return _cache[key][1]
                return {"error": str(e), "fallback": True}
        return wrapper
    return deco


def _get(url: str, timeout: int = 8) -> dict:
    req = Request(url, headers={"User-Agent": "EiffelPulse/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------- MÉTÉO ----------------------------- #

@cached(ttl_seconds=1800)  # 30 min
def get_weather() -> dict:
    """Météo actuelle + prévisions 7 jours sur Paris."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={EIFFEL_LAT}&longitude={EIFFEL_LON}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        "&timezone=Europe/Paris&forecast_days=7"
    )
    data = _get(url)
    current = data.get("current", {})
    code = int(current.get("weather_code", 0))
    label, icon = WMO_CODES.get(code, ("Inconnu", "❓"))

    daily = data.get("daily", {})
    daily_list = []
    days = daily.get("time", [])
    for i in range(len(days)):
        d_code = int(daily["weather_code"][i])
        d_label, d_icon = WMO_CODES.get(d_code, ("Inconnu", "❓"))
        daily_list.append({
            "date": days[i],
            "weather_code": d_code,
            "label": d_label,
            "icon": d_icon,
            "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "precipitation_probability": daily["precipitation_probability_max"][i],
        })

    return {
        "source": "Open-Meteo",
        "location": "Tour Eiffel, Paris",
        "current": {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "weather_code": code,
            "label": label,
            "icon": icon,
            "wind_speed": current.get("wind_speed_10m"),
        },
        "forecast_7d": daily_list,
        "impact_estimate": _weather_impact(code, current.get("precipitation", 0)),
    }


def _weather_impact(code: int, precip: float) -> dict:
    """Estime l'impact météo sur l'affluence (heuristique métier)."""
    if code in (95, 96, 99):  # orages
        return {"factor": 0.55, "label": "Très défavorable", "color": "rose"}
    if code in (61, 63, 65, 80, 81, 82) or precip > 1.0:
        return {"factor": 0.72, "label": "Défavorable", "color": "amber"}
    if code in (51, 53, 55, 45, 48):
        return {"factor": 0.88, "label": "Modéré", "color": "amber"}
    if code in (0, 1):
        return {"factor": 1.18, "label": "Très favorable", "color": "emerald"}
    return {"factor": 1.0, "label": "Neutre", "color": "midnight"}


# ----------------------------- FÉRIÉS ----------------------------- #

@cached(ttl_seconds=86400)  # 24h
def get_holidays(country: str = "FR", upcoming: int = 6) -> dict:
    """Jours fériés à venir pour le pays donné (FR par défaut)."""
    today = date.today()
    year = today.year
    holidays = _get(f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}")
    # Inclut aussi l'année prochaine si on est en fin d'année
    if today.month >= 10:
        holidays += _get(f"https://date.nager.at/api/v3/PublicHolidays/{year + 1}/{country}")

    upcoming_list = []
    for h in holidays:
        d = datetime.strptime(h["date"], "%Y-%m-%d").date()
        if d >= today:
            delta = (d - today).days
            upcoming_list.append({
                "date": h["date"],
                "name_local": h["localName"],
                "name_en": h["name"],
                "days_until": delta,
                "is_weekend_bridge": d.weekday() in (1, 4),  # mardi ou vendredi = pont
            })

    upcoming_list = sorted(upcoming_list, key=lambda h: h["days_until"])[:upcoming]

    # Pays touristiques émetteurs majeurs : agrégation rapide pour planification
    return {
        "source": "date.nager.at",
        "country": country,
        "upcoming": upcoming_list,
    }


def get_holidays_multi(countries: list[str] | None = None) -> dict:
    """Combine jours fériés de plusieurs pays (FR + grands émetteurs touristiques)."""
    countries = countries or ["FR", "GB", "ES", "DE", "IT", "US"]
    result = {}
    for c in countries:
        r = get_holidays(c, upcoming=4)
        result[c] = r.get("upcoming", []) if not r.get("fallback") else []
    return {"source": "date.nager.at", "by_country": result}


# ----------------------------- DEVISES ----------------------------- #

@cached(ttl_seconds=3600)  # 1h
def get_currency_rates() -> dict:
    """Taux EUR vers devises touristiques majeures."""
    symbols = "USD,GBP,JPY,CNY,KRW,CHF,CAD,AUD,BRL,INR"
    data = _get(f"https://api.frankfurter.dev/v1/latest?base=EUR&symbols={symbols}")
    return {
        "source": "Frankfurter",
        "base": "EUR",
        "date": data.get("date"),
        "rates": data.get("rates", {}),
    }


def convert_eur(amount_eur: float, currencies: list[str] | None = None) -> dict:
    """Convertit un montant EUR vers plusieurs devises."""
    rates_data = get_currency_rates()
    rates = rates_data.get("rates", {})
    if not rates:
        return {"error": "Taux indisponibles", "fallback": True}
    currencies = currencies or list(rates.keys())
    conversions = {
        cur: round(amount_eur * rate, 2)
        for cur, rate in rates.items()
        if cur in currencies
    }
    conversions["EUR"] = float(amount_eur)
    return {
        "amount_eur": amount_eur,
        "conversions": conversions,
        "rates_date": rates_data.get("date"),
    }


# ----------------------------- SOLEIL ----------------------------- #

@cached(ttl_seconds=86400)  # 24h
def get_sun_times(target_date: str | None = None) -> dict:
    """Heures lever / coucher / golden hour pour Tour Eiffel."""
    target_date = target_date or date.today().isoformat()
    data = _get(
        f"https://api.sunrise-sunset.org/json?lat={EIFFEL_LAT}&lng={EIFFEL_LON}"
        f"&formatted=0&date={target_date}"
    )
    res = data.get("results", {})

    def parse(iso: str | None) -> str | None:
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            # Conversion vers heure locale Paris (UTC+2 en été, +1 en hiver)
            # Simplification : on demande déjà le bon offset à l'API si possible
            return dt.astimezone().strftime("%H:%M")
        except Exception:
            return iso

    sunset = res.get("sunset")
    # "Golden hour" recommandée : 1h avant le coucher du soleil
    return {
        "source": "Sunrise-Sunset.org",
        "date": target_date,
        "sunrise": parse(res.get("sunrise")),
        "sunset": parse(sunset),
        "solar_noon": parse(res.get("solar_noon")),
        "day_length_hours": round((res.get("day_length") or 0) / 3600, 2),
        "civil_twilight_end": parse(res.get("civil_twilight_end")),
        "recommendation": (
            f"Meilleure heure photo : ~1h avant le coucher du soleil"
            if sunset else "Données indisponibles"
        ),
    }


# ----------------------------- WIKIPÉDIA ----------------------------- #

@cached(ttl_seconds=604800)  # 7 jours
def get_wiki_fact(lang: str = "fr") -> dict:
    """Extrait Wikipedia sur la Tour Eiffel."""
    if lang not in ("fr", "en"):
        lang = "fr"
    title = "Tour_Eiffel" if lang == "fr" else "Eiffel_Tower"
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    data = _get(url)
    return {
        "source": "Wikipedia",
        "language": lang,
        "title": data.get("title"),
        "extract": data.get("extract"),
        "thumbnail": data.get("thumbnail", {}).get("source"),
        "page_url": data.get("content_urls", {}).get("desktop", {}).get("page"),
    }


# ----------------------------- HEALTH ----------------------------- #

def health_check_all() -> dict:
    """Test rapide de toutes les intégrations."""
    return {
        "weather": "ok" if not get_weather().get("fallback") else "degraded",
        "holidays": "ok" if not get_holidays().get("fallback") else "degraded",
        "currency": "ok" if not get_currency_rates().get("fallback") else "degraded",
        "sun_times": "ok" if not get_sun_times().get("fallback") else "degraded",
        "wikipedia": "ok" if not get_wiki_fact().get("fallback") else "degraded",
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
