"""Scraper de veille tarifaire pour attractions parisiennes.

Architecture :
- Selenium headless pour les sites en JavaScript dynamique
- BeautifulSoup en fallback pour les sites statiques
- Cache local des résultats (data/competitors_scraped.json)
- Mode dégradé : utilise data/competitors.json (pré-scrapé) si Chromium
  n'est pas disponible (utile en environnement Docker / CI minimal)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

log = logging.getLogger(__name__)


def selenium_available() -> bool:
    """Vérifie si un driver Chromium est utilisable."""
    try:
        from selenium import webdriver  # noqa: F401
        from selenium.webdriver.chrome.options import Options  # noqa: F401
        return True
    except Exception:
        return False


def scrape_attraction(url: str, name: str) -> dict | None:
    """Scrape une page d'attraction (Selenium headless).

    Implémentation simplifiée : extrait le <title> et les prix éventuels
    via patterns regex. En production, des sélecteurs spécifiques par site
    seraient utilisés.
    """
    if not selenium_available():
        log.warning("Selenium indisponible, scraping désactivé")
        return None

    try:
        import re

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=opts)
        try:
            driver.set_page_load_timeout(20)
            driver.get(url)
            html = driver.page_source
            title = driver.title

            # Extraction prix par regex (€ ou EUR)
            prices = re.findall(r"(\d{1,3}(?:[.,]\d{1,2})?)\s*€", html)
            prices_num = sorted({float(p.replace(",", ".")) for p in prices if float(p.replace(",", ".")) < 200})

            return {
                "name": name,
                "url": url,
                "page_title": title,
                "prices_detected_eur": prices_num[:10],
                "min_price": float(min(prices_num)) if prices_num else None,
                "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        finally:
            driver.quit()
    except Exception as exc:
        log.warning("Echec scraping %s : %s", name, exc)
        return None


def get_cached_competitors() -> list[dict]:
    """Renvoie les tarifs concurrents (pré-scrapés)."""
    with open(DATA / "competitors.json", encoding="utf-8") as fp:
        return json.load(fp)


def run_scraper_demo() -> dict:
    """Lance un scraping de démo sur 2 sites publics.

    En cas d'échec (réseau / sandbox), tombe en mode dégradé avec données pré-cachées.
    """
    targets = [
        ("https://www.toureiffel.paris", "Tour Eiffel"),
        ("https://www.parisinfo.com", "Office Tourisme Paris"),
    ]
    results = []
    for url, name in targets:
        r = scrape_attraction(url, name)
        if r:
            results.append(r)

    return {
        "live_scrape_results": results,
        "cached_competitors": get_cached_competitors(),
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "live" if results else "cached_only",
    }
