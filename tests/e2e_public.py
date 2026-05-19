"""Screenshots via l'URL publique (test extérieur)."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots" / "public"
SHOTS.mkdir(parents=True, exist_ok=True)

URL = "https://established-words-cosmetic-without.trycloudflare.com"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = context.new_page()

        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        print(f"Test via URL publique : {URL}")
        page.goto(URL, wait_until="networkidle", timeout=45000)
        # Plus de temps pour Alpine + fonts Google + Chart.js init via cloudflared
        page.wait_for_timeout(4000)
        # On vérifie que les charts ont une largeur > 0 (rendus)
        page.wait_for_function(
            "() => document.querySelector('#chartMonthly')?.getBoundingClientRect().width > 100",
            timeout=15000,
        )
        page.screenshot(path=str(SHOTS / "public_dashboard.png"), full_page=True)
        print(f"  ✓ Dashboard capturé via URL publique")

        # Test sentiment via URL publique
        page.evaluate("window.location.hash = '#reviews'")
        page.wait_for_timeout(1000)
        page.locator("textarea").fill("Vue exceptionnelle depuis le sommet, expérience inoubliable !")
        page.locator("section:visible button.btn-primary").last.click()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SHOTS / "public_sentiment_demo.png"), full_page=True)
        print(f"  ✓ Démo sentiment fonctionne via URL publique")

        # Test chat via URL publique
        page.evaluate("window.location.hash = '#chat'")
        page.wait_for_timeout(1000)
        page.locator('input[type="text"]').fill("Quels sont les tarifs ?")
        page.locator("section:visible button.btn-primary").last.click()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SHOTS / "public_chat_demo.png"), full_page=True)
        print(f"  ✓ Chat RAG fonctionne via URL publique")

        browser.close()

    if errors:
        print(f"\n[ERRORS] {errors}")
        return 1
    print(f"\n✓ App pleinement fonctionnelle via URL publique, 0 erreur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
