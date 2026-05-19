"""Capture screenshots Playwright + détection erreurs console."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
SHOTS.mkdir(exist_ok=True)

URL = "http://localhost:5050"

PAGES = [
    {"id": "dashboard", "wait_for": "#chartMonthly"},
    {"id": "forecast", "wait_for": "#chartForecast"},
    {"id": "reviews", "wait_for": ".review-card"},
    {"id": "chat", "wait_for": "#chatMessages"},
    {"id": "pricing", "wait_for": "table"},
    {"id": "about", "wait_for": ".tech-block"},
]


def main() -> int:
    console_log: list[dict] = []
    failed = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = context.new_page()

        page.on("console", lambda msg: console_log.append({"type": msg.type, "text": msg.text}))
        page.on("pageerror", lambda err: console_log.append({"type": "pageerror", "text": str(err)}))

        print(f"-> Chargement {URL}")
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1800)  # alpine init + first chart render

        for spec in PAGES:
            print(f"-> Capture page : {spec['id']}")
            try:
                # Navigation par changement de hash (déclenche le hashchange listener Alpine)
                page.evaluate(f"window.location.hash = '#{spec['id']}'")
                page.wait_for_timeout(1100)
                try:
                    page.wait_for_selector(spec["wait_for"], timeout=8000, state="visible")
                except Exception:
                    print(f"   [warn] selecteur {spec['wait_for']} non visible, on continue")

                page.wait_for_timeout(900)  # animations chart

                out = SHOTS / f"{spec['id']}.png"
                page.screenshot(path=str(out), full_page=True)
                size_kb = out.stat().st_size // 1024
                print(f"   -> {out.name} ({size_kb} KB)")
            except Exception as e:
                print(f"   [FAIL] {e}")
                failed.append(spec["id"])

        # Démo sentiment
        try:
            page.evaluate("window.location.hash = '#reviews'")
            page.wait_for_timeout(900)
            page.locator("textarea").fill(
                "La vue depuis le sommet est absolument magnifique au coucher du soleil !"
            )
            # Bouton "Analyser" : dernier btn-primary sur la section visible
            page.locator("section:visible button.btn-primary").last.click()
            page.wait_for_timeout(1500)
            page.screenshot(path=str(SHOTS / "reviews_demo.png"), full_page=True)
            print("   -> reviews_demo.png (avec résultat analyse)")
        except Exception as e:
            print(f"   [warn] demo sentiment : {e}")

        # Démo chat
        try:
            page.evaluate("window.location.hash = '#chat'")
            page.wait_for_timeout(900)
            page.locator('input[type="text"]').fill(
                "Quels sont les horaires d'ouverture ?"
            )
            page.locator("section:visible button.btn-primary").last.click()
            page.wait_for_timeout(1500)
            page.screenshot(path=str(SHOTS / "chat_demo.png"), full_page=True)
            print("   -> chat_demo.png (avec conversation)")
        except Exception as e:
            print(f"   [warn] demo chat : {e}")

        browser.close()

    log_path = SHOTS / "console.json"
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(console_log, fp, ensure_ascii=False, indent=2)

    errors = [m for m in console_log if m["type"] in ("error", "pageerror")]
    warnings = [m for m in console_log if m["type"] == "warning"]
    pngs = sorted(SHOTS.glob("*.png"))

    print(f"\n=== RECAP ===")
    print(f"Screenshots: {len(pngs)}")
    for f in pngs:
        print(f"  - {f.name} ({f.stat().st_size // 1024} KB)")
    print(f"Erreurs console: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    if errors:
        print("\n[ERRORS]")
        for e in errors[:10]:
            print(" -", e["text"][:200])
    if failed:
        print(f"\n[FAILED PAGES] {failed}")
        return 2
    if errors:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
