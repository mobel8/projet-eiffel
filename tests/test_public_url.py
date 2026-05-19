"""Tests end-to-end via l'URL publique cloudflared.

Vérifie que l'application est accessible depuis l'extérieur (latence, JSON,
fonctionnalités principales).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.request import Request, urlopen

DEFAULT_URL = "https://established-words-cosmetic-without.trycloudflare.com"


def req(url: str, payload: dict | None = None, timeout: int = 15, parse_json: bool = True) -> tuple[int, dict | str, float]:
    start = time.time()
    if payload:
        data = json.dumps(payload).encode("utf-8")
        r = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    else:
        r = Request(url)
    try:
        with urlopen(r, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, (json.loads(body) if parse_json else body), time.time() - start
    except Exception as e:
        return 0, {"error": str(e)}, time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"Cible : {base}\n")

    tests = [
        ("GET", "/api/health", None, lambda d: d.get("status") == "ok"),
        (
            "POST", "/api/sentiment/predict",
            {"text": "La vue depuis le sommet est magnifique au coucher du soleil !"},
            lambda d: d.get("sentiment") == "positive",
        ),
        (
            "POST", "/api/sentiment/predict",
            {"text": "File d'attente énorme et tarifs prohibitifs, déception totale."},
            lambda d: d.get("sentiment") == "negative",
        ),
        (
            "POST", "/api/sentiment/predict",
            {"text": "The elevator was broken, awful experience."},
            lambda d: d.get("sentiment") == "negative",
        ),
        ("GET", "/api/forecast/day?date=2026-08-15", None, lambda d: d.get("total_visitors", 0) > 0),
        (
            "GET", "/api/forecast/historical", None,
            lambda d: list(d.get("by_day_of_week", {}).keys())[:2] == ["Lundi", "Mardi"],
        ),
        ("POST", "/api/chat", {"question": "Quels sont les horaires ?"}, lambda d: "9h30" in d.get("answer", "")),
        ("POST", "/api/chat", {"question": "How tall is the tower?"}, lambda d: d.get("confidence") in ("high", "medium")),
        ("GET", "/api/dashboard/reviews", None, lambda d: d.get("total_reviews") == 5000),
        ("GET", "/api/dashboard/competitors", None, lambda d: len(d.get("attractions", [])) > 5),
        ("GET", "/api/chat/faq", None, lambda d: len(d.get("faq", [])) >= 15),
    ]

    passed = 0
    failed = []
    total_time = 0.0

    for method, path, payload, validate in tests:
        url = base + path
        status, data, dt = req(url, payload)
        total_time += dt
        ok = status == 200 and validate(data) if isinstance(data, dict) else False
        flag = "✓" if ok else "✗"
        print(f"  {flag} {method:4s} {path:48s} HTTP {status} · {dt*1000:5.0f} ms")
        if ok:
            passed += 1
        else:
            failed.append(f"{method} {path} : status={status} body={str(data)[:200]}")

    # Test frontend (HTML, on désactive le parsing JSON)
    status, _, dt = req(base + "/", parse_json=False)
    total_time += dt
    # On accepte tout 200 sur "/", on ne parse pas le HTML
    if status == 200:
        passed += 1
        print(f"  ✓ GET  /                                                HTTP {status} · {dt*1000:5.0f} ms (HTML)")
    else:
        failed.append(f"GET / : status={status}")
        print(f"  ✗ GET  /                                                HTTP {status}")

    print(f"\n=== RECAP ===")
    print(f"Tests passés : {passed} / {len(tests) + 1}")
    print(f"Temps total : {total_time:.2f} s ({total_time / (len(tests) + 1) * 1000:.0f} ms / req)")
    if failed:
        print("\nEchecs :")
        for f in failed:
            print(" -", f)
        return 1
    print("\n✓ Tous les tests publics passent — l'application est accessible mondialement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
