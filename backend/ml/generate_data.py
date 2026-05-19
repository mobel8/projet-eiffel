"""Génère des données synthétiques réalistes pour EiffelPulse.

- Avis multilingues visiteurs Tour Eiffel (FR/EN/ES/DE/IT)
- Historique d'affluence horaire 2 ans
- FAQ pour le RAG
- Veille tarifaire attractions concurrentes
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ----------------------------- AVIS VISITEURS ----------------------------- #

REVIEW_TEMPLATES = {
    "fr": {
        "positive": {
            "vue": [
                "La vue depuis le sommet est tout simplement à couper le souffle, Paris s'étend à perte de vue.",
                "Panorama exceptionnel sur Paris, on voit jusqu'au Sacré-Cœur et la Défense.",
                "Vue magnifique au coucher du soleil, un moment inoubliable avec ma famille.",
                "Spectacle incroyable de nuit avec les illuminations, je recommande absolument.",
            ],
            "personnel": [
                "Personnel très accueillant et professionnel, toujours souriant.",
                "Le staff est multilingue et très serviable, parfait pour les touristes.",
                "Accueil chaleureux et équipe efficace, merci pour cette belle visite.",
            ],
            "experience": [
                "Une expérience magique, le monument est encore plus impressionnant en vrai.",
                "Visite parfaite, organisation au top, je recommande vivement.",
                "Moment magique en famille, les enfants ont adoré.",
            ],
        },
        "negative": {
            "file_attente": [
                "File d'attente interminable, plus de 2 heures sous le soleil sans ombre.",
                "Attente beaucoup trop longue malgré la réservation en ligne, c'est inacceptable.",
                "Queue énorme à chaque étage, prévoir une journée entière.",
            ],
            "prix": [
                "Tarifs vraiment excessifs pour ce que c'est, beaucoup trop cher.",
                "Le prix du billet est prohibitif, surtout pour une famille avec enfants.",
                "Trop cher pour la restauration sur place, comptez 25€ un sandwich.",
            ],
            "ascenseur": [
                "Ascenseur en panne le jour de notre visite, déception totale.",
                "Un seul ascenseur fonctionnait, attente énorme pour redescendre.",
                "Ascenseurs souvent en maintenance, prévoyez les escaliers.",
            ],
            "proprete": [
                "Toilettes pas très propres au deuxième étage, à améliorer.",
                "Beaucoup de déchets autour du monument, dommage pour l'image.",
            ],
            "securite": [
                "Trop de vendeurs à la sauvette autour, sentiment d'insécurité.",
                "Pickpockets très présents, faites très attention à vos affaires.",
            ],
        },
        "neutral": {
            "generique": [
                "Visite correcte mais sans plus, attendu plus d'animations.",
                "Monument iconique mais l'expérience reste classique.",
                "Ça vaut le coup de le voir une fois dans sa vie.",
            ],
        },
    },
    "en": {
        "positive": {
            "vue": [
                "The view from the top is absolutely breathtaking, you can see all of Paris.",
                "Stunning panorama, especially at sunset. A must-visit in Paris.",
                "Amazing 360 view, we could see the Sacre-Coeur and even further.",
            ],
            "personnel": [
                "Staff was very friendly and helpful, spoke perfect English.",
                "Great team, super welcoming and professional throughout.",
            ],
            "experience": [
                "Truly magical experience, the monument is even more impressive in person.",
                "Perfect family day, kids loved every minute of it.",
            ],
        },
        "negative": {
            "file_attente": [
                "Waited over 2 hours in line even with online tickets, awful.",
                "The queue was insane, plan a whole day for this visit.",
                "Endless lines at every floor, terrible organization.",
            ],
            "prix": [
                "Ridiculously overpriced, especially food and drinks.",
                "Tickets are way too expensive for a family of four.",
            ],
            "ascenseur": [
                "Elevator broke down during our visit, total disappointment.",
                "Only one lift working, massive wait to go back down.",
            ],
            "securite": [
                "Watch out for pickpockets, they are everywhere around the tower.",
                "Felt unsafe with all the street vendors harassing tourists.",
            ],
        },
        "neutral": {
            "generique": [
                "Decent visit, nothing extraordinary but worth doing once.",
                "Iconic monument, the experience is okay overall.",
            ],
        },
    },
    "es": {
        "positive": {
            "vue": [
                "La vista desde la cima es impresionante, se ve todo París.",
                "Panorama espectacular al atardecer, momento inolvidable.",
            ],
            "experience": [
                "Experiencia mágica, el monumento es aún más impresionante en persona.",
                "Visita perfecta en familia, los niños lo adoraron.",
            ],
        },
        "negative": {
            "file_attente": [
                "Cola interminable de más de 2 horas, una pesadilla.",
                "Demasiada espera incluso con entradas online.",
            ],
            "prix": [
                "Precios excesivos, sobre todo la comida en el lugar.",
                "Entradas demasiado caras para una familia.",
            ],
        },
    },
    "de": {
        "positive": {
            "vue": [
                "Die Aussicht vom Gipfel ist atemberaubend, ganz Paris liegt einem zu Füßen.",
                "Wunderschöner Panoramablick, vor allem bei Sonnenuntergang.",
            ],
            "experience": [
                "Magische Erfahrung, das Monument ist live noch beeindruckender.",
            ],
        },
        "negative": {
            "file_attente": [
                "Endlose Warteschlange, über 2 Stunden trotz Online-Ticket.",
                "Riesige Schlangen auf jeder Ebene, sehr enttäuschend.",
            ],
            "prix": [
                "Viel zu teuer, besonders Essen und Getränke.",
            ],
        },
    },
    "it": {
        "positive": {
            "vue": [
                "Vista mozzafiato dalla cima, si vede tutta Parigi.",
                "Panorama spettacolare al tramonto, momento indimenticabile.",
            ],
        },
        "negative": {
            "file_attente": [
                "Coda interminabile di oltre 2 ore, un incubo.",
            ],
            "prix": [
                "Prezzi eccessivi, soprattutto la ristorazione.",
            ],
        },
    },
}

# Mapping thématique français standardisé
TOPIC_MAP = {
    "vue": "Vue / Panorama",
    "personnel": "Personnel / Accueil",
    "experience": "Expérience globale",
    "file_attente": "File d'attente",
    "prix": "Prix / Tarification",
    "ascenseur": "Ascenseurs",
    "proprete": "Propreté",
    "securite": "Sécurité",
    "generique": "Général",
}


def generate_reviews(n: int = 5000) -> pd.DataFrame:
    """Génère N avis multilingues réalistes."""
    rows = []
    # Pondération réaliste : majoritairement positif (TripAdvisor 4.5/5 réel)
    sentiment_weights = {"positive": 0.62, "negative": 0.22, "neutral": 0.16}
    lang_weights = {"fr": 0.32, "en": 0.40, "es": 0.12, "de": 0.10, "it": 0.06}

    start_date = datetime(2024, 1, 1)

    for i in range(n):
        lang = random.choices(list(lang_weights.keys()), weights=list(lang_weights.values()))[0]
        if lang not in REVIEW_TEMPLATES:
            lang = "en"

        sentiment = random.choices(
            list(sentiment_weights.keys()), weights=list(sentiment_weights.values())
        )[0]

        templates = REVIEW_TEMPLATES[lang].get(sentiment)
        if not templates:
            # fallback
            templates = REVIEW_TEMPLATES["en"][sentiment]

        topic_key = random.choice(list(templates.keys()))
        text = random.choice(templates[topic_key])

        # rating cohérent avec sentiment
        if sentiment == "positive":
            rating = random.choices([4, 5], weights=[0.3, 0.7])[0]
        elif sentiment == "negative":
            rating = random.choices([1, 2], weights=[0.4, 0.6])[0]
        else:
            rating = 3

        date = start_date + timedelta(days=random.randint(0, 700))

        rows.append(
            {
                "review_id": f"R{i:06d}",
                "date": date.strftime("%Y-%m-%d"),
                "language": lang,
                "rating": rating,
                "sentiment": sentiment,
                "topic": TOPIC_MAP[topic_key],
                "topic_key": topic_key,
                "text": text,
                "source": random.choices(
                    ["TripAdvisor", "Google Maps", "Booking"], weights=[0.5, 0.4, 0.1]
                )[0],
            }
        )

    df = pd.DataFrame(rows)
    return df


# ----------------------------- AFFLUENCE ----------------------------- #

def generate_attendance(start: str = "2024-01-01", end: str = "2026-05-19") -> pd.DataFrame:
    """Génère affluence horaire avec patterns saisonniers / hebdo / horaires."""
    dates = pd.date_range(start=start, end=end, freq="h")
    rows = []

    for ts in dates:
        # Ouverture 9h30 -> 23h45
        if ts.hour < 9 or ts.hour >= 24:
            continue

        # Base journalière selon saison
        month = ts.month
        if month in (7, 8):  # haute saison été
            base = 4500
        elif month in (4, 5, 6, 9, 10):  # moyenne saison
            base = 2800
        elif month in (12, 1):  # Noël/NY tourisme
            base = 2200
        else:  # basse saison
            base = 1500

        # Effet weekend
        if ts.weekday() >= 5:
            base *= 1.35

        # Pic horaire : courbe en cloche centrée 15h
        hour_factor = np.exp(-((ts.hour - 15) ** 2) / 40)
        # Soir : pic illumination
        if 20 <= ts.hour <= 22:
            hour_factor = max(hour_factor, 0.85)

        # Bruit aléatoire
        noise = np.random.normal(1.0, 0.12)

        # Météo dégradée occasionnelle (-30%)
        weather_penalty = 1.0
        if random.random() < 0.08:
            weather_penalty = 0.7

        visitors = int(base * hour_factor * noise * weather_penalty)
        visitors = max(0, visitors)

        rows.append(
            {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "date": ts.strftime("%Y-%m-%d"),
                "hour": ts.hour,
                "day_of_week": ts.weekday(),
                "month": month,
                "is_weekend": int(ts.weekday() >= 5),
                "is_holiday_season": int(month in (7, 8, 12)),
                "visitors": visitors,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------- FAQ ----------------------------- #

FAQ_DATA = [
    {
        "question": "Quels sont les horaires d'ouverture de la Tour Eiffel ?",
        "answer": "La Tour Eiffel est ouverte tous les jours de 9h30 à 23h45 (dernière montée à 22h30 en ascenseur, 22h00 à pied). En haute saison (mi-juin à début septembre), elle ouvre dès 9h00.",
        "category": "Horaires",
    },
    {
        "question": "Combien coûte un billet pour la Tour Eiffel ?",
        "answer": "Les tarifs 2026 sont : Adulte sommet 35,30€ (ascenseur), 22,40€ (escalier). Jeune 12-24 ans : 17,70€ (ascenseur sommet). Enfant 4-11 ans : 8,90€. Gratuit pour moins de 4 ans.",
        "category": "Tarifs",
    },
    {
        "question": "Comment réserver mes billets pour la Tour Eiffel ?",
        "answer": "Réservation fortement recommandée sur le site officiel toureiffel.paris jusqu'à 60 jours à l'avance. Cela évite les longues files d'attente sur place.",
        "category": "Billetterie",
    },
    {
        "question": "La Tour Eiffel est-elle accessible aux personnes à mobilité réduite ?",
        "answer": "Oui, les 1er et 2ème étages sont accessibles en fauteuil roulant via les ascenseurs. Le 3ème étage (sommet) n'est malheureusement pas accessible aux fauteuils. Tarif réduit pour les PMR et leur accompagnant.",
        "category": "Accessibilité",
    },
    {
        "question": "Y a-t-il des restaurants dans la Tour Eiffel ?",
        "answer": "Oui, plusieurs options : Madame Brasserie au 1er étage (cuisine bistronomique), Le Jules Verne au 2ème étage (étoilé Michelin, chef Frédéric Anton), et plusieurs buffets/snacks à chaque étage.",
        "category": "Restauration",
    },
    {
        "question": "Combien de temps faut-il prévoir pour la visite ?",
        "answer": "Comptez entre 2h30 et 4h pour une visite complète incluant les 3 étages, les files d'attente et le temps sur place. Plus en haute saison.",
        "category": "Visite",
    },
    {
        "question": "Quelle est la meilleure heure pour visiter la Tour Eiffel ?",
        "answer": "Pour éviter la foule : ouverture (9h30) ou en soirée après 21h. Pour la vue : 1h avant le coucher du soleil pour profiter du jour ET de la nuit avec les illuminations.",
        "category": "Visite",
    },
    {
        "question": "La Tour Eiffel scintille-t-elle vraiment ?",
        "answer": "Oui ! Le scintillement doré a lieu pendant 5 minutes au début de chaque heure, de la tombée de la nuit jusqu'à 23h (1h en été). 20 000 ampoules s'allument simultanément.",
        "category": "Animations",
    },
    {
        "question": "Quelle est la hauteur exacte de la Tour Eiffel ?",
        "answer": "330 mètres avec l'antenne (depuis 2022). 300 mètres jusqu'au sommet de la structure. Construite en 1889 par Gustave Eiffel pour l'Exposition Universelle.",
        "category": "Histoire",
    },
    {
        "question": "Peut-on monter à la Tour Eiffel à pied ?",
        "answer": "Oui, vous pouvez monter par les escaliers jusqu'au 2ème étage (674 marches). Tarif réduit : 22,40€ adulte. Le 3ème étage n'est accessible qu'en ascenseur.",
        "category": "Accès",
    },
    {
        "question": "Y a-t-il un parking à proximité de la Tour Eiffel ?",
        "answer": "Pas de parking dédié. Parkings publics à proximité : Quai Branly, Pont d'Iéna, Tour Eiffel Bir-Hakeim. Le métro (Bir-Hakeim ligne 6) ou RER C (Champ de Mars) est recommandé.",
        "category": "Accès",
    },
    {
        "question": "What are the opening hours of the Eiffel Tower?",
        "answer": "The Eiffel Tower is open daily from 9:30am to 11:45pm (last elevator up at 10:30pm). In high season (mid-June to early September), it opens at 9:00am.",
        "category": "Hours",
    },
    {
        "question": "How much does an Eiffel Tower ticket cost?",
        "answer": "2026 prices: Adult top floor 35.30€ (elevator), 22.40€ (stairs). Youth 12-24: 17.70€. Child 4-11: 8.90€. Free under 4. Book online to skip the queue.",
        "category": "Prices",
    },
    {
        "question": "Is the Eiffel Tower wheelchair accessible?",
        "answer": "Yes, the 1st and 2nd floors are wheelchair accessible via elevators. The summit (3rd floor) is unfortunately not accessible. Reduced rate for disabled visitors and their companion.",
        "category": "Accessibility",
    },
    {
        "question": "Que voit-on depuis le sommet de la Tour Eiffel ?",
        "answer": "Par temps clair, vue panoramique à 360° jusqu'à 70 km : Sacré-Cœur, Notre-Dame, Arc de Triomphe, La Défense, Versailles. Au sommet, le bureau de Gustave Eiffel a été reconstitué.",
        "category": "Visite",
    },
]


# ----------------------------- VEILLE TARIFAIRE ----------------------------- #

COMPETITORS = [
    {"name": "Musée du Louvre", "price_adult": 22, "price_youth": 17, "category": "Musée"},
    {"name": "Château de Versailles", "price_adult": 21, "price_youth": 16, "category": "Monument"},
    {"name": "Arc de Triomphe", "price_adult": 16, "price_youth": 13, "category": "Monument"},
    {"name": "Disneyland Paris (1j)", "price_adult": 105, "price_youth": 95, "category": "Parc"},
    {"name": "Bateaux Mouches", "price_adult": 15, "price_youth": 7, "category": "Croisière"},
    {"name": "Catacombes de Paris", "price_adult": 29, "price_youth": 23, "category": "Monument"},
    {"name": "Sainte-Chapelle", "price_adult": 13, "price_youth": 11, "category": "Monument"},
    {"name": "Centre Pompidou", "price_adult": 15, "price_youth": 12, "category": "Musée"},
    {"name": "Panthéon", "price_adult": 13, "price_youth": 11, "category": "Monument"},
    {"name": "Musée d'Orsay", "price_adult": 16, "price_youth": 13, "category": "Musée"},
]


# ----------------------------- MAIN ----------------------------- #


def main() -> None:
    print("[1/4] Génération des avis multilingues...")
    reviews = generate_reviews(5000)
    reviews.to_csv(DATA_DIR / "reviews.csv", index=False)
    print(f"  -> {len(reviews)} avis générés ({reviews['language'].value_counts().to_dict()})")

    print("[2/4] Génération de l'historique d'affluence (2 ans, horaire)...")
    attendance = generate_attendance()
    attendance.to_csv(DATA_DIR / "attendance.csv", index=False)
    print(f"  -> {len(attendance):,} lignes horaires ({attendance['visitors'].sum():,} visiteurs total)")

    print("[3/4] FAQ pour le RAG...")
    with open(DATA_DIR / "faq.json", "w", encoding="utf-8") as fp:
        json.dump(FAQ_DATA, fp, ensure_ascii=False, indent=2)
    print(f"  -> {len(FAQ_DATA)} entrées FAQ")

    print("[4/4] Tarifs concurrents (veille tarifaire)...")
    with open(DATA_DIR / "competitors.json", "w", encoding="utf-8") as fp:
        json.dump(COMPETITORS, fp, ensure_ascii=False, indent=2)
    print(f"  -> {len(COMPETITORS)} attractions concurrentes")

    print("\nTous les datasets générés dans data/")


if __name__ == "__main__":
    main()
