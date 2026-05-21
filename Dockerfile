# --- Stage 1 : builder ----------------------------------------------------- #
FROM python:3.12-slim AS builder

WORKDIR /app

# Dépendances système nécessaires à la compilation des wheels (scikit-learn etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installe les dépendances dans un venv isolé (gunicorn inclus pour la prod)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# --- Stage 2 : runtime ----------------------------------------------------- #
FROM python:3.12-slim AS runtime

# Sécurité : utilisateur non-root
RUN useradd --create-home --shell /bin/bash --uid 1000 user

WORKDIR /app

# Copie du venv préparé
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Copie du code applicatif et des artefacts ML
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user frontend/ ./frontend/
COPY --chown=user:user data/ ./data/
COPY --chown=user:user ml_artifacts/ ./ml_artifacts/

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health', timeout=3)" || exit 1

# Gunicorn (déjà installé dans le venv builder), 2 workers x 4 threads
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "--timeout", "60", "--access-logfile", "-", "backend.app:app"]
