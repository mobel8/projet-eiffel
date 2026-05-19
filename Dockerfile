# --- Stage 1 : builder ----------------------------------------------------- #
FROM python:3.12-slim AS builder

WORKDIR /app

# Dépendances système nécessaires à la compilation des wheels (scikit-learn etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installe les dépendances dans un venv isolé
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2 : runtime ----------------------------------------------------- #
FROM python:3.12-slim AS runtime

# Sécurité : utilisateur non-root
RUN useradd --create-home --shell /bin/bash eiffel

WORKDIR /app

# Copie du venv préparé
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5050

# Copie du code applicatif et des artefacts ML
COPY --chown=eiffel:eiffel backend/ ./backend/
COPY --chown=eiffel:eiffel frontend/ ./frontend/
COPY --chown=eiffel:eiffel data/ ./data/
COPY --chown=eiffel:eiffel ml_artifacts/ ./ml_artifacts/

USER eiffel

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5050/api/health', timeout=3)" || exit 1

# Gunicorn pour la prod (4 workers, timeout 60s)
RUN pip install --no-cache-dir gunicorn==21.2.0

CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "2", "--threads", "4", "--timeout", "60", "--access-logfile", "-", "backend.app:app"]
