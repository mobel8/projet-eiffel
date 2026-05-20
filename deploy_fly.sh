#!/usr/bin/env bash
# Déploiement automatique EiffelPulse sur Fly.io.
# Pré-requis : être authentifié via `flyctl auth login`.
set -euo pipefail

FLYCTL="${FLYCTL:-$HOME/.fly/bin/flyctl}"

if ! "$FLYCTL" auth whoami >/dev/null 2>&1; then
  echo "Vous n'êtes pas authentifié."
  echo "Lancez d'abord : ! ~/.fly/bin/flyctl auth login"
  exit 1
fi

echo "==> Authentifié en tant que $("$FLYCTL" auth whoami)"

APP_NAME="${APP_NAME:-eiffelpulse-sete}"

# Vérifie si l'app existe déjà
if "$FLYCTL" apps list 2>/dev/null | grep -q "^${APP_NAME}\s"; then
  echo "==> L'application '${APP_NAME}' existe déjà, on déploie dessus."
else
  echo "==> Création de l'application '${APP_NAME}'..."
  if ! "$FLYCTL" apps create "${APP_NAME}" --org personal 2>&1; then
    # Nom déjà pris : on essaie avec un suffixe aléatoire
    APP_NAME="${APP_NAME}-$(date +%s | tail -c 5)"
    echo "==> Tentative avec nom alternatif : ${APP_NAME}"
    "$FLYCTL" apps create "${APP_NAME}" --org personal
  fi
  # Met à jour fly.toml avec le nom retenu
  sed -i "s|^app = .*|app = \"${APP_NAME}\"|" fly.toml
fi

echo "==> Déploiement en cours (région CDG · Paris)..."
"$FLYCTL" deploy --remote-only --ha=false

echo ""
echo "==> Déployé ! URL permanente :"
"$FLYCTL" status --app "${APP_NAME}" | head -20
echo ""
echo "    https://${APP_NAME}.fly.dev"
echo ""
