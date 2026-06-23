#!/usr/bin/env bash
# Bootstrap TRACE on a fresh Ubuntu VPS (run as root).
# Usage: curl -fsSL .../vps_bootstrap.sh | bash
set -euo pipefail

TRACE_DIR="${TRACE_DIR:-/opt/TRACE}"
REPO="${TRACE_REPO:-https://github.com/masoomprakhar/TRACE.git}"

echo "==> Installing Docker..."
apt-get update -qq
apt-get install -y -qq ca-certificates curl git
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | tee /etc/apt/sources.list.d/docker.list >/dev/null
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

echo "==> Cloning TRACE to ${TRACE_DIR}..."
if [[ -d "$TRACE_DIR/.git" ]]; then
  git -C "$TRACE_DIR" pull --ff-only
else
  mkdir -p "$(dirname "$TRACE_DIR")"
  git clone "$REPO" "$TRACE_DIR"
fi
cd "$TRACE_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  cat >> .env <<'EOF'

TRACE_CONFIG=config/roboflow-fast.yaml
TRACE_LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
EOF
  echo ""
  echo "!! Edit ${TRACE_DIR}/.env and set ROBOFLOW_API_KEY, then run:"
  echo "   cd ${TRACE_DIR} && docker compose -f docker-compose.prod.yml up -d --build"
  exit 0
fi

echo "==> Building and starting TRACE (this may take 15–20 minutes)..."
docker compose -f docker-compose.prod.yml up -d --build

echo "==> Seeding demo violations..."
docker compose -f docker-compose.prod.yml exec -T api python -m trace_cv.cli seed-demo -n 40 || true

IP=$(curl -fsSL -4 ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
echo ""
echo "==> TRACE is starting on http://${IP}:8000/#overview"
echo "    Ensure firewall allows TCP 8000 (ufw allow 8000/tcp)"
echo "    Logs: docker compose -f docker-compose.prod.yml logs -f api"
