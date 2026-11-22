#!/usr/bin/env bash
# ==============================================================================
# Google Cloud VM (Ubuntu/Debian) One-Click Watchdog Setup Script
# Installs a minimal Python venv and registers a systemd background daemon.
#
# Usage:
#   export WANDB_API_KEY="your_wandb_api_key"
#   export NTFY_TOPIC="your-private-ntfy-topic"
#   bash monitor/setup_gcp_vm.sh
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv-monitor"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: Please set WANDB_API_KEY first: export WANDB_API_KEY='your_key'"
  exit 1
fi

echo "[1/4] Installing python3-venv and pip..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip

echo "[2/4] Creating lightweight monitor venv at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -r "${REPO_DIR}/requirements-monitor.txt" -q

if [[ ! -f "${REPO_DIR}/monitor/monitor_config.yaml" ]]; then
  cp "${REPO_DIR}/monitor/monitor_config.example.yaml" "${REPO_DIR}/monitor/monitor_config.yaml"
  echo "Created default config: ${REPO_DIR}/monitor/monitor_config.yaml"
fi

ENV_FILE="${REPO_DIR}/.env"
cat > "${ENV_FILE}" <<EOF
WANDB_API_KEY=${WANDB_API_KEY}
NTFY_TOPIC=${NTFY_TOPIC:-}
EOF
chmod 600 "${ENV_FILE}"

echo "[3/4] Registering systemd service (nqs-watchdog.service)..."
SERVICE_FILE="/etc/systemd/system/nqs-watchdog.service"
sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=NQS Google Colab 24/7 Telemetry Watchdog
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${VENV_DIR}/bin/python ${REPO_DIR}/monitor/gcp_watchdog.py --config ${REPO_DIR}/monitor/monitor_config.yaml
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
EOF

echo "[4/4] Enabling and starting nqs-watchdog.service..."
sudo systemctl daemon-reload
sudo systemctl enable --now nqs-watchdog.service
sudo systemctl status nqs-watchdog.service --no-pager

echo ""
echo "Setup complete! Useful commands:"
echo "  Stream live logs : journalctl -u nqs-watchdog -f"
echo "  Print status MD  : ${VENV_DIR}/bin/python ${REPO_DIR}/monitor/gcp_watchdog.py --once --markdown"
