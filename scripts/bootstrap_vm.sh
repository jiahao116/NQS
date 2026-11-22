#!/usr/bin/env bash
# ==============================================================================
# Bootstrap script for a lightweight Linux control host (e.g. 1 GB RAM cloud VM).
# Sets up:
#   1. Permanent 2 GB swapfile (prevents OOM on 1 GB RAM hosts)
#   2. Base packages (git, tmux, curl, python3-venv, python3-pip)
#   3. uv + official google-colab-cli
#   4. NQS repository + lightweight monitor virtualenv
# ==============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jiahao116/NQS.git}"
REPO_DIR="${HOME}/NQS"

echo "[1/5] Ensuring 2 GB swapfile exists..."
if ! swapon --show | grep -q "/swapfile"; then
  if [[ ! -f /swapfile ]]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
  fi
  sudo swapon /swapfile
  if ! grep -q "^/swapfile " /etc/fstab; then
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
  fi
fi
free -h

echo "[2/5] Installing base packages (git, tmux, curl, python3-venv)..."
sudo apt-get update -qq
sudo apt-get install -y -qq git tmux curl python3-venv python3-pip

echo "[3/5] Installing uv and official google-colab-cli..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

if ! grep -q 'HOME/.local/bin' "${HOME}/.bashrc"; then
  echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
fi
if ! grep -q 'HOME/.local/bin' "${HOME}/.profile"; then
  echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${HOME}/.profile"
fi

uv tool install --upgrade google-colab-cli

echo "[4/5] Cloning or updating NQS repository at ${REPO_DIR}..."
if [[ -d "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" pull origin main
else
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

echo "[5/5] Creating lightweight monitor virtualenv..."
VENV_DIR="${REPO_DIR}/.venv-monitor"
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -r "${REPO_DIR}/requirements-monitor.txt" -q

echo ""
echo "Bootstrap complete!"
which colab
colab --help
