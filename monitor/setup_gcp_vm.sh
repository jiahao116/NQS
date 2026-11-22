#!/usr/bin/env bash
# ==============================================================================
# Google Cloud Always-Free VM (e2-micro, Ubuntu/Debian) 一键部署脚本
# 作用：创建轻量 Python 虚拟环境并注册 systemd 守护进程，7x24 小时常驻监控 Colab
# 用法：
#   export WANDB_API_KEY="你的_WANDB_KEY"
#   export NTFY_TOPIC="你自己起的手机推送频道名"
#   bash monitor/setup_gcp_vm.sh
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv-monitor"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "❌ 请先设置环境变量: export WANDB_API_KEY='你的_W&B_API_Key'"
  exit 1
fi

echo "[1/4] 安装基础 Python venv 支持..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip

echo "[2/4] 创建超轻量监控虚拟环境 (${VENV_DIR})..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -r "${REPO_DIR}/requirements-monitor.txt" -q

if [[ ! -f "${REPO_DIR}/monitor/monitor_config.yaml" ]]; then
  cp "${REPO_DIR}/monitor/monitor_config.example.yaml" "${REPO_DIR}/monitor/monitor_config.yaml"
  echo "已生成默认配置: ${REPO_DIR}/monitor/monitor_config.yaml"
fi

ENV_FILE="${REPO_DIR}/.env"
cat > "${ENV_FILE}" <<EOF
WANDB_API_KEY=${WANDB_API_KEY}
NTFY_TOPIC=${NTFY_TOPIC:-}
EOF
chmod 600 "${ENV_FILE}"

echo "[3/4] 注册 systemd 开机自启守护服务 (nqs-watchdog.service)..."
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

echo "[4/4] 启动服务并查看状态..."
sudo systemctl daemon-reload
sudo systemctl enable --now nqs-watchdog.service
sudo systemctl status nqs-watchdog.service --no-pager

echo ""
echo "✅ 部署完成！常用命令："
echo "  查看实时监控日志 : journalctl -u nqs-watchdog -f"
echo "  打印当前实验报表 : ${VENV_DIR}/bin/python ${REPO_DIR}/monitor/gcp_watchdog.py --once --markdown"
