#!/usr/bin/env bash
# ==============================================================================
# Workflow 1: Fast Interactive / Short Verification via Hot Colab GPU Session
# Runs on the GCP Linux VM (nqs-vm), invoked via `ssh nqs-vm "bash ~/NQS/scripts/colab_quick.sh ..."`.
#
# Subcommands:
#   start [GPU]                          Allocate hot session `nqs-quick` (default: T4)
#   run <config.yaml> [--fresh] [--no-wandb]  Execute on hot session (streams live output)
#   status                               Show active Colab sessions & status
#   usage                                Show Colab Compute Units balance and burn rate
#   stop                                 Release `nqs-quick` GPU immediately to save Units
# ==============================================================================
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${COLAB_QUICK_SESSION:-nqs-quick}"
AUTH_MODE="${COLAB_AUTH_MODE:-adc}"

# Load optional WANDB_API_KEY / NTFY_TOPIC from local .env (never committed to Git)
if [[ -f "${REPO_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.env"
  set +a
fi

cmd="${1:-help}"
shift || true

case "${cmd}" in
  start)
    gpu="${1:-T4}"
    echo "[Quick] Provisioning hot Colab session '${SESSION_NAME}' with --gpu ${gpu}..."
    colab --auth="${AUTH_MODE}" new -s "${SESSION_NAME}" --gpu "${gpu}"
    ;;

  run)
    config="${1:-configs/tfim_1d_rbm.yaml}"
    shift || true
    fresh="0"
    no_wandb="0"
    for arg in "$@"; do
      case "${arg}" in
        --fresh) fresh="1" ;;
        --no-wandb) no_wandb="1" ;;
      esac
    done

    git -C "${REPO_DIR}" pull --ff-only origin main || true

    env_args=(
      --env "NQS_CONFIG=${config}"
      --env "NQS_FRESH=${fresh}"
      --env "NQS_NO_WANDB=${no_wandb}"
    )
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      env_args+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")
    fi

    echo "[Quick] Running '${config}' on hot session '${SESSION_NAME}'..."
    colab --auth="${AUTH_MODE}" exec \
      -s "${SESSION_NAME}" \
      --timeout 0 \
      "${env_args[@]}" \
      -f "${REPO_DIR}/scripts/colab_remote_runner.py"
    ;;

  status)
    colab --auth="${AUTH_MODE}" sessions
    colab --auth="${AUTH_MODE}" status -s "${SESSION_NAME}" || true
    ;;

  usage)
    colab --auth="${AUTH_MODE}" usage
    ;;

  stop)
    echo "[Quick] Stopping hot Colab session '${SESSION_NAME}' to save Compute Units..."
    colab --auth="${AUTH_MODE}" stop -s "${SESSION_NAME}"
    ;;

  *)
    echo "Usage: bash scripts/colab_quick.sh {start [T4|L4|A100] | run <config> [--fresh] [--no-wandb] | status | usage | stop}"
    exit 1
    ;;
esac
