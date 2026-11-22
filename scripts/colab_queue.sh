#!/usr/bin/env bash
# ==============================================================================
# Workflow 2: Unattended Long / Batch Queue inside tmux on a remote Linux host
# Uses `colab run` so each job provisions a fresh Colab GPU, syncs W&B curves
# and checkpoints, and AUTOMATICALLY releases the GPU immediately on exit.
# Survives local machine power-off / network disconnect completely!
#
# Subcommands:
#   submit <GPU> <config1.yaml> [config2.yaml ...]   Launch detached tmux queue
#   status                                           Show tmux status & latest log tail
#   stop                                             Kill running tmux queue
# ==============================================================================
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
TMUX_SESSION="nqs-queue"
AUTH_MODE="${COLAB_AUTH_MODE:-adc}"

mkdir -p "${LOG_DIR}"

if [[ -f "${REPO_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.env"
  set +a
fi

_notify() {
  local title="$1"
  local body="$2"
  local prio="${3:-default}"
  if [[ -n "${NTFY_TOPIC:-}" ]]; then
    curl -fsS -H "Title: ${title}" -H "Priority: ${prio}" -d "${body}" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null || true
  fi
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  submit)
    if [[ $# -lt 2 ]]; then
      echo "Usage: bash scripts/colab_queue.sh submit <T4|L4|A100> <config1.yaml> [config2.yaml ...]"
      exit 1
    fi
    gpu="$1"
    shift
    configs=("$@")

    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "ERROR: tmux session '${TMUX_SESSION}' is already running. Check status or stop it first."
      exit 1
    fi

    git -C "${REPO_DIR}" pull --ff-only origin main || true
    ts="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_DIR}/queue_${ts}.log"
    ln -sf "${log_file}" "${LOG_DIR}/latest_queue.log"

    # Write a standalone runner script for tmux to execute cleanly
    job_script="${LOG_DIR}/run_queue_${ts}.sh"
    {
      echo '#!/usr/bin/env bash'
      echo 'set -uo pipefail'
      echo "export PATH=\"\${HOME}/.local/bin:\${PATH}\""
      echo "REPO_DIR=\"${REPO_DIR}\""
      echo "AUTH_MODE=\"${AUTH_MODE}\""
      echo "if [[ -f \"\${REPO_DIR}/.env\" ]]; then set -a; source \"\${REPO_DIR}/.env\"; set +a; fi"
      echo "echo \"[Queue] Started at \$(date -u) | GPU=${gpu} | Jobs=${#configs[@]}\""
      echo "FAILED=0"
      for cfg in "${configs[@]}"; do
        echo "echo \"============================================================\""
        echo "echo \"[Queue] Launching ephemeral Colab ${gpu} job: ${cfg}\""
        echo "ENV_ARGS=(--env \"NQS_CONFIG=${cfg}\")"
        echo "if [[ -n \"\${WANDB_API_KEY:-}\" ]]; then ENV_ARGS+=(--env \"WANDB_API_KEY=\${WANDB_API_KEY}\"); fi"
        echo "if colab --auth=\"\${AUTH_MODE}\" run --gpu \"${gpu}\" --timeout 0 \"\${ENV_ARGS[@]}\" \"\${REPO_DIR}/scripts/colab_remote_runner.py\"; then"
        echo "  echo \"[Queue] SUCCESS: ${cfg} (Colab GPU auto-released)\""
        echo "else"
        echo "  echo \"[Queue] FAILED: ${cfg} (Colab GPU auto-released)\""
        echo "  FAILED=\$((FAILED + 1))"
        echo "fi"
      done
      echo "echo \"[Queue] Finished all jobs at \$(date -u) | Failed=\${FAILED}\""
      echo "if [[ -n \"\${NTFY_TOPIC:-}\" ]]; then"
      echo "  if [[ \"\${FAILED}\" -eq 0 ]]; then"
      echo "    curl -fsS -H 'Title: NQS Queue Completed' -d \"All ${#configs[@]} Colab ${gpu} jobs finished cleanly!\" \"https://ntfy.sh/\${NTFY_TOPIC}\" >/dev/null || true"
      echo "  else"
      echo "    curl -fsS -H 'Title: NQS Queue Finished With Errors' -H 'Priority: high' -d \"\${FAILED}/${#configs[@]} jobs failed. Check logs on the control host.\" \"https://ntfy.sh/\${NTFY_TOPIC}\" >/dev/null || true"
      echo "  fi"
      echo "fi"
    } > "${job_script}"
    chmod +x "${job_script}"

    tmux new-session -d -s "${TMUX_SESSION}" "bash '${job_script}' 2>&1 | tee -a '${log_file}'"
    echo "[Queue] Launched detached tmux session '${TMUX_SESSION}' with ${#configs[@]} job(s) on GPU=${gpu}."
    echo "[Queue] Log file: ${log_file}"
    echo "[Queue] Safe to shut down your local PC now!"
    ;;

  status)
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "[Queue] tmux session '${TMUX_SESSION}' is RUNNING."
    else
      echo "[Queue] tmux session '${TMUX_SESSION}' is NOT running."
    fi
    if [[ -f "${LOG_DIR}/latest_queue.log" ]]; then
      echo "--- Latest Queue Log (last 30 lines) ---"
      tail -n 30 "${LOG_DIR}/latest_queue.log"
    fi
    ;;

  stop)
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      tmux kill-session -t "${TMUX_SESSION}"
      echo "[Queue] Killed tmux session '${TMUX_SESSION}'."
    else
      echo "[Queue] No active tmux session '${TMUX_SESSION}'."
    fi
    # Killing the local `colab run` client does NOT release its remote GPU:
    # explicitly stop every ephemeral `run-*` session left behind by the queue.
    sleep 3
    for s in $(colab --auth="${AUTH_MODE}" sessions 2>/dev/null | grep -o '^\[run-[^]]*\]' | tr -d '[]'); do
      echo "[Queue] Releasing orphaned Colab session '${s}'..."
      colab --auth="${AUTH_MODE}" stop -s "${s}" || true
    done
    colab --auth="${AUTH_MODE}" sessions || true
    ;;

  *)
    echo "Usage: bash scripts/colab_queue.sh {submit <GPU> <config1> [config2 ...] | status | stop}"
    exit 1
    ;;
esac
