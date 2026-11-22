# AGENTS.md — NQS Project Handover & Operational Rules (v0.2.0)

> **For any new AI agent taking over this workspace:** Read this file first. It is the
> authoritative, live specification of the repository's security invariants, coding rules,
> verified cloud architecture, compatibility pins, and operational commands.

---

## 1. Non-Negotiable Operational Rules (Strict Priority Order)

### Rule #1 — Public GitHub Secret & Billing Safety (HIGHEST PRIORITY)
This repository (`https://github.com/jiahao116/NQS`) is **PUBLIC**. Preventing credential leaks
and unexpected cloud/credit-card charges overrides every other goal:
1. **NEVER commit or echo secrets into any tracked file**, including:
   - API keys / tokens (`WANDB_API_KEY`, Google/GCP keys, GitHub PATs, OpenAI keys, bearer tokens)
   - Credential files (`.env`, `application_default_credentials.json`, `*service-account*.json`, `*oauth*.json`, `monitor/monitor_config.yaml`, `monitor/.watchdog_state.json`)
   - SSH keys (`id_ed25519*`, `id_rsa*`, `*.pem`, `*.key`)
   - Sensitive private identifiers: GCP Project ID, GCP VM External IP, private `NTFY_TOPIC`, or personal account emails.
2. **Where secrets actually live (out-of-repo only):**
   - Local Windows PC: SSH private key & host alias `nqs-vm` live exclusively in `~/.ssh/config` and `~/.ssh/id_ed25519_nqs`.
   - GCP VM (`nqs-vm`): `WANDB_API_KEY` & optional `NTFY_TOPIC` live in `~/NQS/.env` (`chmod 600`); Google Colab CLI ADC credentials live in `~/.config/gcloud/application_default_credentials.json`.
   - Web Colab fallback: `WANDB_API_KEY` lives exclusively in Colab's **Secrets** sidebar (`google.colab.userdata`).
3. **Mandatory gate before EVERY `git commit` / `git push`:**
   - Run `python scripts/check_secrets.py` (also enforced automatically by `.git/hooks/pre-commit`) and `python tests/test_workflow_local.py`.
   - Never use `git commit --no-verify` to bypass the secret guard.
   - Never run `git add -A` blindly if untracked local files exist; inspect `git status --short` first.
4. **Never click "Save to GitHub" in the Colab web UI** (it can overwrite the clean notebook with execution outputs or temporary cells).

### Rule #2 — Language Split
- **Conversation with the user**: Always reply in **Chinese**.
- **All repository code artifacts**: **100% English only** — every `.py`, `.sh`, `.yaml`, `.ipynb`, `pyproject.toml`, `requirements*.txt`, comment, log line, alert message, and commit message. Enforced automatically by `tests/test_workflow_local.py::test_code_and_config_files_are_english_only`.
- Human-oriented documentation in `docs/WORKFLOW_GUIDE.md` may be written in Chinese; `AGENTS.md` and `README.md` are maintained in English.

### Rule #3 — Living Documentation Hygiene (Update Every Workflow Change & Prune Obsolete Content)
Whenever you modify the workflow, scripts, dependencies, models, or cloud infrastructure:
1. Update **`AGENTS.md`**, **`README.md`**, and **`docs/WORKFLOW_GUIDE.md`** in the **same commit**.
2. **Delete obsolete sections, deprecated flags, dead scripts, and outdated workarounds immediately** — do not leave historical clutter that could mislead the next agent.

### Rule #4 — Protect Colab Compute Units & GCP Always-Free Tier
- **Colab Compute Units (Colab Pro+)**:
  - Always call `bash NQS/scripts/colab_quick.sh stop` immediately after finishing a short verification session.
  - Use `scripts/colab_queue.sh` (`colab run`) for unattended/long jobs so each Colab GPU VM is **automatically torn down on exit (even on error)**.
  - Verify `Active assignments: 0` via `ssh nqs-vm "~/.local/bin/colab --auth=adc sessions"` after interactive work.
- **GCP VM (`nqs-vm`)**:
  - Always-Free `e2-micro` (`us-central1`, Ubuntu 24.04 LTS, 20 GB Standard persistent disk, 2 GB `/swapfile`). Do not upgrade machine type, attach GPUs, reserve paid static IPs, or switch disk type without explicit user approval.
  - Never install heavy browser/IDE servers on `nqs-vm` (1 GB physical RAM).

---

## 2. Current Verified Architecture (v0.2.0 — Option A Dual Workflow)

The user's local Windows PC may frequently lose power/internet, so **zero long-running compute or monitoring depends on the local PC**:

```text
Local Windows PC (Antigravity + passwordless `ssh nqs-vm`)
   │
   ├── git push ──► GitHub (public: jiahao116/NQS, single source of truth)
   │
   └── ssh nqs-vm ──► GCP Always-Free e2-micro Linux VM (`nqs-vm`, 24/7 online, no Antigravity)
                         ├── Official `google-colab-cli` (`--auth=adc`, installed via `uv tool`)
                         │     ├── Workflow 1 (Short Verification): `scripts/colab_quick.sh`
                         │     │     `colab new` -> hot `colab exec --timeout 0` -> `colab stop`
                         │     └── Workflow 2 (Long / Batch Queue): `scripts/colab_queue.sh`
                         │           Detached `tmux` session running sequential `colab run --timeout 0`
                         │           (auto-provisions GPU -> syncs W&B -> auto-releases GPU)
                         ├── `nqs-watchdog.service` (systemd 24/7 daemon, < 40 MB RAM)
                         │     Polls W&B heartbeats every 2 min -> alerts phone via `ntfy.sh`
                         └── Google Colab Pro+ GPU (T4 / L4 / A100) ◄──► Weights & Biases (`nqs-colab`)
                               Drive-free live curves + atomic `.mpack` checkpoint Artifacts
```

### Verified Environment & Hard-Won Compatibility Pins (Colab 2026 / Python 3.13)
Verified end-to-end on Colab `T4 GPU` (both web notebook and headless `google-colab-cli`):
- **Verified package versions**: `JAX 0.11.1` (Colab native CUDA 13), `Flax 0.12.12`, `NetKet 3.23.0`, `optax 0.2.9`, `beartype 0.22.9`, `plum-dispatch 2.7.1`, `wandb 0.31.2`.
- **Critical Gotcha #1 (CUDA 13 vs CUDA 12)**: Colab 2026 ships with native CUDA 13 (`jax-cuda13-plugin`). **NEVER install `netket[cuda]`** (which pulls `jax-cuda12-plugin` and causes `ALREADY_EXISTS: PJRT_Api already exists for device type cuda`). Always install plain `netket`.
- **Critical Gotcha #2 (`beartype 0.23` dispatch break)**: `beartype>=0.23` changed `is_subhint(Any, T)`, breaking `plum-dispatch` inside `netket/utils/numbers.py` (`TypeError: cannot deduce dtype of object type <class 'int'>`). Always pin **`beartype<0.23`** and **`plum-dispatch<2.8`**.
- **Critical Gotcha #3 (NetKet 3.23 VMC API)**: `driver.iter(chunk)` was removed in NetKet 3.23. Always use **`driver.run(n_iter=chunk, out=(), show_progress=False)`** (already implemented in `src/nqs/train.py`).
- **Critical Gotcha #4 (Clean subprocess + live streaming)**: `scripts/colab_remote_runner.py` runs `python -u -m nqs.train` inside a clean `subprocess.Popen` child process piping `stdout` line-by-line to Python `sys.stdout`. This simultaneously avoids stale in-memory `beartype` imports (no kernel restart needed) and streams live `[Step ...]` logs through Jupyter/Colab CLI.
- **Critical Gotcha #5 (`--timeout 0`)**: `colab exec` and `colab run` default to `--timeout 30.0` seconds of stdout silence, which kills JAX JIT compilation. Always pass **`--timeout 0`**.
- **Critical Gotcha #6 (Windows PowerShell -> SSH quoting)**: Wrap remote SSH commands in single quotes `'...'` to prevent PowerShell from expanding `$` variables locally.

---

## 3. Exact Operational Command Cheat Sheet

> All remote commands assume local `~/.ssh/config` defines `Host nqs-vm`. On Antigravity Windows sandbox, `git push` and `ssh nqs-vm` require `BypassSandbox: true` (try sandboxed first per runtime rules).

### A. Local Pre-Push Verification (Mandatory Before Every Push)
```powershell
python scripts/check_secrets.py
python tests/test_workflow_local.py
```

### B. Workflow 1 — Short Verification / Fast Iteration (Hot Colab Session)
```powershell
# 1. Start a hot Colab GPU session (`nqs-quick`)
ssh nqs-vm 'bash NQS/scripts/colab_quick.sh start T4'

# 2. Run / iterate rapidly (pulls latest GitHub `main`, skips pip install after first run)
ssh nqs-vm 'bash NQS/scripts/colab_quick.sh run configs/tfim_1d_rbm.yaml --fresh'

# 3. ALWAYS stop the session when finished to protect Compute Units!
ssh nqs-vm 'bash NQS/scripts/colab_quick.sh stop'
```

### C. Workflow 2 — Long / Batch Unattended Queue (Survives Local PC Shutdown)
```powershell
# Launch detached tmux queue on nqs-vm (each job uses `colab run` & auto-releases GPU)
ssh nqs-vm 'bash NQS/scripts/colab_queue.sh submit T4 configs/tfim_1d_rbm.yaml configs/tfim_1d_gpt.yaml'

# Check queue status & latest log tail
ssh nqs-vm 'bash NQS/scripts/colab_queue.sh status'

# Abort queue if needed (also check `colab sessions` afterwards)
ssh nqs-vm 'bash NQS/scripts/colab_queue.sh stop'
```

### D. Monitoring, W&B Status Report, & Compute Unit Balance
```powershell
# Check remaining Colab Compute Units & verify zero leaked GPU sessions
ssh nqs-vm '~/.local/bin/colab --auth=adc usage && ~/.local/bin/colab --auth=adc sessions'

# Print Markdown summary table of recent W&B runs
ssh nqs-vm 'set -a; source NQS/.env; set +a; NQS/.venv-monitor/bin/python NQS/monitor/gcp_watchdog.py --config NQS/monitor/monitor_config.yaml --once --markdown'

# Check 24/7 watchdog daemon status
ssh nqs-vm 'systemctl is-active nqs-watchdog.service'
```

### E. Web Notebook Fallback
- Direct URL: `https://colab.research.google.com/github/jiahao116/NQS/blob/main/notebooks/colab_runner.ipynb`

---

## 4. Repository File Map

| Path | Role |
| :--- | :--- |
| `AGENTS.md` | **This file** — Master handover specification & mandatory agent rules (keep always current). |
| `README.md` | Public GitHub repository overview (English, zero private identifiers). |
| `docs/WORKFLOW_GUIDE.md` | Human-facing operational handbook (Chinese, always synced with v0.2 workflow). |
| `src/nqs/config.py` | Experiment dataclasses (`SystemConfig`, `ModelConfig`, `TrainingConfig`, `TelemetryConfig`) & validation. |
| `src/nqs/physics.py` | Quantum spin Hamiltonians (`tfim_1d`, `heisenberg_1d`, `heisenberg_2d`) & Lanczos Exact Diagonalization. |
| `src/nqs/models.py` | Variational ansatzes: `RBM` (+ SR), `ARNNDense`, and `CausalTransformerWaveFunction` (`gpt`, exact direct sampling). |
| `src/nqs/checkpoint.py` | Atomic `.mpack` + JSON checkpointing with Drive-free W&B Artifact upload & auto-restore. |
| `src/nqs/train.py` | Main VMC driver loop (`driver.run`), Exact ED relative error, V-score, heartbeats, NaN guard, convergence plotting. |
| `configs/*.yaml` | Reproducible experiment configs (`tfim_1d_rbm.yaml`, `tfim_1d_gpt.yaml`, `heisenberg_1d_gpt.yaml`). |
| `scripts/colab_remote_runner.py` | Remote Colab VM runner invoked by `colab exec` / `colab run` (idempotent env + streaming subprocess). |
| `scripts/colab_quick.sh` | Workflow 1 wrapper on `nqs-vm` for short interactive hot-GPU sessions. |
| `scripts/colab_queue.sh` | Workflow 2 wrapper on `nqs-vm` for detached `tmux` unattended batch queues (`colab run`). |
| `scripts/bootstrap_vm.sh` | One-shot Ubuntu GCP `e2-micro` setup (2 GB swap, `tmux`, `uv`, `google-colab-cli`, `.venv-monitor`). |
| `scripts/check_secrets.py` | Secret & forbidden-file scanner (working tree, staged changes, and full Git history). |
| `scripts/install_git_hooks.sh` | Installs `.git/hooks/pre-commit` secret guard locally. |
| `monitor/gcp_watchdog.py` | 24/7 lightweight W&B heartbeat watchdog (`< 40 MB` RAM) + mobile `ntfy.sh` / webhook alerts + `--markdown` reporter. |
| `monitor/setup_gcp_vm.sh` | Writes local `.env` (`chmod 600`) and registers `nqs-watchdog.service` via `systemd`. |
| `monitor/monitor_config.example.yaml` | Safe template for `monitor/monitor_config.yaml` (real file is gitignored). |
| `notebooks/colab_runner.ipynb` | Verified browser fallback notebook for Google Colab GPU. |
| `tests/test_workflow_local.py` | Host-side test suite (zero-secret check, English-only code check, AST/ipynb parse, config validation, watchdog state machine). |

---

## 5. Verified Baseline Benchmarks (Colab T4 GPU)

1D Critical Transverse-Field Ising Model ($N = 12, h = 1.0, \text{PBC}$), Exact Lanczos ED $E_0 = -15.32259515$ ($\Delta E = 0.13108693$):
- **`configs/tfim_1d_rbm.yaml`** (`RBM` + `SR`, 324 params, 300 steps): `E = -15.322866 ± 0.002724`, **Relative Error `1.77e-05`**, V-score `3.51e-04`, ~36s (~`0.03` Compute Units via headless `colab-cli`).
- **`configs/tfim_1d_gpt.yaml`** (`CausalTransformerWaveFunction`, 17,698 params, 400 steps): `E = -15.272748 ± 0.015628`, **Relative Error `3.25e-03`**, V-score `1.08e-02`, ~112s.
