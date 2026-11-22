# NQS Cloud Workflow (JAX / NetKet + Google Colab CLI + GCP Watchdog)

Version `0.2.0` — A cloud-first Neural Quantum States (NQS) computation and experiment management workflow built on **JAX + NetKet 3**, orchestrated via the official **`google-colab-cli`** on an Always-Free **GCP Linux VM**, with **Drive-free Weights & Biases** telemetry and automated secret-leak protection.

> **AI Agent Handover**: If you are an AI agent taking over this repository, read **[AGENTS.md](AGENTS.md)** first for mandatory security rules, verified compatibility pins, and exact operational commands.

## Architecture Overview

* **Workflow 1 — Short Verification (`scripts/colab_quick.sh`)**: Reuses a hot Google Colab GPU session (`colab new` -> `colab exec --timeout 0` -> `colab stop`) for rapid code iteration with live streamed training logs.
* **Workflow 2 — Long / Batch Unattended Queue (`scripts/colab_queue.sh`)**: Launches a detached `tmux` queue on the 24/7 GCP Linux VM (`nqs-vm`). Each job runs via `colab run --timeout 0` (provisions fresh Colab GPU -> runs NQS -> syncs W&B curves & checkpoints -> **auto-releases GPU immediately on exit**), completely independent of local PC power or network state.
* **Drive-Free Cloud Telemetry & Checkpoints (`Weights & Biases`)**: Live energy, relative error vs. Lanczos Exact Diagonalization, V-score, convergence plots, and periodic atomic `.mpack` checkpoints sync automatically to W&B Artifacts.
* **24/7 Lightweight Sentinel (`monitor/gcp_watchdog.py`)**: Runs as a `systemd` daemon (`< 40 MB` RAM) polling W&B heartbeats and sending mobile alerts on completion, disconnect, or `NaN` divergence.
* **Web Fallback (`notebooks/colab_runner.ipynb`)**: Verified one-click Google Colab GPU notebook for ad-hoc browser execution.
* **Built-In Secret Guard (`scripts/check_secrets.py`)**: Scans tracked/staged files and full Git history to block any API key, token, private key, or credential file from reaching public GitHub.

## Project Structure

```text
NQS/
├── AGENTS.md                        # Master agent handover manual & mandatory project rules
├── README.md                        # Public repository overview
├── docs/WORKFLOW_GUIDE.md           # Chinese operational guide
├── .agents/skills/                  # Auto-discovered NQS research skills (`quantax-core`, `measure`, `zero-var-extrap`, `symm-test`, `time-bench`)
├── configs/
│   ├── tfim_1d_rbm.yaml             # Starter benchmark: 1D Critical TFIM (L=12) + RBM + SR (rel_err ~1.8e-5)
│   ├── tfim_1d_gpt.yaml             # Autoregressive Causal Transformer on 1D Critical TFIM (L=12)
│   └── heisenberg_1d_gpt.yaml       # Complex-phase Causal Transformer on 1D Heisenberg chain (L=12)
├── src/nqs/                         # Core NQS package (config, physics, models, checkpoint, train)
├── scripts/
│   ├── bootstrap_vm.sh              # GCP e2-micro bootstrap (2 GB swap, tmux, uv, google-colab-cli)
│   ├── colab_remote_runner.py       # Remote Colab execution wrapper (idempotent env + live streaming)
│   ├── colab_quick.sh               # Workflow 1: hot-GPU quick verification
│   ├── colab_queue.sh               # Workflow 2: detached tmux unattended queue (`colab run`)
│   ├── check_secrets.py             # Automated API-key / credential leak scanner
│   └── install_git_hooks.sh         # Installs local `.git/hooks/pre-commit` secret guard
├── monitor/
│   ├── gcp_watchdog.py              # 24/7 W&B heartbeat watchdog & Markdown status reporter
│   ├── setup_gcp_vm.sh              # systemd service setup for nqs-watchdog
│   └── monitor_config.example.yaml  # Config template (real config is gitignored)
├── notebooks/colab_runner.ipynb     # Verified browser fallback notebook
└── tests/test_workflow_local.py     # Local test suite (security, English-only code, AST, watchdog)
```

## Quick Checks (Local)

```bash
# Install local pre-commit secret guard
bash scripts/install_git_hooks.sh

# Run secret scan + full local test suite
python scripts/check_secrets.py
python tests/test_workflow_local.py
```

See **[AGENTS.md](AGENTS.md)** for full technical reference and **[docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md)** for the Chinese operational handbook.
