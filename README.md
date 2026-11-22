# NQS Cloud Workflow (JAX / NetKet on Google Colab GPU + GCP Watchdog)

A cloud-first Neural Quantum States (NQS) workflow built on **JAX + NetKet 3**, designed to run on **Google Colab GPU** without relying on a local computer staying powered on or mounting Google Drive.

## Key Features

* **GPU-Accelerated NQS (`JAX` + `NetKet 3`)**: Includes the classic **RBM + Stochastic Reconfiguration** starter benchmark (Carleo & Troyer, *Science* 2017) plus modern autoregressive wavefunctions (`ARNNDense` and `CausalTransformerWaveFunction`), each automatically verified against Lanczos Exact Diagonalization ($E_{\text{exact}}$).
* **Drive-Free Cloud Checkpoints & Live Curves (`Weights & Biases`)**: Periodic atomic checkpoints (`.mpack`) and live energy/V-score curves sync automatically to W&B so you can inspect progress from your phone and resume after any disconnect.
* **Colab Compute Units Protection**: Includes an optional `AUTO_DISCONNECT_WHEN_DONE` switch (`runtime.unassign()`) in [colab_runner.ipynb](notebooks/colab_runner.ipynb) to release GPU resources immediately after completion.
* **24/7 Cloud Watchdog (`monitor/gcp_watchdog.py`)**: Lightweight Python daemon (`< 40 MB` RAM, no browser) for a Google Cloud VM that polls W&B heartbeats and sends mobile notifications on completion, disconnect, or `NaN` divergence.

## Project Layout

```text
NQS/
├── configs/
│   ├── tfim_1d_rbm.yaml        # Default starter: 1D Critical TFIM (L=12) + RBM + SR (~20-40s on T4 GPU)
│   ├── tfim_1d_gpt.yaml        # Autoregressive Causal Transformer on 1D Critical TFIM (L=12)
│   └── heisenberg_1d_gpt.yaml  # Complex-phase Causal Transformer on 1D Heisenberg chain (L=12)
├── src/nqs/
│   ├── config.py               # Experiment configuration dataclasses & validation
│   ├── physics.py              # Quantum spin Hamiltonians & Lanczos exact solver
│   ├── models.py               # RBM, ARNNDense, and CausalTransformerWaveFunction
│   ├── checkpoint.py           # Atomic local/W&B checkpoint save & auto-restore
│   └── train.py                # VMC optimization loop, V-score/rel_error metrics, plotting
├── notebooks/
│   └── colab_runner.ipynb      # One-click Google Colab GPU notebook
├── monitor/
│   ├── gcp_watchdog.py         # 24/7 W&B heartbeat watchdog for GCP VM
│   ├── setup_gcp_vm.sh         # One-click systemd setup script for Ubuntu/Debian GCP VM
│   └── monitor_config.example.yaml
└── docs/
    └── WORKFLOW_GUIDE.md       # Step-by-step Chinese user guide
```

## Quick Start

* Open directly in Google Colab: https://colab.research.google.com/github/jiahao116/NQS/blob/main/notebooks/colab_runner.ipynb
* Step-by-step walkthrough (in Chinese): [docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md)
