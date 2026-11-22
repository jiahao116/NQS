"""Experiment configuration schema and loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class SystemConfig:
    """Quantum spin system specification."""

    hamiltonian: str = "tfim_1d"  # Options: "tfim_1d", "heisenberg_1d", "heisenberg_2d"
    n_spins: int = 12             # Number of spins L (for 2d, n_spins = Lx * Ly)
    lx: int = 4                   # Used only when hamiltonian == "heisenberg_2d"
    ly: int = 3                   # Used only when hamiltonian == "heisenberg_2d"
    h_field: float = 1.0          # Transverse field strength h (critical point h = 1.0 for 1D TFIM)
    j_coupling: float = 1.0       # Exchange coupling J
    periodic: bool = True         # Periodic boundary conditions (PBC)
    compute_exact: bool = True    # Compute Lanczos Exact Diagonalization ground truth (recommended for N <= 20)


@dataclass
class ModelConfig:
    """Neural Quantum State wavefunction ansatz configuration."""

    ansatz: str = "gpt"           # Options: "gpt" (Causal Transformer), "arnn" (NetKet ARNNDense), "rbm" (Classic RBM)
    # GPT / Causal Transformer hyperparameters
    d_model: int = 32             # Embedding & attention hidden dimension
    n_heads: int = 4              # Number of causal attention heads (must divide d_model)
    n_layers: int = 2             # Number of Transformer blocks
    mlp_ratio: int = 2            # Feed-forward expansion factor
    use_phase: bool = False       # Output complex phase in [ -pi, pi ] (False for TFIM, True for Heisenberg)
    # ARNNDense hyperparameters
    arnn_layers: int = 3
    arnn_features: int = 32
    # RBM hyperparameters
    rbm_alpha: int = 2            # Hidden unit density (n_hidden = alpha * n_spins)


@dataclass
class TrainingConfig:
    """Variational Monte Carlo (VMC) optimization parameters."""

    seed: int = 42
    n_samples: int = 1024         # Monte Carlo / autoregressive direct batch size per step
    n_discard_per_chain: int = 64 # Used only by MCMC samplers (e.g. RBM MetropolisLocal)
    n_steps: int = 400            # Total VMC optimization steps
    optimizer: str = "adam"       # Options: "adam", "sgd"
    learning_rate: float = 2e-3
    use_sr: bool = False          # Stochastic Reconfiguration preconditioner (recommended True for RBM, False for GPT)
    diag_shift: float = 1e-2      # Diagonal regularization for SR
    target_rel_error: float = 1e-4  # Optional early-celebration threshold vs exact energy


@dataclass
class TelemetryConfig:
    """Checkpointing & W&B cloud monitoring settings."""

    experiment_name: str = "tfim1d_L12_gpt"
    wandb_project: str = "nqs-colab"
    wandb_entity: str | None = None
    use_wandb: bool = True
    log_every: int = 10           # Push metrics & heartbeat every N steps
    checkpoint_every: int = 50    # Save resumable checkpoint every N steps
    checkpoint_dir: str = "checkpoints"  # Override in Colab to /content/drive/MyDrive/NQS_checkpoints
    upload_wandb_checkpoint: bool = True # Also back up latest checkpoint as a W&B Artifact


@dataclass
class ExperimentConfig:
    """Top-level NQS experiment configuration."""

    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    def validate(self) -> None:
        valid_hamiltonians = {"tfim_1d", "heisenberg_1d", "heisenberg_2d"}
        if self.system.hamiltonian not in valid_hamiltonians:
            raise ValueError(
                f"Unknown hamiltonian '{self.system.hamiltonian}'. Choose from {sorted(valid_hamiltonians)}."
            )
        if self.system.hamiltonian == "heisenberg_2d":
            self.system.n_spins = self.system.lx * self.system.ly

        if self.system.n_spins < 2:
            raise ValueError("system.n_spins must be >= 2.")
        if self.system.compute_exact and self.system.n_spins > 22:
            raise ValueError(
                f"system.n_spins={self.system.n_spins} is too large for Lanczos Exact Diagonalization on Colab RAM; "
                "set system.compute_exact: false for N > 22."
            )

        valid_ansatz = {"gpt", "arnn", "rbm"}
        if self.model.ansatz not in valid_ansatz:
            raise ValueError(f"Unknown model.ansatz '{self.model.ansatz}'. Choose from {sorted(valid_ansatz)}.")

        if self.model.ansatz == "gpt" and self.model.d_model % self.model.n_heads != 0:
            raise ValueError(
                f"model.d_model ({self.model.d_model}) must be divisible by model.n_heads ({self.model.n_heads})."
            )

        valid_opts = {"adam", "sgd"}
        if self.training.optimizer not in valid_opts:
            raise ValueError(f"Unknown training.optimizer '{self.training.optimizer}'. Choose from {sorted(valid_opts)}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an ExperimentConfig from a YAML or JSON file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    raw_text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required to load .yaml files (`pip install pyyaml`).") from exc
        data = yaml.safe_load(raw_text) or {}

    cfg = ExperimentConfig(
        system=SystemConfig(**data.get("system", {})),
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        telemetry=TelemetryConfig(**data.get("telemetry", {})),
    )
    cfg.validate()
    return cfg
