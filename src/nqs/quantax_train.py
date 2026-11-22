"""Quantax (v0.4.0+, JAX/Equinox) VMC training & scientific verification runner.

Implements the complete two-stage symmetry-projected NQS workflow:
  1. QuSpin Exact Diagonalization reference (energy, gap, staggered order)
  2. Stage 1: Translation-invariant ResConv optimization via SR / MinSR
  3. Symmetry overlap diagnostic (`symm-test`) on candidate point-group & Z2 ops
  4. Stage 2: Symmetry-projected fine-tuning loaded from Stage 1 + thermalized chains
  5. Exact dense overlap/fidelity (small clusters), autocorrelation binning (`measure`),
     and ODR zero-variance extrapolation (`zero-var-extrap`)
  6. Live W&B telemetry, heartbeat summaries compatible with `monitor/gcp_watchdog.py`,
     publication-grade 3-panel plot, and W&B Artifact upload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from nqs.quantax_analysis import (
    build_staggered_mag_sq_operator,
    run_binning_measurement,
    run_symmetry_diagnostics,
    run_zero_variance_extrapolation,
    save_quantax_results_panel,
)


@dataclass
class QuantaxSystemConfig:
    """2D/1D lattice spin system specification for Quantax."""

    lattice: str = "square"          # Options: "square"
    lx: int = 4                      # Linear extent L for L x L square lattice
    hamiltonian: str = "heisenberg"  # Options: "heisenberg", "j1j2"
    j1: float = 1.0
    j2: float = 0.0
    msr: bool = True                 # Marshall sign rule basis rotation on nearest neighbors
    compute_exact: bool = True       # QuSpin Exact Diagonalization benchmark (for N <= 20)

    @property
    def n_spins(self) -> int:
        return self.lx * self.lx


@dataclass
class QuantaxModelConfig:
    """Quantax variational wavefunction architecture."""

    ansatz: str = "resconv"          # Options: "resconv", "rbm_conv"
    nblocks: int = 2
    channels: int = 8
    kernel_size: int = 3
    final_activation: str = "exp"    # Options: "exp" (positive), "sinhp1_by_log" (real signed)
    disable_tf32: bool = False       # Recommended True for RBM or strict fp32 symmetry checks


@dataclass
class QuantaxTrainingConfig:
    """Two-stage VMC SR/MinSR optimization hyperparameters."""

    seed: int = 42
    nsamples: int = 1024
    stage1_steps: int = 250
    stage2_steps: int = 250
    learning_rate: float = 5e-3
    stage2_learning_rate: float = 5e-3
    optimizer: str = "sr"            # Options: "sr", "spring"
    spring_mu: float = 0.9
    reweight: float = 2.0
    max_parallel_stage1: int = 20000
    max_parallel_stage2: int = 1024
    sampler_neighbors: list[int] = field(default_factory=lambda: [1])


@dataclass
class QuantaxAnalysisConfig:
    """Post-training scientific verification controls."""

    run_symm_test: bool = True
    run_binning: bool = True
    measure_nchains: int = 512
    measure_nsweeps: int = 64
    measure_sweep_steps: int | None = None
    run_zero_var_extrap: bool = True


@dataclass
class QuantaxTelemetryConfig:
    """W&B telemetry, heartbeat, and artifact settings."""

    experiment_name: str = "qtx_square4x4_heisenberg"
    wandb_project: str = "nqs-colab"
    wandb_entity: str | None = None
    use_wandb: bool = True
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    upload_wandb_artifact: bool = True


@dataclass
class QuantaxExperimentConfig:
    """Top-level Quantax experiment specification."""

    backend: str = "quantax"
    system: QuantaxSystemConfig = field(default_factory=QuantaxSystemConfig)
    model: QuantaxModelConfig = field(default_factory=QuantaxModelConfig)
    training: QuantaxTrainingConfig = field(default_factory=QuantaxTrainingConfig)
    analysis: QuantaxAnalysisConfig = field(default_factory=QuantaxAnalysisConfig)
    telemetry: QuantaxTelemetryConfig = field(default_factory=QuantaxTelemetryConfig)

    def validate(self) -> None:
        if self.backend != "quantax":
            raise ValueError(f"Expected backend='quantax', got '{self.backend}'.")
        if self.system.lattice not in {"square"}:
            raise ValueError(f"Unsupported Quantax lattice '{self.system.lattice}'.")
        if self.system.lx < 2 or self.system.lx % 2 != 0:
            raise ValueError(f"system.lx must be an even integer >= 2, got {self.system.lx}.")
        if self.system.hamiltonian not in {"heisenberg", "j1j2"}:
            raise ValueError(f"Unsupported Quantax hamiltonian '{self.system.hamiltonian}'.")
        if self.system.compute_exact and self.system.n_spins > 20:
            raise ValueError(
                f"system.n_spins={self.system.n_spins} exceeds safe QuSpin ED limit (20); "
                "set system.compute_exact: false."
            )
        if self.model.ansatz not in {"resconv", "rbm_conv"}:
            raise ValueError(f"Unsupported Quantax ansatz '{self.model.ansatz}'.")
        if self.model.final_activation not in {"exp", "sinhp1_by_log"}:
            raise ValueError(f"Unsupported final_activation '{self.model.final_activation}'.")
        if self.training.optimizer not in {"sr", "spring"}:
            raise ValueError(f"Unsupported Quantax optimizer '{self.training.optimizer}'.")
        if self.training.stage1_steps < 1 or self.training.stage2_steps < 1:
            raise ValueError("Both stage1_steps and stage2_steps must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_quantax_config(path: str | Path) -> QuantaxExperimentConfig:
    """Load and validate a QuantaxExperimentConfig from YAML or JSON."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    raw = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        import yaml  # type: ignore

        data = yaml.safe_load(raw) or {}

    cfg = QuantaxExperimentConfig(
        backend=data.get("backend", "quantax"),
        system=QuantaxSystemConfig(**data.get("system", {})),
        model=QuantaxModelConfig(**data.get("model", {})),
        training=QuantaxTrainingConfig(**data.get("training", {})),
        analysis=QuantaxAnalysisConfig(**data.get("analysis", {})),
        telemetry=QuantaxTelemetryConfig(**data.get("telemetry", {})),
    )
    cfg.validate()
    return cfg


def _build_quantax_model(m_cfg: QuantaxModelConfig) -> Any:
    import quantax as qtx

    if m_cfg.ansatz == "rbm_conv":
        return qtx.model.RBM_Conv(channels=m_cfg.channels)

    kwargs: dict[str, Any] = {
        "nblocks": m_cfg.nblocks,
        "channels": m_cfg.channels,
        "kernel_size": m_cfg.kernel_size,
    }
    if m_cfg.final_activation == "sinhp1_by_log":
        kwargs["final_activation"] = qtx.nn.sinhp1_by_log
    return qtx.model.ResConv(**kwargs)


def _build_quantax_optimizer(t_cfg: QuantaxTrainingConfig, state: Any, hamiltonian: Any) -> Any:
    import quantax as qtx

    if t_cfg.optimizer == "spring":
        return qtx.optimizer.SPRING(state, hamiltonian, mu=t_cfg.spring_mu)
    return qtx.optimizer.SR(state, hamiltonian)


def _print_symm_table(label: str, report: dict[str, list[dict[str, float]]]) -> None:
    print(f"[Symm-Test] {label}:", flush=True)
    for name, elems in report.items():
        summary = ", ".join(f"g{e['element']}={e['overlap']:.4f}±{e['error']:.4f}" for e in elems)
        print(f"  {name:16s}: {summary}", flush=True)


def run_quantax_experiment(
    cfg: QuantaxExperimentConfig,
    checkpoint_dir_override: str | None = None,
    no_wandb: bool = False,
    fresh_start: bool = False,
) -> dict[str, Any]:
    """Run a complete two-stage Quantax VMC training & verification experiment."""
    del fresh_start  # Quantax process always runs cleanly from scratch per Sites rule

    import jax

    if cfg.model.disable_tf32 or cfg.model.ansatz == "rbm_conv":
        jax.config.update("jax_default_matmul_precision", "float32")

    import jax.numpy as jnp
    import numpy as np
    import quantax as qtx
    from quantax.symmetry import Flip, Rotation, SpinInverse, TransND

    qtx.set_random_seed(cfg.training.seed)

    ckpt_root = Path(checkpoint_dir_override or cfg.telemetry.checkpoint_dir)
    exp_dir = ckpt_root / cfg.telemetry.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg.training.stage1_steps + cfg.training.stage2_steps
    lx = cfg.system.lx
    n_spins = cfg.system.n_spins

    print("=" * 72, flush=True)
    print(f"[Quantax] Experiment : {cfg.telemetry.experiment_name}", flush=True)
    print(f"[Quantax] JAX Devices: {jax.devices()} (backend={jax.default_backend()})", flush=True)
    print(
        f"[Quantax] System     : {lx}x{lx} Square ({cfg.system.hamiltonian}, "
        f"J1={cfg.system.j1}, J2={cfg.system.j2}, msr={cfg.system.msr})",
        flush=True,
    )
    print(
        f"[Quantax] Model      : {cfg.model.ansatz} (blocks={cfg.model.nblocks}, "
        f"ch={cfg.model.channels}, act={cfg.model.final_activation})",
        flush=True,
    )
    print(
        f"[Quantax] Schedule   : Stage1={cfg.training.stage1_steps} + "
        f"Stage2(Symm)={cfg.training.stage2_steps} | nsamples={cfg.training.nsamples}",
        flush=True,
    )
    print("=" * 72, flush=True)

    # 1. Global Lattice & Hamiltonian (exactly one Sites instance per process)
    lattice = qtx.sites.Square(lx, Nparticles=(n_spins // 2, n_spins // 2))
    if abs(cfg.system.j2) > 0.0:
        hamiltonian = qtx.operator.Heisenberg(
            J=[cfg.system.j1, cfg.system.j2],
            n_neighbor=[1, 2],
            msr=cfg.system.msr,
        )
    else:
        hamiltonian = qtx.operator.Heisenberg(
            J=cfg.system.j1,
            n_neighbor=1,
            msr=cfg.system.msr,
        )

    m_stag2_op = build_staggered_mag_sq_operator(lattice)
    full_symm = TransND() @ Rotation(np.pi / 2) @ Flip() @ SpinInverse()
    proj_symm = Rotation(np.pi / 2) @ Flip() @ SpinInverse()

    # 2. Exact Diagonalization benchmark (QuSpin)
    exact_energy: float | None = None
    exact_gap: float | None = None
    exact_mstag2: float | None = None
    dense_exact: Any = None
    if cfg.system.compute_exact:
        evals, evecs = hamiltonian.diagonalize(full_symm, k=2)
        exact_energy = float(evals[0])
        exact_gap = float(evals[1] - evals[0]) if len(evals) > 1 else None
        dense_exact = qtx.state.DenseState(evecs[:, 0], full_symm).normalize()
        exact_mstag2 = float(jnp.asarray(dense_exact @ m_stag2_op @ dense_exact).real)
        print(
            f"[Exact ED] E0 = {exact_energy:.8f} | dE = {exact_gap:.6f} | "
            f"<M_stag^2>_ED = {exact_mstag2:.6f}",
            flush=True,
        )

    # 3. Initialize Weights & Biases (watchdog-compatible schema)
    wandb_run = None
    use_wandb = cfg.telemetry.use_wandb and not no_wandb
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.telemetry.wandb_project,
            entity=cfg.telemetry.wandb_entity,
            name=cfg.telemetry.experiment_name,
            config=cfg.to_dict(),
        )
        if exact_energy is not None:
            wandb_run.summary["exact_ground_energy"] = exact_energy
        if exact_mstag2 is not None:
            wandb_run.summary["exact_mstag2"] = exact_mstag2

    t_start = time.time()
    try:
        # ------------------------------------------------------------------
        # Stage 1: Translation-invariant ResConv (unprojected)
        # ------------------------------------------------------------------
        model = _build_quantax_model(cfg.model)
        state_s1 = qtx.state.Variational(model, max_parallel=cfg.training.max_parallel_stage1)
        print(f"[Model] Total variational parameters: {state_s1.nparams:,}", flush=True)

        sampler_s1 = qtx.sampler.SpinExchange(
            state_s1,
            nsamples=cfg.training.nsamples,
            reweight=cfg.training.reweight,
            n_neighbor=cfg.training.sampler_neighbors,
        )
        opt_s1 = _build_quantax_optimizer(cfg.training, state_s1, hamiltonian)

        stage1_rows: list[list[float]] = []
        log_every = max(1, cfg.telemetry.log_every)
        t_chunk = time.time()

        print(f"\n--- Stage 1: Unprojected {cfg.model.ansatz} ({cfg.training.stage1_steps} steps) ---", flush=True)
        for step in range(1, cfg.training.stage1_steps + 1):
            samples = sampler_s1.sweep()
            update = opt_s1.get_step(samples)
            state_s1.update(update * cfg.training.learning_rate)

            e_val = float(np.asarray(opt_s1.energy).real)
            v_val = float(np.asarray(opt_s1.VarE).real)
            if math.isnan(e_val) or math.isinf(e_val) or math.isnan(v_val):
                if wandb_run is not None:
                    wandb_run.summary.update({"status": "diverged_nan", "last_step": step, "heartbeat_unix": time.time()})
                raise RuntimeError(f"NaN/Inf detected during Stage 1 at step {step}!")

            stage1_rows.append([float(step - 1), e_val, v_val])

            if step % log_every == 0 or step == cfg.training.stage1_steps:
                dt = time.time() - t_chunk
                t_chunk = time.time()
                v_score = (n_spins * v_val) / max(e_val**2, 1e-12)
                now_ts = time.time()
                metrics: dict[str, Any] = {
                    "step": step,
                    "stage": 1,
                    "energy_mean": e_val,
                    "energy_variance": v_val,
                    "v_score": v_score,
                    "heartbeat_unix": now_ts,
                    "status": "running",
                }
                rel_str = ""
                if exact_energy is not None:
                    rel_err = abs(e_val - exact_energy) / max(abs(exact_energy), 1e-12)
                    metrics["rel_error"] = rel_err
                    rel_str = f" | rel_err={rel_err:.2e}"
                print(
                    f"[Stage1 {step:4d}/{cfg.training.stage1_steps}] "
                    f"E = {e_val:.6f} | VarE = {v_val:.4e} | V-score = {v_score:.3e}"
                    f"{rel_str} ({dt / log_every * 1000:.1f} ms/step)",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                    wandb_run.summary.update(
                        {
                            "last_step": step,
                            "total_steps": total_steps,
                            "heartbeat_unix": now_ts,
                            "status": "running",
                            "energy_mean": e_val,
                            "v_score": v_score,
                            **({"rel_error": metrics["rel_error"]} if "rel_error" in metrics else {}),
                        }
                    )

        stage1_ckpt = exp_dir / "stage1.eqx"
        state_s1.save(stage1_ckpt)
        chains_s1 = qtx.utils.to_replicated_numpy(sampler_s1._spins)
        np.save(exp_dir / "chains_stage1.npy", chains_s1)
        stage1_arr = np.asarray(stage1_rows, dtype=float)
        np.savetxt(exp_dir / "training_curve_stage1.txt", stage1_arr, header="step energy VarE")

        # ------------------------------------------------------------------
        # Symmetry Overlap Diagnostic (Before Projection)
        # ------------------------------------------------------------------
        symm_before: dict[str, list[dict[str, float]]] = {}
        if cfg.analysis.run_symm_test:
            print("\n--- Running Symmetry Overlap Test (Before Projection) ---", flush=True)
            symm_before = run_symmetry_diagnostics(
                model=model,
                param_file=stage1_ckpt,
                symm_projected=None,
                initial_spins=chains_s1,
                nsamples=cfg.training.nsamples,
                n_neighbor=cfg.training.sampler_neighbors,
            )
            _print_symm_table("Bare ResConv (Stage 1)", symm_before)

        # ------------------------------------------------------------------
        # Stage 2: Symmetry-Projected Fine-Tuning (C4v x Z2)
        # ------------------------------------------------------------------
        print(
            f"\n--- Stage 2: Symmetry-Projected ResConv (nsymm={proj_symm.nsymm}, "
            f"{cfg.training.stage2_steps} steps) ---",
            flush=True,
        )
        state_s2 = qtx.state.Variational(
            model,
            param_file=stage1_ckpt,
            symm=proj_symm,
            max_parallel=cfg.training.max_parallel_stage2,
        )
        sampler_s2 = qtx.sampler.SpinExchange(
            state_s2,
            nsamples=cfg.training.nsamples,
            reweight=cfg.training.reweight,
            n_neighbor=cfg.training.sampler_neighbors,
            initial_spins=jnp.asarray(chains_s1),
            thermal_steps=5 * lattice.Nmodes,
        )
        opt_s2 = _build_quantax_optimizer(cfg.training, state_s2, hamiltonian)

        stage2_rows: list[list[float]] = []
        t_chunk = time.time()

        for step in range(1, cfg.training.stage2_steps + 1):
            samples = sampler_s2.sweep()
            update = opt_s2.get_step(samples)
            state_s2.update(update * cfg.training.stage2_learning_rate)

            e_val = float(np.asarray(opt_s2.energy).real)
            v_val = float(np.asarray(opt_s2.VarE).real)
            if math.isnan(e_val) or math.isinf(e_val) or math.isnan(v_val):
                if wandb_run is not None:
                    wandb_run.summary.update(
                        {
                            "status": "diverged_nan",
                            "last_step": cfg.training.stage1_steps + step,
                            "heartbeat_unix": time.time(),
                        }
                    )
                raise RuntimeError(f"NaN/Inf detected during Stage 2 at step {step}!")

            stage2_rows.append([float(step - 1), e_val, v_val])
            global_step = cfg.training.stage1_steps + step

            if step % log_every == 0 or step == cfg.training.stage2_steps:
                dt = time.time() - t_chunk
                t_chunk = time.time()
                v_score = (n_spins * v_val) / max(e_val**2, 1e-12)
                now_ts = time.time()
                metrics = {
                    "step": global_step,
                    "stage": 2,
                    "energy_mean": e_val,
                    "energy_variance": v_val,
                    "v_score": v_score,
                    "heartbeat_unix": now_ts,
                    "status": "running" if global_step < total_steps else "completed",
                }
                rel_str = ""
                if exact_energy is not None:
                    rel_err = abs(e_val - exact_energy) / max(abs(exact_energy), 1e-12)
                    metrics["rel_error"] = rel_err
                    rel_str = f" | rel_err={rel_err:.2e}"
                print(
                    f"[Stage2 {step:4d}/{cfg.training.stage2_steps}] "
                    f"E = {e_val:.6f} | VarE = {v_val:.4e} | V-score = {v_score:.3e}"
                    f"{rel_str} ({dt / log_every * 1000:.1f} ms/step)",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
                    wandb_run.summary.update(
                        {
                            "last_step": global_step,
                            "total_steps": total_steps,
                            "heartbeat_unix": now_ts,
                            "status": metrics["status"],
                            "energy_mean": e_val,
                            "v_score": v_score,
                            **({"rel_error": metrics["rel_error"]} if "rel_error" in metrics else {}),
                        }
                    )

        final_ckpt = exp_dir / "final_symm.eqx"
        state_s2.save(final_ckpt)
        chains_final = qtx.utils.to_replicated_numpy(sampler_s2._spins)
        np.save(exp_dir / "chains_final.npy", chains_final)
        stage2_arr = np.asarray(stage2_rows, dtype=float)
        np.savetxt(exp_dir / "training_curve_stage2.txt", stage2_arr, header="step energy VarE")

        # ------------------------------------------------------------------
        # Exact Dense Verification (small clusters)
        # ------------------------------------------------------------------
        dense_energy: float | None = None
        dense_rel_err: float | None = None
        dense_fidelity: float | None = None
        dense_mstag2: float | None = None
        if cfg.system.compute_exact and dense_exact is not None:
            dense_nqs = state_s2.todense(full_symm).normalize()
            dense_energy = float(jnp.asarray(dense_nqs @ hamiltonian @ dense_nqs).real)
            dense_rel_err = abs(dense_energy - exact_energy) / max(abs(exact_energy), 1e-12)  # type: ignore[operator]
            dense_fidelity = float(abs(dense_nqs @ dense_exact) ** 2)
            dense_mstag2 = float(jnp.asarray(dense_nqs @ m_stag2_op @ dense_nqs).real)
            print(
                f"\n[Exact Dense Check] E_NQS = {dense_energy:.8f} | E_ED = {exact_energy:.8f} | "
                f"rel_err = {dense_rel_err:.2e} | Fidelity = {dense_fidelity:.8f} | "
                f"<M_stag^2>_NQS = {dense_mstag2:.6f} (ED = {exact_mstag2:.6f})",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Post-Projection Symmetry Verification
        # ------------------------------------------------------------------
        symm_after: dict[str, list[dict[str, float]]] = {}
        if cfg.analysis.run_symm_test:
            print("\n--- Running Symmetry Overlap Test (After Projection) ---", flush=True)
            symm_after = run_symmetry_diagnostics(
                model=model,
                param_file=final_ckpt,
                symm_projected=proj_symm,
                initial_spins=chains_final,
                nsamples=cfg.training.nsamples,
                n_neighbor=cfg.training.sampler_neighbors,
            )
            _print_symm_table("Symmetry-Projected ResConv (Stage 2)", symm_after)

        # ------------------------------------------------------------------
        # Production Autocorrelation Binning Measurement (`measure` skill)
        # ------------------------------------------------------------------
        binning_res: dict[str, Any] = {}
        if cfg.analysis.run_binning:
            print(
                f"\n--- Running Production Binning Measurement "
                f"({cfg.analysis.measure_nchains} chains x {cfg.analysis.measure_nsweeps} sweeps) ---",
                flush=True,
            )
            binning_res = run_binning_measurement(
                state=state_s2,
                hamiltonian=hamiltonian,
                m_stag2_op=m_stag2_op,
                initial_spins=chains_final,
                nchains=cfg.analysis.measure_nchains,
                nsweeps=cfg.analysis.measure_nsweeps,
                sweep_steps=cfg.analysis.measure_sweep_steps,
                n_neighbor=cfg.training.sampler_neighbors,
            )
            eb = binning_res["energy"]
            mb = binning_res["m_stag2"]
            print(
                f"[Measure] E = {eb['mean']:.6f} ± {eb['err']:.2e} "
                f"(tau_int={eb['tau_int']:.2f}, plateaued={eb['plateaued']})",
                flush=True,
            )
            print(
                f"[Measure] <M_stag^2> = {mb['mean']:.6f} ± {mb['err']:.2e} "
                f"(tau_int={mb['tau_int']:.2f}, plateaued={mb['plateaued']})",
                flush=True,
            )
            print(
                f"[Measure] Stationarity: sweep_scatter={binning_res['sweep_scatter']:.2e}, "
                f"expected~{binning_res['expected_scatter']:.2e}, zero_weight_samples={binning_res['n_zero_r']}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Zero-Variance ODR Extrapolation (`zero-var-extrap` skill)
        # ------------------------------------------------------------------
        zve_res: dict[str, Any] = {"available": False}
        if cfg.analysis.run_zero_var_extrap:
            print("\n--- Running Zero-Variance ODR Extrapolation (Stage 2 Tail) ---", flush=True)
            zve_res = run_zero_variance_extrapolation(stage2_arr)
            if not zve_res.get("available"):
                print(
                    f"[Zero-Var-Extrap] Stage 2 tail notice: {zve_res.get('reason')} "
                    "Trying Stage 1 tail for visualization...",
                    flush=True,
                )
                zve_s1 = run_zero_variance_extrapolation(stage1_arr)
                if zve_s1.get("available"):
                    zve_s1["source_stage"] = 1
                    zve_res = zve_s1
            else:
                zve_res["source_stage"] = 2

            if zve_res.get("available"):
                print(
                    f"[Zero-Var-Extrap] E0 = {zve_res['e0_mean']:.6f} ± "
                    f"{zve_res['stat_err']:.6f} (stat) ± {zve_res['syst_err']:.6f} (window) | "
                    f"R_med={zve_res['median_lever_arm']:.1f}, "
                    f"kept={zve_res['n_kept_windows']}/{zve_res['n_total_windows']}",
                    flush=True,
                )

        # ------------------------------------------------------------------
        # Save 3-Panel Publication Plot, Summary JSON, and W&B Artifact
        # ------------------------------------------------------------------
        panel_path = exp_dir / "results_panel.png"
        model_label = (
            f"4x4 Square J1-J2 (J2/J1 = {cfg.system.j2:g})"
            if abs(cfg.system.j2) > 0.0
            else "4x4 Square Heisenberg (J2/J1 = 0)"
        )
        save_quantax_results_panel(
            stage1_curve=stage1_arr,
            stage2_curve=stage2_arr,
            exact_energy=exact_energy,
            dense_energy=dense_energy,
            dense_fidelity=dense_fidelity,
            zve_result=zve_res,
            out_path=panel_path,
            title=f"Quantax ResConv + MinSR/SR — {model_label}",
        )

        elapsed = round(time.time() - t_start, 2)
        final_e = float(stage2_arr[-1, 1])
        final_vare = float(stage2_arr[-1, 2])
        final_vscore = (n_spins * final_vare) / max(final_e**2, 1e-12)

        summary: dict[str, Any] = {
            "experiment": cfg.telemetry.experiment_name,
            "backend": "quantax",
            "status": "completed",
            "elapsed_sec": elapsed,
            "n_params": int(state_s1.nparams),
            "total_steps": total_steps,
            "system": asdict(cfg.system),
            "exact_ed": {
                "energy": exact_energy,
                "gap": exact_gap,
                "m_stag2": exact_mstag2,
            },
            "final_vmc": {
                "energy": final_e,
                "var_e": final_vare,
                "v_score": final_vscore,
            },
            "exact_dense_nqs": {
                "energy": dense_energy,
                "rel_error": dense_rel_err,
                "fidelity": dense_fidelity,
                "m_stag2": dense_mstag2,
            },
            "symm_test_before": symm_before,
            "symm_test_after": symm_after,
            "binning_measurement": binning_res,
            "zero_var_extrap": {k: v for k, v in zve_res.items() if k != "best_window"},
            "zero_var_extrap_best": (
                {k: v for k, v in zve_res["best_window"].items() if k not in {"x", "y", "x_err", "y_err", "step"}}
                if zve_res.get("best_window")
                else None
            ),
            "artifacts_dir": str(exp_dir),
            "panel_path": str(panel_path),
        }

        summary_path = exp_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if wandb_run is not None:
            import wandb

            final_summary_updates: dict[str, Any] = {
                "last_step": total_steps,
                "total_steps": total_steps,
                "heartbeat_unix": time.time(),
                "status": "completed",
                "energy_mean": dense_energy if dense_energy is not None else final_e,
                "v_score": final_vscore,
                "elapsed_sec": elapsed,
            }
            if dense_rel_err is not None:
                final_summary_updates["rel_error"] = dense_rel_err
            if dense_fidelity is not None:
                final_summary_updates["exact_fidelity"] = dense_fidelity
            if binning_res:
                final_summary_updates["mc_energy"] = binning_res["energy"]["mean"]
                final_summary_updates["mc_energy_err"] = binning_res["energy"]["err"]
                final_summary_updates["mc_mstag2"] = binning_res["m_stag2"]["mean"]
                final_summary_updates["mc_mstag2_err"] = binning_res["m_stag2"]["err"]
            if zve_res.get("available"):
                final_summary_updates["zve_e0"] = zve_res["e0_mean"]
                final_summary_updates["zve_stat_err"] = zve_res["stat_err"]
                final_summary_updates["zve_syst_err"] = zve_res["syst_err"]

            wandb_run.summary.update(final_summary_updates)
            if panel_path.exists():
                wandb_run.log({"results_panel": wandb.Image(str(panel_path))}, step=total_steps)

            if cfg.telemetry.upload_wandb_artifact:
                art = wandb.Artifact(name=f"nqs-results-{cfg.telemetry.experiment_name}", type="results")
                for fname in (
                    "summary.json",
                    "results_panel.png",
                    "training_curve_stage1.txt",
                    "training_curve_stage2.txt",
                    "final_symm.eqx",
                    "chains_final.npy",
                ):
                    fpath = exp_dir / fname
                    if fpath.exists():
                        art.add_file(str(fpath))
                wandb_run.log_artifact(art)

        print("=" * 72, flush=True)
        print(f"[Quantax] Completed in {elapsed:.1f}s! Artifacts: {exp_dir}", flush=True)
        print("=" * 72, flush=True)
        return summary

    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Quantax two-stage NQS experiment with full diagnostics.")
    parser.add_argument("--config", type=str, required=True, help="Path to Quantax YAML/JSON config file.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint/output root directory.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases telemetry.")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (always default for Quantax).")
    args = parser.parse_args()

    cfg = load_quantax_config(args.config)
    run_quantax_experiment(
        cfg=cfg,
        checkpoint_dir_override=args.checkpoint_dir,
        no_wandb=args.no_wandb,
        fresh_start=args.fresh,
    )


if __name__ == "__main__":
    main()
