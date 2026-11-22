"""Main NQS training entry point with resumable checkpoints, W&B live curves, and heartbeats."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from nqs.checkpoint import CheckpointManager
from nqs.config import ExperimentConfig, load_config
from nqs.models import build_variational_state
from nqs.physics import build_quantum_system


def _save_convergence_plot(
    history: list[dict[str, Any]],
    exact_energy: float | None,
    out_path: Path,
    title: str,
) -> None:
    """Render an energy convergence & relative error plot for offline/Colab inspection."""
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = [row["step"] for row in history]
    energies = [row["energy_mean"] for row in history]
    errors = [row["energy_error"] for row in history]

    has_exact = exact_energy is not None
    fig, axes = plt.subplots(1, 2 if has_exact else 1, figsize=(11 if has_exact else 6, 4.2))
    ax_e = axes[0] if has_exact else axes

    ax_e.errorbar(steps, energies, yerr=errors, fmt="-o", markersize=3, alpha=0.85, label="NQS Variational E")
    if exact_energy is not None:
        ax_e.axhline(exact_energy, color="crimson", linestyle="--", linewidth=1.8, label=f"Exact E0 = {exact_energy:.6f}")
    ax_e.set_xlabel("VMC Step")
    ax_e.set_ylabel("Energy")
    ax_e.set_title(f"{title} — Ground State Energy")
    ax_e.legend()
    ax_e.grid(True, alpha=0.3)

    if has_exact:
        ax_err = axes[1]
        rel_errs = [max(row.get("rel_error", 1e-16), 1e-16) for row in history]
        ax_err.semilogy(steps, rel_errs, "-s", markersize=3, color="darkorange", label="|E - E_exact| / |E_exact|")
        ax_err.set_xlabel("VMC Step")
        ax_err.set_ylabel("Relative Error (log scale)")
        ax_err.set_title("Relative Error vs. Exact Diagonalization")
        ax_err.legend()
        ax_err.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_experiment(
    cfg: ExperimentConfig,
    checkpoint_dir_override: str | None = None,
    no_wandb: bool = False,
    fresh_start: bool = False,
) -> dict[str, Any]:
    """Execute or resume a complete NQS VMC optimization run."""
    import jax
    import netket as nk

    ckpt_root = checkpoint_dir_override or cfg.telemetry.checkpoint_dir
    ckpt_mgr = CheckpointManager(base_dir=ckpt_root, experiment_name=cfg.telemetry.experiment_name)

    print("=" * 68)
    print(f"[NQS] Experiment : {cfg.telemetry.experiment_name}")
    print(f"[NQS] JAX Devices: {jax.devices()} (default backend: {jax.default_backend()})")
    print(f"[NQS] System     : {cfg.system.hamiltonian} | N={cfg.system.n_spins} spins")
    print(f"[NQS] Ansatz     : {cfg.model.ansatz.upper()} | Samples/step={cfg.training.n_samples}")
    print("=" * 68)

    # 1. Build quantum Hamiltonian & Exact Diagonalization reference
    qsys = build_quantum_system(cfg.system)
    if qsys.exact_ground_energy is not None:
        print(f"[Exact ED] Ground State E0 = {qsys.exact_ground_energy:.8f}")
        if qsys.exact_gap is not None:
            print(f"[Exact ED] Energy Gap  dE  = {qsys.exact_gap:.8f}")

    # 2. Build Variational State, Optimizer, and VMC Driver
    vstate, optimizer, preconditioner = build_variational_state(cfg, qsys.hilbert)
    print(f"[Model] Total variational parameters: {vstate.n_parameters:,}")

    driver_kwargs: dict[str, Any] = {
        "hamiltonian": qsys.hamiltonian,
        "optimizer": optimizer,
        "variational_state": vstate,
    }
    if preconditioner is not None:
        driver_kwargs["preconditioner"] = preconditioner
    driver = nk.driver.VMC(**driver_kwargs)

    # 3. Check for existing checkpoint (auto-resume after Colab disconnect)
    start_step = 0
    existing_meta = None if fresh_start else ckpt_mgr.load_meta()
    if not fresh_start:
        start_step = ckpt_mgr.try_restore(
            vstate=vstate,
            wandb_project=cfg.telemetry.wandb_project if cfg.telemetry.use_wandb and not no_wandb else None,
            wandb_entity=cfg.telemetry.wandb_entity,
        )
        if existing_meta is None:
            existing_meta = ckpt_mgr.load_meta()

    if start_step >= cfg.training.n_steps:
        print(f"[NQS] Experiment already completed ({start_step}/{cfg.training.n_steps} steps). Nothing to do!")
        return existing_meta or {"step": start_step, "status": "already_completed"}

    # 4. Initialize Weights & Biases (resuming the same chart if disconnected mid-run!)
    wandb_run = None
    use_wandb = cfg.telemetry.use_wandb and not no_wandb
    if use_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            print("[W&B] WANDB_API_KEY not found in environment; attempting interactive login or offline fallback...")
        import wandb

        prev_run_id = existing_meta.get("wandb_run_id") if existing_meta else None
        wandb_init_kwargs: dict[str, Any] = {
            "project": cfg.telemetry.wandb_project,
            "entity": cfg.telemetry.wandb_entity,
            "name": cfg.telemetry.experiment_name,
            "config": cfg.to_dict(),
        }
        if prev_run_id:
            wandb_init_kwargs["id"] = prev_run_id
            wandb_init_kwargs["resume"] = "allow"

        wandb_run = wandb.init(**wandb_init_kwargs)
        if qsys.exact_ground_energy is not None:
            wandb_run.summary["exact_ground_energy"] = qsys.exact_ground_energy

    # 5. Chunked Optimization Loop with Heartbeats, NaN Guard, and Checkpointing
    history_path = ckpt_mgr.exp_dir / "history.jsonl"
    history: list[dict[str, Any]] = []
    if start_step > 0 and history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.append(json.loads(line))

    current_step = start_step
    total_steps = cfg.training.n_steps
    log_every = max(1, cfg.telemetry.log_every)
    last_ckpt_step = start_step
    status = "completed"

    t_run_start = time.time()
    try:
        while current_step < total_steps:
            chunk = min(log_every, total_steps - current_step)
            t_chunk = time.time()

            # Advance VMC optimization by `chunk` steps (compatible with NetKet 3.23+)
            driver.run(n_iter=chunk, out=(), show_progress=False)
            current_step += chunk
            dt = time.time() - t_chunk

            # Evaluate current energy statistics (already cached on driver.energy from last step)
            e_stats = driver.energy
            e_mean = float(e_stats.mean.real)
            e_err = float(e_stats.error_of_mean) if hasattr(e_stats, "error_of_mean") else float(e_stats.error)
            e_var = float(e_stats.variance.real)

            # NaN / Inf circuit breaker
            if math.isnan(e_mean) or math.isinf(e_mean) or math.isnan(e_var):
                status = "diverged_nan"
                err_payload = {
                    "step": current_step,
                    "status": status,
                    "heartbeat_unix": time.time(),
                }
                if wandb_run is not None:
                    wandb_run.summary.update(err_payload)
                    wandb_run.log(err_payload, step=current_step)
                raise RuntimeError(f"NaN/Inf detected in variational energy at step {current_step}! Halting safely.")

            # Compute physics quality metrics: V-score & Exact Relative Error
            n_spins = cfg.system.n_spins
            v_score = (n_spins * e_var) / max(e_mean**2, 1e-12)
            now_ts = time.time()

            metrics: dict[str, Any] = {
                "step": current_step,
                "progress_pct": round(100.0 * current_step / total_steps, 2),
                "energy_mean": e_mean,
                "energy_error": e_err,
                "energy_variance": e_var,
                "v_score": v_score,
                "sec_per_step": dt / chunk,
                "heartbeat_unix": now_ts,
                "status": "running" if current_step < total_steps else "completed",
            }
            if qsys.exact_ground_energy is not None:
                abs_err = abs(e_mean - qsys.exact_ground_energy)
                rel_err = abs_err / max(abs(qsys.exact_ground_energy), 1e-12)
                metrics["exact_energy"] = qsys.exact_ground_energy
                metrics["abs_error"] = abs_err
                metrics["rel_error"] = rel_err

            # Append local JSONL history
            history.append(metrics)
            with history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metrics) + "\n")

            # Console progress line
            rel_str = f" | rel_err={metrics['rel_error']:.2e}" if "rel_error" in metrics else ""
            print(
                f"[Step {current_step:4d}/{total_steps}] "
                f"E = {e_mean:.6f} ± {e_err:.6f} | Var = {e_var:.4e} | V-score = {v_score:.3e}"
                f"{rel_str} ({dt / chunk * 1000:.1f} ms/step)"
            )

            # Push live curve + heartbeat to W&B
            if wandb_run is not None:
                wandb_run.log(metrics, step=current_step)
                wandb_run.summary.update(
                    {
                        "last_step": current_step,
                        "total_steps": total_steps,
                        "heartbeat_unix": now_ts,
                        "status": metrics["status"],
                        "energy_mean": e_mean,
                        "v_score": v_score,
                        **({"rel_error": metrics["rel_error"]} if "rel_error" in metrics else {}),
                    }
                )

            # Periodic atomic checkpoint
            if (current_step - last_ckpt_step) >= cfg.telemetry.checkpoint_every or current_step == total_steps:
                ckpt_mgr.save(
                    vstate=vstate,
                    step=current_step,
                    metrics=metrics,
                    wandb_run=wandb_run,
                    upload_artifact=cfg.telemetry.upload_wandb_checkpoint,
                )
                last_ckpt_step = current_step

        # Save final convergence figure
        plot_path = ckpt_mgr.exp_dir / "convergence.png"
        _save_convergence_plot(
            history=history,
            exact_energy=qsys.exact_ground_energy,
            out_path=plot_path,
            title=cfg.telemetry.experiment_name,
        )
        if wandb_run is not None and plot_path.exists():
            import wandb

            wandb_run.log({"convergence_plot": wandb.Image(str(plot_path))}, step=current_step)

        elapsed = time.time() - t_run_start
        summary = {
            "experiment": cfg.telemetry.experiment_name,
            "status": status,
            "final_step": current_step,
            "elapsed_sec": round(elapsed, 2),
            "final_metrics": history[-1] if history else {},
            "checkpoint_dir": str(ckpt_mgr.exp_dir),
            "plot_path": str(plot_path),
        }
        (ckpt_mgr.exp_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("=" * 68)
        print(f"[NQS] Finished in {elapsed:.1f}s! Final E = {history[-1]['energy_mean']:.6f}")
        if "rel_error" in history[-1]:
            print(
                f"[NQS] Exact E0 = {qsys.exact_ground_energy:.6f} | "
                f"Relative Error = {history[-1]['rel_error']:.2e}"
            )
        print(f"[NQS] Artifacts saved to: {ckpt_mgr.exp_dir}")
        print("=" * 68)
        return summary

    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume an NQS Variational Monte Carlo experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tfim_1d_rbm.yaml",
        help="Path to experiment YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (e.g. /content/drive/MyDrive/NQS_checkpoints on Colab).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases cloud logging (local run only).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints and start optimization from step 0.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(
        cfg=cfg,
        checkpoint_dir_override=args.checkpoint_dir,
        no_wandb=args.no_wandb,
        fresh_start=args.fresh,
    )


if __name__ == "__main__":
    main()
