"""Self-contained entrypoint executed on remote Google Colab VMs via `colab exec` or `colab run`.

Design goals:
  1. Idempotent environment check: installs matched NetKet/Flax/beartype<0.23/plum-dispatch<2.8
     on fresh VMs and skips pip install in ~0.1s on reused hot sessions.
  2. Clean subprocess execution: runs `python -m nqs.train` in a fresh child process so
     downgraded packages (e.g. beartype<0.23) load cleanly without requiring a kernel restart.
  3. Configurable via either CLI arguments (`colab run`) or `--env` variables (`colab exec`).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys


def _parse_ver(pkg: str) -> tuple[int, int] | None:
    try:
        parts = importlib.metadata.version(pkg).split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def _ensure_environment() -> None:
    env_ready = (
        _parse_ver("NetKet") is not None
        and _parse_ver("wandb") is not None
        and (_parse_ver("beartype") or (99, 0)) < (0, 23)
        and (_parse_ver("plum-dispatch") or (99, 0)) < (2, 8)
        and _parse_ver("jax-cuda12-plugin") is None
    )
    if env_ready:
        print("[Remote] Matched NetKet + JAX + Flax environment already installed. Skipping pip install.", flush=True)
        return

    has_gpu = shutil.which("nvidia-smi") is not None
    print(f"[Remote] Setting up NetKet environment (GPU detected: {has_gpu})...", flush=True)

    if _parse_ver("jax-cuda12-plugin") is not None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "jax-cuda12-plugin", "jax-cuda12-pjrt"]
        )

    # Use plain `netket` (not `netket[cuda]`) so Colab's native CUDA 13 JAX plugin stays intact.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "netket",
            "flax",
            "optax",
            "beartype<0.23",
            "plum-dispatch<2.8",
            "wandb",
            "pyyaml",
            "matplotlib",
        ]
    )
    print("[Remote] Environment setup complete.", flush=True)


def _sync_repository(repo_url: str, branch: str, work_dir: str) -> None:
    if os.path.isdir(os.path.join(work_dir, ".git")):
        print(f"[Remote] Pulling latest {branch} in {work_dir}...", flush=True)
        subprocess.check_call(["git", "-C", work_dir, "fetch", "origin", branch])
        subprocess.check_call(["git", "-C", work_dir, "checkout", branch])
        subprocess.check_call(["git", "-C", work_dir, "pull", "--ff-only", "origin", branch])
    else:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        print(f"[Remote] Cloning {repo_url} ({branch}) into {work_dir}...", flush=True)
        subprocess.check_call(["git", "clone", "-b", branch, repo_url, work_dir])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an NQS experiment on a remote Google Colab VM.")
    parser.add_argument(
        "--config",
        default=os.environ.get("NQS_CONFIG", "configs/tfim_1d_rbm.yaml"),
        help="Experiment YAML config path relative to repository root.",
    )
    parser.add_argument(
        "--repo-url",
        default=os.environ.get("NQS_REPO_URL", "https://github.com/jiahao116/NQS.git"),
        help="Git repository URL to clone/pull on the Colab VM.",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("NQS_BRANCH", "main"),
        help="Git branch to check out.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=os.environ.get("NQS_FRESH", "").lower() in {"1", "true", "yes"},
        help="Ignore existing checkpoints and train from step 0.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=os.environ.get("NQS_NO_WANDB", "").lower() in {"1", "true", "yes"},
        help="Disable Weights & Biases telemetry.",
    )
    args, _ = parser.parse_known_args()

    work_dir = "/content/NQS"
    _ensure_environment()
    _sync_repository(args.repo_url, args.branch, work_dir)

    env = os.environ.copy()
    src_dir = os.path.join(work_dir, "src")
    env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nqs.train",
        "--config",
        args.config,
        "--checkpoint-dir",
        os.path.join(work_dir, "checkpoints"),
    ]
    if args.fresh:
        cmd.append("--fresh")
    if args.no_wandb or not env.get("WANDB_API_KEY"):
        if not args.no_wandb and not env.get("WANDB_API_KEY"):
            print("[Remote] WANDB_API_KEY not provided; running with --no-wandb.", flush=True)
        cmd.append("--no-wandb")

    print(f"[Remote] Launching clean subprocess: {' '.join(cmd)}", flush=True)
    # Pipe child stdout/stderr through Python sys.stdout so Jupyter/Colab CLI
    # streams every training step and W&B URL live to the caller terminal.
    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
