"""Self-contained entrypoint executed on remote Google Colab VMs via `colab exec` or `colab run`.

Design goals:
  1. Multi-backend support: automatically detects `backend: quantax` vs `netket`
     from the experiment config file and installs only the matching environment
     (idempotently skipping pip install in ~0.1s on reused hot sessions).
  2. Clean subprocess execution: runs `python -u -m nqs.quantax_train` or
     `python -u -m nqs.train` in a fresh child process, respecting Quantax's
     one-Sites-per-process invariant and NetKet's beartype pin without kernel restarts.
  3. Configurable via either CLI arguments (`colab run`) or `--env` variables (`colab exec`).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

QUANTAX_BACKEND_RE = re.compile(r"^\s*backend\s*:\s*['\"]?quantax['\"]?\s*(#.*)?$", re.MULTILINE)


def _parse_ver(pkg: str) -> tuple[int, int] | None:
    try:
        parts = importlib.metadata.version(pkg).split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def detect_backend(config_path: str | Path) -> str:
    """Detect experiment backend (`quantax` or `netket`) without requiring PyYAML."""
    p = Path(config_path)
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="replace")
        if QUANTAX_BACKEND_RE.search(text):
            return "quantax"
    return "netket"


def _ensure_netket_environment() -> None:
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
    print("[Remote] NetKet environment setup complete.", flush=True)


def _ensure_quantax_environment() -> None:
    env_ready = (
        (_parse_ver("quantax") or (0, 0)) >= (0, 4)
        and _parse_ver("quspin") is not None
        and _parse_ver("odrpack") is not None
        and _parse_ver("wandb") is not None
        and _parse_ver("PyYAML") is not None
        and _parse_ver("jax-cuda12-plugin") is None
    )
    if env_ready:
        print("[Remote] Matched Quantax v0.4+ environment already installed. Skipping pip install.", flush=True)
        return

    has_gpu = shutil.which("nvidia-smi") is not None
    print(f"[Remote] Setting up Quantax v0.4+ environment (GPU detected: {has_gpu})...", flush=True)

    if _parse_ver("jax-cuda12-plugin") is not None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "jax-cuda12-plugin", "jax-cuda12-pjrt"]
        )

    # Omit --upgrade so Colab's native CUDA 13 jax / jaxlib / jax-cuda13-plugin stay untouched.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "quantax[full]>=0.4.0",
            "odrpack",
            "wandb",
            "pyyaml",
            "matplotlib",
        ]
    )
    print("[Remote] Quantax environment setup complete.", flush=True)


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
    _sync_repository(args.repo_url, args.branch, work_dir)

    cfg_full = os.path.join(work_dir, args.config)
    backend = detect_backend(cfg_full)
    print(f"[Remote] Detected backend '{backend}' for config '{args.config}'.", flush=True)

    if backend == "quantax":
        _ensure_quantax_environment()
        module_name = "nqs.quantax_train"
    else:
        _ensure_netket_environment()
        module_name = "nqs.train"

    env = os.environ.copy()
    src_dir = os.path.join(work_dir, "src")
    env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        module_name,
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
