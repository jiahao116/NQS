"""Atomic checkpointing for Google Drive + Weights & Biases Artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any


class CheckpointManager:
    """Manages atomic save/restore of NetKet variational states across Colab disconnects."""

    def __init__(self, base_dir: str | Path, experiment_name: str) -> None:
        self.exp_dir = Path(base_dir) / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.vars_path = self.exp_dir / "latest_variables.mpack"
        self.meta_path = self.exp_dir / "latest_meta.json"
        self.artifact_name = f"ckpt-{experiment_name}"

    def save(
        self,
        vstate: Any,
        step: int,
        metrics: dict[str, Any],
        wandb_run: Any | None = None,
        upload_artifact: bool = True,
    ) -> Path:
        """Atomically serialize vstate.variables + metadata to disk (and optionally W&B)."""
        import flax.serialization

        # 1. Atomic write of Flax variables (.mpack)
        tmp_vars = self.exp_dir / "latest_variables.mpack.tmp"
        tmp_vars.write_bytes(flax.serialization.to_bytes(vstate.variables))
        os.replace(tmp_vars, self.vars_path)

        # 2. Atomic write of JSON metadata
        meta = {
            "step": int(step),
            "saved_at_unix": time.time(),
            "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
            "last_metrics": metrics,
        }
        tmp_meta = self.exp_dir / "latest_meta.json.tmp"
        tmp_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_meta, self.meta_path)

        # 3. Optional cloud backup via W&B Artifact (survives even without Google Drive!)
        if wandb_run is not None and upload_artifact:
            import wandb

            artifact = wandb.Artifact(name=self.artifact_name, type="nqs-checkpoint", metadata=meta)
            artifact.add_file(str(self.vars_path))
            artifact.add_file(str(self.meta_path))
            wandb_run.log_artifact(artifact, aliases=["latest", f"step-{step}"])

        return self.vars_path

    def load_meta(self) -> dict[str, Any] | None:
        """Read local/Drive metadata if a checkpoint exists."""
        if self.meta_path.exists() and self.vars_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def try_restore(
        self,
        vstate: Any,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ) -> int:
        """Restore vstate.variables from local/Drive or fallback to W&B Artifact.

        Returns the resumed step number (0 if starting fresh).
        """
        import flax.serialization

        meta = self.load_meta()

        # Fallback: try pulling latest checkpoint from W&B Artifacts if local/Drive is empty
        if meta is None and wandb_project and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                api = wandb.Api()
                prefix = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
                artifact = api.artifact(f"{prefix}/{self.artifact_name}:latest")
                dl_dir = Path(artifact.download(root=str(self.exp_dir / "_wandb_restore")))
                dl_vars = dl_dir / "latest_variables.mpack"
                dl_meta = dl_dir / "latest_meta.json"
                if dl_vars.exists() and dl_meta.exists():
                    self.vars_path.write_bytes(dl_vars.read_bytes())
                    self.meta_path.write_text(dl_meta.read_text(encoding="utf-8"), encoding="utf-8")
                    meta = self.load_meta()
                    print(f"[Checkpoint] Pulled cloud checkpoint from W&B Artifact ({self.artifact_name}:latest).")
            except Exception:
                # No remote checkpoint yet (first run) or offline
                pass

        if meta is None:
            return 0

        raw_bytes = self.vars_path.read_bytes()
        vstate.variables = flax.serialization.from_bytes(vstate.variables, raw_bytes)
        resumed_step = int(meta.get("step", 0))
        print(f"[Checkpoint] Resumed from step {resumed_step} ({self.vars_path})")
        return resumed_step
