"""Local smoke tests (no GPU or NetKet required) for configs, notebook syntax, English-only code rule, and watchdog."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from monitor.gcp_watchdog import classify_run_health
from nqs.config import ExperimentConfig, ModelConfig, SystemConfig

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class WorkflowSmokeTests(unittest.TestCase):
    def test_all_python_sources_compile(self) -> None:
        py_files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "monitor").rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 6)
        for path in py_files:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))

    def test_code_and_config_files_are_english_only(self) -> None:
        targets = (
            list((ROOT / "src").rglob("*.py"))
            + list((ROOT / "monitor").rglob("*"))
            + list((ROOT / "scripts").rglob("*"))
            + list((ROOT / "configs").rglob("*.yaml"))
            + list((ROOT / "tests").rglob("*.py"))
            + [
                ROOT / "notebooks" / "colab_runner.ipynb",
                ROOT / "pyproject.toml",
                ROOT / "requirements-colab.txt",
                ROOT / "requirements-monitor.txt",
            ]
        )
        for path in targets:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                CJK_RE.search(text),
                msg=f"Found non-English CJK characters in code/config file: {path}",
            )

    def test_colab_notebook_valid_and_cells_parse(self) -> None:
        nb_path = ROOT / "notebooks" / "colab_runner.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 4)
        for idx, cell in enumerate(code_cells):
            code_str = "".join(cell["source"])
            ast.parse(code_str, filename=f"colab_runner_cell_{idx + 1}.py")

    def test_config_validation_rules(self) -> None:
        cfg = ExperimentConfig()
        cfg.validate()

        bad_heads = ExperimentConfig(model=ModelConfig(ansatz="gpt", d_model=32, n_heads=3))
        with self.assertRaises(ValueError):
            bad_heads.validate()

        too_big_ed = ExperimentConfig(system=SystemConfig(n_spins=30, compute_exact=True))
        with self.assertRaises(ValueError):
            too_big_ed.validate()

    def test_yaml_configs_if_pyyaml_available(self) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed in bare host Python")

        from nqs.config import load_config

        for cfg_file in (ROOT / "configs").glob("*.yaml"):
            cfg = load_config(cfg_file)
            self.assertGreaterEqual(cfg.system.n_spins, 2)

    def test_watchdog_health_classifier(self) -> None:
        now = 10_000.0
        # 1. Healthy active run
        key, _ = classify_run_health(
            {
                "id": "r1",
                "name": "tfim_rbm",
                "wandb_state": "running",
                "status": "running",
                "last_step": 100,
                "total_steps": 300,
                "heartbeat_unix": now - 45,
                "energy_mean": -15.31,
                "rel_error": 8e-4,
            },
            now_unix=now,
            stale_timeout_sec=300,
        )
        self.assertEqual(key, "healthy_running")

        # 2. Silent Colab disconnect (heartbeat older than stale_timeout_sec)
        key, msg = classify_run_health(
            {
                "id": "r1",
                "name": "tfim_rbm",
                "wandb_state": "running",
                "status": "running",
                "last_step": 150,
                "total_steps": 300,
                "heartbeat_unix": now - 600,
                "energy_mean": -15.32,
            },
            now_unix=now,
            stale_timeout_sec=300,
        )
        self.assertEqual(key, "stale_150")
        self.assertIn("DISCONNECTED", msg)

        # 3. NaN divergence
        key, _ = classify_run_health(
            {
                "id": "r2",
                "name": "bad_lr",
                "wandb_state": "failed",
                "status": "diverged_nan",
                "last_step": 30,
                "total_steps": 300,
            },
            now_unix=now,
            stale_timeout_sec=300,
        )
        self.assertEqual(key, "diverged_nan")

        # 4. Completed
        key, _ = classify_run_health(
            {
                "id": "r3",
                "name": "done_run",
                "wandb_state": "finished",
                "status": "completed",
                "last_step": 300,
                "total_steps": 300,
                "energy_mean": -15.32259,
                "rel_error": 1.2e-5,
            },
            now_unix=now,
            stale_timeout_sec=300,
        )
        self.assertEqual(key, "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
