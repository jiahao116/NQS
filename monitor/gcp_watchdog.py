"""Lightweight 24/7 Cloud Watchdog for monitoring Google Colab NQS runs from a GCP VM.

Memory footprint: < 40 MB RAM (no browser, no GPU, ToS-compliant via W&B telemetry API).
Detects:
  1. Run completion + physics accuracy summary (E_NQS vs E_exact, V-score)
  2. Colab disconnect / stale heartbeat
  3. Run resumption after checkpoint recovery
  4. NaN divergence or crash
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.request


@dataclass
class WatchdogConfig:
    wandb_project: str = "nqs-colab"
    wandb_entity: str | None = None
    poll_interval_sec: int = 120
    stale_timeout_sec: int = 360   # 6 minutes without heartbeat => Colab likely disconnected
    max_runs_to_check: int = 10
    ntfy_topic: str | None = None  # e.g. "nqs-alert-jiahao116-9823" -> https://ntfy.sh/<topic>
    webhook_url: str | None = None # Optional Discord / Slack / custom webhook URL
    state_file: str = "monitor/.watchdog_state.json"


def load_watchdog_config(path: str | Path | None) -> WatchdogConfig:
    cfg = WatchdogConfig()
    if path and Path(path).exists():
        raw = Path(path).read_text(encoding="utf-8")
        if str(path).endswith(".json"):
            data = json.loads(raw)
        else:
            import yaml  # type: ignore

            data = yaml.safe_load(raw) or {}
        for k, v in data.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

    if os.environ.get("WANDB_PROJECT"):
        cfg.wandb_project = os.environ["WANDB_PROJECT"]
    if os.environ.get("WANDB_ENTITY"):
        cfg.wandb_entity = os.environ["WANDB_ENTITY"]
    if os.environ.get("NTFY_TOPIC"):
        cfg.ntfy_topic = os.environ["NTFY_TOPIC"]
    if os.environ.get("WEBHOOK_URL"):
        cfg.webhook_url = os.environ["WEBHOOK_URL"]
    return cfg


def classify_run_health(
    run_info: dict[str, Any],
    now_unix: float,
    stale_timeout_sec: int,
) -> tuple[str, str]:
    """Pure function classifying a run's health into (event_key, human_message)."""
    name = run_info.get("name") or run_info.get("id", "unknown")
    w_state = run_info.get("wandb_state", "unknown")
    summary_status = run_info.get("status", "")
    step = run_info.get("last_step", 0)
    total = run_info.get("total_steps", 0)
    hb = run_info.get("heartbeat_unix")
    e_mean = run_info.get("energy_mean")
    rel_err = run_info.get("rel_error")
    v_score = run_info.get("v_score")
    url = run_info.get("url", "")

    metrics_tail = []
    if e_mean is not None:
        metrics_tail.append(f"E={e_mean:.6f}")
    if rel_err is not None:
        metrics_tail.append(f"rel_err={rel_err:.2e}")
    if v_score is not None:
        metrics_tail.append(f"V-score={v_score:.2e}")
    metrics_str = ", ".join(metrics_tail) if metrics_tail else "no metrics yet"

    # 1. Explicit NaN divergence
    if summary_status == "diverged_nan":
        return (
            "diverged_nan",
            f"[ALERT: NaN Diverged] Run '{name}' hit NaN at step {step}/{total}. "
            f"Consider lowering learning_rate. {url}",
        )

    # 2. Completed all target steps
    if summary_status == "completed" or (total and step >= total and w_state == "finished"):
        return (
            "completed",
            f"[DONE] Run '{name}' completed all {step}/{total} steps ({metrics_str}). {url}",
        )

    # 3. Crashed or failed unexpectedly before completion
    if w_state in {"crashed", "failed", "killed"} and (not total or step < total):
        return (
            f"stopped_{step}",
            f"[INTERRUPTED] Run '{name}' stopped at step {step}/{total} "
            f"(state={w_state}, {metrics_str}). Cloud checkpoint is saved; reopen Colab and Run All to resume. {url}",
        )

    # 4. Still marked "running" on W&B, check if heartbeat is stale (silent Colab disconnect)
    if w_state == "running" and hb is not None:
        age_sec = max(0.0, now_unix - float(hb))
        if age_sec > stale_timeout_sec:
            mins = int(age_sec // 60)
            return (
                f"stale_{step}",
                f"[DISCONNECTED] Run '{name}' has no heartbeat for {mins}m "
                f"(paused at step {step}/{total}, {metrics_str}). "
                f"Reopen Colab and click Run All to resume from the latest checkpoint. {url}",
            )

    # 5. Healthy active run
    return (
        "healthy_running",
        f"[RUNNING] '{name}' progress {step}/{total} ({metrics_str})",
    )


def send_notification(cfg: WatchdogConfig, title: str, message: str, priority: str = "default") -> None:
    """Push alert to phone via ntfy.sh and/or custom webhook using stdlib urllib."""
    if cfg.ntfy_topic:
        ntfy_url = f"https://ntfy.sh/{cfg.ntfy_topic}"
        safe_title = title.encode("ascii", errors="replace").decode("ascii")
        req = urllib.request.Request(
            ntfy_url,
            data=message.encode("utf-8"),
            headers={
                "Title": safe_title,
                "Priority": priority,
                "Tags": "atom_symbol,chart_with_downwards_trend",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            print(f"[Watchdog] ntfy push failed: {exc}")

    if cfg.webhook_url:
        payload = json.dumps({"content": f"**{title}**\n{message}", "text": f"{title}\n{message}"}).encode("utf-8")
        req = urllib.request.Request(
            cfg.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            print(f"[Watchdog] webhook push failed: {exc}")


def fetch_recent_runs(cfg: WatchdogConfig) -> list[dict[str, Any]]:
    """Query recent NQS runs from Weights & Biases API."""
    import wandb

    api = wandb.Api(timeout=20)
    path = f"{cfg.wandb_entity}/{cfg.wandb_project}" if cfg.wandb_entity else cfg.wandb_project
    runs = api.runs(path=path, per_page=cfg.max_runs_to_check)

    results: list[dict[str, Any]] = []
    for idx, run in enumerate(runs):
        if idx >= cfg.max_runs_to_check:
            break
        s = run.summary
        cfg_dict = run.config or {}
        total_steps = s.get("total_steps") or cfg_dict.get("training", {}).get("n_steps", 0)
        results.append(
            {
                "id": run.id,
                "name": run.name,
                "wandb_state": run.state,
                "status": s.get("status", ""),
                "last_step": s.get("last_step", s.get("step", 0)),
                "total_steps": total_steps,
                "heartbeat_unix": s.get("heartbeat_unix"),
                "energy_mean": s.get("energy_mean"),
                "rel_error": s.get("rel_error"),
                "v_score": s.get("v_score"),
                "url": run.url,
            }
        )
    return results


def poll_once(cfg: WatchdogConfig, print_markdown: bool = False) -> str:
    """Poll W&B once, fire deduplicated alerts on state transitions, and return status table."""
    state_path = Path(cfg.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    prev_state: dict[str, str] = {}
    if state_path.exists():
        try:
            prev_state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_state = {}

    runs = fetch_recent_runs(cfg)
    now_ts = time.time()
    new_state = dict(prev_state)

    md_lines = [
        "| Run Name | W&B State | Step | Energy | Rel Error | V-score | Health |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]

    for r in runs:
        run_id = r["id"]
        event_key, msg = classify_run_health(r, now_unix=now_ts, stale_timeout_sec=cfg.stale_timeout_sec)
        old_key = prev_state.get(run_id)

        if event_key != old_key:
            if event_key != "healthy_running":
                prio = "high" if ("diverged" in event_key or "stale" in event_key or "stopped" in event_key) else "default"
                print(f"[Alert] {msg}")
                send_notification(cfg, title=f"NQS Monitor: {r['name']}", message=msg, priority=prio)
            elif old_key and (old_key.startswith("stale_") or old_key.startswith("stopped_")):
                resume_msg = (
                    f"[RESUMED] Run '{r['name']}' reconnected and resumed "
                    f"(step {r['last_step']}/{r['total_steps']})."
                )
                print(f"[Alert] {resume_msg}")
                send_notification(cfg, title=f"NQS Resumed: {r['name']}", message=resume_msg, priority="default")
            new_state[run_id] = event_key

        e_str = f"{r['energy_mean']:.6f}" if r["energy_mean"] is not None else "-"
        rel_str = f"{r['rel_error']:.2e}" if r["rel_error"] is not None else "-"
        v_str = f"{r['v_score']:.2e}" if r["v_score"] is not None else "-"
        md_lines.append(
            f"| `{r['name']}` | {r['wandb_state']} | {r['last_step']}/{r['total_steps']} | "
            f"{e_str} | {rel_str} | {v_str} | {event_key} |"
        )

    state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    report = "\n".join(md_lines)
    if print_markdown:
        print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="GCP VM 24/7 Watchdog for Google Colab NQS runs.")
    parser.add_argument("--config", type=str, default="monitor/monitor_config.yaml", help="Watchdog YAML config.")
    parser.add_argument("--once", action="store_true", help="Check once and exit (useful for CLI / Agent inspection).")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown summary table of recent runs.")
    args = parser.parse_args()

    cfg = load_watchdog_config(args.config)

    if args.once:
        poll_once(cfg, print_markdown=args.markdown or True)
        return

    print(
        f"[Watchdog] Started 24/7 NQS Sentinel for project='{cfg.wandb_project}' "
        f"(poll={cfg.poll_interval_sec}s, stale_timeout={cfg.stale_timeout_sec}s, ntfy={cfg.ntfy_topic or 'off'})"
    )
    while True:
        try:
            poll_once(cfg, print_markdown=args.markdown)
        except Exception as exc:
            print(f"[Watchdog] Poll error (will retry in {cfg.poll_interval_sec}s): {exc}")
        time.sleep(cfg.poll_interval_sec)


if __name__ == "__main__":
    main()
