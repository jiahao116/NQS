"""Zero-dependency secret & credential leak scanner for Git working tree, staged files, and history.

Run before every `git commit` / `git push` (and automatically via `.git/hooks/pre-commit` + unit tests).
Exit code 0 = clean, Exit code 1 = potential secret or forbidden file detected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Filenames / path patterns that must NEVER be committed to a public repository.
FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"(^|/)\.env(\..+)?$"),
    re.compile(r"(^|/)monitor/monitor_config\.yaml$"),
    re.compile(r"(^|/)monitor/\.watchdog_state\.json$"),
    re.compile(r"(^|/)id_(rsa|ed25519|ecdsa|dsa)(\.pub)?$"),
    re.compile(r"\.(pem|key|p12|pfx|jks|kdbx)$", re.IGNORECASE),
    re.compile(r"application_default_credentials\.json$", re.IGNORECASE),
    re.compile(r"(service[_-]?account|client[_-]?secret|oauth[_-]?config).*\.json$", re.IGNORECASE),
    re.compile(r"\.colab-cli-oauth", re.IGNORECASE),
]

# Content patterns matching real API keys, tokens, private keys, or cloud credentials.
SECRET_CONTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Private Key Header", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("W&B v1 API Key", re.compile(r"\bwandb_v1_[A-Za-z0-9_-]{20,}\b")),
    ("W&B 40-hex API Key Assignment", re.compile(r"(?i)WANDB_API_KEY\s*[:=]\s*['\"]?[0-9a-f]{40}['\"]?")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Google OAuth Access Token", re.compile(r"\bya29\.[0-9A-Za-z_-]{20,}\b")),
    ("GitHub Personal Access Token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("GitHub Fine-Grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("OpenAI / Generic sk- Key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("AWS Access Key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GCP Service Account JSON Key", re.compile(r'"private_key"\s*:\s*"-----BEGIN')),
    ("SSH Public Key Material", re.compile(r"\bssh-(ed25519|rsa)\s+AAAA[0-9A-Za-z+/=]{30,}")),
    ("Hardcoded Bearer Token", re.compile(r"(?i)Authorization:\s*Bearer\s+[A-Za-z0-9._-]{25,}")),
]


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _check_paths(paths: list[str]) -> list[str]:
    findings: list[str] = []
    for rel in paths:
        norm = rel.replace("\\", "/")
        for pat in FORBIDDEN_PATH_PATTERNS:
            if pat.search(norm):
                findings.append(f"[FORBIDDEN PATH] {norm} (matched {pat.pattern})")
    return findings


def _check_text(label: str, text: str) -> list[str]:
    findings: list[str] = []
    for name, pat in SECRET_CONTENT_PATTERNS:
        for match in pat.finditer(text):
            snippet = match.group(0)
            redacted = snippet[:10] + "..." + snippet[-4:] if len(snippet) > 18 else "[REDACTED]"
            findings.append(f"[SECRET: {name}] in {label} ({redacted})")
    return findings


def scan_staged() -> list[str]:
    staged = [p for p in _git_output(["diff", "--cached", "--name-only", "--diff-filter=ACM"]).splitlines() if p.strip()]
    findings = _check_paths(staged)
    for rel in staged:
        if rel.replace("\\", "/") == "scripts/check_secrets.py":
            continue
        try:
            blob = _git_output(["show", f":{rel}"])
        except subprocess.CalledProcessError:
            continue
        findings.extend(_check_text(f"staged:{rel}", blob))
    return findings


def scan_tracked() -> list[str]:
    tracked = [p for p in _git_output(["ls-files"]).splitlines() if p.strip()]
    findings = _check_paths(tracked)
    for rel in tracked:
        if rel.replace("\\", "/") == "scripts/check_secrets.py":
            continue
        file_path = ROOT / rel
        if not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8", errors="replace")
        findings.extend(_check_text(rel, text))
    return findings


def scan_history() -> list[str]:
    findings: list[str] = []
    commits = [c for c in _git_output(["rev-list", "--all"]).splitlines() if c.strip()]
    for commit in commits:
        files = [p for p in _git_output(["diff-tree", "--no-commit-id", "--name-only", "-r", commit]).splitlines() if p.strip()]
        findings.extend([f"[HISTORY {commit[:7]}] {f}" for f in _check_paths(files)])
        for rel in files:
            if rel.replace("\\", "/") == "scripts/check_secrets.py":
                continue
            try:
                blob = _git_output(["show", f"{commit}:{rel}"])
            except subprocess.CalledProcessError:
                continue
            for item in _check_text(f"commit:{commit[:7]}:{rel}", blob):
                findings.append(item)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan repository for accidental API keys, tokens, or credential files.")
    parser.add_argument("--staged", action="store_true", help="Scan only Git staged changes (for pre-commit hook).")
    parser.add_argument("--history", action="store_true", help="Also scan all historical Git commits.")
    args = parser.parse_args()

    findings: list[str] = []
    if args.staged:
        findings.extend(scan_staged())
    else:
        findings.extend(scan_tracked())
        if args.history:
            findings.extend(scan_history())

    if findings:
        print("SECURITY CHECK FAILED! Potential secrets or forbidden files detected:", file=sys.stderr)
        for item in findings:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)

    scope = "staged files" if args.staged else ("tracked files + full Git history" if args.history else "tracked files")
    print(f"[Security] OK — zero API keys, tokens, or forbidden credential files found in {scope}.")


if __name__ == "__main__":
    main()
