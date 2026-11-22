#!/usr/bin/env bash
# ==============================================================================
# Installs a local Git pre-commit hook that blocks any accidental commit of
# API keys, tokens, private keys, or forbidden credential files.
# Uses the resolved Python interpreter path directly in the shebang so it works
# natively on Linux, macOS, and Windows (avoiding MSYS2/env sandbox issues).
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_PATH="${REPO_DIR}/.git/hooks/pre-commit"
PYTHON_BIN="$(command -v python3 || command -v python)"

cat > "${HOOK_PATH}" <<EOF
#!${PYTHON_BIN}
import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, "scripts/check_secrets.py", "--staged"]))
EOF

chmod +x "${HOOK_PATH}"
echo "Installed pre-commit secret guard at ${HOOK_PATH} (python: ${PYTHON_BIN})."
