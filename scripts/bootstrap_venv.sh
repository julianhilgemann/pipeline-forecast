#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

choose_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi

  for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done

  echo "No Python interpreter found. Install Python 3.11+ and retry." >&2
  exit 1
}

if [[ ! -d "$VENV_DIR" ]]; then
  PYTHON_CMD="$(choose_python)"
  echo "Creating virtual environment with: $PYTHON_CMD"
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

if [[ $# -eq 0 ]]; then
  cat <<EOF
Virtual environment ready: $VENV_DIR
Activate it:
  source "$VENV_DIR/bin/activate"

Run a Python command inside the venv:
  ./scripts/bootstrap_venv.sh pipeline_forecast.py
  ./scripts/bootstrap_venv.sh -m unittest discover -s tests -v
EOF
  exit 0
fi

exec "$VENV_PYTHON" "$@"
