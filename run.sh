#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_BIN:-python3}"
fi

case "${1:-gui}" in
  gui)
    exec "$PYTHON" gui.py "${@:2}"
    ;;
  convert)
    exec "$PYTHON" pdf_to_json.py "${@:2}"
    ;;
  train)
    exec "$PYTHON" train_and_qa.py train-from-folders "${@:2}"
    ;;
  ask)
    exec "$PYTHON" train_and_qa.py ask "${@:2}"
    ;;
  *)
    echo "Usage: ./run.sh [gui|convert|train|ask] [options]" >&2
    exit 2
    ;;
esac
