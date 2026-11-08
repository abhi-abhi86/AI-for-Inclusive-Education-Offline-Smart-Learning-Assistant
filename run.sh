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
    exec "$PYTHON" progress_gui.py \
      --title "TXT to JSON" -- \
      "$PYTHON" pdf_to_json.py "${@:2}"
    ;;
  text)
    exec "$PYTHON" progress_gui.py \
      --title "Extracting PDFs to Text" -- \
      "$PYTHON" pdf_to_text.py "${@:2}"
    ;;
  train)
    exec "$PYTHON" progress_gui.py \
      --title "PDF to TXT to JSON and Train Offline AI" -- \
      "$PYTHON" train_and_qa.py train-from-folders "${@:2}"
    ;;
  ask)
    exec "$PYTHON" train_and_qa.py ask "${@:2}"
    ;;
  *)
    echo "Usage: ./run.sh [gui|text|convert|train|ask] [options]" >&2
    exit 2
    ;;
esac
