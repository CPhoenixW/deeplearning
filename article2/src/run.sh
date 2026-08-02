#!/usr/bin/env bash
set -euo pipefail

# Project root = parent of src/ (run all commands from there).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Usage:
#   ./src/run.sh configs/ag_news.json
#   ./src/run.sh configs/pipeline_smoke.json --dry-run
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pipeline-config.json> [--dry-run]" >&2
  exit 2
fi

CONFIG_PATH="$1"
shift
PYTHON_BIN="$ROOT/dl/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
exec "$PYTHON_BIN" -m src.pipeline --config "$CONFIG_PATH" "$@"
