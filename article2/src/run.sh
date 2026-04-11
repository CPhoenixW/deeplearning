#!/usr/bin/env bash
set -euo pipefail

# Project root = parent of src/ (run all commands from there).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Usage:
#   ./src/run.sh
#   ./src/run.sh --rounds 50
#   ./src/run.sh --attacks gn,bd --defenses avg,mk
exec python3 -m src.run_matrix "$@"

