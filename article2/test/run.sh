#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Usage:
#   ./run.sh
#   ./run.sh --rounds 50
#   ./run.sh --attacks gn,bd --defenses avg,mk
python3 "run_matrix.py" "$@"

