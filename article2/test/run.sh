#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Usage:
#   ./run.sh
#   ./run.sh --rounds 50
#   ./run.sh --attacks gaussian_noise,backdoor --defenses fedavg,multi_krum
python3 "run_matrix.py" "$@"

