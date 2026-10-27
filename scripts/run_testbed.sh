#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "${1:-}" = "--mininet" ]; then
    shift
    exec python scripts/run_mininet.py "$@"
fi

python scripts/collect_data.py
python scripts/train_gnn.py
python scripts/train_rl.py
python scripts/evaluate.py
