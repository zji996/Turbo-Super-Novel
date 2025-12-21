#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
PID_DIR="${PID_DIR:-$DATA_DIR/tsn_runtime/pids}"

"$SCRIPT_DIR/tsn_manage.sh" stop all || true
rm -f "$PID_DIR"/*.pid 2>/dev/null || true
