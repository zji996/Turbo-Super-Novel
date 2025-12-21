#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

# Ensure a clean slate: stop previous processes and clear logs.
"$SCRIPT_DIR/tsn_manage.sh" stop all >/dev/null 2>&1 || true
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

exec "$SCRIPT_DIR/tsn_manage.sh" start all
