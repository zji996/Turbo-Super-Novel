#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
PID_DIR="${PID_DIR:-$DATA_DIR/tsn_runtime/pids}"

UV_BIN="${UV_BIN:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1" >&2
    exit 127
  fi
}

resolve_uv() {
  if [[ -n "$UV_BIN" ]]; then
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi

  local candidates=()
  candidates+=("$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "/usr/local/bin/uv" "/usr/bin/uv")
  if [[ -n "${SUDO_USER:-}" ]]; then
    candidates+=("/home/$SUDO_USER/.local/bin/uv" "/home/$SUDO_USER/.cargo/bin/uv")
  fi

  local cand
  for cand in "${candidates[@]}"; do
    if [[ -x "$cand" ]]; then
      UV_BIN="$cand"
      return 0
    fi
  done

  echo "missing command: uv" >&2
  echo "hint: uv is installed at ~/.local/bin/uv; avoid running with sudo, or set UV_BIN=/path/to/uv" >&2
  exit 127
}

apply_runtime_defaults() {
  LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
  DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
  MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
  PID_DIR="${PID_DIR:-$DATA_DIR/tsn_runtime/pids}"

  export LOG_DIR DATA_DIR MODELS_DIR

  mkdir -p "$LOG_DIR" "$PID_DIR"
  if [[ ! -w "$PID_DIR" ]]; then
    echo "PID_DIR is not writable: $PID_DIR" >&2
    echo "fix: chown it to your user (e.g. sudo chown -R $USER:$USER \"$PID_DIR\") or set PID_DIR to a writable path" >&2
    exit 1
  fi
}

load_env() {
  local service="$1"
  local env_file="${TSN_ENV_FILE:-}"

  if [[ -z "$env_file" ]]; then
    if [[ -f "$REPO_ROOT/.env" ]]; then
      env_file="$REPO_ROOT/.env"
    elif [[ -f "$REPO_ROOT/apps/$service/.env" ]]; then
      env_file="$REPO_ROOT/apps/$service/.env"
    else
      env_file="$REPO_ROOT/apps/$service/env.example"
    fi
  fi

  if [[ ! -f "$env_file" ]]; then
    echo "env file not found: $env_file" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a

  apply_runtime_defaults
}

pidfile() {
  echo "$PID_DIR/$1.pid"
}

is_running() {
  local pid="$1"
  kill -0 "-$pid" >/dev/null 2>&1
}

maybe_uv_sync() {
  local project_dir="$1"
  if [[ "${TSN_SKIP_SYNC:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -d "$REPO_ROOT/$project_dir/.venv" ]]; then
    return 0
  fi
  "$UV_BIN" sync --project "$project_dir"
}

start_service() {
  local name="$1"
  local log_path="$2"
  local cmd="$3"

  local pf
  pf="$(pidfile "$name")"

  if [[ -f "$pf" ]]; then
    local existing
    existing="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$existing" ]] && is_running "$existing"; then
      echo "$name already running (pid=$existing)"
      return 0
    fi
    rm -f "$pf"
  fi

  echo "starting $name; logs: $log_path"
  (
    cd "$REPO_ROOT"
    exec setsid bash -lc "$cmd" >>"$log_path" 2>&1
  ) &
  local pid="$!"
  echo "$pid" >"$pf"
  echo "$name started (pid=$pid)"
}

stop_service() {
  local name="$1"
  local pf
  pf="$(pidfile "$name")"

  if [[ ! -f "$pf" ]]; then
    echo "$name not running (no pidfile)"
    return 0
  fi

  local pid
  pid="$(cat "$pf" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$pf"
    echo "$name not running (empty pidfile)"
    return 0
  fi

  if ! is_running "$pid"; then
    rm -f "$pf"
    echo "$name not running (stale pidfile pid=$pid)"
    return 0
  fi

  echo "stopping $name (pid=$pid)"
  kill -TERM "-$pid" >/dev/null 2>&1 || true

  local i
  for i in {1..50}; do
    if ! is_running "$pid"; then
      rm -f "$pf"
      echo "$name stopped"
      return 0
    fi
    sleep 0.2
  done

  echo "$name did not stop in time; killing (pid=$pid)" >&2
  kill -KILL "-$pid" >/dev/null 2>&1 || true
  rm -f "$pf"
}

status_service() {
  local name="$1"
  local pf
  pf="$(pidfile "$name")"
  if [[ ! -f "$pf" ]]; then
    echo "$name: stopped"
    return 0
  fi
  local pid
  pid="$(cat "$pf" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && is_running "$pid"; then
    echo "$name: running (pid=$pid)"
  else
    echo "$name: stopped (stale pidfile)"
  fi
}

api_cmd() {
  local host="${API_HOST:-0.0.0.0}"
  local port="${API_PORT:-8000}"
  local reload="${API_RELOAD:-1}"
  local reload_arg=""
  if [[ "$reload" == "1" ]]; then
    reload_arg="--reload"
  fi
  echo "\"$UV_BIN\" run --project apps/api --directory apps/api uvicorn main:app --host $host --port $port $reload_arg"
}

worker_cmd() {
  local concurrency="${CELERY_CONCURRENCY:-5}"
  local prefetch="${CELERY_PREFETCH_MULTIPLIER:-1}"
  echo "\"$UV_BIN\" run --project apps/worker --directory apps/worker celery -A celery_app:celery_app worker -l ${CELERY_LOG_LEVEL:-info} --concurrency ${concurrency} --prefetch-multiplier ${prefetch}"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/tsn_manage.sh start  [api|worker|all]
  scripts/tsn_manage.sh stop   [api|worker|all]
  scripts/tsn_manage.sh restart [api|worker|all]
  scripts/tsn_manage.sh status

Env:
  TSN_ENV_FILE=...        Optional env file to source (default: .env or apps/<svc>/.env or env.example)
  TSN_SKIP_SYNC=1         Skip `uv sync` checks
  LOG_DIR=...             Log directory (default: ./logs)
  DATA_DIR=...            Data directory for pidfiles (default: ./data)

API:
  API_HOST=0.0.0.0
  API_PORT=8000
  API_RELOAD=1            Use uvicorn --reload (default: 1)

Worker:
  CELERY_LOG_LEVEL=info
  CELERY_CONCURRENCY=5
  CELERY_PREFETCH_MULTIPLIER=1
EOF
}

main() {
  require_cmd setsid
  resolve_uv

  if [[ "${EUID}" -eq 0 && -n "${SUDO_USER:-}" && "${TSN_ALLOW_SUDO:-0}" != "1" ]]; then
    echo "This script should not run as root; re-running as $SUDO_USER." >&2
    exec sudo -u "$SUDO_USER" -H --preserve-env=TSN_ENV_FILE,TSN_SKIP_SYNC,LOG_DIR,DATA_DIR,MODELS_DIR,PID_DIR,UV_BIN "$0" "$@"
  fi

  apply_runtime_defaults

  local action="${1:-}"
  local target="${2:-all}"

  case "$action" in
    start)
      case "$target" in
        api)
          load_env api
          maybe_uv_sync apps/api
          start_service "api" "$LOG_DIR/api.log" "$(api_cmd)"
          ;;
        worker)
          load_env worker
          maybe_uv_sync apps/worker
          start_service "worker" "$LOG_DIR/worker.log" "$(worker_cmd)"
          ;;
        all)
          load_env api
          maybe_uv_sync apps/api
          start_service "api" "$LOG_DIR/api.log" "$(api_cmd)"
          load_env worker
          maybe_uv_sync apps/worker
          start_service "worker" "$LOG_DIR/worker.log" "$(worker_cmd)"
          ;;
        *)
          usage
          exit 2
          ;;
      esac
      ;;
    stop)
      case "$target" in
        api) stop_service "api" ;;
        worker) stop_service "worker" ;;
        all)
          stop_service "worker"
          stop_service "api"
          ;;
        *)
          usage
          exit 2
          ;;
      esac
      ;;
    restart)
      "$0" stop "$target"
      "$0" start "$target"
      ;;
    status)
      status_service "api"
      status_service "worker"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
