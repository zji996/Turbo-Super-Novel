#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
PID_DIR="${PID_DIR:-$DATA_DIR/tsn_runtime/pids}"

UV_BIN="${UV_BIN:-}"
PNPM_BIN="${PNPM_BIN:-}"

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

resolve_pnpm() {
  if [[ -n "$PNPM_BIN" ]]; then
    return 0
  fi
  if command -v pnpm >/dev/null 2>&1; then
    PNPM_BIN="$(command -v pnpm)"
    return 0
  fi

  local candidates=()
  candidates+=("$HOME/.local/share/pnpm/pnpm" "$HOME/.npm-global/bin/pnpm" "/usr/local/bin/pnpm" "/usr/bin/pnpm")
  if [[ -n "${SUDO_USER:-}" ]]; then
    candidates+=("/home/$SUDO_USER/.local/share/pnpm/pnpm")
  fi

  local cand
  for cand in "${candidates[@]}"; do
    if [[ -x "$cand" ]]; then
      PNPM_BIN="$cand"
      return 0
    fi
  done

  echo "missing command: pnpm" >&2
  echo "hint: install pnpm with 'npm install -g pnpm' or set PNPM_BIN=/path/to/pnpm" >&2
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
    elif [[ -f "$REPO_ROOT/.env.example" ]]; then
      env_file="$REPO_ROOT/.env.example"
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

resolve_gpu_mode() {
  # GPU_MODE 是高层语义配置，自动推断底层参数
  # 用户也可以手动覆盖任何底层参数
  local mode="${GPU_MODE:-balanced}"
  
  case "$mode" in
    fast)
      # 🚀 速度优先: 模型常驻 + 并发 + threads 池
      export TD_RESIDENT_GPU="${TD_RESIDENT_GPU:-1}"
      export CELERY_CONCURRENCY="${CELERY_CONCURRENCY:-2}"
      export CELERY_POOL="${CELERY_POOL:-threads}"
      export TD_CUDA_CLEANUP="${TD_CUDA_CLEANUP:-0}"
      echo "[GPU_MODE=fast] 模型常驻显存, 并发=${CELERY_CONCURRENCY}, 池=${CELERY_POOL}"
      ;;
    balanced)
      # ⚖️ 平衡模式: 模型常驻 + 单任务
      export TD_RESIDENT_GPU="${TD_RESIDENT_GPU:-1}"
      export CELERY_CONCURRENCY="${CELERY_CONCURRENCY:-1}"
      export CELERY_POOL="${CELERY_POOL:-threads}"
      export TD_CUDA_CLEANUP="${TD_CUDA_CLEANUP:-0}"
      echo "[GPU_MODE=balanced] 模型常驻显存, 单任务处理"
      ;;
    lowvram)
      # 💾 显存优先: 按需加载 + 单任务
      export TD_RESIDENT_GPU="${TD_RESIDENT_GPU:-0}"
      export CELERY_CONCURRENCY="${CELERY_CONCURRENCY:-1}"
      export CELERY_POOL="${CELERY_POOL:-}"
      export TD_CUDA_CLEANUP="${TD_CUDA_CLEANUP:-1}"
      echo "[GPU_MODE=lowvram] 按需加载模型, 节省显存"
      ;;
    *)
      echo "[warn] 未知的 GPU_MODE='$mode', 使用 balanced 模式" >&2
      GPU_MODE=balanced
      resolve_gpu_mode
      return
      ;;
  esac
}

worker_cmd() {
  # 先解析 GPU_MODE，设置默认值
  resolve_gpu_mode

  local concurrency="${CELERY_CONCURRENCY:-1}"
  local prefetch="${CELERY_PREFETCH_MULTIPLIER:-1}"
  local pool="${CELERY_POOL:-}"
  local resident_gpu="${TD_RESIDENT_GPU:-0}"

  # 如果用户手动设置了底层参数但没设置 pool，自动推断
  if [[ -z "$pool" && "$resident_gpu" == "1" ]]; then
    pool="threads"
  fi

  local extra_args=""
  if [[ -n "${CELERY_MAX_TASKS_PER_CHILD:-}" ]]; then
    extra_args+=" --max-tasks-per-child ${CELERY_MAX_TASKS_PER_CHILD}"
  fi
  if [[ -n "${CELERY_MAX_MEMORY_PER_CHILD:-}" ]]; then
    extra_args+=" --max-memory-per-child ${CELERY_MAX_MEMORY_PER_CHILD}"
  fi
  local pool_arg=""
  if [[ -n "$pool" ]]; then
    pool_arg=" --pool ${pool}"
  fi
  echo "\"$UV_BIN\" run --project apps/worker --directory apps/worker celery -A celery_app:celery_app worker -l ${CELERY_LOG_LEVEL:-info}${pool_arg} --concurrency ${concurrency} --prefetch-multiplier ${prefetch}${extra_args}"
}

web_cmd() {
  local port="${WEB_PORT:-5173}"
  echo "cd \"$REPO_ROOT/apps/web\" && \"$PNPM_BIN\" dev --port $port"
}

maybe_pnpm_install() {
  local project_dir="$1"
  if [[ "${TSN_SKIP_SYNC:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -d "$REPO_ROOT/$project_dir/node_modules" ]]; then
    return 0
  fi
  (cd "$REPO_ROOT/$project_dir" && "$PNPM_BIN" install)
}

usage() {
  cat <<'EOF'
用法:
  scripts/tsn_manage.sh start   [api|worker|web|all]
  scripts/tsn_manage.sh stop    [api|worker|web|all]
  scripts/tsn_manage.sh restart [api|worker|web|all]
  scripts/tsn_manage.sh status

GPU 模式 (最重要的配置):
  GPU_MODE=fast       🚀 速度优先 (24GB+ 显存, 并发处理)
  GPU_MODE=balanced   ⚖️ 平衡模式 (16-24GB 显存, 推荐)
  GPU_MODE=lowvram    💾 显存优先 (12GB 及以下)

环境变量:
  TSN_ENV_FILE=...    指定 env 文件 (默认: .env)
  TSN_SKIP_SYNC=1     跳过 uv sync 检查

目录:
  MODELS_DIR=...      模型目录 (默认: ./models)
  DATA_DIR=...        数据目录 (默认: ./data)
  LOG_DIR=...         日志目录 (默认: ./logs)

服务端口:
  API_PORT=8000       API 端口
  WEB_PORT=5173       前端端口
EOF
}

main() {
  require_cmd setsid
  resolve_uv
  resolve_pnpm

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
          maybe_pnpm_install apps/web
          start_service "web" "$LOG_DIR/web.log" "$(web_cmd)"
          ;;
        web)
          maybe_pnpm_install apps/web
          start_service "web" "$LOG_DIR/web.log" "$(web_cmd)"
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
        web) stop_service "web" ;;
        all)
          stop_service "web"
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
      status_service "web"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
