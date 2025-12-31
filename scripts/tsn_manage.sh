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
  local group_args=()
  local sync_args=()
  local groups=""
  local cuda_home=""
  local cudacxx=""
  local cutlass_dir=""
  if [[ "$project_dir" == "apps/worker" ]]; then
    # When TurboDiffusion ops are enabled, keep extra packages (like editable builds) instead of pruning them.
    if [[ "${TSN_BUILD_TD_OPS:-0}" == "1" ]]; then
      sync_args+=(--inexact)
    fi

    # Worker is GPU inference oriented; default to installing the `cuda` group to enable FlashAttention etc.
    # Override examples:
    #   TSN_WORKER_UV_GROUPS=cuda,sagesla   # install both groups
    #   TSN_WORKER_UV_GROUPS=               # install no extra groups
    groups="${TSN_WORKER_UV_GROUPS:-cuda,sagesla}"

    # SageSLA / TurboDiffusion ops compilation requires nvcc matching torch CUDA (e.g. torch +cu128 -> nvcc 12.8).
    # Prefer a user-installed toolkit at ~/.local/cuda-<torch.cuda>, falling back to system nvcc.
    cuda_home="${CUDA_HOME:-}"
    cudacxx="${CUDACXX:-}"
    if [[ -z "$cuda_home" ]]; then
      local torch_cuda
      torch_cuda="$("$UV_BIN" run --project apps/worker --directory apps/worker python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)"
      if [[ -n "$torch_cuda" && -d "$HOME/.local/cuda-$torch_cuda" ]]; then
        cuda_home="$HOME/.local/cuda-$torch_cuda"
      fi
    fi
    if [[ -n "$cuda_home" && -z "$cudacxx" && -x "$cuda_home/bin/nvcc" ]]; then
      cudacxx="$cuda_home/bin/nvcc"
    fi

    # CUTLASS is required to build `turbo_diffusion_ops` (used by quantized Wan2.2 checkpoints).
    cutlass_dir="${CUTLASS_DIR:-$DATA_DIR/cutlass-v4.3.0}"
    if [[ ! -d "$cutlass_dir/include" ]]; then
      mkdir -p "$DATA_DIR"
      if ! command -v curl >/dev/null 2>&1; then
        echo "missing command: curl (required to fetch CUTLASS)" >&2
        exit 127
      fi
      if ! command -v tar >/dev/null 2>&1; then
        echo "missing command: tar (required to extract CUTLASS)" >&2
        exit 127
      fi
      echo "[info] CUTLASS not found; downloading v4.3.0 into $cutlass_dir" >&2
      rm -rf "$cutlass_dir"
      local cutlass_tgz="$DATA_DIR/cutlass-v4.3.0.tar.gz"
      local cutlass_tmp="$DATA_DIR/.cutlass-v4.3.0.tmp"
      rm -rf "$cutlass_tmp" "$cutlass_tgz"
      curl -L --fail -o "$cutlass_tgz" "https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.3.0"
      mkdir -p "$cutlass_tmp"
      tar -xzf "$cutlass_tgz" -C "$cutlass_tmp"
      mv "$cutlass_tmp"/cutlass-* "$cutlass_dir"
      rm -rf "$cutlass_tmp" "$cutlass_tgz"
    fi

    if [[ -n "$groups" ]]; then
      local g
      IFS=',' read -r -a _tsn_groups <<<"$groups"
      for g in "${_tsn_groups[@]}"; do
        g="$(echo "$g" | xargs)"
        if [[ -n "$g" ]]; then
          group_args+=(--group "$g")
        fi
      done
    fi
  fi
  if [[ -n "$cuda_home" || -n "$cudacxx" ]]; then
    if CUDA_HOME="$cuda_home" CUDACXX="$cudacxx" CUTLASS_DIR="$cutlass_dir" "$UV_BIN" sync "${sync_args[@]}" --project "$project_dir" "${group_args[@]}"; then
      return 0
    fi
  else
    if CUTLASS_DIR="$cutlass_dir" "$UV_BIN" sync "${sync_args[@]}" --project "$project_dir" "${group_args[@]}"; then
      return 0
    fi
  fi

  # If SageSLA build fails (nvcc/CUDA mismatch is common), automatically fall back to `cuda` only.
  if [[ "$project_dir" == "apps/worker" && "$groups" == *"sagesla"* ]]; then
    echo "[warn] uv sync for worker failed with TSN_WORKER_UV_GROUPS='${groups}'; retrying without 'sagesla'." >&2
    if [[ -n "$cuda_home" || -n "$cudacxx" ]]; then
      CUDA_HOME="$cuda_home" CUDACXX="$cudacxx" CUTLASS_DIR="$cutlass_dir" "$UV_BIN" sync "${sync_args[@]}" --project "$project_dir" --group cuda
    else
      CUTLASS_DIR="$cutlass_dir" "$UV_BIN" sync "${sync_args[@]}" --project "$project_dir" --group cuda
    fi
    return 0
  fi

  return 1
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
  local concurrency_explicit="0"
  if [[ -n "${CELERY_CONCURRENCY+x}" ]]; then
    concurrency_explicit="1"
  fi

  local pool_explicit="0"
  if [[ -n "${CELERY_POOL+x}" ]]; then
    pool_explicit="1"
  fi

  local capabilities="${WORKER_CAPABILITIES:-tts,videogen}"
  local queues="celery"
  local cap_count="0"
  IFS=',' read -r -a _tsn_caps <<<"$capabilities"
  local cap
  for cap in "${_tsn_caps[@]}"; do
    cap="$(echo "$cap" | xargs)"
    if [[ -n "$cap" ]]; then
      queues="${queues},cap.${cap}"
      cap_count=$((cap_count + 1))
    fi
  done

  export CAP_GPU_MODE="${CAP_GPU_MODE:-ondemand}"

  local concurrency="${CELERY_CONCURRENCY:-1}"
  local prefetch="${CELERY_PREFETCH_MULTIPLIER:-1}"
  local pool="${CELERY_POOL:-}"

  # Capability switching requires a single consumer to avoid concurrent GPU use.
  if [[ "$cap_count" -gt 1 && "$concurrency_explicit" == "0" ]]; then
    concurrency="1"
    if [[ "$pool_explicit" == "0" ]]; then
      pool=""
    fi
  fi

  # If user increases concurrency without explicitly setting pool, default to threads.
  if [[ -z "$pool" && "$concurrency" != "1" ]]; then
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
  echo "\"$UV_BIN\" run --project apps/worker --directory apps/worker celery -A celery_app:celery_app worker -l ${CELERY_LOG_LEVEL:-info}${pool_arg} -Q ${queues} --concurrency ${concurrency} --prefetch-multiplier ${prefetch}${extra_args}"
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

maybe_build_turbodiffusion_ops() {
  if [[ "${TSN_BUILD_TD_OPS:-0}" != "1" ]]; then
    return 0
  fi
  if [[ ! -x "$REPO_ROOT/scripts/build_turbodiffusion_ops.sh" ]]; then
    echo "missing script: $REPO_ROOT/scripts/build_turbodiffusion_ops.sh" >&2
    exit 1
  fi
  echo "[info] TSN_BUILD_TD_OPS=1: ensuring turbo_diffusion_ops is installed..." >&2
  "$REPO_ROOT/scripts/build_turbodiffusion_ops.sh"
}

nvcc_release() {
  # Prints "<major>.<minor>" (e.g. "12.8") or empty.
  "$1" -V 2>/dev/null | awk '/release/ {for (i=1;i<=NF;i++) if ($i=="release") {print $(i+1) ; exit}}' | sed 's/,//g'
}

ensure_worker_cuda_toolkit_env() {
  # Ensure the worker runtime can find libcudart.so.* and (optionally) a matching nvcc.
  if [[ "${TSN_SET_CUDA_TOOLKIT_ENV:-1}" != "1" ]]; then
    return 0
  fi

  local torch_cuda=""
  torch_cuda="$("$UV_BIN" run --project apps/worker --directory apps/worker python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)"

  local preferred_toolkit=""
  if [[ -n "$torch_cuda" && -d "$HOME/.local/cuda-$torch_cuda" ]]; then
    preferred_toolkit="$HOME/.local/cuda-$torch_cuda"
  fi

  if [[ "${TSN_KEEP_CUDA_HOME:-0}" != "1" && -n "$preferred_toolkit" ]]; then
    if [[ -z "${CUDA_HOME:-}" ]]; then
      export CUDA_HOME="$preferred_toolkit"
    elif [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
      local current_release=""
      current_release="$(nvcc_release "${CUDA_HOME}/bin/nvcc" || true)"
      if [[ -n "$current_release" && -n "$torch_cuda" && "$current_release" != "$torch_cuda" ]]; then
        export CUDA_HOME="$preferred_toolkit"
      fi
    fi
  fi

  if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
}

ensure_worker_torch_ld_library_path() {
  # TurboDiffusion ops is a native extension; importing it needs Torch's lib dir on LD_LIBRARY_PATH.
  if [[ "${TSN_SET_TORCH_LD_LIBRARY_PATH:-1}" != "1" ]]; then
    return 0
  fi
  local torch_lib
  torch_lib="$("$UV_BIN" run --project apps/worker --directory apps/worker python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || true)"
  if [[ -z "$torch_lib" || ! -d "$torch_lib" ]]; then
    return 0
  fi
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$torch_lib:"*) ;;
    *) export LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}" ;;
  esac
}

usage() {
  cat <<'EOF'
用法:
  scripts/tsn_manage.sh start   [api|worker|web|all]
  scripts/tsn_manage.sh stop    [api|worker|web|all]
  scripts/tsn_manage.sh restart [api|worker|web|all]
  scripts/tsn_manage.sh status

GPU 模式 (最重要的配置):
  CAP_GPU_MODE=resident  模型常驻（直到切换 capability）
  CAP_GPU_MODE=ondemand  每个任务完成后卸载（默认）

环境变量:
  TSN_ENV_FILE=...    指定 env 文件 (默认: .env)
  TSN_SKIP_SYNC=1     跳过 uv sync 检查
  TSN_BUILD_TD_OPS=1  启动 worker 前构建 turbo_diffusion_ops
  TSN_SET_TORCH_LD_LIBRARY_PATH=0  禁用自动设置 torch lib 到 LD_LIBRARY_PATH
  TSN_KEEP_CUDA_HOME=1  不自动覆盖 CUDA_HOME
  TSN_SET_CUDA_TOOLKIT_ENV=0  禁用自动设置 CUDA toolkit 环境变量

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
          maybe_build_turbodiffusion_ops
          ensure_worker_cuda_toolkit_env
          ensure_worker_torch_ld_library_path
          start_service "worker" "$LOG_DIR/worker.log" "$(worker_cmd)"
          ;;
        all)
          load_env api
          maybe_uv_sync apps/api
          start_service "api" "$LOG_DIR/api.log" "$(api_cmd)"
          load_env worker
          maybe_uv_sync apps/worker
          maybe_build_turbodiffusion_ops
          ensure_worker_cuda_toolkit_env
          ensure_worker_torch_ld_library_path
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
