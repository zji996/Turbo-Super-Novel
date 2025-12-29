#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"

CUDA_HOME="${CUDA_HOME:-}"
MAX_JOBS="${MAX_JOBS:-5}"
USE_NINJA="${USE_NINJA:-1}"
TSN_KEEP_CUDA_HOME="${TSN_KEEP_CUDA_HOME:-0}"
TSN_FORCE_REBUILD="${TSN_FORCE_REBUILD:-0}"

nvcc_release() {
  # Prints "<major>.<minor>" (e.g. "12.8") or empty.
  "$1" -V 2>/dev/null | awk '/release/ {for (i=1;i<=NF;i++) if ($i=="release") {print $(i+1) ; exit}}' | sed 's/,//g'
}

UV="${UV_BIN:-}"
if [[ -z "$UV" ]]; then
  UV="$(command -v uv || true)"
fi
if [[ -z "$UV" ]]; then
  echo "missing command: uv" >&2
  exit 127
fi

ensure_cutlass() {
  if [[ -n "${CUTLASS_DIR:-}" && -d "${CUTLASS_DIR}/include" ]]; then
    return 0
  fi

  local cutlass_dir="$DATA_DIR/cutlass-v4.3.0"
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

  export CUTLASS_DIR="$cutlass_dir"
}

torch_cuda="$(
  "$UV" run --project "$REPO_ROOT/apps/worker" --directory "$REPO_ROOT/apps/worker" \
    python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true
)"

if [[ -n "$torch_cuda" ]]; then
  preferred_toolkit="$HOME/.local/cuda-$torch_cuda"
else
  preferred_toolkit=""
fi

if [[ -z "$CUDA_HOME" && -n "$preferred_toolkit" && -d "$preferred_toolkit" ]]; then
  CUDA_HOME="$preferred_toolkit"
fi

if [[ "$TSN_KEEP_CUDA_HOME" != "1" && -n "$torch_cuda" && -n "$CUDA_HOME" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  cuda_home_release="$(nvcc_release "$CUDA_HOME/bin/nvcc" || true)"
  if [[ -n "$cuda_home_release" && "$cuda_home_release" != "$torch_cuda" && -n "$preferred_toolkit" && -d "$preferred_toolkit" ]]; then
    echo "[warn] CUDA_HOME=$CUDA_HOME (nvcc $cuda_home_release) mismatches torch CUDA $torch_cuda; overriding to $preferred_toolkit" >&2
    CUDA_HOME="$preferred_toolkit"
  fi
fi

if [[ -z "$CUDA_HOME" ]]; then
  echo "CUDA_HOME is not set, and no matching toolkit found at ~/.local/cuda-<torch.version.cuda>." >&2
  echo "Fix: export CUDA_HOME=\$HOME/.local/cuda-12.8 (or your matching toolkit) and re-run." >&2
  exit 1
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "nvcc not found at: $CUDA_HOME/bin/nvcc" >&2
  echo "Fix: set CUDA_HOME to a CUDA toolkit directory that contains bin/nvcc." >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/libs/ai/ops/setup.py" ]]; then
  echo "missing package: $REPO_ROOT/libs/ai/ops" >&2
  exit 1
fi

export CUDA_HOME
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS
export USE_NINJA

ensure_cutlass

echo "[info] Using CUDA_HOME=$CUDA_HOME"
echo "[info] nvcc_release=$(nvcc_release "$CUDA_HOME/bin/nvcc")"
echo "[info] CUTLASS_DIR=$CUTLASS_DIR"

# Add torch's lib directory to LD_LIBRARY_PATH to avoid missing libc10.so when importing the extension.
torch_lib="$(
  "$UV" run --project "$REPO_ROOT/apps/worker" --directory "$REPO_ROOT/apps/worker" \
    python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
)"
export LD_LIBRARY_PATH="$torch_lib:$LD_LIBRARY_PATH"

if [[ "$TSN_FORCE_REBUILD" != "1" ]]; then
  if "$UV" run --project "$REPO_ROOT/apps/worker" --directory "$REPO_ROOT/apps/worker" \
    python -c "import turbo_diffusion_ops" >/dev/null 2>&1; then
    echo "[info] turbo_diffusion_ops already importable; skipping build (set TSN_FORCE_REBUILD=1 to rebuild)." >&2
    exit 0
  fi
fi

# Ensure the worker venv has the build tooling.
"$UV" run --project "$REPO_ROOT/apps/worker" --directory "$REPO_ROOT/apps/worker" python -m ensurepip --upgrade
"$UV" run --project "$REPO_ROOT/apps/worker" --directory "$REPO_ROOT/apps/worker" \
  python -m pip install -U pip setuptools wheel ninja

"$UV" run --project "$REPO_ROOT/apps/worker" --directory "$REPO_ROOT/apps/worker" \
  python -m pip install -e "$REPO_ROOT/libs/ai/ops" --no-build-isolation

"$UV" run --project "$REPO_ROOT/apps/worker" --directory "$REPO_ROOT/apps/worker" \
  python -c "import turbo_diffusion_ops; print('turbo_diffusion_ops import ok')"
