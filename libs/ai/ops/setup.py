from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _require_dir(path: Path, *, hint: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"Missing required directory: {path}\nFix: {hint}")


repo_dir = Path(__file__).resolve().parent
ops_dir = (repo_dir / "csrc").resolve()

cutlass_dir = os.environ.get("CUTLASS_DIR", "")
if not cutlass_dir:
    raise RuntimeError(
        "CUTLASS_DIR is not set.\n"
        "Fix: set CUTLASS_DIR to a CUTLASS checkout (e.g. $DATA_DIR/cutlass-v4.3.0), "
        "or run scripts/build_turbodiffusion_ops.sh which auto-downloads it."
    )
cutlass_dir = Path(cutlass_dir).expanduser().resolve()

_require_dir(ops_dir, hint="Expected sources under libs/ai/ops/csrc")
_require_dir(cutlass_dir / "include", hint="Point CUTLASS_DIR at the CUTLASS repo root")
_require_dir(
    cutlass_dir / "tools" / "util" / "include", hint="Ensure CUTLASS repo is complete"
)

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=--verbose,--warn-on-local-memory-usage",
    "-lineinfo",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-Xcompiler",
    "-fPIC",
]

cc_flag = [
    "-gencode",
    "arch=compute_120a,code=sm_120a",
    "-gencode",
    "arch=compute_90,code=sm_90",
    "-gencode",
    "arch=compute_89,code=sm_89",
    "-gencode",
    "arch=compute_80,code=sm_80",
]

ext_modules = [
    CUDAExtension(
        name="turbo_diffusion_ops",
        sources=[
            "csrc/bindings.cpp",
            "csrc/quant/quant.cu",
            "csrc/norm/rmsnorm.cu",
            "csrc/norm/layernorm.cu",
            "csrc/gemm/gemm.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags + ["-DEXECMODE=0"] + cc_flag + ["--threads", "4"],
        },
        include_dirs=[
            str(cutlass_dir / "include"),
            str(cutlass_dir / "tools" / "util" / "include"),
            str(ops_dir),
        ],
        libraries=["cuda"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
