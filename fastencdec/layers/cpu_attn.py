import functools
import os
import platform
import subprocess
import sys

import torch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_CSRC = os.path.join(_REPO_ROOT, "csrc")
_CPU = os.path.join(_CSRC, "cpu")

# Head dims with a 32-wide kernel; 48/80/112 only have the 16-wide one.
_HEAD_DIMS_32 = {32, 64, 96, 128, 160, 192, 224, 256, 512}
_HEAD_DIMS_16 = {48, 80, 112}
_FLOAT_DTYPES = {torch.float32, torch.bfloat16, torch.float16}


def _x86_kernel_ok() -> bool:
    """Return True if this x86 CPU supports AVX-512 and AVX-512 BF16."""
    # Guard against SIGILL since x86 kernels are compiled with -mavx512*.
    try:
        return bool(torch.cpu._is_avx512_supported()) and bool(
            torch.cpu._is_avx512_bf16_supported()
        )
    except Exception:  # noqa: BLE001
        return False


def select_isa(dtype, block_size: int, head_dim: int) -> str | None:
    """Return the vLLM CPU kernel ISA ('neon', 'vec', or 'vec16') if supported, else None."""
    if dtype not in _FLOAT_DTYPES or block_size % 16 != 0:
        return None
    if platform.machine().lower() in ("arm64", "aarch64"):
        if head_dim in _HEAD_DIMS_32:
            return "neon"
        return "vec16" if head_dim in _HEAD_DIMS_16 else None
    if not _x86_kernel_ok():
        return None
    if head_dim in _HEAD_DIMS_32 and block_size % 32 == 0:
        return "vec"
    return "vec16" if head_dim in (_HEAD_DIMS_32 | _HEAD_DIMS_16) else None


def _compile_flags():
    """Return architecture-specific C++ compiler flags for the CPU attention extension."""
    base = [
        "-O3",
        "-fopenmp",
        "-DVLLM_NUMA_DISABLED",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-function",
    ]
    if platform.machine().lower() in ("arm64", "aarch64"):
        return base + [
            "-march=armv8.2-a+bf16+dotprod+fp16+i8mm",
            "-DARM_BF16_SUPPORT",
            "-DARM_I8MM_SUPPORT",
        ]
    return base + [
        "-mavx2",
        "-mavx512f",
        "-mavx512bf16",
        "-mavx512vl",
        "-mavx512dq",
        "-mavx512bw",
    ]


def _build():
    """JIT-compile and load the vendored vLLM CPU attention C++ extension."""
    generated = os.path.join(_CPU, "cpu_attn_dispatch_generated.h")
    if not os.path.exists(generated):
        subprocess.run(
            [sys.executable, os.path.join(_CPU, "generate_cpu_attn_dispatch.py")],
            check=True,
        )
    # Ensure venv-installed ninja is on PATH for torch's cpp_extension loader.
    os.environ["PATH"] = (
        os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    )
    from torch.utils.cpp_extension import load

    return load(
        name="fastencdec_cpu_attn",
        sources=[
            os.path.join(_CSRC, "bindings.cpp"),
            os.path.join(_CPU, "cpu_attn.cpp"),
            os.path.join(_CPU, "utils.cpp"),
            os.path.join(_CPU, "cpu_isa.cpp"),
        ],
        extra_include_paths=[_CSRC, _CPU],
        extra_cflags=_compile_flags(),
        extra_ldflags=["-fopenmp"],
        with_cuda=False,
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def module():
    """Return the compiled CPU attention extension module, or None if build fails."""
    try:
        return _build()
    except Exception:  # noqa: BLE001
        return None


def require():
    """Return the compiled CPU attention extension module, raising RuntimeError if unavailable."""
    built = module()
    if built is None:
        raise RuntimeError(
            "vLLM CPU attention backend requested but csrc/ failed to build; "
            "install a C++ toolchain + ninja, or use attn_backend='auto'/'sdpa'."
        )
    return built
