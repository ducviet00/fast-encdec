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


@functools.lru_cache(maxsize=1)
def _compiler_supports_amx() -> bool:
    """Return True if the C++ compiler accepts the AMX-BF16 tile flags."""
    import shutil

    cxx = (
        os.environ.get("CXX")
        or shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("cc")
        or "cc"
    )
    try:
        r = subprocess.run(
            [cxx, "-mamx-tile", "-mamx-bf16", "-E", "-"],
            input="#include <immintrin.h>\nint x = __AMX_BF16__;\n",
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return False
    # -E exits 0 even for unknown -m flags on some drivers, so also reject
    # the "unrecognized option" diagnostic rather than trusting returncode.
    return r.returncode == 0 and "unrecognized" not in r.stderr


@functools.lru_cache(maxsize=1)
def _amx_usable() -> bool:
    """Return True if the CPU and OS permit AMX-BF16 tile instructions."""
    # torch.cpu._init_amx() checks CPUID for AMX and requests XFEATURE_XTILEDATA
    # permission from the Linux kernel via arch_prctl(ARCH_REQ_XCOMP_PERM, 18),
    # without which the kernel traps tile instructions with SIGILL.
    try:
        return bool(torch.cpu._init_amx()) and _compiler_supports_amx()
    except Exception:  # noqa: BLE001
        return False


def select_isa(dtype, block_size: int, head_dim: int) -> str | None:
    """Return the vLLM CPU kernel ISA ('amx', 'neon', 'vec', or 'vec16') if supported, else None."""
    if dtype not in _FLOAT_DTYPES or block_size % 16 != 0:
        return None
    if platform.machine().lower() in ("arm64", "aarch64"):
        if head_dim in _HEAD_DIMS_32:
            return "neon"
        return "vec16" if head_dim in _HEAD_DIMS_16 else None
    if not _x86_kernel_ok():
        return None
    # AMX tile GEMMs only specialize BFloat16 with head_dim % 32 == 0 and a
    # 32-aligned page; fp32/fp16 and odd head dims stay on the AVX-512 path.
    if (
        dtype == torch.bfloat16
        and head_dim in _HEAD_DIMS_32
        and block_size % 32 == 0
        and _amx_usable()
    ):
        return "amx"
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
    return (
        base
        + [
            "-mavx2",
            "-mavx512f",
            "-mavx512bf16",
            "-mavx512vl",
            "-mavx512dq",
            "-mavx512bw",
        ]
        + _amx_flags()
    )


def _amx_flags():
    """Return AMX-BF16 tile flags if the compiler accepts them, else []."""
    # The tile kernel is compiled in whenever the toolchain allows it;
    # select_isa only dispatches to it on CPUs with amx_bf16+amx_tile, so the
    # unused code never faults on older x86 hosts.
    if _compiler_supports_amx():
        return ["-mamx-tile", "-mamx-bf16", "-DCPU_CAPABILITY_AMXBF16"]
    return []


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
        name="fastencdec_cpu_attn_v2",
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
        if platform.machine().lower() not in ("arm64", "aarch64"):
            _amx_usable()
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
