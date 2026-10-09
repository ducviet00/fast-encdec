"""JIT build + wrapper for the vendored vLLM CPU attention kernels (``csrc/``).

The extension is built on first use with ``torch.utils.cpp_extension`` and
cached under torch's extensions directory.  :func:`module` returns ``None``
when the build is unavailable (no compiler/ninja), so ``attn_backend="auto"``
can fall back to the pure-PyTorch SDPA path.
"""

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

# head dims with a 32-wide kernel; 48/80/112 only have the 16-wide one.
_HEAD_DIMS_32 = {32, 64, 96, 128, 160, 192, 224, 256, 512}
_HEAD_DIMS_16 = {48, 80, 112}
_FLOAT_DTYPES = {torch.float32, torch.bfloat16, torch.float16}


def _x86_kernel_ok() -> bool:
    """Whether this x86 host can run the AVX-512 kernels we compile.

    The x86 build passes ``-mavx512*``, so returning an ISA on a host without
    AVX-512 (or AVX-512 BF16) would SIGILL; the pure-PyTorch path is used
    instead.  ``select_isa`` is the only gate, and the runner only builds/calls
    the extension when it returns an ISA.
    """
    try:
        return bool(torch.cpu._is_avx512_supported()) and bool(
            torch.cpu._is_avx512_bf16_supported()
        )
    except Exception:  # noqa: BLE001 - older torch without these probes
        return False


def select_isa(dtype, block_size: int, head_dim: int) -> str | None:
    """Kernel ISA for a ``(dtype, block_size, head_dim)`` triple, else ``None``.

    aarch64 prefers the NEON BFMMLA kernel (head dims divisible by 32); x86 uses
    the 32-wide ``"vec"`` kernel, falling back to the 16-wide ``"vec16"`` one
    (which exists for every supported head dim, so ``block_size=16`` works).
    """
    if dtype not in _FLOAT_DTYPES:
        return None
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        if head_dim in _HEAD_DIMS_32 and block_size % 16 == 0:
            return "neon"
        if head_dim in _HEAD_DIMS_16 and block_size % 16 == 0:
            return "vec16"
        return None
    if not _x86_kernel_ok():
        return None
    if head_dim in _HEAD_DIMS_32 and block_size % 32 == 0:
        return "vec"
    if (
        head_dim in _HEAD_DIMS_32 or head_dim in _HEAD_DIMS_16
    ) and block_size % 16 == 0:
        return "vec16"
    return None


def _compile_flags():
    """Compiler flags mirroring vLLM's ``cmake/cpu_extension.cmake``."""
    base = [
        "-O3",
        "-fopenmp",
        "-DVLLM_NUMA_DISABLED",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-function",
    ]
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        # Without ARM_BF16_SUPPORT the NEON path uses an untested bf16
        # conversion; the march flags match vLLM's for a BF16/I8MM core.
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
    generated = os.path.join(_CPU, "cpu_attn_dispatch_generated.h")
    if not os.path.exists(generated):
        subprocess.run(
            [sys.executable, os.path.join(_CPU, "generate_cpu_attn_dispatch.py")],
            check=True,
        )
    # torch's loader shells out to ninja; make sure a venv-installed ninja is seen.
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
    """The built extension, or ``None`` if it could not be built."""
    try:
        return _build()
    except Exception:  # noqa: BLE001 - any build failure disables the backend
        return None


def require():
    """The built extension, raising if unavailable (for ``attn_backend="vllm"``)."""
    built = module()
    if built is None:
        raise RuntimeError(
            "vLLM CPU attention backend requested but csrc/ failed to build; "
            "install a C++ toolchain + ninja, or use attn_backend='auto'/'sdpa'."
        )
    return built
