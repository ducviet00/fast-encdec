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

# Process-static CPU facts resolved once (see _resolve_isa); the per-call
# dispatch path only reads plain globals, so Dynamo never traces into the
# torch.cpu capability probes on compiled runs.
_ON_ARM = platform.machine().lower() in ("arm64", "aarch64")
_ISA_RESOLVED = False
_ISA_X86_OK = False


def _resolve_isa() -> None:
    """Resolve this CPU's ISA capability flags once, before the kernels run."""
    global _ISA_RESOLVED, _ISA_X86_OK
    if _ISA_RESOLVED:
        return
    if not _ON_ARM:
        _ISA_X86_OK = _x86_kernel_ok()
    _ISA_RESOLVED = True


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
    _resolve_isa()
    if dtype not in _FLOAT_DTYPES or block_size % 16 != 0:
        return None
    if _ON_ARM:
        if head_dim in _HEAD_DIMS_32:
            return "neon"
        return "vec16" if head_dim in _HEAD_DIMS_16 else None
    if not _ISA_X86_OK:
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


_BUILT = None


def module():
    """Return the compiled CPU attention extension module, or None if build fails."""
    # A plain global instead of functools.lru_cache: Dynamo traces cache-wrapped
    # bodies (reaching _build()'s os.path.exists) once per call site.
    global _BUILT
    if _BUILT is None:
        try:
            _resolve_isa()
            _BUILT = _build()
        except Exception:  # noqa: BLE001
            _BUILT = False
    return _BUILT or None


def require():
    """Return the compiled CPU attention extension module, raising RuntimeError if unavailable."""
    built = module()
    if built is None:
        raise RuntimeError(
            "vLLM CPU attention backend requested but csrc/ failed to build; "
            "install a C++ toolchain + ninja, or use attn_backend='auto'/'sdpa'."
        )
    return built


def _isa(dtype, block_size: int, head_dim: int) -> str:
    """Return the kernel ISA for this configuration, raising if unsupported."""
    isa = select_isa(dtype, block_size, head_dim)
    if isa is None:
        raise RuntimeError(
            f"the vLLM CPU attention kernel does not support dtype={dtype}, "
            f"block_size={block_size}, head_dim={head_dim}"
        )
    return isa


def _paged_attention(
    query, k_cache, v_cache, block_table, cache_seqlens, scale, isa, dynamic_causal=None
):
    """Run the paged kernel over an already-populated K/V cache."""
    mod = require()
    cache_seqlens = cache_seqlens.to(torch.int32)
    query_start_loc = torch.arange(
        query.shape[0] + 1, dtype=torch.int32, device=query.device
    )
    metadata = mod.build_metadata(
        query.shape[0],
        query.shape[1],
        query.shape[1],
        query.shape[2],
        cache_seqlens,
        query.dtype,
        query_start_loc,
        False,
        -1,
        isa,
        True,
        dynamic_causal,
    )
    return mod.attention_forward(
        query,
        k_cache,
        v_cache,
        query_start_loc,
        cache_seqlens,
        scale,
        False,
        -1,
        block_table,
        metadata,
        dynamic_causal,
    )


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter new key/value [T, H, D] into the paged cache at slot_mapping."""
    require().reshape_and_cache(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        _isa(key.dtype, k_cache.shape[2], key.shape[-1]),
    )


def self_attention(
    query, key, value, k_cache, v_cache, slot_mapping, block_table, context_lens, scale
):
    """Write new K/V into the paged cache and run causal self-attention."""
    isa = _isa(query.dtype, k_cache.shape[2], query.shape[-1])
    require().reshape_and_cache(key, value, k_cache, v_cache, slot_mapping, isa)
    # After the write every prefill row has length 1 and every decode row > 1.
    dynamic_causal = (context_lens == 1).to(torch.int32)
    return _paged_attention(
        query,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        scale,
        isa,
        dynamic_causal,
    )


def cross_attention(query, k_cache, v_cache, block_table, cache_seqlens, scale):
    """Run non-causal cross-attention over a staged, paged encoder K/V cache."""
    isa = _isa(query.dtype, k_cache.shape[2], query.shape[-1])
    return _paged_attention(
        query, k_cache, v_cache, block_table, cache_seqlens, scale, isa
    )
