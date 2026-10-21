"""Loader and thin wrapper for the Mojo paged-attention kernel.

The kernel is built from ``mojo/src`` into ``mojo/build/fastencdec_mojo_attn.so``
(``pixi run -m mojo/pixi.toml build``). Set ``FASTENCDEC_MOJO_ATTN`` to point at
another build. Everything here is optional: without the shared library (or on an
unsupported configuration) ``module()`` is ``None`` and ``attn_backend="auto"``
falls back to the vendored C++ kernel.
"""

import importlib.util
import os

import torch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DEFAULT_PATH = os.path.join(_REPO_ROOT, "mojo", "build", "fastencdec_mojo_attn.so")

# The kernel specializes on head_dim at compile time; anything else raises.
_HEAD_DIMS = {32, 64, 128}
_DTYPE_CODES = {torch.float32: 0, torch.bfloat16: 1}

_BUILT = None


def _load():
    path = os.environ.get("FASTENCDEC_MOJO_ATTN", _DEFAULT_PATH)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("fastencdec_mojo_attn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module():
    """Return the compiled Mojo kernel module, or None if it is unavailable."""
    global _BUILT
    if _BUILT is None:
        try:
            _BUILT = _load()
        except Exception:  # noqa: BLE001
            _BUILT = False
    return _BUILT or None


def supported(dtype, block_size: int, head_dim: int) -> bool:
    """Return True if the Mojo kernel can run this dtype / page size / head dim."""
    return (
        dtype in _DTYPE_CODES
        and head_dim in _HEAD_DIMS
        and block_size % 16 == 0
        and module() is not None
    )


def require():
    """Return the Mojo kernel module, raising RuntimeError if it is unavailable."""
    built = module()
    if built is None:
        raise RuntimeError(
            "Mojo attention backend requested but fastencdec_mojo_attn.so is "
            "missing; build it with `pixi run --manifest-path mojo/pixi.toml build`."
        )
    return built


_ATTENTION = None


def _attention():
    """The extension's single entry point, resolved once (saves a lookup per call)."""
    global _ATTENTION
    if _ATTENTION is None:
        _ATTENTION = require().attention
    return _ATTENTION


_PHYSICAL = None
# Read once: an os.environ.get on the hot path costs ~0.35 us per call, i.e. >1% of
# a short-context attention call. Harnesses that sweep the worker count should
# patch ``_OVERRIDE`` (or ``_workers``) directly, not the environment.
_OVERRIDE = os.environ.get("FASTENCDEC_MOJO_WORKERS")


def _physical_cores() -> int:
    """Physical cores visible to this process, from the topology sysfs files."""
    global _PHYSICAL
    if _PHYSICAL is None:
        cores = set()
        try:
            for cpu in range(os.cpu_count() or 1):
                base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
                with open(f"{base}/core_id") as f:
                    core = f.read().strip()
                with open(f"{base}/physical_package_id") as f:
                    cores.add((f.read().strip(), core))
        except OSError:
            cores = set()
        _PHYSICAL = len(cores) or (os.cpu_count() or 1)
    return _PHYSICAL


def _workers(num_tasks: int, pages: int = 1) -> int:
    """Workers for one call: the physical-core cap, inline only for tiny work.

    The kernel dispatches on torch's own OpenMP pool (``GOMP_parallel``), so a
    fork-join costs ~2.4 us warm and ~2.7 us with a 1 ms gap -- there is no
    sleeping second pool to wake, which is why the pool now wins almost
    everywhere: measured cold, one worker (inline, 0.36 us) beats 8 only up to
    ~32 KV positions per call (24.5 vs 31.0 us at batch 1 / ctx 64), while at
    batch 1 / ctx 128 the pool is already 1.3x faster and at batch 8 / ctx 256
    it is 8x. The cap at the *physical* core count matters on SMT boxes (8
    workers on a 4-core Xeon: 1.5x slower at ctx 1024); torch already sizes its
    pool that way, so it only bites when a parent sets a larger
    ``OMP_NUM_THREADS``. ``FASTENCDEC_MOJO_WORKERS`` overrides everything.
    """
    if _OVERRIDE:
        return max(1, min(int(_OVERRIDE), num_tasks))
    workers = min(torch.get_num_threads(), _physical_cores(), num_tasks)
    if workers > 1 and num_tasks * pages <= 32:
        return 1
    return workers


def _call(
    query,
    key,
    value,
    k_cache,
    v_cache,
    out,
    slots,
    block_table,
    lens,
    scale,
    do_store: bool,
):
    """Run one kernel call; ``key``/``value``/``slots`` are ignored without a store."""
    attention = _attention()
    zero = 0
    num_seqs, num_heads, head_dim = query.shape
    q_stride = query.stride()
    if q_stride[-1] != 1 or q_stride[-2] != head_dim:
        query = query.contiguous()
        q_stride = query.stride()
    # The fused q/k/v projection hands us chunks of one buffer: the head dim is
    # contiguous but the token stride is 3 * d_model, which the kernel takes
    # explicitly (a head-major layout check would silently copy).
    kv_stride = 0
    if do_store:
        k_stride = key.stride()
        if k_stride[-1] != 1 or k_stride[-2] != head_dim or value.stride() != k_stride:
            key, value = key.contiguous(), value.contiguous()
            k_stride = key.stride()
        kv_stride = k_stride[0]
    num_tasks = num_seqs * num_heads
    attention(
        query.data_ptr(),
        key.data_ptr() if do_store else zero,
        value.data_ptr() if do_store else zero,
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        out.data_ptr(),
        slots.data_ptr() if do_store else zero,
        block_table.data_ptr(),
        lens.data_ptr(),
        num_seqs,
        num_heads,
        k_cache.shape[1],
        head_dim,
        k_cache.shape[2],
        block_table.shape[1],
        q_stride[0],
        kv_stride,
        out.stride(0),
        _DTYPE_CODES[query.dtype],
        scale,
        _workers(num_tasks, block_table.shape[1]),
        do_store,
        True,
    )
    return out


def self_attention(
    query, key, value, k_cache, v_cache, slot_mapping, block_table, context_lens, scale
):
    """Write new K/V into the paged cache and run attention over cached keys."""
    return _call(
        query,
        key,
        value,
        k_cache,
        v_cache,
        torch.empty_like(query),
        slot_mapping,
        block_table,
        context_lens,
        scale,
        do_store=True,
    )


def cross_attention(query, k_cache, v_cache, block_table, cache_seqlens, scale):
    """Run cross-attention over a staged, paged encoder K/V cache."""
    return _call(
        query,
        None,
        None,
        k_cache,
        v_cache,
        torch.empty_like(query),
        None,
        block_table,
        cache_seqlens,
        scale,
        do_store=False,
    )


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter key/value [T, H, D] into the paged cache at slot_mapping."""
    # The `q`/`dst`/`lens` slots of the shared entry point are unused here: the
    # kernel only stores when ``do_attend`` is False.
    attention = _attention()
    num_tokens, num_heads, head_dim = key.shape
    if key.stride()[-1] != 1 or key.stride()[-2] != head_dim:
        key, value = key.contiguous(), value.contiguous()
    attention(
        key.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        0,
        slot_mapping.data_ptr(),
        0,
        0,
        num_tokens,
        num_heads,
        k_cache.shape[1],
        head_dim,
        k_cache.shape[2],
        0,
        0,
        key.stride(0),
        0,
        _DTYPE_CODES[key.dtype],
        1.0,
        _workers(num_tokens * num_heads),
        True,
        False,
    )
