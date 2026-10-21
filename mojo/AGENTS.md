# AGENTS.md — Mojo paged-attention port (handover)

Everything learned building `mojo/src/attention.mojo`: a Mojo port of the hot
path of `csrc/` (the vendored vLLM CPU attention kernel), wired into the engine
as `attn_backend="mojo"`. Read this before touching the kernel; the root
`AGENTS.md` covers the engine side.

## 1. Status in one line

One source file, compile-time per-ISA paths (AVX512-BF16 `vdpbf16ps` + fp32 P@V
on x86; NEON `bfdot` QK **and** P@V on ARM), bit-identical to `csrc/` on x86,
≤1e-3 on ARM (bf16 probabilities). Beats `csrc/` at `cross b8/ctx256` by 1.4× on
the 4-core Xeon and 2.0× on Graviton 5, parity-or-better elsewhere. It runs on
torch's own OpenMP pool (`GOMP_parallel`), which makes the engine **faster than
the vendored C++ backend end to end** on the laptop: 392-396 vs 364-377 tok/s at
8x128 (§9.3), 29.8 vs 90.7 µs per attention call at b8/ctx256 (§7).

## 2. Layout

```
mojo/
  pixi.toml                 Mojo nightly, no MAX; platforms = linux-64, linux-aarch64
  src/attention.mojo        the kernel (Job, store, generic + dot attention, dispatch)
  src/fastencdec_mojo_attn.mojo  Python extension: PyInit_..., one `attention(...)` entry
  tests/test_attention.mojo standalone parity test vs a scalar fp32 reference
  tests/bench_qk.mojo       isolated QK micro-benchmark (fp32 vs dpbf16ps)
  README.md                 user-facing: design, per-box tables, "what not to redo"
```

Python side (see root `AGENTS.md`): `fastencdec/layers/mojo_attn.py` (loader +
call wrapper), `mojo_ops.py` (opaque custom ops), dispatch in
`layers/attention.py` (`_OPAQUE`, `uses_custom_ops`), compile gating in
`engine/model_runner.py` (`compiled_backend`), test `tests/test_mojo_attn.py`.

## 3. Build, run, test

```bash
cd mojo
pixi run build        # mojo build src/... --emit shared-lib -o build/... -Xlinker -lgomp
pixi run test         # mojo build -I src tests/test_attention.mojo -o build/test_attention \
                      #     -Xlinker -lgomp && build/test_attention        -> 11 OK lines
```

`-Xlinker -lgomp` is a hard build dependency: the kernel dispatches on libgomp
(§4), so the `.so` records `DT_NEEDED libgomp.so.1` and the standalone test
binary links it too. Two consequences:
`mojo run` **cannot** run the kernel (the JIT cannot resolve `GOMP_parallel` /
`omp_get_thread_num` and fails with `Symbols not found`), which is why the `test`
task builds a binary and runs it; and the backend is Linux-only, so `pixi.toml`
lists `linux-64`/`linux-aarch64` only. The env depends on `mojo` alone — no
`max` package, no MAX thread pool, no `MODULAR_THREAD_BUSY_WAIT_US`.

From the repo root (needs the `.so` above):

```bash
PYTHONPATH=. .venv/bin/python tests/test_mojo_attn.py            # 45 checks, engine parity
PYTHONPATH=. .venv/bin/python benchmarks/attn_backend.py --mode kernel
PYTHONPATH=. .venv/bin/python benchmarks/kernel_ab.py            # drift-controlled A/B
```

`mojo build` targets the **host CPU** (`--print-effective-target` shows
`znver4 +avx512bf16` here), so rebuild per machine. `FASTENCDEC_MOJO_ATTN`
overrides the `.so` path; `FASTENCDEC_MOJO_WORKERS` overrides the worker count.

## 4. The kernel in one page

Contract (enforced by the engine): **exactly one query token per sequence**,
attending to every cached position. That makes the causal mask the cache tail.

- Work item = one `(sequence, head)` pair, dispatched with `GOMP_parallel` on
  **torch's own OpenMP pool**, `num_workers = min(torch threads, physical cores,
  tasks)`; one worker (inline, no region) when `num_tasks * pages <= 32`. Each
  task owns one head's cache slice end to end → no KV split, no scratch pad, no
  cross-thread reduction.
- KV is scored in **tiles** of `tile` positions (`32` when `block_size % 32 == 0`,
  else `16`) and `GROUP = 4` tiles per online-softmax step.
- A **page** holds `block_size / tile` tiles. The group loop resolves
  `(page, offset)` per tile into `tile_page[]`/`tile_off[]` and shares it between
  Q@K^T and P@V. **Walking pages instead of tiles is the bug from §11.2.**
- Page layouts (all inside one flat page of `block_size * head_dim` elements):
  - V, portable/x86: `[block_size, head_dim]` row-major.
  - V, ARM dot path: `[block_size/2, head_dim, 2]` — `(v[j][d], v[j+1][d])`
    adjacent, because `bfdot` contracts two adjacent values.
  - K: `[head_dim/2, block_size, 2]` (dot paths) or `[head_dim, block_size]`
    (fp32 fallback). One token is a column.
- Online softmax: `m' = max(m, max(group))`, `l = l·exp(m-m') + Σexp(group-m')`,
  `acc = acc·exp(m-m') + P@V`. Probabilities stay **in registers**; the C++
  kernel stores and reloads them (a 4-byte reload out of a 64-byte store = a
  store-forwarding stall per position).
- `scale` is folded into Q on the fp32 path; on the dot path Q stays bf16 and
  `scale` multiplies the logits (never round Q through a scaled multiply).
- **Strides:** `q`/`k`/`v` are chunks of the fused `qkv_proj` output, so the
  token stride is `3 * d_model`, not `num_heads * head_dim`. The kernel takes
  `q_stride`/`kv_stride`/`dst_stride` explicitly. Assuming contiguity silently
  gives every sequence sequence 0's output (§11.1).
- Supported: `head_dim` 32/64/128 (compile-time specialisations), fp32/bf16,
  `block_size` a multiple of 16. fp16 is rejected (no `avx512_fp16` on these
  boxes).

## 5. One source, two ISAs

```mojo
comptime X86_DOT  = CompilationTarget.is_x86() and CompilationTarget._has_feature["avx512bf16"]()
comptime NEON_DOT = CompilationTarget.has_neon() and CompilationTarget._has_feature["bf16"]()
comptime BF16_DOT = X86_DOT or NEON_DOT
comptime LANES = simd_width_of[DType.float32]()   # 16 AVX-512, 4 NEON
```

One dot instruction consumes `2 * LANES` bf16 (= two head dims of `LANES`
positions), so `tile / LANES` accumulators per page — 2 on x86, 8 on NEON for a
32-token page. The intrinsic is the only per-ISA code:

| ISA | intrinsic | signature |
|---|---|---|
| x86 | `llvm.x86.avx512bf16.dpbf16ps.512` | `<16 x f32>, <32 x bf16>, <32 x bf16>` |
| ARM | `llvm.aarch64.neon.bfdot` | `<4 x f32>, <8 x bf16>, <8 x bf16>` |

Both via `std.sys.intrinsics.llvm_intrinsic[..., has_side_effect=False]`.
**Without `has_side_effect=False` the calls are ordered and the phase is ~4×
slower.** The fp32 broadcast-FMA path stays as the fallback (AVX2, pre-BF16
ARM).

Verify both targets without the hardware (this is the cheap CI check):

```bash
cd mojo
pixi run mojo build --emit llvm src/fastencdec_mojo_attn.mojo -o /tmp/x86.ll
grep -c dpbf16ps /tmp/x86.ll                     # 73 on the current tree
pixi run mojo build --emit llvm --target-triple aarch64-unknown-linux-gnu \
    --target-cpu neoverse-v3 src/fastencdec_mojo_attn.mojo -o /tmp/arm.ll
grep -c neon.bfdot /tmp/arm.ll                   # 5665 on the current tree
```

The `--target-cpu` is **not optional**: the default aarch64 target has no
`+bf16`, so `_has_feature["bf16"]` is false and the build silently uses the
portable path while still compiling. Same trap on x86 (`--target-cpu
x86-64-v2` → fp32 path, `0` dots).

Checking a built `.so`: `objdump -d build/fastencdec_mojo_attn.so | grep -c
vdpbf16ps` (72 here) or `grep -c bfdot` (5664 on Graviton).

## 6. Mojo language notes (things that cost time here)

- `fn` is gone → `def`; `@parameter` → `@__parameter`, and **do not** put
  `@__parameter` on nested closures: use unified closures with capture lists
  (`{imm}`, `{mut x, imm}`, `{var x}`). A nested `def` with no capture list
  fails with "Could not infer capture convention".
- **No mutable globals.** `var X = ...` at module scope is an error. Module-level
  `comptime` constants are fine. This is what blocks a hand-rolled thread pool
  without a C shim (state can be smuggled as a Python-held `Int` address, §9.3).
- **`comptime` ternaries do not fold into comptime integers.**
  `comptime LANES = 16 if X86 else 4` stays a runtime expression and every
  `SIMD[.., LANES]` fails to match. `simd_width_of[DType.float32]()` folds.
- **`comptime if` arms are type-checked even when dead.** A 32-wide value cannot
  appear in a `tile`-generic function, so the dot kernel is a separate
  `_attention_head_dot` with `tile` fixed per instantiation, not a branch inside
  the generic kernel. (Module-level comptime booleans *are* fine as conditions
  when both arms typecheck for every instantiation.)
- Runtime index into a `SIMD` (`v[i]` with runtime `i`) spills the vector to the
  stack — use `comptime for i in range(N)` so indices are constants.
- `SIMD` comparison operators are **reducing** (`a < b` → `Bool`). For per-lane
  masks use `.lt/.le/.gt/.ge/.eq/.ne` (→ `SIMD[bool, N]`) and then
  `mask.select(true_vec, false_vec)`. `iota[DType.int32, N](offset)` builds an
  index vector.
- SIMD helpers used here: `.cast[DType]()` (widening bf16→f32), `.join` (concat),
  `.interleave` (`a0,b0,a1,b1,...` — how the broadcast pair operand is built),
  `SIMD[dtype, N](scalar)` (splat), `.reduce_max()`, `.reduce_add()`.
- Memory: `stack_allocation[count, dtype]()`; `Array[T, N](uninitialized=True)`
  (promoted to registers when every index is comptime);
  `ptr.unsafe_load[width=W](offset)` / `unsafe_store[width=W](offset, val)`
  (`width` defaults to 1; a value-only store overload exists).
  `Pointer` is non-nullable; build one with `unsafe_from_address=`, origin
  `MutUntrackedOrigin`; `Int(ptr)` gives the address.
- FFI: `external_call["symbol", RetType](args...)`; `@export def f() abi("C")`;
  a thin function value can be taken (`comptime entry = worker_entry`) and passed
  to C — that is enough for `pthread_create`/`pthread_join` and for the whole
  libgomp dispatch (`GOMP_parallel(entry, Int(state), Int32(n), Int32(0))`).
- **`fn` is a reserved word**: a field, parameter or `comptime` binding named
  `fn` is a parse error (`var fn: ...` → "expected name for 'var' declaration";
  `comptime fn = f` → "'comptime' cannot be used with 'fn'"). Name them
  `task_fn` / `entry`.
- A struct that holds a **thin function value** (`comptime TaskFn = def(Job, Int,
  Bool, Bool) thin -> None`) is assignable and callable through the field —
  including when the struct is reached through a raw pointer — but it must
  conform to `ImplicitlyCopyable` (as must the `Job` it carries), otherwise
  `state[] = Dispatch(...)` fails with "cannot be implicitly copied".
- **libgomp slices must use `omp_get_num_threads()`**, not the requested worker
  count: `GOMP_parallel` clamps the request to `OMP_NUM_THREADS`, which torch
  sets via `set_num_threads`, and a slice computed from the request leaves work
  unexecuted when the clamp bites. The region is synchronous, so the call
  state can live on the caller's stack.
- Python extension: `PythonModuleBuilder`, `def_function`, args are
  `PythonObject`, convert with `Int(py=...)` / `Float64(py=...)`. 23 int args
  cost 0.68 µs total, so marshalling is not worth micro-optimising.
  `std.runtime.initialize_runtime()` (MAX's runtime) is **gone**: nothing uses a
  Mojo thread pool now. A shared library must not define `main`.
- `mojo format src tests` exists and is expected to be clean; `mojo build
  --emit asm|llvm`, `--print-effective-target`, `--target-triple/--target-cpu`.
- Prefer `stack_allocation` or torch tensors over `List` for buffers a kernel
  reads through raw pointers; the tests do that.

## 7. Python-extension / host-side cost (measured on the Ryzen, b8/ctx256)

| item | µs |
|---|---|
| empty call, `num_seqs=0` (23 args + dispatch) | 0.68 |
| full Python wrapper (`mojo_attn._call`) | 2.8-3.6 |
| `torch.empty_like(query)` | 0.54 |
| 9 × `data_ptr()` | 0.51 |
| `_attention()` (extension callable, bound once) | 0.03 |
| `_workers()` (env override cached at import) | 0.16 |
| dtype lookup | 0.08 |
| **`GOMP_parallel`, 8 workers, warm** | **2.4** |
| **`GOMP_parallel`, 8 workers, 1 ms idle gap** | **2.7** |
| inline (1 worker, no region) | 0.36 |
| `max.algorithm.parallelize`, warm (removed) | 6.3 |
| same, 1 ms gap (removed) | 103-140 |
| attention call in-engine, ours / C++ backend (0.8 ms gap) | **29.8** / 90.7 |

Consequences: the dispatch is no longer the story, so `_workers()` inlines only
`num_tasks * pages <= 32` (24.5 vs 31.0 µs at b1/ctx64) — from b1/ctx128 up the
pool wins (1.3x at b1/ctx128, 8x at b8/ctx256). The cap at the **physical** core
count stays (8 workers on a 4-core Xeon: 1.5x slower at ctx 1024 and unstable at
batch 8), though torch now already sizes its pool that way, so it only bites when
a parent sets a larger `OMP_NUM_THREADS`. The old MAX-pool workarounds are gone
with the pool: no spin knob, no `initialize_runtime()`. Interleaved with GEMMs at
the engine's cadence the totals are now 88.8 µs per call (ours) vs 116.6 (C++),
i.e. attention 29.8-49.6 + interference +39.2 against 90.7 + 25.9.

## 8. Engine integration

- Backend name `mojo`, opt-in; `auto` still resolves to the vendored C++ kernel
  so all recorded baselines keep their meaning.
- **Public API mirrors flash-attn**: `flash_attn_with_kvcache(...)` and
  `flash_attn_func(...)` take flash's full argument list; anything the kernel
  cannot do (`qv`, `rotary_*`, `window_size`, `softcap`, `cache_leftpad`,
  `cu_seqlens_*`, `*_descale`, `return_softmax_lse`, `seqlen > 1`, dense
  multi-token) raises `NotImplementedError` rather than being ignored, and the
  tuning hints are no-ops.  The cache layout stays ours (head-major, K
  transposed in the page), not flash's page-major one, and `causal` has no
  effect with one query token.  The engine's per-layer path calls the thin
  aliases `self_attention`/`cross_attention` instead: the checked wrapper costs
  ~4% end to end at 24 calls/step (~2% with `check_inputs=False`), measured.
- `attention.uses_custom_ops(backend)` gates the frozen-compile path (the ops are
  `torch.library.custom_op` with `mutates_args={"k_cache","v_cache"}` so Dynamo
  never traces the kernel or the `Context` reads).
- `ModelRunner.compiled_backend` records which backend the graph was built with,
  so a later `attention.set_backend(...)` override falls back to eager.
- `attention.stage_cross(layers, request_ids, backend)` stages the paged encoder
  cache for whichever CPU kernel is active; the page layout is ours, so
  `stage_cross_kv` must stage through `mojo_attn.reshape_and_cache` (not the
  torch scatter) for the Mojo backend.

## 9. Performance: where the time goes

1. **x86 dot path** is worth 1.87× on the QK phase (33.8 vs 63.4 ns per 32-token
   page, isolated). P@V stays fp32 on x86 on purpose (bit-exactness).
2. **GROUP = 4** amortizes the softmax bookkeeping; per-page softmax cost was
   4× more max/exp/rescale work at short contexts. Measured neutral 2/4/8 on
   both servers afterwards.
3. **Per-call fixed cost dominated short contexts — solved by running on torch's
   pool.** With a private pool it was ~12.6 µs of a 75 µs call on the 4-core Xeon
   at b8/ctx256 and ~6 µs of a 33 µs call on the Ryzen, and every attempt to
   shrink it failed: MAX's `parallelize` 6.3 warm / 103-140 cold, a hand-rolled
   pthread pool 9.4 / 103, a 200 µs worker spin that cost 4.5% end to end.
   `GOMP_parallel` on the libgomp pool torch keeps busy between calls is 2.4 µs
   warm / 2.7 µs cold, and it removed the problem instead of shaving it: the
   in-engine call at b8/ctx256 went 180-196 → **29.8 µs** against the C++
   backend's 90.7-167, and 8x128 end to end 344 → **392-396 tok/s** (C++:
   364-377). The backend is now faster end to end on the laptop.
4. **Do not re-try**: a private thread pool in any form, including a spin
   (`GOMP_parallel` is the answer, §9.3) — and note `max.algorithm.parallelize`
   never had an OpenMP backend to select: it enqueues on AsyncRT's own
   `CPUDevice` work queue (`std::thread` + semaphore, `WorkQueueType` is only
   `kSingleThread`/`kThreadPool`), `libAsyncRTMojoBindings.so`/`libmax.so` contain
   no `GOMP_*`/`omp_*`/`__kmpc` symbols, and the only knob is
   `MODULAR_THREAD_BUSY_WAIT_US`; software prefetch of the next page's K/V (3 strides,
   neutral-to-worse — the pages are 4 KB sequential streams), `GROUP` 2/8,
   6/8 workers (`GROUP` beyond 4 registers-spills; >physical threads oversubscribes
   the fork-join), AMX.
5. **AMX is useless for this workload**: Q@K^T and P@V are M=1 per head for MHA
   (each head owns its own K *and* V page), and AMX tiles are 16 rows;
   `csrc`'s AMX path (`MaxQHeadNumPerIteration = 32`) only pays with GQA. On the
   Xeon the built `csrc` extension contains **0** AMX instructions and our
   wrapper cannot select `isa="amx"` anyway.
6. **Bandwidth is the wall on the servers**: Xeon 133 GB/s from L3 at
   b8/ctx256 (C++ runs 81 GB/s, compute-bound) and 62-70 GB/s DRAM at ctx 1024;
   Graviton 298 GB/s at 8.4 MB, 231 at 33.5 MB, 199 at 134 MB. Measured by
   pointing every sequence at one shared page set (1 MB unique vs 8.4 MB): the
   shared set is 1.6× faster at ctx 256 and 4× at ctx 1024.
7. **MAX's own CPU attention** (`max.kernels.nn.attention.cpu.mha`, measured with
   a script that went away with the `max` dependency) is the wrong yardstick: not paged, no block table, and its bf16 path is **5-9× slower than
   fp32** on AVX-512 (355 vs 39 µs at b8/ctx256) because AVX-512 has no native
   bf16 add/mul/fma — a bf16-native kernel scalarizes. Widening to fp32 and using
   bf16 only for the dot instruction is what makes this port fast.

## 10. Final three-box numbers (ratio `mojo/vllm`, <1 = faster)

`benchmarks/attn_backend.py --mode kernel` + `benchmarks/kernel_ab.py`, two runs
each, threads = physical cores, workers = `_workers()`. Measured with the old
private pool; the libgomp switch removes ~4 µs of dispatch per call (a few % of
the short-context rows, nothing at 1024), so treat these as slightly
conservative for the current tree:

| box (threads/workers) | self b8/256 | cross b8/256 | self b8/1024 | cross b8/1024 | self b32/1024 |
|---|---:|---:|---:|---:|---:|
| Ryzen 7840H (8/8) | 0.76-0.88 | 1.02-1.18 | 0.84-0.92 | 0.92-0.96 | 0.86 |
| Xeon 6975P-C (4/4) | 0.67-0.70 | 0.71-0.84 | 0.89-0.90 | 1.00-1.02 | 0.85-0.90 |
| Graviton 5 (8/8) | 0.43-0.48 | 0.43-0.51 | 0.60-0.79 | 0.65-0.77 | 0.86-0.95 |

Absolute µs at `cross b8/enc256`: Xeon 116.7/114.4 → 82.5/85.5; Graviton
65.9/65.6 → 33.6/33.3; Ryzen 29.6/32.5 → 32.8/38.3 (parity). The laptop is
*faster in absolute terms* than the Xeon because it has 8 full-speed cores and
32 MB L3 vs a 4-core cloud slice; the repo already records that ordering for
decode GEMMs. What matters is the ratio on the same box.

Config findings worth remembering: on the Xeon `block_size=128` is ~9% faster at
ctx 1024 (a `Config`/CLI knob, not a kernel change); on Graviton the aarch64
default of 16 is also the C++ kernel's best point (67 vs 76 µs at block 32).

## 11. Failure modes we hit (each has a regression test now)

1. **Token strides.** `q/k/v` come from `torch.chunk(3, dim=-1)` + `reshape`, so
   `stride(0) == 3 * d_model` and the tensor is *not* contiguous. The first
   in-engine run gave every sequence sequence 0's output. Passing strides
   explicitly fixed it; `tests/test_mojo_attn.py` covers it via the engine.
2. **Page-vs-tile walk.** The group loop advanced by whole pages while scoring
   only `tile` positions of each page: correct only when `block_size == tile`.
   For `block_size ∈ {48,64,128,256}` (accepted by `supported()`) it scored the
   first 32 positions of every page and skipped the rest — silent garbage
   (2046/2048 elements wrong at block 64). The engine test missed it because the
   tiny random model decodes the same token everywhere and the logits check only
   ran at `lens == 1`. `tests/test_attention.mojo` now covers block 48/64 with
   `lens` 190-200 (multi-tile pages, groups straddling pages).
3. **The K-page/V-page layout must match the selected path.** Store and load both
   read the same `comptime` predicate (`BF16_DOT`, `NEON_DOT`, `dtype == bf16`,
   `tile % LANES == 0`). A mismatch shows up as bit-exact-but-wrong results for
   one dtype or block size only.
4. **bf16 probabilities on ARM** (only place parity with `csrc` is given up):
   ≤1e-3 vs the fp32 reference, matching `csrc`'s ARM `prob_buffer_t`. x86 stays
   bit-identical (`max|diff| 0.0`).
5. **Engine tests can be blind.** `tests/test_mojo_attn.py`'s token comparison is
   degenerate on the tiny model; the standalone test against a scalar fp32
   reference is the real gate. Keep both.

## 12. Box access and environment

| box | IP | notes |
|---|---|---|
| Ryzen 7840H (dev) | local | 8 physical/16 SMT, avx512_bf16, 32 MB L3; repo `.venv` or `~/.venvs/torch-cpu` |
| Xeon 6975P-C | `ubuntu@13.210.181.255` | 8 vCPU = **4 physical cores**, `avx512_bf16` + `amx_bf16`; `~/venv`; torch default 4 threads |
| Graviton 5 | `ubuntu@13.236.70.191` | 8 cores, ASIMD + `bf16`; `~/venv`; torch default 8 threads |

- Sync with `tar czf - --exclude=.pixi --exclude=.venv --exclude='*.so'
  fastencdec mojo/src mojo/tests mojo/pixi.toml tests benchmarks csrc | ssh box
  'tar xzf - -C ~/fast-encdec'`. **Never `rm -rf ~/fast-encdec` on the servers** —
  their `~/venv` is a symlink into `~/fast-encdec/.venv` and `mojo/.pixi` holds
  the 1.4 GB Mojo/MAX env.
- Both servers need `pixi` (`~/.pixi/bin/pixi`), `gcc/g++/ninja` (for the `csrc`
  JIT), and `pixi run build` after every sync. `mojo/pixi.toml` lists
  `linux-64`, `linux-aarch64` only (libgomp dispatch) and the env is `mojo`
  alone, ~1.2 GB. `pixi run test` needs no `gcc`: it links the conda-forge
  libgomp that ships with the env, and the `.so` records an RPATH into it.
- `benchmarks/kernel_ab.py --threads N --workers N` pins the knobs; `0` means
  "torch default / the kernel's physical-core rule". These boxes drift ±10% and
  Graviton's 1024-token rows ±25%, so always run it twice and report the band.

## 13. Ideas not tried (in rough priority order)

1. **bf16 P@V on x86** (probabilities quantized like MAX/`csrc`-AMX): halves the
   P@V multiply count, ~30% at short contexts; costs bit-parity with `csrc/`.
   Only worth it if the engine, not the kernel table, is the target.
2. **Chunked prefill / multi-token queries**: needs the mask logic generalised
   (the port assumes one query token at the last position). MAX's
   `MaskStrategy.start_column` / `total_iters` is the structure to copy.
3. **GQA/MQA**: one KV page shared by `q_heads_per_kv` heads would let one task
   amortise the K/V loads over several query heads — and only then does AMX
   (16-row tiles) become interesting. Not needed for BART/Florence-2.
4. ~~Own thread pool~~ **Done, differently**: not a pool of our own (no win,
   §9.3) but `GOMP_parallel` on torch's pool. A hand-rolled pool cannot beat it
   because the win is *reusing* the pool that is already awake.
5. **`partial_simd_load`** for tails instead of `iota`+`select` (cosmetic).
6. **T5** (root `AGENTS.md`): same paged path, but its per-layer pointwise chains
   are what the frozen compile graphs fuse, so the CPU gains would be larger.
