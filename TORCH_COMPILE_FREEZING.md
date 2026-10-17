# Why `torch.compile` + graph freezing does (does not) produce `mkldnn._linear_pointwise` weight prepacking

Findings, 2026-10-17. Environment: dev box (AMD Ryzen 7 7840H, Zen 4, `avx512_bf16`,
no AMX), Python 3.11, **torch 2.14.0+cpu**, fast-encdec @ `1092eb5`.

Answers for the three questions that motivated this doc:

1. **Does `torch.compile` + graph freezing emit `torch.ops.mkldnn._linear_pointwise`
   with prepacked weights?** On torch 2.14, **yes** — for bf16, for bf16 under
   `torch.amp.autocast`, and even with dynamic batch sizes. It also does it on
   *this engine*, graph breaks and all: the frozen decoder emits and executes
   **48 packed GEMMs per decoder forward** (runtime-verified). The prior note in
   AGENTS.md §7 ("[the recipe] does work, but only on models that are pure
   torch", "It does not pay off on this engine") was right about the outcome
   and wrong about the mechanism: the pack fires on this engine, graph breaks
   and all — what loses is the per-step Python overhead of a ~70-break
   shattered graph.
2. **Where does "no `_linear_pointwise`" actually come from?** Five concrete,
   silent traps (§3). The two that apply to us: freezing is *off by default* and
   in torch 2.14 the weight-pack pass exists **only inside the freezing
   pipeline** — `config.cpp.weight_prepack=True` alone does nothing; and
   compiling a *free function* (the `enable_compile` helpers, the "linear+gelu
   post-op fusion" plumbing) can never pack because the weights arrive as
   **user arguments**, not frozen constants.
3. **Do we need `torch.amp()`?** **No.** Autocast is *sufficient* but not
   *necessary*. The pass keys off (weight = frozen constant, dtype bf16/fp16, or
   fp32 with `mkldnn.matmul.fp32_precision` fast-math) — not off autocast.
   A module already cast to bf16 packs identically (verified).

---

## 1. The gate chain in torch 2.14 source

Weight prepacking on CPU is implemented as *freezing* FX patterns. This is why
the 2024 blogs' excerpt (`torch/_inductor/mkldnn.py:mkldnn_fuse_fx` /
`pack_module`) is stale: that entry point no longer exists in 2.14
(`grep -r mkldnn_fuse_fx torch/_inductor/` → 0 hits).

Decision tree, all paths pinned to the installed source
(`torch/_inductor/`, line numbers from the installed 2.14 wheel):

```
torch.compile(model)  ──  first forward call is the trigger
  │
  ├─ config.freezing == True AND torch.is_grad_enabled() == False   [*]
  │      → compile_fx.py picks fw_compiler_freezing
  │      → freezing._freeze() replaces every non-mutated parameter
  │        placeholder with a graph constant (`_frozen_paramN`)
  │      → fx_passes.freezing_patterns.freezing_passes() runs
  │            └─ lazy_init() registers _mkldnn_weight_pack_init()   [**]
  │                  └─ pattern aten.addmm / aten.mm (pass_number=1)
  │                        extra_check = _is_packable_linear         [***]
  │                        →  pack_linear_weight  (mkldnn._reorder_linear_weight,
  │                                             constant-folded offline during
  │                                             compile → another _frozen_param)
  │                        →  pack_linear (op choice, §2)
  │
  └─ otherwise: plain fw_compiler_base(is_inference=True)
         → the weight-pack pass is NEVER REGISTERED
         → all linears stay `extern_kernels.addmm` (verified on disk)
```

- `[*] compile_fx.py:3181`
  `if config.freezing and not torch.is_grad_enabled():` — grad mode is sampled
  **at trace time** (inside the first call). Note that the choice between
  `fw_compiler` (grad) and `inference_compiler` is made here; AOTAutograd later
  picks the inference compiler when the graph doesn't need grad.
- `[**] fx_passes/freezing_patterns.py:91-101` —
  `lazy_init()` registers `_mkldnn_weight_pack_init()` **only** if
  `torch._C._has_mkldnn and config.cpp.weight_prepack`; these patterns are
  registered into `pass_patterns[]` which is applied *only* from
  `freezing_passes()` (line 37-89), i.e. only inside `freezing._freeze()`
  (`freezing.py:74-144`).
- `[***] fx_passes/mkldnn_fusion.py:1288-1370` — `_is_packable_linear`:
  - weight arg must be a **graph constant** (`get_attr`) or a `cat` of constants
    (`is_const_or_cat_by_const`, lines 1293-1298) — the frozen params satisfy
    this, *user-supplied tensors do not*;
  - both args 2-D, on CPU, not fp64;
  - fp32 requires MKL (x86) or ACL (aarch64), lines 1332-1339;
  - bf16/fp16 require oneDNN's low-precision support
    (`is_mkldnn_bf16_supported`, `utils.py:4289`).

Defaults (torch 2.14, `_inductor/config.py`): `cpp.weight_prepack` is **already
True** (line 1822); `freezing` is **False** (line 1672) — so of the pair in the
classic recipe, the relay that actually matters is `freezing`.

## 2. Which packed op you get, by dtype

From `mkldnn_fusion.py:109-159` (`pack_linear_weight` / `pack_linear`):

| weight dtype | condition | packed op |
|---|---|---|
| bf16 / fp16 | always | `torch.ops.mkldnn._linear_pointwise` |
| fp32 | `torch._C.has_mkl` and no fast-math | `torch.ops.mkl._mkl_linear` |
| fp32 | `torch.backends.mkldnn.matmul.fp32_precision ∈ {bf16, tf32}`* | `torch.ops.mkldnn._linear_pointwise` |
| fp32 on aarch64 | ACL supported | `torch.ops.mkldnn._linear_pointwise` |
| dynamic batch (`has_free_symbols`) | linear kernels only | force the mkldnn path (`reorder` with `optimization_hint(batch)`) |

\* this is exactly the setting the AWS Graviton blog imports via
`DNNL_DEFAULT_FPMATH_MODE=BF16`; setting
`torch.backends.mkldnn.matmul.fp32_precision = "bf16"` switched our fp32 toy's
packed op from `mkl._mkl_linear` to `mkldnn._linear_pointwise` (verified by
grepping the emitted code).

So expecting `mkldnn._linear_pointwise` from an **fp32** recipe without
fast-math/autocast is itself a mistaken expectation — you get `mkl._mkl_linear`
(prepacking still happens, just under the MKL namespace).

## 3. The traps that silently yield *no* `_linear_pointwise`

Each one exercised on this box (see §4 evidence column):

1. **`config.freezing` not set** (torch 2.14 default False, `config.py:1672`).
   The pack pass lives only in the freeze pipeline; `weight_prepack=True` alone
   does nothing, and the GEMMs compile to plain `extern_kernels.addmm`.
2. **The first (compiling) call made with grad enabled** — `freezing` silently
   degrades to the non-freezing compiler per the `[*]` gate. In our test the
   *call made under `no_grad`* afterwards recompiles with guards on grad mode
   and packs then, so you pay for two compiles instead of one (24-pack before
   and 24 after; useful first-call lesson, not a correctness trap).
3. **`torch.compile` over a free function** (weights are plain *user args*,
   e.g. our `models/bart.py:enable_compile` helpers `_add_layer_norm(x, …, w, b)`
   or the "fuse linear+gelu" helper idea): `params_flat` is empty, weights are
   never frozen constants, `is_const_or_cat_by_const` fails → **zero packing**.
   This is inherent, not fixable by flags; the model's own forward must be the
   compiled entry point so AOT sees module parameters.
4. **Manual prepacking (`layers/packed_linear.py`) replaces `nn.Linear.forward`
   with a closure wrapped in `@torch.compiler.disable`.** Those GEMMs are
   invisible to Dynamo by construction — Inductor cannot pack weights it never
   sees, and the frozen pipeline cannot fuse around a disabled region. The two
   routes are mutually exclusive in the current wiring (they need not be: see
   §6 — and note that on torch 2.14 Inductor *can* consume the opaque blocked
   weight of an *explicit* `mkldnn._linear_pointwise` call: the "fails on
   `stride=[1, 0]`" observation in the old Florence-2 note did not reproduce on
   a chained-packed-forward toy compiled today; worth rechecking next time
   `compile_mm_encoder` is touched).
5. **dtype mismatch of expectations** (§2): fp32 without fast-math packs into
   `mkl._mkl_linear`; scanning `prof` for `mkldnn._linear_pointwise` in an fp32
   run finds none even though packing all happened.

Not a trap (we back this now with the engine run): **graph breaks don't stop
packing.** Each broken subgraph is frozen and packed independently.

## 4. Empirical evidence

### 4.1 Toy module — 12×[Linear(1024→4096), GELU, Linear(4096→1024)], batch 8

Metric: `torch._dynamo.utils.counters["inductor"]
["mkldnn_linear_weight_pack_matcher_count"]` — one per `aten.mm`/`aten.addmm`
node rewritten to `reorder_linear_weight (+ bias) + pointwise linear`.
24 GEMMs = 24/24 packed. Timings wall-clock per call, CPU.

| variant | packed | op in code | compiled ms | eager ms |
|---|---|---|---:|---:|
| bf16 (weights pre-cast, **no autocast**), `no_grad`, freezing | 24 | `_linear_pointwise` | 4.53 | 5.99 |
| bf16 but first call grad-enabled | 24 (2nd compile) | same | 4.92 | 6.40 |
| fp32, `no_grad`, freezing | 24 | `mkl._mkl_linear` | 17.80 | 21.60 |
| fp32 **+ `torch.amp.autocast("cpu", bfloat16)`, traced under amp** | 24 | `_linear_pointwise` | 4.56 | 38.18 (with autocast) |
| fp32 + `mkldnn.matmul.fp32_precision="bf16"` | 24 | `_linear_pointwise` | — | — |
| bf16 + `dynamic=True`, batch 8/1/3 | 24 | `_linear_pointwise` | 4.48 | 6.04 |
| bf16 + opaque-packaged weights + explicit `_linear_pointwise`, **no** `compiler.disable` | n/a | `_linear_pointwise` extern (60× in code) | 4.58 | 4.53 |
| `torch.compile` over a free function (weights = args) | **0** | plain `mm`/`addmm` externs | — | — |

Blogs' environment check: the same recipe was measured on Graviton (ACL) and on
x86 with `DNNL_DEFAULT_FPMATH_MODE` (cf. §2); those posts show the packed op
`mkldnn::_linear_pointwise` in their torch.profiler / code tables.

### 4.2 The engine — `bart-large-cnn`, bf16, batch 8, 256→256 greedy,
manual prepack stubbed out (`packed_linear.apply ≡ 0`), `Config.compile_model=False`,
then `torch.compile(engine.runner.decoder, dynamic=False)`,
`torch._inductor.config.freezing = True`, decode runs under the engine's own
`torch.no_grad()` (`engine/model_runner.py:151`):

- compile-time: `mkldnn_linear_weight_pack_matcher_count = 48` — the 5 GEMMs per
  decoder layer × 12 (qkv, self-out, cross-q, fc1, fc2).
- emitted code (TORCH_COMPILE_DEBUG dump): 47 output modules; the packed lines
  look like
  `buf0 = torch.ops.mkldnn._linear_pointwise.default(arg2_1, _frozen_param3, _frozen_param1, 'none', [-1], '')`
  — the `_reorder_linear_weight` call was constant-folded at compile time into
  a new `_frozen_param`, i.e. *the same load-time reorder our `pack_weights`
  performs, done per compile instead of once at load*.
- runtime (torch.profiler, 24-token generate, 8 seqs):
  `mkldnn::_linear_pointwise count=1152` (= 48 × 24 decode steps), while the
  same generate *without* freezing had `aten::addmm 1788` (the frozen run drops
  it to 636: the remainder is the encoder + LM head + per-step staging GEMMs
  which live outside the compiled decoder).
- wall time (256→256, b8, best of 3): **compiled 7006 ms vs eager 5919 ms
  (0.84×)** — the *same* breakdown as the AGENTS.md table (7490 vs 6700 there),
  even though packing demonstrably happened. Compile time itself is tiny
  (~0.1 s per subgraph, a few s total), so the regression is entirely in the
  hot path. (After the §6.1 capability-hoist fix the same experiment reads
  6467 vs 5860 ms = 0.91×, with the break census below.)

Where the wall time goes (24-token generate profile; the break census is from
dynamo's compile report):

| overhead signal | count/generate |
|---|---:|
| graph breaks per decoder-forward trace (compile-time census) | ~70 pre-fix → ~53 post-fix §6.1 |
| `TorchDynamo Cache Lookup` (guard checks) | 8040 |
| `Pregraph bytecode` | 2640 |
| `AOTDispatcher Runtime Wrapper Prologue` | 2640 |
| `Torch-Compiled Region` dispatches | dozens per step (~12/layer) |
| `aten::clone` / `aten::contiguous` between subgraphs | 432 / 420 |

Of the ~70 breaks/forward, **52 are ours to remove today**: 39 from
`torch.cpu._is_avx512_supported` and 13 from `torch.cpu._init_amx` — both are
re-checked inside our `layers/cpu_attn.py` `_x86_kernel_ok()` / `_amx_usable()`,
which `select_isa()` reaches from the op-dispatch path on *every* attention
call (`layers/attention.py:42,168,284`). They are torch CPU-API calls resolved
per process, so caching the answers as plain bools on the module removes the
breaks (the existing `lru_cache` on `_amx_usable` only protects the compiler
probe subprocess, not the torch checks; and an `lru_cache` would not help under
Dynamo anyway — it re-fires per fake-tensor frame).
The remaining 17 (9 `attention_forward`, 8 `reshape_and_cache`) are the pybind
kernels that need `torch.library.custom_op` + `register_fake`, as AGENTS.md
§7 already diagnosed.

**Conclusion for the engine:** the recipe works; the engine's *cost structure*
(per-step Python dispatch across a shattered graph) is what loses to eager,
since our `pack_weights` reaches the identical oneDNN blocked-weight op with
zero compile machinery. Freezing reaches exactly the same packed weights — it
is not a different optimization we are missing.

## 5. Do you need `torch.amp`?

Short answer: **no**.

- The packing condition depends on frozen constants + GEMM dtype — not on
  autocast. bf16 module + `no_grad` + `freezing=True` packs 24/24 (§4.1 row 1).
- `torch.amp.autocast("cpu", torch.bfloat16)` is the recommended recipe in the
  blogs because *their models are fp32 modules* wanting bf16 GEMMs. Under
  compile, autocast's `weight` casts get constant-folded during freezing, so an
  fp32+autocast run packs too (§4.1 row 4). If you instead cast the module (as
  our engine does), you need no autocast at all.
- For fp32 kernels there is one autocast-*adjacent* switch worth knowing:
  `torch.backends.mkldnn.matmul.fp32_precision = "bf16"` (or
  `DNNL_DEFAULT_FPMATH_MODE=BF16` from the Graviton post) — it moves fp32 GEMMs
  (and their packed form) from `mkl._mkl_linear` to the oneDNN
  `_linear_pointwise` bf16-accelerated path (§2). It trades exact fp32 for
  reduced-precision accumulation — the engine's exact-fp32 parity (the fp32
  rows of `benchmarks/verify_parity.py`) would no longer hold.

## 6. What to actually do (if/when revisiting compile on this engine)

1. ~~**Freeze the capability checks.**~~ **Done** (same-day): `cpu_attn.py`
   resolves AVX-512/AMX capability once into module globals (`_resolve_isa`)
   and caches the built extension in a plain global (`_BUILT`) instead of
   `lru_cache` — Dynamo ignores cache wrappers and was tracing `_build()` per
   call site (41 `posix.stat` breaks from `os.path.exists`). Post-fix break
   census on the engine: **53, all pybind** (`reshape_and_cache` ×40,
   `attention_forward` ×9, `build_metadata` ×4); compile+freezing now measures
   **6467 ms vs eager 5860 (0.91x, was 0.84x)** with packing unchanged (48),
   and the eager default is marginally faster too (the baseline table in
   AGENTS.md was refreshed). Tests pass (`test_parity`, `test_attn_backend`,
   `test_beam`, `test_packed_linear`, `test_florence2`).
2. **Verdict on replacing the manual prepack with compile+freezing: no.**
   With packing proven equal (§4.2), the compile path still loses at
   `256->256, batch=8` (6467 ms vs the fresh eager baseline 5061 ms with manual
   prepack — 28% behind) because the remaining pybind-call sites keep the graph
   shattered and every decode step pays the subgraph dispatch. The manual
   `packed_linear` backend stays until the attention ops are `torch.library.custom_op`s
   *and* the per-step dispatch overhead shrinks; only then is
   removing `pack_weights` worth re-measuring.
3. Never expect packing from compiled *helpers* (trap §3.3); compile the
   module's own forward.

## 7. Repro

Minimal toy check (all signals in-process; ~10 s):

```python
import torch

torch._inductor.config.freezing = True  # cpp.weight_prepack is already True
from torch._dynamo.utils import counters

model = torch.nn.Sequential(
    *[
        m
        for _ in range(12)
        for m in (
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        )
    ]
).to(torch.bfloat16)  # no autocast needed
model = torch.compile(model)

with torch.no_grad():  # must wrap the FIRST call
    model(torch.randn(8, 1024, dtype=torch.bfloat16))

print(
    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"]
)  # 24 = every GEMM
# to see which packed op was emitted (per §2):
#   $ TORCH_LOGS=+output_code python this_script.py 2>&1 | grep -E "_linear_pointwise|_mkl_linear"
```

Full experiment scripts (toy grid, engine timing, engine runtime profile) are
kept in `/tmp/opencode` (`prepack_recipes.py`, `engine_compile.py`,
`engine_ops.py`); expected results in §4.

- Counters meanings: `mkldnn_linear_weight_pack_matcher_count` = one per
  `aten.mm`/`aten.addmm` node rewritten into
  `mkldnn._reorder_linear_weight` (+ bias) + `mkldnn._linear_pointwise`
  / `mkl._mkl_linear`.
- Emitted code on disk: `TORCH_COMPILE_DEBUG=1` writes
  `torch_compile_debug/run_*/torchinductor/model__*/output_code.py`
  (grep `_linear_pointwise`, `_frozen_param`, `extern_kernels.addmm`).
- Runtime op counts: `torch.profiler` — `mkldnn::_linear_pointwise` vs
  `aten::addmm`/`aten::linear`.


## 8. References

- PyTorch blog, *Accelerated CPU Inference with PyTorch Inductor* — the
  `weight_prepack`/oneDNN `_linear_pointwise` design and (at the time)
  `mkldnn_fuse_fx`: https://pytorch.org/blog/accelerated-cpu-inference/
- Intel version of the same post: https://www.intel.com/content/www/us/en/developer/articles/technical/accelerated-cpu-inference-with-pytorch-inductor.html
- PyTorch blog, *Accelerated PyTorch inference … on AWS Graviton* — recipe
  `config.cpp.weight_prepack=True; config.freezing=True` under
  `torch.no_grad()`, T5 profile showing `mkldnn::_linear_pointwise`, and
  `DNNL_DEFAULT_FPMATH_MODE=BF16`:
  https://pytorch.org/blog/accelerated-pytorch-inference/
- PyTorch tutorial, *Using Max-Autotune Compilation on CPU* — "only support
  frozen model with `torch.no_grad` or the inference mode … set
  `TORCHINDUCTOR_FREEZING=1` and ensure both compile and inference steps are
  executed within `torch.no_grad`", example wrapped in
  `torch.no_grad(), torch.cpu.amp.autocast(bf16)`:
  https://docs.pytorch.org/tutorials/unstable/max_autotune_on_CPU_tutorial.html
- Source (torch 2.14 layout, same files in the installed wheel):
  - `torch/_inductor/config.py` (1672: `freezing` default False; 1822:
    `cpp.weight_prepack` default True)
  - `torch/_inductor/compile_fx.py:3181` — the `freezing and not grad_enabled`
    gate; `:2308` `fw_compiler_freezing`
  - `torch/_inductor/freezing.py:74-144` — `freeze`/`_freeze`, params→constants
  - `torch/_inductor/fx_passes/freezing_patterns.py:37-101` — the pass
    registration home of weight packing
  - `torch/_inductor/fx_passes/mkldnn_fusion.py` (109-159 op choice; 1288-1370
    `_is_packable_linear`; 1510-1574 the `addmm`/`mm` patterns)
  - `torch/_dynamo/config.py:121` — `recompile_limit = 8`
