"""Check the pipelined greedy path against the synchronous loop (CUDA).

Uses the benchmark's greedy decoding config (``no_repeat_ngram_size=3``), which
the Triton n-gram processor now keeps on the device.
"""

import sys
import time

import torch

sys.path.insert(0, "benchmarks")
sys.path.insert(0, ".")
import benchmark_vllm as bv
from fastencdec import LLM, SamplingParams
from fastencdec.engine.llm_engine import LLMEngine
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(bv.MODEL)
prompt = bv.make_prompt(tok, 256)
llm = LLM(
    bv.MODEL,
    num_blocks=1024,
    block_size=256,
    max_num_seqs=8,
    dtype=torch.bfloat16,
    device="cuda",
)

OUT = int(sys.argv[1]) if len(sys.argv) > 1 else 256
params = SamplingParams(
    max_tokens=OUT,
    num_beams=1,
    min_length=OUT + 1,
    eos_token_id=llm.eos_token_id,
    **bv.DECODE,
)


def timed(fn, warm=1, rep=3):
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(rep):
        torch.cuda.synchronize()
        s = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, r


def gen():
    out = llm.generate([prompt] * 8, params, use_tqdm=False)
    return [o.outputs[0].token_ids for o in out]


orig = LLMEngine._batch_can_pipeline
pipe_ms, pipe_tokens = timed(gen)
LLMEngine._batch_can_pipeline = lambda self: False  # force the synchronous loop
sync_ms, sync_tokens = timed(gen)
LLMEngine._batch_can_pipeline = orig

n = 8 * OUT
print(f"256->{OUT}  batch=8  bf16  greedy+ngram=3")
print(f"  pipelined   {pipe_ms:8.1f} ms   {n / (pipe_ms / 1e3):7.1f} tok/s")
print(
    f"  synchronous {sync_ms:8.1f} ms   {n / (sync_ms / 1e3):7.1f} tok/s"
    f"   speedup {sync_ms / pipe_ms:.2f}x"
)
print("  token parity:", "exact" if pipe_tokens == sync_tokens else "MISMATCH")
if pipe_tokens != sync_tokens:
    for i, (a, b) in enumerate(zip(sync_tokens, pipe_tokens)):
        if a != b:
            d = next(
                (j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
            )
            print(f"  request {i}: first diff at {d}")
            break
