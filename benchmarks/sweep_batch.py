"""Batch sweep: pipelined vs synchronous greedy decode (ours), CUDA.

Shows how the fixed per-step host round-trip fades as the batch grows, so the
pipelined path matters at small batch and stops mattering when the GPU dominates.
"""

import sys, time, torch

sys.path.insert(0, "benchmarks")
sys.path.insert(0, ".")
import benchmark_vllm as bv
from fastencdec import LLM, SamplingParams
from fastencdec.config import default_block_size
from fastencdec.engine.llm_engine import LLMEngine
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(bv.MODEL)
prompt = bv.make_prompt(tok, 256)
enc = len(tok(prompt, add_special_tokens=False)["input_ids"])
OUT = int(sys.argv[1]) if len(sys.argv) > 1 else 128
block_size = default_block_size("cuda")

print(f"ours, 256->{OUT}, greedy bf16, no ngram")
print(
    f"{'batch':>6}{'sync ms':>10}{'pipe ms':>10}{'sync tok/s':>12}{'pipe tok/s':>12}{'speedup':>9}"
)
for batch in (8, 32, 64, 128):
    num_blocks = bv.blocks_for([{"batch": batch, "beams": 1}], enc, OUT, block_size)
    llm = LLM(
        bv.MODEL,
        num_blocks=num_blocks + 32,
        block_size=block_size,
        max_num_seqs=batch,
        dtype=torch.bfloat16,
        device="cuda",
    )
    params = SamplingParams(
        max_tokens=OUT, num_beams=1, min_length=OUT + 1, eos_token_id=llm.eos_token_id
    )

    def gen():
        llm.generate([prompt] * batch, params, use_tqdm=False)

    def timed(warm=1, rep=3):
        for _ in range(warm):
            gen()
        best = float("inf")
        for _ in range(rep):
            torch.cuda.synchronize()
            s = time.perf_counter()
            gen()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - s)
        return best * 1e3

    pipe = timed()
    orig = LLMEngine._pipeline_mode
    LLMEngine._pipeline_mode = lambda self: None
    sync = timed()
    LLMEngine._pipeline_mode = orig
    n = batch * OUT
    print(
        f"{batch:>6}{sync:>10.1f}{pipe:>10.1f}{n / (sync / 1e3):>12.0f}"
        f"{n / (pipe / 1e3):>12.0f}{sync / pipe:>8.2f}x"
    )
    del llm
