import torch

_NEG_INF = float("-inf")


def _repetition_penalty(scores: torch.Tensor, input_ids: torch.Tensor, penalty: float):
    safe = input_ids.clamp_min(0)
    present = torch.gather(scores, 1, safe)
    fixed = torch.where(present < 0, present * penalty, present / penalty)
    scores.scatter_(1, safe, torch.where(input_ids >= 0, fixed, present))


def _no_repeat_ngram(scores: torch.Tensor, input_ids: torch.Tensor, n: int):
    cur_len = input_ids.shape[1]
    if cur_len < n:
        return
    grams = input_ids.unfold(1, n, 1)
    match = (grams[..., :-1] == input_ids[:, cur_len - n + 1 :].unsqueeze(1)).all(-1)
    match = match & (grams[..., -1] >= 0)
    banned = torch.zeros_like(scores).scatter_reduce(
        1, grams[..., -1].clamp_min(0), match.float(), reduce="amax", include_self=True
    )
    scores.masked_fill_(banned > 0, _NEG_INF)


class LogitsProcessor:
    def __init__(
        self,
        params,
        *,
        vocab_size: int,
        device,
        max_length: int,
        begin_index: int,
        is_beam: bool,
    ):
        self.params = params
        self.max_length = max_length
        self.begin_index = begin_index
        self.is_beam = is_beam
        self.eos_ids = sorted(params.eos_ids())
        self.masks = {
            name: torch.zeros(vocab_size, dtype=torch.bool, device=device)
            for name in ("eos", "suppress", "begin_suppress", "bos_only", "eos_only")
        }
        self.masks["eos"][self.eos_ids] = True
        if params.suppress_tokens:
            self.masks["suppress"][params.suppress_tokens] = True
        if params.begin_suppress_tokens:
            self.masks["begin_suppress"][params.begin_suppress_tokens] = True
        if params.forced_bos_token_id is not None:
            self.masks["bos_only"][params.forced_bos_token_id] = True
        if params.forced_eos_token_id is not None:
            self.masks["eos_only"][params.forced_eos_token_id] = True

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, lengths=None
    ) -> torch.Tensor:
        # lengths: true per-row history lengths for left-padded (-1) input_ids,
        # as a CPU tensor; None means dense (every row fills the width).
        p = self.params
        masks = self.masks
        if p.repetition_penalty != 1.0:
            _repetition_penalty(scores, input_ids, p.repetition_penalty)
        if p.no_repeat_ngram_size > 0:
            _no_repeat_ngram(scores, input_ids, p.no_repeat_ngram_size)
        if lengths is None:
            cur = input_ids.shape[1]
            if p.min_length > 0 and self.eos_ids and cur < p.min_length:
                scores.masked_fill_(masks["eos"], _NEG_INF)
            if p.forced_bos_token_id is not None and cur == 1:
                scores.copy_(torch.where(masks["bos_only"], 0.0, _NEG_INF))
            if p.forced_eos_token_id is not None and cur == self.max_length - 1:
                scores.copy_(torch.where(masks["eos_only"], 0.0, _NEG_INF))
            if p.suppress_tokens:
                scores.masked_fill_(masks["suppress"], _NEG_INF)
            if p.begin_suppress_tokens and cur == self.begin_index:
                scores.masked_fill_(masks["begin_suppress"], _NEG_INF)
        else:
            dev = scores.device
            if p.min_length > 0 and self.eos_ids:
                sel = lengths < p.min_length
                if sel.any():
                    scores.masked_fill_(sel.to(dev)[:, None] & masks["eos"], _NEG_INF)
            if p.forced_bos_token_id is not None:
                sel = lengths == 1
                if sel.any():
                    fill = torch.where(masks["bos_only"], 0.0, _NEG_INF)
                    scores = torch.where(sel.to(dev)[:, None], fill, scores)
            if p.forced_eos_token_id is not None:
                sel = lengths == self.max_length - 1
                if sel.any():
                    fill = torch.where(masks["eos_only"], 0.0, _NEG_INF)
                    scores = torch.where(sel.to(dev)[:, None], fill, scores)
            if p.suppress_tokens:
                scores.masked_fill_(masks["suppress"], _NEG_INF)
            if p.begin_suppress_tokens:
                sel = lengths == self.begin_index
                if sel.any():
                    scores.masked_fill_(
                        sel.to(dev)[:, None] & masks["begin_suppress"], _NEG_INF
                    )

        if p.temperature > 0:
            min_tokens = len(self.eos_ids) + 1 if self.is_beam else 1
            if p.temperature != 1.0:
                scores = scores / p.temperature
            if p.top_k > 0:
                top_k = min(p.top_k, scores.shape[-1])
                keep = torch.topk(scores, top_k)[0][..., -1, None]
                scores = scores.masked_fill(scores < keep, _NEG_INF)
            if p.top_p < 1.0:
                order = torch.sort(scores, descending=False)
                drop = order.values.softmax(dim=-1).cumsum(dim=-1) <= (1 - p.top_p)
                drop[..., -min_tokens:] = False
                scores = scores.masked_fill(
                    drop.scatter(1, order.indices, drop), _NEG_INF
                )
        return scores
