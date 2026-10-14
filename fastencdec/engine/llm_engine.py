from collections import defaultdict
from dataclasses import replace

import torch

from ..config import Config
from ..layers.logits_processor import build_logits_processor
from ..layers.sampler import Sampler
from .block_manager import BlockManager
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence


def _length_penalized(seq: Sequence, length_penalty: float) -> float:
    return seq.cum_logprob / (max(seq.num_generated, 1) ** length_penalty)


def select_beam_candidates(
    stopped: list[bool], num_beams: int
) -> tuple[list[int], list[int]]:
    # Stopped candidates (EOS/max_len) only finish if ranked in the top num_beams.
    finished = [r for r, stop in enumerate(stopped) if stop and r < num_beams]
    live = [r for r, stop in enumerate(stopped) if not stop][:num_beams]
    return finished, live


class LLMEngine:
    def __init__(self, model, config: Config, eos_token_id: int | None = None):
        self.block_manager = BlockManager(config.num_blocks, config.block_size)
        self.runner = ModelRunner(
            model,
            self.block_manager,
            config.dtype,
            attn_backend=config.attn_backend,
            device=config.device,
            max_num_seqs=config.max_num_seqs,
            use_cudagraph=config.use_cudagraph,
            compile_model=config.compile_model,
        )
        self.scheduler = Scheduler(config)
        self.sampler = Sampler()
        self.eos_token_id = eos_token_id
        self.device = torch.device(config.device)
        self.results: dict[int, list[int]] = {}
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)
        self._logits_processor = None
        self._next_request_id = 0

    def generate(self, requests, params) -> list[int]:
        requests = list(requests)
        if not requests:
            return []
        if params.eos_token_id is None and self.eos_token_id is not None:
            params = replace(params, eos_token_id=self.eos_token_id)

        prompt_len = len(requests[0][1])
        if any(len(decoder_ids) != prompt_len for _, decoder_ids, _ in requests):
            raise ValueError("decoder prompt length must be uniform across the batch")
        if prompt_len != 1:
            raise ValueError("the decoder prompt must be a single token")
        begin_index = prompt_len + int(params.forced_bos_token_id is not None)
        self._logits_processor = build_logits_processor(
            params,
            max_length=prompt_len + params.max_tokens,
            begin_index=begin_index,
            is_beam=params.num_beams > 1,
            device=self.device,
        )

        request_ids = []
        for encoder_token_ids, decoder_token_ids, pixel_values in requests:
            request_id = self._next_request_id
            self._next_request_id += 1
            self.scheduler.add(
                Sequence(
                    request_id=request_id,
                    decoder_prompt_ids=decoder_token_ids,
                    encoder_token_ids=encoder_token_ids,
                    sampling=params,
                    pixel_values=pixel_values,
                )
            )
            request_ids.append(request_id)
        self.run()
        return request_ids

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step(self) -> None:
        seqs = self.scheduler.schedule()
        logits = self.runner.run(seqs)
        self._postprocess(seqs, logits)

    def run(self) -> None:
        while not self.is_finished():
            self.step()

    def _postprocess(self, seqs, logits) -> None:
        processor = self._logits_processor
        by_request: dict[int, list[int]] = defaultdict(list)
        for index, seq in enumerate(seqs):
            by_request[seq.request_id].append(index)

        device = logits.device
        pin = device.type == "cuda"
        beam_groups = defaultdict(list)
        buckets = defaultdict(list)
        for request_id, indices in by_request.items():
            params = seqs[indices[0]].sampling
            length = len(seqs[indices[0]].token_ids)
            if params.num_beams > 1:
                # Group by (num_beams, live_beam_count, decoder_len) for a single batched topk.
                beam_groups[(params.num_beams, len(indices), length)].append(
                    (request_id, indices)
                )
            else:
                buckets[length].extend(indices)

        for (num_beams, beam_count, _), group in beam_groups.items():
            self._beam_group(
                seqs, group, num_beams, beam_count, processor, logits, device, pin
            )

        for indices in buckets.values():
            input_ids = torch.tensor(
                [seqs[i].token_ids for i in indices],
                dtype=torch.long,
                pin_memory=pin,
            ).to(device, non_blocking=True)
            scores = processor(input_ids, logits[indices].float())
            tokens = self.sampler(
                scores, seqs[indices[0]].sampling.temperature
            ).tolist()
            for row, i in enumerate(indices):
                self._sample(seqs[i], seqs[i].sampling, tokens[row])

    def _sample(self, seq: Sequence, params, token: int) -> None:
        seq.append_token(token)
        if token in params.eos_ids() or seq.num_completion_tokens >= params.max_tokens:
            self.scheduler.remove(seq)
            self.block_manager.deallocate(seq)
            self.runner.clear_request(seq.request_id)
            self.results[seq.request_id] = seq.completion_token_ids

    def _fork(self, parent: Sequence, token: int, score: float) -> Sequence:
        child = Sequence(
            request_id=parent.request_id,
            decoder_prompt_ids=parent.decoder_prompt_ids,
            encoder_token_ids=parent.encoder_token_ids,
            sampling=parent.sampling,
            pixel_values=parent.pixel_values,
        )
        child.token_ids = parent.token_ids + [token]
        child.num_cached_tokens = parent.num_cached_tokens
        child.cum_logprob = score
        self.block_manager.fork(parent, child)
        return child

    def _beam_group(
        self, seqs, group, num_beams, beam_count, processor, logits, device, pin
    ) -> None:
        # Batched log_softmax + processor + topk over [num_requests, beam_count * vocab].
        vocab = logits.shape[-1]
        rows = [i for _, seq_indices in group for i in seq_indices]
        input_ids = torch.tensor(
            [seqs[i].token_ids for i in rows], dtype=torch.long, pin_memory=pin
        ).to(device, non_blocking=True)
        logprobs = processor(input_ids, torch.log_softmax(logits[rows].float(), dim=-1))
        cumulative = torch.tensor(
            [[seqs[i].cum_logprob for i in seq_indices] for _, seq_indices in group],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        scores = (logprobs.reshape(len(group), beam_count, vocab) + cumulative).reshape(
            len(group), -1
        )
        params = seqs[group[0][1][0]].sampling
        keep = min(max(2, 1 + len(params.eos_ids())) * num_beams, scores.shape[-1])
        top_values, top_indices = torch.topk(scores, keep, dim=-1)
        values, indices = top_values.tolist(), top_indices.tolist()
        for row, (request_id, seq_indices) in enumerate(group):
            self._beam_advance(
                request_id,
                [seqs[i] for i in seq_indices],
                params,
                [k // vocab for k in indices[row]],
                [k % vocab for k in indices[row]],
                values[row],
            )

    def _beam_advance(self, request_id, beams, params, parents, tokens, top_values):
        num_beams = params.num_beams
        eos_ids = params.eos_ids()

        stopped = [
            token in eos_ids or beams[parent].num_generated + 1 >= params.max_tokens
            for parent, token in zip(parents, tokens)
        ]
        finished_ranks, live_ranks = select_beam_candidates(stopped, num_beams)

        # Fork children before freeing parents so shared blocks stay alive.
        new_finished = [
            self._fork(beams[parents[r]], tokens[r], top_values[r])
            for r in finished_ranks
        ]
        live = [
            self._fork(beams[parents[r]], tokens[r], top_values[r]) for r in live_ranks
        ]

        for beam in beams:
            self.scheduler.remove(beam)
            self.block_manager.deallocate(beam)

        self._finished_beams[request_id].extend(new_finished)
        self._prune_finished(request_id, params)
        for child in live:
            self.scheduler.add_running(child)

        if not live or self._should_stop(request_id, params, live):
            self._finish_beam_request(request_id, params)

    def _prune_finished(self, request_id: int, params) -> None:
        finished = self._finished_beams[request_id]
        if len(finished) <= params.num_beams:
            return
        finished.sort(
            key=lambda seq: _length_penalized(seq, params.length_penalty), reverse=True
        )
        for seq in finished[params.num_beams :]:
            self.block_manager.deallocate(seq)
        del finished[params.num_beams :]

    def _should_stop(self, request_id: int, params, live) -> bool:
        # HF early_stopping=False: stop when no live beam can beat the worst finished score.
        finished = self._finished_beams[request_id]
        if len(finished) < params.num_beams:
            return False
        worst_finished = min(
            _length_penalized(seq, params.length_penalty) for seq in finished
        )
        best_possible = max(seq.cum_logprob for seq in live) / (
            max(live[0].num_generated, 1) ** params.length_penalty
        )
        return worst_finished >= best_possible

    def _finish_beam_request(self, request_id: int, params) -> None:
        finished = self._finished_beams.pop(request_id, [])
        live = [s for s in list(self.scheduler.running) if s.request_id == request_id]

        best = max(
            finished,
            key=lambda seq: _length_penalized(seq, params.length_penalty),
            default=None,
        )

        for seq in live:
            self.scheduler.remove(seq)
        for seq in finished + live:
            self.block_manager.deallocate(seq)
        self.runner.clear_request(request_id)
        self.results[request_id] = best.completion_token_ids if best else []
