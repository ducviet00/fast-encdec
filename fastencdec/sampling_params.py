from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # temperature <= 0 selects greedy decoding; num_beams > 1 selects beam search.
    max_tokens: int = 64
    num_beams: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    eos_token_id: int | list[int] | None = None
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    forced_bos_token_id: int | None = None
    forced_eos_token_id: int | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None

    def eos_ids(self) -> set[int]:
        if self.eos_token_id is None:
            return set()
        return (
            {self.eos_token_id}
            if isinstance(self.eos_token_id, int)
            else set(self.eos_token_id)
        )

    @classmethod
    def from_generation_config(
        cls,
        config,
        eos_token_id: int | None = None,
        decoder_prompt_len: int = 1,
    ) -> "SamplingParams":
        val = lambda x, default: default if x is None else x
        temperature = 0.0 if not config.do_sample else val(config.temperature, 1.0)
        if config.max_new_tokens is not None:
            max_tokens = config.max_new_tokens
        elif config.max_length is not None:
            max_tokens = config.max_length - decoder_prompt_len
        else:
            max_tokens = 64

        return cls(
            max_tokens=max_tokens,
            num_beams=val(config.num_beams, 1),
            temperature=temperature,
            top_p=val(config.top_p, 1.0),
            top_k=val(config.top_k, 0),
            repetition_penalty=val(config.repetition_penalty, 1.0),
            length_penalty=val(config.length_penalty, 1.0),
            no_repeat_ngram_size=val(config.no_repeat_ngram_size, 0),
            min_length=val(config.min_length, 0),
            eos_token_id=val(config.eos_token_id, eos_token_id),
            forced_bos_token_id=config.forced_bos_token_id,
            forced_eos_token_id=config.forced_eos_token_id,
            suppress_tokens=list(config.suppress_tokens)
            if config.suppress_tokens is not None
            else None,
            begin_suppress_tokens=list(config.begin_suppress_tokens)
            if config.begin_suppress_tokens is not None
            else None,
        )
