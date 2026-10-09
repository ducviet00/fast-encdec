"""Completion outputs, mirroring the vLLM ``RequestOutput`` shape.

``LLM.generate`` returns one :class:`RequestOutput` per request; the generated
text is ``output.outputs[0].text``, exactly as in vLLM.
"""

from dataclasses import dataclass, field


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request."""

    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: None = None
    finish_reason: str | None = None
    stop_reason: int | None = None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, text={self.text!r}, "
            f"token_ids={self.token_ids}, finish_reason={self.finish_reason!r})"
        )


@dataclass
class RequestOutput:
    """The output data of a completion request to the :class:`LLM`."""

    request_id: int
    prompt: str | None
    prompt_token_ids: list[int] | None
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = True
    encoder_prompt: str | None = None
    encoder_prompt_token_ids: list[int] | None = None

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, prompt={self.prompt!r}, "
            f"outputs={self.outputs}, finished={self.finished})"
        )
