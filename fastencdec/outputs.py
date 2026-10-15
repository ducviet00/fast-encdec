from dataclasses import dataclass, field


@dataclass
class CompletionOutput:
    """Output data for one generated completion hypothesis."""

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
    """Output data for a completed generation request."""

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
