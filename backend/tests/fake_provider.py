from collections.abc import AsyncIterator

from pydantic import BaseModel

from providers.base import LLMProvider, LLMResult, StreamDelta, StreamDone


class FakeProvider(LLMProvider):
    """Returns scripted replies, in order, and remembers every message it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.messages: list[str] = []

    async def generate_structured(
        self, message: str, output_type: type[BaseModel]
    ) -> LLMResult:
        self.messages.append(message)
        return LLMResult(
            text=self.replies.pop(0), model="fake", input_tokens=1, output_tokens=1
        )

    async def generate(self, message: str) -> LLMResult:
        raise NotImplementedError

    def stream(self, message: str) -> AsyncIterator[StreamDelta | StreamDone]:
        raise NotImplementedError
