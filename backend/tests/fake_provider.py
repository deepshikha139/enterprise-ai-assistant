from collections.abc import AsyncIterator

from pydantic import BaseModel

from providers.base import (
    LLMProvider,
    LLMResult,
    Message,
    StreamDelta,
    StreamDone,
    ToolCall,
    ToolSpec,
    ToolTurn,
)


def text_turn(text: str) -> ToolTurn:
    return ToolTurn(text=text, model="fake", input_tokens=10, output_tokens=5)


def tool_turn(*calls: ToolCall) -> ToolTurn:
    return ToolTurn(tool_calls=list(calls), model="fake", input_tokens=10, output_tokens=5)


class FakeProvider(LLMProvider):
    """Returns scripted replies, in order, and remembers what it was sent."""

    def __init__(
        self, replies: list[str] | None = None, turns: list[ToolTurn] | None = None
    ) -> None:
        self.replies = list(replies or [])
        self.turns = list(turns or [])
        self.messages: list[str] = []  # for generate_structured
        self.conversations: list[list[Message]] = []  # for generate_with_tools
        self.tools_offered: list[list[str]] = []
        self.systems: list[str | None] = []

    async def generate_structured(
        self, message: str, output_type: type[BaseModel]
    ) -> LLMResult:
        self.messages.append(message)
        return LLMResult(
            text=self.replies.pop(0), model="fake", input_tokens=1, output_tokens=1
        )

    async def generate_with_tools(
        self, messages: list[Message], tools: list[ToolSpec], system: str | None = None
    ) -> ToolTurn:
        # Copy: the caller keeps adding to its list after this call.
        self.conversations.append(list(messages))
        self.systems.append(system)
        self.tools_offered.append([tool.name for tool in tools])
        return self.turns.pop(0)

    async def generate(self, message: str) -> LLMResult:
        raise NotImplementedError

    def stream(self, message: str) -> AsyncIterator[StreamDelta | StreamDone]:
        raise NotImplementedError
