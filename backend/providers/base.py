from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel


class LLMResult(BaseModel):
    """One answer from a model, in the same shape whichever provider produced it."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


class StreamDelta(BaseModel):
    """One piece of an answer that is still being written."""

    text: str


class StreamDone(BaseModel):
    """Sent once, last, when a streamed answer has finished normally."""

    model: str
    input_tokens: int
    output_tokens: int


class ToolSpec(BaseModel):
    """A tool the model may ask for. input_model makes the schema and checks the inputs."""

    name: str
    description: str
    input_model: type[BaseModel]


class ToolCall(BaseModel):
    """The model asking us to run one tool."""

    id: str
    name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    """What we send back for one ToolCall. is_error tells the model the tool failed."""

    tool_call_id: str
    content: str
    is_error: bool = False


class UserMessage(BaseModel):
    text: str


class AssistantMessage(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []


class ToolResultsMessage(BaseModel):
    results: list[ToolResult]


# A conversation in our own shape; each provider translates it to its own format.
Message = UserMessage | AssistantMessage | ToolResultsMessage


class ToolTurn(BaseModel):
    """One model reply that may contain text, tool requests, or both."""

    text: str = ""
    tool_calls: list[ToolCall] = []
    model: str
    input_tokens: int
    output_tokens: int


class ProviderError(Exception):
    """The provider failed in a way the user can't fix by retrying."""


class ProviderBusyError(ProviderError):
    """The provider is temporarily busy or unreachable; trying again later may work."""


class LLMProvider(ABC):
    """The rule every provider (Anthropic, Bedrock, ...) must follow."""

    @abstractmethod
    async def generate(self, message: str) -> LLMResult:
        """Send one user message and return the model's answer."""

    @abstractmethod
    async def generate_structured(
        self, message: str, output_type: type[BaseModel]
    ) -> LLMResult:
        """Like generate, but the model is asked to reply as JSON matching output_type.

        The reply comes back as raw text in LLMResult.text. It is NOT validated here:
        the caller checks it, because the model can still break rules the schema can't
        express (for example a number range).
        """

    @abstractmethod
    async def generate_with_tools(
        self, messages: list[Message], tools: list[ToolSpec], system: str | None = None
    ) -> ToolTurn:
        """Make ONE model call with the conversation so far and the tools on offer.

        system is an optional instruction for the model, kept apart from the messages.
        The reply may ask for tools. Running them and calling again is the caller's job.
        """

    @abstractmethod
    def stream(self, message: str) -> AsyncIterator[StreamDelta | StreamDone]:
        """Send one user message and yield the answer piece by piece.

        Yields StreamDelta items, then one StreamDone. Raises ProviderError
        (or ProviderBusyError) if it fails, possibly after some pieces were yielded.
        """
