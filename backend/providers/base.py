from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

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
    def stream(self, message: str) -> AsyncIterator[StreamDelta | StreamDone]:
        """Send one user message and yield the answer piece by piece.

        Yields StreamDelta items, then one StreamDone. Raises ProviderError
        (or ProviderBusyError) if it fails, possibly after some pieces were yielded.
        """
