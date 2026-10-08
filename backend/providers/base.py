from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResult(BaseModel):
    """One answer from a model, in the same shape whichever provider produced it."""

    text: str
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
