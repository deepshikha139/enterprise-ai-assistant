import logging
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from providers.base import (
    LLMProvider,
    LLMResult,
    ProviderBusyError,
    ProviderError,
    StreamDelta,
    StreamDone,
)
from settings import settings

logger = logging.getLogger(__name__)


def log_api_error(error: anthropic.APIError) -> None:
    # Only details from the error itself; never the client or request headers.
    logger.error(
        "Anthropic API call failed: type=%s status=%s request_id=%s message=%s",
        type(error).__name__,
        getattr(error, "status_code", None),
        getattr(error, "request_id", None),
        error.message,
    )


class AnthropicProvider(LLMProvider):
    """Calls Claude through the Anthropic API."""

    def __init__(self) -> None:
        # The SDK retries temporary failures by itself, up to max_retries times.
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_seconds,
        )

    async def generate(self, message: str) -> LLMResult:
        try:
            response = await self.client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                messages=[{"role": "user", "content": message}],
            )
        # These are only raised after the SDK's retries have run out.
        except (
            anthropic.RateLimitError,
            anthropic.OverloadedError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        ) as error:
            log_api_error(error)
            raise ProviderBusyError() from error
        except anthropic.APIError as error:
            log_api_error(error)
            raise ProviderError() from error

        # A response can hold several content blocks; keep only the text ones.
        text = "".join(block.text for block in response.content if block.type == "text")

        return LLMResult(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def stream(self, message: str) -> AsyncIterator[StreamDelta | StreamDone]:
        try:
            async with self.client.messages.stream(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                messages=[{"role": "user", "content": message}],
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamDelta(text=text)
                final = await stream.get_final_message()
        except (
            anthropic.RateLimitError,
            anthropic.OverloadedError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        ) as error:
            log_api_error(error)
            raise ProviderBusyError() from error
        except anthropic.APIError as error:
            log_api_error(error)
            raise ProviderError() from error

        yield StreamDone(
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
