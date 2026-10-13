import logging
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic, transform_schema
from pydantic import BaseModel

from providers.base import (
    AssistantMessage,
    LLMProvider,
    LLMResult,
    Message,
    ProviderBusyError,
    ProviderError,
    StreamDelta,
    StreamDone,
    ToolCall,
    ToolSpec,
    ToolTurn,
    UserMessage,
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


def to_anthropic_message(message: Message) -> dict:
    """Translate one of our messages into Anthropic's format."""
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.text}

    if isinstance(message, AssistantMessage):
        blocks: list[dict] = []
        if message.text:
            blocks.append({"type": "text", "text": message.text})
        blocks += [
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            for call in message.tool_calls
        ]
        return {"role": "assistant", "content": blocks}

    # All results for one turn go back together in a single user message.
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.results
        ],
    }


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

    async def generate_structured(
        self, message: str, output_type: type[BaseModel]
    ) -> LLMResult:
        try:
            response = await self.client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                messages=[{"role": "user", "content": message}],
                # transform_schema turns the Pydantic class into a schema the API accepts.
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": transform_schema(output_type),
                    }
                },
            )
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

        text = "".join(block.text for block in response.content if block.type == "text")

        return LLMResult(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def generate_with_tools(
        self, messages: list[Message], tools: list[ToolSpec], system: str | None = None
    ) -> ToolTurn:
        try:
            response = await self.client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                # Left out entirely when there is no system prompt.
                **({"system": system} if system else {}),
                messages=[to_anthropic_message(m) for m in messages],
                tools=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_model.model_json_schema(),
                    }
                    for tool in tools
                ],
            )
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

        return ToolTurn(
            text="".join(block.text for block in response.content if block.type == "text"),
            tool_calls=[
                ToolCall(id=block.id, name=block.name, input=block.input)
                for block in response.content
                if block.type == "tool_use"
            ],
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
