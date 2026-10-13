import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

from pydantic import BaseModel

from providers.base import (
    LLMProvider,
    LLMResult,
    Message,
    ProviderBusyError,
    ProviderError,
    StreamDelta,
    StreamDone,
    ToolSpec,
    ToolTurn,
)
from providers.pricing import estimate_cost_usd
from settings import settings

logger = logging.getLogger("llm_calls")


def log_call(
    kind: str,
    status: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_seconds: float,
    first_piece_seconds: float | None,
    tools_requested: list[str] | None = None,
) -> None:
    # One line per call. Never the API key, the message, or the answer text.
    cost = None
    if input_tokens is not None and output_tokens is not None:
        cost = estimate_cost_usd(model, input_tokens, output_tokens)

    # Only tool calls have this field: the names of the tools the model asked for.
    # Passed as a log argument, not pasted into the format, since the model chose the names.
    tools_format = "" if tools_requested is None else " tools_requested=%s"
    tools_args = () if tools_requested is None else (",".join(tools_requested) or "-",)

    logger.info(
        "kind=%s status=%s model=%s input_tokens=%s output_tokens=%s "
        "total_ms=%d first_piece_ms=%s cost_usd=%s" + tools_format,
        kind,
        status,
        model,
        "?" if input_tokens is None else input_tokens,
        "?" if output_tokens is None else output_tokens,
        total_seconds * 1000,
        "-" if first_piece_seconds is None else round(first_piece_seconds * 1000),
        "n/a" if cost is None else f"{cost:.6f}",
        *tools_args,
    )


class LoggingProvider(LLMProvider):
    """Wraps any provider and logs one line for every call made through it."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner

    async def generate(self, message: str) -> LLMResult:
        start = time.perf_counter()
        # Until the provider answers, the best we know is the model we asked for.
        model = settings.llm_model
        try:
            result = await self.inner.generate(message)
        except ProviderBusyError:
            log_call("generate", "error:busy", model, None, None, time.perf_counter() - start, None)
            raise
        except ProviderError:
            log_call("generate", "error:provider", model, None, None, time.perf_counter() - start, None)
            raise

        log_call(
            "generate",
            "ok",
            result.model,
            result.input_tokens,
            result.output_tokens,
            time.perf_counter() - start,
            None,
        )
        return result

    async def generate_structured(
        self, message: str, output_type: type[BaseModel]
    ) -> LLMResult:
        start = time.perf_counter()
        model = settings.llm_model
        try:
            result = await self.inner.generate_structured(message, output_type)
        except ProviderBusyError:
            log_call("structured", "error:busy", model, None, None, time.perf_counter() - start, None)
            raise
        except ProviderError:
            log_call("structured", "error:provider", model, None, None, time.perf_counter() - start, None)
            raise

        log_call(
            "structured",
            "ok",
            result.model,
            result.input_tokens,
            result.output_tokens,
            time.perf_counter() - start,
            None,
        )
        return result

    async def generate_with_tools(
        self, messages: list[Message], tools: list[ToolSpec], system: str | None = None
    ) -> ToolTurn:
        start = time.perf_counter()
        model = settings.llm_model
        try:
            turn = await self.inner.generate_with_tools(messages, tools, system)
        except ProviderBusyError:
            log_call("tools", "error:busy", model, None, None, time.perf_counter() - start, None)
            raise
        except ProviderError:
            log_call("tools", "error:provider", model, None, None, time.perf_counter() - start, None)
            raise

        log_call(
            "tools",
            "ok",
            turn.model,
            turn.input_tokens,
            turn.output_tokens,
            time.perf_counter() - start,
            None,
            # Names only, never the tool inputs.
            tools_requested=[call.name for call in turn.tool_calls],
        )
        return turn

    async def stream(self, message: str) -> AsyncIterator[StreamDelta | StreamDone]:
        start = time.perf_counter()
        model = settings.llm_model
        input_tokens = output_tokens = first_piece = None
        # If the caller stops reading, we are closed at the yield and never reach "ok".
        status = "stopped_early"

        try:
            # aclosing makes sure the inner stream is closed too if we are closed early.
            async with aclosing(self.inner.stream(message)) as pieces:
                async for item in pieces:
                    if isinstance(item, StreamDelta) and first_piece is None:
                        first_piece = time.perf_counter() - start
                    elif isinstance(item, StreamDone):
                        status = "ok"
                        model = item.model
                        input_tokens = item.input_tokens
                        output_tokens = item.output_tokens
                    yield item
        except ProviderBusyError:
            status = "error:busy"
            raise
        except ProviderError:
            status = "error:provider"
            raise
        finally:
            log_call(
                "stream",
                status,
                model,
                input_tokens,
                output_tokens,
                time.perf_counter() - start,
                first_piece,
            )
