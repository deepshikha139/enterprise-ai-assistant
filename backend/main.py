import json
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from classifier import Classification, classify
from providers.anthropic_provider import AnthropicProvider
from providers.base import (
    LLMProvider,
    ProviderBusyError,
    ProviderError,
    StreamDelta,
    StreamDone,
)
from providers.logging_provider import LoggingProvider
from settings import settings

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_allowed_origin],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Created once when the server starts, then reused by every request.
provider: LLMProvider = LoggingProvider(AnthropicProvider())


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    # Busy must be checked first, because it is also a kind of ProviderError.
    try:
        result = await provider.generate(request.message)
    except ProviderBusyError:
        raise HTTPException(status_code=503, detail="Assistant is busy, please try again.")
    except ProviderError:
        raise HTTPException(status_code=500, detail="Something went wrong.")

    return ChatResponse(
        answer=result.text,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@app.post("/classify")
async def classify_question(request: ChatRequest) -> Classification:
    try:
        return await classify(provider, request.message)
    except ProviderBusyError:
        raise HTTPException(status_code=503, detail="Assistant is busy, please try again.")
    except ProviderError:
        raise HTTPException(status_code=500, detail="Something went wrong.")


def sse(event: str, data: dict) -> str:
    # One Server-Sent Event: an event name, a JSON data line, then a blank line.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    pieces = provider.stream(request.message)

    # Wait for the first piece before replying, so a failure at the very start
    # can still become a real 503/500. After this, the status is already sent.
    try:
        first = await anext(pieces)
    except ProviderBusyError:
        raise HTTPException(status_code=503, detail="Assistant is busy, please try again.")
    except ProviderError:
        raise HTTPException(status_code=500, detail="Something went wrong.")

    async def events() -> AsyncIterator[str]:
        try:
            item = first
            while True:
                if isinstance(item, StreamDelta):
                    yield sse("delta", {"text": item.text})
                elif isinstance(item, StreamDone):
                    yield sse("done", item.model_dump())
                item = await anext(pieces)
        except StopAsyncIteration:
            pass
        # Too late for an error status, so tell the client inside the stream.
        except ProviderBusyError:
            yield sse("error", {"message": "Assistant is busy, please try again."})
        except ProviderError:
            yield sse("error", {"message": "Something went wrong."})
        finally:
            # If the client leaves early, close the stream now so the call is logged.
            await pieces.aclose()

    return StreamingResponse(events(), media_type="text/event-stream")
