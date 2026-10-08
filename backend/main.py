import logging

import anthropic
from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Created once when the server starts, then reused by every request.
# The SDK retries temporary failures by itself, up to max_retries times.
client = AsyncAnthropic(
    api_key=settings.anthropic_api_key.get_secret_value(),
    max_retries=settings.llm_max_retries,
    timeout=settings.llm_timeout_seconds,
)


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


def log_api_error(error: anthropic.APIError) -> None:
    # Only details from the error itself; never the client or request headers.
    logger.error(
        "Anthropic API call failed: type=%s status=%s request_id=%s message=%s",
        type(error).__name__,
        getattr(error, "status_code", None),
        getattr(error, "request_id", None),
        error.message,
    )


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            messages=[{"role": "user", "content": request.message}],
        )
    # These are only raised after the SDK's retries have run out.
    except (
        anthropic.RateLimitError,
        anthropic.OverloadedError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
    ) as error:
        log_api_error(error)
        raise HTTPException(status_code=503, detail="Assistant is busy, please try again.")
    except anthropic.APIError as error:
        log_api_error(error)
        raise HTTPException(status_code=500, detail="Something went wrong.")

    # A response can hold several content blocks; keep only the text ones.
    answer = "".join(block.text for block in response.content if block.type == "text")

    return ChatResponse(
        answer=answer,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
