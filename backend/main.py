import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from providers.anthropic_provider import AnthropicProvider
from providers.base import LLMProvider, ProviderBusyError, ProviderError

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Created once when the server starts, then reused by every request.
provider: LLMProvider = AnthropicProvider()


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
