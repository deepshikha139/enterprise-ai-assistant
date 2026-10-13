import logging

from pydantic import BaseModel

from providers.base import (
    AssistantMessage,
    LLMProvider,
    Message,
    ToolResultsMessage,
    UserMessage,
)
from tools import TOOL_SPECS, execute_tool

logger = logging.getLogger(__name__)

# How many times one question may go "model asks for tools, we run them, model continues".
MAX_TOOL_ROUNDS = 3

LIMIT_MESSAGE = "I couldn't complete that request."

SYSTEM_PROMPT = (
    "Answer from your own knowledge when you can. If a question needs live information "
    "or an action that none of your tools provides (for example the weather), "
    "say you can't do that instead of guessing."
)


class ChatResult(BaseModel):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int


async def chat_with_tools(provider: LLMProvider, question: str) -> ChatResult:
    """Answer a question, running any tools the model asks for.

    One round = the model asks for one or more tools and we send all their results back.
    After MAX_TOOL_ROUNDS rounds the model gets no more turns and we return a safe message.
    Provider errors are not caught here; the caller turns them into 503/500.
    """
    messages: list[Message] = [UserMessage(text=question)]
    input_tokens = output_tokens = 0

    # One call to start, then one more after each round.
    for round_number in range(MAX_TOOL_ROUNDS + 1):
        turn = await provider.generate_with_tools(messages, TOOL_SPECS, SYSTEM_PROMPT)
        input_tokens += turn.input_tokens
        output_tokens += turn.output_tokens

        if not turn.tool_calls:
            return ChatResult(
                answer=turn.text,
                model=turn.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        if round_number == MAX_TOOL_ROUNDS:
            break

        messages.append(AssistantMessage(text=turn.text, tool_calls=turn.tool_calls))
        messages.append(
            ToolResultsMessage(results=[execute_tool(call) for call in turn.tool_calls])
        )

    logger.warning("tool round limit reached: max_rounds=%d", MAX_TOOL_ROUNDS)
    return ChatResult(
        answer=LIMIT_MESSAGE,
        model=turn.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
