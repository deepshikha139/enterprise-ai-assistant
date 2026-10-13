from datetime import datetime

import pytest

import tools
from assistant import LIMIT_MESSAGE, MAX_TOOL_ROUNDS, SYSTEM_PROMPT, chat_with_tools
from providers.base import ToolCall, ToolResultsMessage
from tests.fake_provider import FakeProvider, text_turn, tool_turn

TIME_CALL = ToolCall(id="t1", name="get_current_time", input={})


def convert_call(**inputs) -> ToolCall:
    return ToolCall(id="c1", name="convert_currency", input=inputs)


def last_results(provider: FakeProvider, call_index: int):
    """The tool results the assistant sent to the model on the given call."""
    last_message = provider.conversations[call_index][-1]
    assert isinstance(last_message, ToolResultsMessage)
    return last_message.results


async def test_question_without_tool_gets_a_direct_answer():
    provider = FakeProvider(turns=[text_turn("Paris.")])

    result = await chat_with_tools(provider, "Capital of France?")

    assert result.answer == "Paris."
    assert len(provider.conversations) == 1
    # The tools were on offer, the model just did not ask for any.
    assert provider.tools_offered[0] == ["get_current_time", "convert_currency"]


async def test_system_prompt_is_sent_on_every_model_call():
    provider = FakeProvider(turns=[tool_turn(TIME_CALL), text_turn("Done.")])

    await chat_with_tools(provider, "What time is it?")

    assert provider.systems == [SYSTEM_PROMPT, SYSTEM_PROMPT]


async def test_tool_request_then_final_answer(monkeypatch):
    fixed = datetime(2026, 10, 13, 14, 5, 9, tzinfo=tools.TIME_ZONE)
    monkeypatch.setattr(tools, "_now", lambda: fixed)
    provider = FakeProvider(turns=[tool_turn(TIME_CALL), text_turn("It is 2:05 pm.")])

    result = await chat_with_tools(provider, "What time is it?")

    assert result.answer == "It is 2:05 pm."
    assert len(provider.conversations) == 2
    (tool_result,) = last_results(provider, 1)
    assert tool_result.tool_call_id == "t1"
    assert not tool_result.is_error
    assert "2026-10-13 14:05:09" in tool_result.content
    assert "UTC+11:00" in tool_result.content
    # Tokens are added up over both model calls.
    assert result.input_tokens == 20
    assert result.output_tokens == 10


@pytest.mark.parametrize(
    ("inputs", "expected_in_error"),
    [
        ({"amount": -5, "from_currency": "USD", "to_currency": "AED"}, "greater than 0"),
        ({"amount": 0, "from_currency": "USD", "to_currency": "AED"}, "greater than 0"),
        ({"amount": 10, "from_currency": "INR", "to_currency": "AED"}, "'INR' is not supported"),
        ({"amount": 10, "from_currency": "USD", "to_currency": "XYZ"}, "'XYZ' is not supported"),
        ({"amount": 10, "from_currency": "USD"}, "to_currency"),
    ],
)
async def test_bad_currency_inputs_become_an_error_result(inputs, expected_in_error):
    provider = FakeProvider(
        turns=[tool_turn(convert_call(**inputs)), text_turn("Sorry, I can't convert that.")]
    )

    result = await chat_with_tools(provider, "Convert some money")

    # No crash: the model got the error and the conversation finished.
    assert result.answer == "Sorry, I can't convert that."
    (tool_result,) = last_results(provider, 1)
    assert tool_result.is_error
    assert expected_in_error in tool_result.content


async def test_unknown_tool_becomes_an_error_result():
    provider = FakeProvider(
        turns=[
            tool_turn(ToolCall(id="x", name="delete_everything", input={})),
            text_turn("I can't do that."),
        ]
    )

    result = await chat_with_tools(provider, "Do something")

    assert result.answer == "I can't do that."
    (tool_result,) = last_results(provider, 1)
    assert tool_result.is_error
    assert "Unknown tool" in tool_result.content


async def test_round_limit_stops_the_loop_with_a_safe_message():
    # The model asks for a tool every single time.
    provider = FakeProvider(turns=[tool_turn(TIME_CALL)] * (MAX_TOOL_ROUNDS + 2))

    result = await chat_with_tools(provider, "What time is it?")

    assert result.answer == LIMIT_MESSAGE
    # The first call plus one call after each of the allowed rounds, and no more.
    assert len(provider.conversations) == MAX_TOOL_ROUNDS + 1
