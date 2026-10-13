from providers.base import ToolCall
from tools import execute_tool


def convert(amount, from_currency, to_currency) -> str:
    call = ToolCall(
        id="c",
        name="convert_currency",
        input={"amount": amount, "from_currency": from_currency, "to_currency": to_currency},
    )
    result = execute_tool(call)
    assert not result.is_error
    return result.content


def test_converts_with_the_fixed_rates_and_rounds_per_currency():
    assert "= 367.25 AED" in convert(100, "USD", "AED")
    assert "= 38.450 OMR" in convert(100, "usd", "OMR")  # code is case-insensitive
    assert "= 10.00 USD" in convert(36.725, "AED", "USD")


def test_result_says_the_rates_are_not_live():
    assert "not live" in convert(1, "USD", "AED")
