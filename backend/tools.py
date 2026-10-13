import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError, field_validator

from providers.base import ToolCall, ToolResult, ToolSpec

logger = logging.getLogger("tool_calls")

TIME_ZONE = ZoneInfo("Australia/Melbourne")

# Fixed demo rates: how many units of each currency equal 1 USD. Not live rates.
RATES_PER_USD: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "AED": Decimal("3.6725"),
    "OMR": Decimal("0.3845"),
}
# Decimal places to show per currency. OMR uses three.
DECIMALS: dict[str, int] = {"USD": 2, "AED": 2, "OMR": 3}


def _now() -> datetime:
    return datetime.now(TIME_ZONE)


class GetCurrentTimeInput(BaseModel):
    """No inputs."""


class ConvertCurrencyInput(BaseModel):
    amount: float = Field(gt=0, allow_inf_nan=False, description="Amount to convert.")
    from_currency: str = Field(description="Currency code to convert from, e.g. USD.")
    to_currency: str = Field(description="Currency code to convert to, e.g. AED.")

    @field_validator("from_currency", "to_currency")
    @classmethod
    def must_be_supported(cls, value: str) -> str:
        code = value.strip().upper()
        if code not in RATES_PER_USD:
            supported = ", ".join(sorted(RATES_PER_USD))
            raise ValueError(f"currency '{value}' is not supported; use one of: {supported}")
        return code


def get_current_time(_: GetCurrentTimeInput) -> str:
    now = _now()
    offset = now.strftime("%z")  # e.g. +1100
    return (
        f"{now:%Y-%m-%d %H:%M:%S} "
        f"({TIME_ZONE.key}, UTC{offset[:3]}:{offset[3:]})"
    )


def convert_currency(data: ConvertCurrencyInput) -> str:
    amount = Decimal(str(data.amount))
    rate = RATES_PER_USD[data.to_currency] / RATES_PER_USD[data.from_currency]

    places = Decimal(1).scaleb(-DECIMALS[data.to_currency])
    converted = (amount * rate).quantize(places, rounding=ROUND_HALF_UP)

    return (
        f"{amount} {data.from_currency} = {converted} {data.to_currency}. "
        f"Fixed rate 1 {data.from_currency} = {rate:.6f} {data.to_currency}; "
        "these are fixed rates, not live market rates."
    )


@dataclass
class Tool:
    spec: ToolSpec
    run: Callable[[BaseModel], str]


TOOLS: dict[str, Tool] = {
    "get_current_time": Tool(
        spec=ToolSpec(
            name="get_current_time",
            description=(
                "Returns the current date and time in Melbourne, Australia. "
                "It does not give weather, and it does not give the time in other cities."
            ),
            input_model=GetCurrentTimeInput,
        ),
        run=get_current_time,
    ),
    "convert_currency": Tool(
        spec=ToolSpec(
            name="convert_currency",
            description=(
                "Convert an amount between currencies using fixed rates. "
                f"Supported currencies: {', '.join(sorted(RATES_PER_USD))}."
            ),
            input_model=ConvertCurrencyInput,
        ),
        run=convert_currency,
    ),
}

TOOL_SPECS: list[ToolSpec] = [tool.spec for tool in TOOLS.values()]


def describe_problems(error: ValidationError) -> str:
    # Field and rule only, so the model can fix its call. No input values.
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc']) or 'input'}: "
        f"{e['msg'].removeprefix('Value error, ')}"
        for e in error.errors(include_input=False)
    )


def execute_tool(call: ToolCall) -> ToolResult:
    """Run one tool request. Any problem becomes an error result for the model."""
    tool = TOOLS.get(call.name)
    if tool is None:
        logger.info("tool=%s status=error:unknown", call.name[:50])
        available = ", ".join(TOOLS)
        return ToolResult(
            tool_call_id=call.id,
            content=f"Unknown tool. Available tools: {available}.",
            is_error=True,
        )

    try:
        data = tool.spec.input_model.model_validate(call.input)
    except ValidationError as error:
        logger.info("tool=%s status=error:invalid_input", call.name)
        return ToolResult(
            tool_call_id=call.id,
            content=f"Invalid input: {describe_problems(error)}",
            is_error=True,
        )

    try:
        content = tool.run(data)
    except Exception as error:
        # Type only: the message could contain input values.
        logger.error("tool=%s status=error:crashed type=%s", call.name, type(error).__name__)
        return ToolResult(
            tool_call_id=call.id,
            content="The tool failed unexpectedly.",
            is_error=True,
        )

    logger.info("tool=%s status=ok", call.name)
    return ToolResult(tool_call_id=call.id, content=content)
