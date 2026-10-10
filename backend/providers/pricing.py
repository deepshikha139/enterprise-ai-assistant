# Dollars per million tokens: model name -> (input price, output price).
# Add a row when a new model (e.g. a Bedrock one) is used.
PRICES_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """A rough cost estimate; ignores caching discounts. None if the model has no price."""
    prices = PRICES_PER_MILLION_TOKENS.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
