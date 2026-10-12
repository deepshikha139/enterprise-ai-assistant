from classifier import FALLBACK, Classification, classify
from tests.fake_provider import FakeProvider

VALID = '{"category": "hr", "confidence": 0.9, "reason": "Asks about leave."}'
BAD_CONFIDENCE = '{"category": "hr", "confidence": 1.7, "reason": "Asks about leave."}'
NOT_JSON = "I think this is about HR."


async def test_valid_reply_is_returned_after_one_call():
    provider = FakeProvider([VALID])

    result = await classify(provider, "How many leave days do I get?")

    assert result == Classification(
        category="hr", confidence=0.9, reason="Asks about leave."
    )
    assert len(provider.messages) == 1


async def test_bad_confidence_is_retried_with_the_error_in_the_message():
    provider = FakeProvider([BAD_CONFIDENCE, VALID])

    result = await classify(provider, "How many leave days do I get?")

    assert result.category == "hr"
    assert len(provider.messages) == 2
    retry_message = provider.messages[1]
    assert BAD_CONFIDENCE in retry_message
    assert "confidence" in retry_message


async def test_reply_that_stays_broken_falls_back_to_other():
    # One first try plus two retries, all invalid.
    provider = FakeProvider([BAD_CONFIDENCE, NOT_JSON, BAD_CONFIDENCE])

    result = await classify(provider, "How many leave days do I get?")

    assert result == FALLBACK
    assert result.category == "other"
    assert len(provider.messages) == 3
