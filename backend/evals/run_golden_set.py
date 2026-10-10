"""Golden set v0: send each question to POST /chat, mark it PASS/FAIL, print a score.

Run from backend/ with the server already running:
    uv run python -m evals.run_golden_set
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic

from settings import settings

GOLDEN_SET = Path(__file__).parent / "golden_set_v0.json"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_PROMPT = """You are grading an AI assistant's answer against an answer key.

Question: {question}

Assistant's answer:
{answer}

The answer must: {must_include}
The answer must NOT: {must_not}

Reply on one line, either "PASS: <short reason>" or "FAIL: <short reason>"."""


def check_contains(q: dict, answer: str) -> tuple[bool, str]:
    # Not glued to other letters or digits, so "30" does not match "300" or "1.30".
    pattern = rf"(?<![\w.,]){re.escape(q['must_include'])}(?!\w)"
    if re.search(pattern, answer, re.IGNORECASE):
        return True, ""
    return False, f"answer does not contain '{q['must_include']}'"


def check_yes_only(q: dict, answer: str) -> tuple[bool, str]:
    cleaned = re.sub(r"[^\w]", "", answer).lower()
    if cleaned == "yes":
        return True, ""
    return False, f"expected only 'yes', got: {answer.strip()[:60]!r}"


def check_max_sentences(q: dict, answer: str) -> tuple[bool, str]:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    if len(sentences) <= q["max_sentences"]:
        return True, ""
    return False, f"{len(sentences)} sentences, at most {q['max_sentences']} allowed"


async def check_judge(judge: AsyncAnthropic, q: dict, answer: str) -> tuple[bool, str]:
    response = await judge.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=q["question"],
                    answer=answer,
                    must_include=q["must_include"],
                    must_not=q["must_not"],
                ),
            }
        ],
    )
    verdict = "".join(b.text for b in response.content if b.type == "text").strip()
    if verdict.upper().startswith("PASS"):
        return True, ""
    return False, verdict.removeprefix("FAIL:").strip() or "judge said FAIL"


async def run_question(
    http: httpx.AsyncClient, judge: AsyncAnthropic, q: dict
) -> tuple[bool, str]:
    try:
        response = await http.post("/chat", json={"message": q["question"]})
        response.raise_for_status()
        answer = response.json()["answer"]

        if q["check"] == "contains":
            return check_contains(q, answer)
        if q["check"] == "yes_only":
            return check_yes_only(q, answer)
        if q["check"] == "max_sentences":
            return check_max_sentences(q, answer)
        if q["check"] == "judge":
            return await check_judge(judge, q, answer)
        return False, f"unknown check '{q['check']}'"
    except Exception as error:
        # One broken question should not stop the rest of the run.
        return False, f"error: {type(error).__name__}: {error}"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden set against POST /chat.")
    parser.add_argument("--url", default="http://localhost:8000", help="backend address")
    args = parser.parse_args()

    questions = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["questions"]
    judge = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    passed = 0
    async with httpx.AsyncClient(base_url=args.url, timeout=60) as http:
        for q in questions:
            ok, reason = await run_question(http, judge, q)
            passed += ok
            print(f"{q['id']} {'PASS' if ok else 'FAIL'}  {q['category']:<22} {q['question']}")
            if not ok:
                print(f"      reason: {reason}")

    print(f"\nScore: {passed}/{len(questions)}")
    return 0 if passed == len(questions) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
