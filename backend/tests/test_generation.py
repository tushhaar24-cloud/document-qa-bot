"""Tests for Groq error handling in the generation service, using a fake client."""

from types import SimpleNamespace

import groq
import httpx
import pytest

from app.services import generation
from app.services.generation import AnswerGenerationError, generate_answer

GROQ_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def make_client(*, result=None, error: Exception | None = None) -> SimpleNamespace:
    async def create(**kwargs):
        if error is not None:
            raise error
        return result

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def make_completion(content: str | None, finish_reason: str = "stop") -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def status_error(cls: type[groq.APIStatusError], status: int) -> groq.APIStatusError:
    response = httpx.Response(status, request=GROQ_REQUEST)
    return cls("secret provider detail gsk_should_not_leak", response=response, body=None)


async def test_returns_stripped_answer() -> None:
    client = make_client(result=make_completion("  RAG combines retrieval with generation.  "))
    assert await generate_answer("What is RAG?", client=client) == "RAG combines retrieval with generation."


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (groq.APITimeoutError(request=GROQ_REQUEST), 504),
        (groq.APIConnectionError(request=GROQ_REQUEST), 503),
        (status_error(groq.AuthenticationError, 401), 502),
        (status_error(groq.RateLimitError, 429), 429),
        (status_error(groq.InternalServerError, 500), 502),
    ],
)
async def test_maps_groq_errors_without_leaking_details(error: Exception, expected_status: int) -> None:
    with pytest.raises(AnswerGenerationError) as exc_info:
        await generate_answer("What is RAG?", client=make_client(error=error))

    assert exc_info.value.status_code == expected_status
    assert "gsk_" not in exc_info.value.message
    assert "secret" not in exc_info.value.message


@pytest.mark.parametrize("content", [None, "", "   "])
async def test_empty_answer_is_an_error(content: str | None) -> None:
    with pytest.raises(AnswerGenerationError) as exc_info:
        await generate_answer("What is RAG?", client=make_client(result=make_completion(content)))
    assert exc_info.value.status_code == 502


@pytest.mark.parametrize("content", ["", "Employees must give 60 days notice, except"])
async def test_cut_off_answer_is_an_error(content: str) -> None:
    # Groq reports finish_reason="length" when max_completion_tokens is reached
    # (verified live: the text is empty or ends mid-sentence).
    completion = make_completion(content, finish_reason="length")

    with pytest.raises(AnswerGenerationError) as exc_info:
        await generate_answer("Summarise everything.", client=make_client(result=completion))

    assert exc_info.value.status_code == 502
    assert "too long" in exc_info.value.message


async def test_missing_api_key_is_a_safe_503(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    no_key = dataclasses.replace(generation.get_settings(), groq_api_key="")
    monkeypatch.setattr(generation, "get_settings", lambda: no_key)
    generation.get_groq_client.cache_clear()
    try:
        with pytest.raises(AnswerGenerationError) as exc_info:
            await generate_answer("What is RAG?")
    finally:
        generation.get_groq_client.cache_clear()

    assert exc_info.value.status_code == 503
    assert exc_info.value.message == "The answer service is not configured."
