"""Answer generation using the Groq chat completions API."""

import logging
from functools import lru_cache

import groq
from groq import AsyncGroq

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question clearly and accurately. "
    "If you are not sure about something, say so instead of guessing."
)


class AnswerGenerationError(Exception):
    """A generation failure that is safe to show to API clients.

    `message` never contains provider error details or credentials.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@lru_cache
def get_groq_client() -> AsyncGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise AnswerGenerationError("The answer service is not configured.", status_code=503)
    return AsyncGroq(
        api_key=settings.groq_api_key,
        timeout=settings.groq_timeout_seconds,
        max_retries=2,
    )


async def close_groq_client() -> None:
    # Only close a client that was actually created.
    if get_groq_client.cache_info().currsize:
        await get_groq_client().close()
        get_groq_client.cache_clear()


async def generate_answer(
    question: str, client: AsyncGroq | None = None, system_prompt: str = SYSTEM_PROMPT
) -> str:
    settings = get_settings()
    client = client or get_groq_client()

    try:
        completion = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_completion_tokens=settings.groq_max_completion_tokens,
        )
    # Order matters: APITimeoutError is a subclass of APIConnectionError.
    except groq.APITimeoutError as exc:
        logger.warning("Groq request timed out: %s", type(exc).__name__)
        raise AnswerGenerationError("The answer service timed out. Please try again.", 504) from exc
    except groq.APIConnectionError as exc:
        logger.warning("Could not reach Groq: %s", type(exc).__name__)
        raise AnswerGenerationError("The answer service is unreachable. Please try again later.", 503) from exc
    except groq.AuthenticationError as exc:
        logger.error("Groq rejected the API key (status %s)", exc.status_code)
        raise AnswerGenerationError("The answer service is misconfigured.", 502) from exc
    except groq.RateLimitError as exc:
        logger.warning("Groq rate limit reached")
        raise AnswerGenerationError("Too many requests. Please wait and try again.", 429) from exc
    except groq.APIStatusError as exc:
        logger.error("Groq returned an error (status %s)", exc.status_code)
        raise AnswerGenerationError("The answer service returned an error.", 502) from exc

    choice = completion.choices[0] if completion.choices else None
    # A cut-off answer can silently drop conditions or exceptions, so never return one.
    # With reasoning models the hidden reasoning also counts towards this limit.
    if choice is not None and choice.finish_reason == "length":
        logger.warning("Groq answer hit the %d-token limit", settings.groq_max_completion_tokens)
        raise AnswerGenerationError(
            "The answer was too long to complete. Please ask a more specific question.", 502
        )
    answer = (choice.message.content or "").strip() if choice is not None else ""
    if not answer:
        logger.error("Groq returned an empty answer for model %s", settings.groq_model)
        raise AnswerGenerationError("The answer service returned an empty answer.", 502)
    return answer
