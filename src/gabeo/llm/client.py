"""Thin OpenAI client with structured-output validation, retries, and cost telemetry.

Why this layer instead of LangChain: it's ~80 lines, swappable in one place, and
keeps the prompts readable. Every call returns a `LLMResponse` with the raw text,
parsed pydantic model (when a `response_model` is given), token counts, and
estimated USD cost.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

# Errors that will never succeed on retry; raise immediately so callers can fall
# back to the mock client without waiting through three exponential backoffs.
_TERMINAL_ERRORS: tuple[type[BaseException], ...] = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
)


def _is_quota_error(exc: BaseException) -> bool:
    if not isinstance(exc, RateLimitError):
        return False
    code = getattr(exc, "code", None)
    if code == "insufficient_quota":
        return True
    msg = str(exc).lower()
    return "insufficient_quota" in msg or "exceeded your current quota" in msg


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, _TERMINAL_ERRORS):
        return False
    if _is_quota_error(exc):
        return False
    return isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError))

T = TypeVar("T", bound=BaseModel)


# USD per 1K tokens. Public list prices; recompute on launch day.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":  (0.00015, 0.0006),
    "gpt-4o":       (0.0025,  0.01),
    "gpt-4.1-mini": (0.0004,  0.0016),
    "gpt-4.1":      (0.002,   0.008),
}


@dataclass
class LLMResponse:
    text: str
    parsed: BaseModel | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = _PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens / 1000.0) * in_price + (completion_tokens / 1000.0) * out_price


class LLMClient:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.0, max=8.0),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_model: type[T] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        start = time.perf_counter()
        kwargs: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_model is not None:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)

        parsed: BaseModel | None = None
        if response_model is not None:
            try:
                parsed = response_model.model_validate(json.loads(text))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"LLM returned text that does not parse as {response_model.__name__}: {exc}\n"
                    f"Raw text: {text[:500]}"
                ) from exc

        return LLMResponse(
            text=text,
            parsed=parsed,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=elapsed_ms,
        )
