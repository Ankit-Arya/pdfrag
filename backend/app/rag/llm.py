from __future__ import annotations

import random
import time
from typing import Any

from app.config import get_settings
from app.rag.prompts import SYSTEM_PROMPT

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        OpenAI,
        RateLimitError,
    )
except ImportError:  # Allows extraction/retrieval tests in minimal environments.
    OpenAI = None  # type: ignore[assignment,misc]

    class _MissingOpenAIError(Exception):
        pass

    APIConnectionError = _MissingOpenAIError  # type: ignore[assignment,misc]
    APITimeoutError = _MissingOpenAIError  # type: ignore[assignment,misc]
    InternalServerError = _MissingOpenAIError  # type: ignore[assignment,misc]
    RateLimitError = _MissingOpenAIError  # type: ignore[assignment,misc]


class LlmConfigurationError(RuntimeError):
    pass


_TRANSIENT_LLM_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    TimeoutError,
    ConnectionError,
)


class LlmService:
    def __init__(self) -> None:
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if OpenAI is None:
                raise LlmConfigurationError(
                    "The openai Python package is not installed. Install project requirements."
                )
            settings = get_settings()
            if not settings.openai_api_key and not settings.llm_base_url:
                raise LlmConfigurationError(
                    "No LLM is configured. Set OPENAI_API_KEY or an OpenAI-compatible LLM_BASE_URL."
                )
            kwargs: dict[str, object] = {
                "api_key": settings.openai_api_key or "local-not-required",
                "timeout": settings.llm_timeout_seconds,
                "max_retries": 0,
            }
            if settings.llm_base_url:
                kwargs["base_url"] = settings.llm_base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                return self._generate_once(
                    system_prompt, user_prompt, max_output_tokens=max_output_tokens
                )
            except _TRANSIENT_LLM_ERRORS as exc:
                last_error = exc
                if attempt == 2:
                    raise
                delay = min(4.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25)
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _generate_once(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        settings = get_settings()
        token_limit = max_output_tokens or settings.max_output_tokens
        if settings.llm_base_url:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=token_limit,
            )
            content = response.choices[0].message.content
            return (content or "").strip()

        response = self.client.responses.create(
            model=settings.llm_model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=token_limit,
        )
        return response.output_text.strip()

    def answer(self, user_prompt: str) -> str:
        return self.generate(SYSTEM_PROMPT, user_prompt)


llm_service = LlmService()
