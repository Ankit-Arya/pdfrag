from __future__ import annotations

import logging
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

from app.config import get_settings
from app.rag.prompts import SYSTEM_PROMPT
from app.rag.progress import emit_progress

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


logger = logging.getLogger(__name__)


class LlmConfigurationError(RuntimeError):
    pass


class LlmRateLimitError(RuntimeError):
    """Raised only after rate-limit-aware retries have been exhausted."""


_TRANSIENT_LLM_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    TimeoutError,
    ConnectionError,
)

_RETRY_MESSAGE_RE = re.compile(
    r"(?:try again|retry)\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|secs|seconds?)?",
    re.IGNORECASE,
)
_DURATION_PART_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)", re.IGNORECASE)


@dataclass(slots=True)
class _RateLimitState:
    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    reset_at: float = 0.0


class LlmService:
    def __init__(self) -> None:
        self._client: Any | None = None
        self._state_lock = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}
        self._rate_states: dict[str, _RateLimitState] = {}

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
                # Retry centrally below so token-rate retries can honor the server's
                # reset window instead of layering SDK retries on top of ours.
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
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        settings = get_settings()
        selected_model = model or settings.llm_model
        token_limit = max_output_tokens or settings.max_output_tokens
        estimated_request_tokens = self._estimate_reserved_tokens(
            system_prompt,
            user_prompt,
            token_limit,
        )

        # A single Uvicorn worker can still serve several chat threadpool calls at
        # once. Serialize calls to the same model so two large summaries do not
        # race each other into the same TPM bucket. Different models remain free to
        # progress independently.
        model_lock = self._model_lock(selected_model)
        lock_context = model_lock if settings.llm_serialize_model_requests else _NullLock()
        if settings.llm_serialize_model_requests and model_lock.locked():
            emit_progress(
                "model_queue",
                "Waiting for model capacity",
                f"{selected_model} is finishing another request",
            )

        last_error: BaseException | None = None
        total_rate_wait = 0.0
        with lock_context:
            for attempt in range(settings.llm_max_retries + 1):
                try:
                    remaining_wait_budget = max(
                        0.0, settings.llm_rate_limit_total_wait_seconds - total_rate_wait
                    )
                    total_rate_wait += self._wait_for_token_capacity(
                        selected_model,
                        estimated_request_tokens,
                        max_wait_seconds=remaining_wait_budget,
                    )
                    return self._generate_once(
                        system_prompt,
                        user_prompt,
                        max_output_tokens=token_limit,
                        model=selected_model,
                        reasoning_effort=reasoning_effort,
                    )
                except _TRANSIENT_LLM_ERRORS as exc:
                    last_error = exc
                    if isinstance(exc, RateLimitError):
                        response = getattr(exc, "response", None)
                        headers = getattr(response, "headers", None)
                        if headers:
                            self._record_rate_limit_headers(selected_model, headers)
                    if attempt >= settings.llm_max_retries:
                        break
                    delay = self._retry_delay(exc, attempt)
                    if isinstance(exc, RateLimitError):
                        remaining_wait_budget = max(
                            0.0, settings.llm_rate_limit_total_wait_seconds - total_rate_wait
                        )
                        if remaining_wait_budget <= 0:
                            break
                        delay = min(delay, remaining_wait_budget)
                        total_rate_wait += delay
                    if isinstance(exc, RateLimitError):
                        emit_progress(
                            "rate_limit",
                            "Waiting for API token capacity",
                            f"Retrying {selected_model} in {delay:.1f}s",
                        )
                    else:
                        emit_progress(
                            "model_retry",
                            "Retrying the model request",
                            f"Temporary {type(exc).__name__}; retrying in {delay:.1f}s",
                        )
                    logger.warning(
                        "Transient LLM error for %s (%s); retrying in %.2fs (%d/%d)",
                        selected_model,
                        type(exc).__name__,
                        delay,
                        attempt + 1,
                        settings.llm_max_retries,
                    )
                    time.sleep(delay)

        if isinstance(last_error, RateLimitError):
            raise LlmRateLimitError(
                "The language-model token rate limit is temporarily exhausted after safe retries. "
                "Please retry shortly."
            ) from last_error
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM request failed without an error")

    def _generate_once(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str:
        settings = get_settings()

        if settings.llm_base_url:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=max_output_tokens,
            )
            content = response.choices[0].message.content
            return (content or "").strip()

        request: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort and _supports_reasoning(model):
            request["reasoning"] = {"effort": reasoning_effort}

        # Use the documented raw-response wrapper so the application can consume
        # x-ratelimit-* headers and pace later calls before they are rejected.
        raw_response = self.client.responses.with_raw_response.create(**request)
        self._record_rate_limit_headers(model, raw_response.headers)
        response = raw_response.parse()
        request_id = getattr(response, "_request_id", None)
        if request_id:
            logger.debug("OpenAI request %s completed for %s", request_id, model)
        return response.output_text.strip()

    def _model_lock(self, model: str) -> threading.Lock:
        key = model.casefold()
        with self._state_lock:
            lock = self._model_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._model_locks[key] = lock
            return lock

    def _estimate_reserved_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> int:
        settings = get_settings()
        prompt_chars = len(system_prompt) + len(user_prompt)
        estimated_input = math.ceil(prompt_chars / settings.llm_chars_per_token_estimate)
        return estimated_input + max_output_tokens + settings.llm_rate_limit_safety_tokens

    def _wait_for_token_capacity(
        self,
        model: str,
        requested_tokens: int,
        *,
        max_wait_seconds: float | None = None,
    ) -> float:
        settings = get_settings()
        if settings.llm_base_url or not settings.llm_proactive_rate_limit_enabled:
            return 0.0

        key = model.casefold()
        waited = 0.0
        wait_cap = (
            settings.llm_rate_limit_max_wait_seconds
            if max_wait_seconds is None
            else min(settings.llm_rate_limit_max_wait_seconds, max_wait_seconds)
        )
        while True:
            wait_seconds = 0.0
            now = time.monotonic()
            with self._state_lock:
                state = self._rate_states.get(key)
                if state is None or state.remaining_tokens is None:
                    return waited
                if state.reset_at and now >= state.reset_at:
                    # x-ratelimit-reset-tokens indicates when token capacity has
                    # refreshed. Let the next response replace this optimistic
                    # local estimate with the authoritative header value.
                    state.remaining_tokens = state.limit_tokens
                    state.reset_at = 0.0
                if state.remaining_tokens is None:
                    return waited
                if requested_tokens <= state.remaining_tokens:
                    state.remaining_tokens = max(0, state.remaining_tokens - requested_tokens)
                    return waited
                if state.reset_at > now:
                    wait_seconds = state.reset_at - now + settings.llm_rate_limit_safety_seconds
                else:
                    # If the server did not provide a reset horizon, do not spin
                    # locally. Let the request/retry path obtain authoritative
                    # retry timing from a 429 response.
                    return waited

            remaining_wait_budget = wait_cap - waited
            if remaining_wait_budget <= 0:
                return waited
            wait_seconds = min(wait_seconds, remaining_wait_budget)
            if wait_seconds <= 0:
                return waited
            logger.info(
                "Pacing %s request for %.2fs to stay within the token rate limit",
                model,
                wait_seconds,
            )
            emit_progress(
                "rate_limit",
                "Waiting for API token capacity",
                f"Token window refresh in about {wait_seconds:.1f}s",
            )
            time.sleep(wait_seconds)
            waited += wait_seconds

    def _record_rate_limit_headers(self, model: str, headers: Mapping[str, str]) -> None:
        settings = get_settings()
        if settings.llm_base_url:
            return

        limit = _parse_int_header(headers, "x-ratelimit-limit-tokens")
        remaining = _parse_int_header(headers, "x-ratelimit-remaining-tokens")
        reset_seconds = _duration_seconds(_header(headers, "x-ratelimit-reset-tokens"))
        if limit is None and remaining is None and reset_seconds is None:
            return

        with self._state_lock:
            state = self._rate_states.setdefault(model.casefold(), _RateLimitState())
            if limit is not None:
                state.limit_tokens = limit
            if remaining is not None:
                state.remaining_tokens = remaining
            if reset_seconds is not None:
                state.reset_at = time.monotonic() + reset_seconds

        logger.debug(
            "Rate limit for %s: limit=%s remaining=%s reset=%s",
            model,
            limit,
            remaining,
            reset_seconds,
        )

    def _retry_delay(self, error: BaseException, attempt: int) -> float:
        settings = get_settings()
        exponential = min(
            settings.llm_retry_max_seconds,
            settings.llm_retry_base_seconds * (2**attempt),
        )
        server_delay = _server_retry_delay(error)
        delay = max(exponential, server_delay or 0.0)
        delay += settings.llm_rate_limit_safety_seconds + random.uniform(0.0, 0.20)
        cap = (
            settings.llm_rate_limit_max_wait_seconds
            if isinstance(error, RateLimitError)
            else settings.llm_retry_max_seconds
        )
        return min(delay, cap)

    def answer(self, user_prompt: str) -> str:
        settings = get_settings()
        return self.generate(
            SYSTEM_PROMPT,
            user_prompt,
            model=settings.llm_model,
            reasoning_effort=settings.llm_reasoning_effort,
        )

    def plan(self, system_prompt: str, user_prompt: str, *, max_output_tokens: int = 700) -> str:
        settings = get_settings()
        return self.generate(
            system_prompt,
            user_prompt,
            max_output_tokens=max_output_tokens,
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )

    def summarize(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        settings = get_settings()
        return self.generate(
            system_prompt,
            user_prompt,
            max_output_tokens=max_output_tokens or settings.summary_max_output_tokens,
            model=settings.summary_model,
            reasoning_effort=settings.summary_reasoning_effort,
        )


def _supports_reasoning(model: str) -> bool:
    lowered = model.casefold()
    return lowered.startswith(("gpt-5", "o1", "o3", "o4"))


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.title())
    return str(value).strip() if value is not None else None


def _parse_int_header(headers: Mapping[str, str], name: str) -> int | None:
    value = _header(headers, name)
    if not value:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip().casefold()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    total = 0.0
    matched = False
    for raw_amount, unit in _DURATION_PART_RE.findall(text):
        matched = True
        amount = float(raw_amount)
        if unit.casefold() == "ms":
            total += amount / 1000.0
        elif unit.casefold() == "s":
            total += amount
        elif unit.casefold() == "m":
            total += amount * 60.0
        elif unit.casefold() == "h":
            total += amount * 3600.0
    return total if matched else None


def _server_retry_delay(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = _header(headers, "retry-after")
        if retry_after:
            parsed = _duration_seconds(retry_after)
            if parsed is not None:
                return parsed
        reset = _duration_seconds(_header(headers, "x-ratelimit-reset-tokens"))
        if reset is not None:
            return reset

    match = _RETRY_MESSAGE_RE.search(str(error))
    if not match:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "s").casefold()
    return amount / 1000.0 if unit == "ms" else amount


class _NullLock:
    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


llm_service = LlmService()
