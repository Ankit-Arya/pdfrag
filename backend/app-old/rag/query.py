from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import get_settings
from app.rag.llm import LlmConfigurationError, llm_service
from app.rag.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_query_rewrite_prompt,
)
from app.rag.types import QueryPlan

logger = logging.getLogger(__name__)


class QueryPlanner:
    def plan(self, question: str, enabled: bool | None = None) -> QueryPlan:
        settings = get_settings()
        should_rewrite = settings.query_rewrite_enabled if enabled is None else enabled
        normalized = " ".join(question.split())
        fallback = QueryPlan(
            original_question=normalized,
            rewritten_question=normalized,
            search_queries=[normalized],
            keywords=[],
            used_ai_rewrite=False,
        )
        if not should_rewrite:
            return fallback

        try:
            raw = llm_service.generate(
                QUERY_REWRITE_SYSTEM_PROMPT,
                build_query_rewrite_prompt(normalized, settings.query_rewrite_max_variants),
                max_output_tokens=400,
            )
            payload = _parse_json_object(raw)
            return _validate_plan(normalized, payload, settings.query_rewrite_max_variants)
        except (LlmConfigurationError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Question rewrite unavailable; using the original question", exc_info=True)
            return fallback
        except Exception:
            # Retrieval should still work if an optional rewrite call fails.
            logger.exception("Question rewrite failed; using the original question")
            return fallback


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Question rewrite did not return JSON")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Question rewrite JSON must be an object")
    return parsed


def _validate_plan(original: str, payload: dict[str, Any], max_variants: int) -> QueryPlan:
    rewritten = _clean_string(payload.get("rewritten_question")) or original
    raw_queries = payload.get("search_queries")
    queries = [_clean_string(item) for item in raw_queries] if isinstance(raw_queries, list) else []
    queries = [query for query in queries if query]
    queries = _unique([original, rewritten, *queries])[:max_variants]

    raw_keywords = payload.get("keywords")
    keywords = [_clean_string(item) for item in raw_keywords] if isinstance(raw_keywords, list) else []
    keywords = _unique([keyword for keyword in keywords if keyword])[:24]

    # A rewrite that is radically longer is more likely to have changed intent.
    if len(rewritten) > max(600, len(original) * 4):
        rewritten = original
    return QueryPlan(
        original_question=original,
        rewritten_question=rewritten,
        search_queries=queries or [original],
        keywords=keywords,
        used_ai_rewrite=rewritten != original or len(queries) > 1,
    )


def _clean_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:1000]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


query_planner = QueryPlanner()
