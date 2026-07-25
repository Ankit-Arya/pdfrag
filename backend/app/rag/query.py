from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import get_settings
from app.rag.llm import LlmConfigurationError, llm_service
from app.rag.prompts import QUERY_REWRITE_SYSTEM_PROMPT, build_query_rewrite_prompt
from app.rag.types import QueryPlan

logger = logging.getLogger(__name__)

# Terms such as FDS, OCC, HVAC, ATP, PEA, and TIMS are domain-dependent.
# They must not be expanded using general model knowledge.
_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,9}(?![A-Za-z0-9])")


class QueryPlanner:
    def plan(self, question: str, enabled: bool | None = None) -> QueryPlan:
        settings = get_settings()
        should_rewrite = settings.query_rewrite_enabled if enabled is None else enabled
        normalized = " ".join(question.split())

        fallback = QueryPlan(
            original_question=normalized,
            rewritten_question=normalized,
            search_queries=[normalized],
            keywords=_extract_acronyms(normalized),
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
            logger.warning(
                "Question rewrite unavailable; using the original question",
                exc_info=True,
            )
            return fallback
        except Exception:
            # Retrieval must still work when the optional rewrite call fails.
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


def _validate_plan(
    original: str,
    payload: dict[str, Any],
    max_variants: int,
) -> QueryPlan:
    acronyms = _extract_acronyms(original)
    candidate_rewrite = _clean_string(payload.get("rewritten_question")) or original

    # The rewrite model has no PDF context at this stage. It cannot safely infer
    # what a domain acronym means, so preserve the original interpreted question.
    if acronyms:
        if candidate_rewrite != original:
            logger.info(
                "Preserving original question because it contains domain acronyms: %s",
                ", ".join(acronyms),
            )
        rewritten = original
    else:
        rewritten = candidate_rewrite

    # Very long rewrites are likely to have changed intent.
    if len(rewritten) > max(600, len(original) * 4):
        rewritten = original

    raw_queries = payload.get("search_queries")
    model_queries = (
        [_clean_string(item) for item in raw_queries]
        if isinstance(raw_queries, list)
        else []
    )
    model_queries = [query for query in model_queries if query]

    candidate_queries = [original, rewritten, *model_queries]

    # When acronyms are present, discard model variants that removed them.
    # This removes incorrect variants such as "Fire Dynamics Simulator..." for FDS.
    if acronyms:
        candidate_queries = [
            query
            for query in candidate_queries
            if query == original or _contains_all_acronyms(query, acronyms)
        ]

    queries = _unique(candidate_queries)[:max_variants]

    raw_keywords = payload.get("keywords")
    model_keywords = (
        [_clean_string(item) for item in raw_keywords]
        if isinstance(raw_keywords, list)
        else []
    )
    keywords = _unique([*acronyms, *[item for item in model_keywords if item]])[:24]

    return QueryPlan(
        original_question=original,
        rewritten_question=rewritten,
        search_queries=queries or [original],
        keywords=keywords,
        used_ai_rewrite=rewritten != original or len(queries) > 1,
    )


def _extract_acronyms(value: str) -> list[str]:
    return _unique(_ACRONYM_PATTERN.findall(value))


def _contains_all_acronyms(value: str, required: list[str]) -> bool:
    present = set(_extract_acronyms(value))
    return all(acronym in present for acronym in required)


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
