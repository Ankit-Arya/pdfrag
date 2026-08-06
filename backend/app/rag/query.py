from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import get_settings
from app.rag.llm import LlmConfigurationError, llm_service
from app.rag.normalization import canonical_phrase, number_word_variant
from app.rag.prompts import QUERY_REWRITE_SYSTEM_PROMPT, build_query_rewrite_prompt
from app.rag.types import QueryPlan

logger = logging.getLogger(__name__)

# Terms such as FDS, OCC, HVAC, ATP, PEA, and TIMS are domain-dependent.
# They must not be expanded using general model knowledge.
_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,9}(?![A-Za-z0-9])")
_TERM_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_CONTEXT_PATTERN = re.compile(
    r"\b(?:class|coach|line|mode|model|phase|train|type|unit|variant|version)"
    r"\s+[A-Za-z0-9][A-Za-z0-9._/-]*\b",
    re.IGNORECASE,
)
_VALID_INTENTS = {
    "comparison",
    "definition",
    "fact_lookup",
    "list",
    "procedure",
    "requirement",
    "summary",
    "troubleshooting",
}
_FOCUS_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


class QueryPlanner:
    def plan(self, question: str, enabled: bool | None = None) -> QueryPlan:
        settings = get_settings()
        should_rewrite = settings.query_rewrite_enabled if enabled is None else enabled
        normalized = " ".join(question.split())
        fallback_intent = _infer_intent(normalized)

        fallback = QueryPlan(
            original_question=normalized,
            rewritten_question=normalized,
            search_queries=_unique([normalized, *_structural_queries(normalized, fallback_intent)])[
                : settings.query_rewrite_max_variants
            ],
            keywords=_extract_acronyms(normalized),
            intent=fallback_intent,
            response_mode=_infer_response_mode(normalized),
            focus_terms=_extract_focus_terms(normalized),
            context_terms=_extract_context_terms(normalized),
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
        [_clean_string(item) for item in raw_queries] if isinstance(raw_queries, list) else []
    )
    model_queries = [query for query in model_queries if query]

    raw_intent = _clean_string(payload.get("intent")).casefold()
    intent = raw_intent if raw_intent in _VALID_INTENTS else _infer_intent(original)
    structural_queries = _structural_queries(original, intent)
    candidate_queries = [original, rewritten, *structural_queries, *model_queries]

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
        [_clean_string(item) for item in raw_keywords] if isinstance(raw_keywords, list) else []
    )
    keywords = _unique([*acronyms, *[item for item in model_keywords if item]])[:24]

    heuristic_focus = _extract_focus_terms(original)
    model_focus = _validated_original_terms(payload.get("focus_terms"), original)
    focus_terms = _unique([*heuristic_focus, *model_focus])[:32]

    model_context = _validated_original_terms(payload.get("context_terms"), original)
    context_terms = _unique([*_extract_context_terms(original), *model_context])[:20]

    return QueryPlan(
        original_question=original,
        rewritten_question=rewritten,
        search_queries=queries or [original],
        keywords=keywords,
        intent=intent,
        response_mode=_infer_response_mode(original),
        focus_terms=focus_terms,
        context_terms=context_terms,
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


def _validated_original_terms(value: Any, original: str) -> list[str]:
    if not isinstance(value, list):
        return []
    original_terms = {token.casefold() for token in _TERM_PATTERN.findall(original)}
    result: list[str] = []
    for item in value:
        clean = _clean_string(item)
        item_terms = {token.casefold() for token in _TERM_PATTERN.findall(clean)}
        if clean and item_terms and item_terms.issubset(original_terms):
            result.append(clean)
    return _unique(result)


def _extract_focus_terms(value: str) -> list[str]:
    return _unique(
        [
            token
            for token in _TERM_PATTERN.findall(value)
            if len(token) > 1 and token.casefold() not in _FOCUS_STOPWORDS
        ]
    )


def _extract_context_terms(value: str) -> list[str]:
    codes = [
        token
        for token in _TERM_PATTERN.findall(value)
        if any(char.isdigit() for char in token)
        or (len(token) >= 2 and token.upper() == token and any(char.isalpha() for char in token))
    ]
    return _unique([*_ACRONYM_PATTERN.findall(value), *_CONTEXT_PATTERN.findall(value), *codes])


def _infer_intent(value: str) -> str:
    lowered = value.casefold()
    terms = {token.casefold() for token in _TERM_PATTERN.findall(value)}
    if {"compare", "comparison", "difference", "differences", "versus", "vs"} & terms:
        return "comparison"
    if {"troubleshoot", "troubleshooting", "fault", "failure", "error", "alarm"} & terms:
        return "troubleshooting"
    if (
        lowered.startswith("how ")
        or {"process", "procedure", "steps", "reset", "isolate", "operate"} & terms
    ):
        return "procedure"
    if {"required", "requirement", "requirements", "must", "shall", "prerequisite"} & terms:
        return "requirement"
    if lowered.startswith("what is ") or lowered.startswith("what does ") or "define" in terms:
        return "definition"
    if {"summarize", "summary", "overview"} & terms:
        return "summary"
    if lowered.startswith("list ") or {"list", "types"} & terms:
        return "list"
    return "fact_lookup"


def _structural_queries(value: str, intent: str) -> list[str]:
    """Add evidence-shape variants without changing the question being answered."""
    normalized = " ".join(value.split())
    lowered = normalized.casefold()
    variants: list[str] = []

    canonical = canonical_phrase(normalized)
    if canonical and canonical.casefold() != normalized.casefold():
        variants.append(canonical)
    number_variant = number_word_variant(normalized)
    if number_variant.casefold() != normalized.casefold():
        variants.append(number_variant)

    if re.search(r"\bprocess\b", lowered):
        subject = re.sub(r"\bprocess\b", "", normalized, flags=re.IGNORECASE)
        subject = " ".join(subject.split()).strip(" -:;,.?")
        if subject:
            variants.extend(
                [
                    f"{subject} test",
                    f"{subject} procedure steps checks examination",
                ]
            )
    elif intent == "procedure":
        subject = re.sub(r"\bprocedure\b", "", normalized, flags=re.IGNORECASE)
        subject = " ".join(subject.split()).strip(" -:;,.?")
        if subject:
            variants.append(f"{subject} steps instructions checks")

    terms = {token.casefold() for token in _TERM_PATTERN.findall(normalized)}
    if {"wake", "wake-up", "wakeup"} & terms:
        variants.append("wake up test examination train functions safety devices")

    return _unique(variants)


def _infer_response_mode(value: str) -> str:
    normalized = " ".join(value.split())
    lowered = normalized.casefold()
    terms = [token.casefold() for token in _TERM_PATTERN.findall(normalized)]
    question_starts = (
        "can ",
        "do ",
        "does ",
        "how ",
        "is ",
        "should ",
        "what ",
        "when ",
        "where ",
        "which ",
        "who ",
        "why ",
    )
    is_specific_question = normalized.endswith("?") or lowered.startswith(question_starts)
    if not is_specific_question and len(terms) <= 5:
        return "evidence"
    return "concise"


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
