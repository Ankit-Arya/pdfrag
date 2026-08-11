from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from app.config import get_settings
from app.rag.llm import LlmConfigurationError, llm_service
from app.rag.normalization import canonical_phrase, number_word_variant
from app.rag.types import QueryPlan

logger = logging.getLogger(__name__)

_TERM_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._/:#-][A-Za-z0-9]+)*")
_UPPER_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,9}(?![A-Za-z0-9])")
_CONTEXT_PATTERN = re.compile(
    r"\b(?:class|coach|line|mode|model|phase|train|type|unit|variant|version)"
    r"\s+[A-Za-z0-9][A-Za-z0-9._/-]*\b",
    re.IGNORECASE,
)
_HINT_TOKEN_PATTERN = re.compile(r"^([A-Z][A-Z0-9/-]{1,9})\b")
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
_QUESTION_STARTS = (
    "can ",
    "could ",
    "do ",
    "does ",
    "explain ",
    "how ",
    "is ",
    "should ",
    "tell me ",
    "what ",
    "when ",
    "where ",
    "which ",
    "who ",
    "why ",
)
_REFERENCE_CUES = {
    "document",
    "documents",
    "docs",
    "find",
    "mention",
    "mentioned",
    "mentions",
    "occurrence",
    "occurrences",
    "reference",
    "references",
    "search",
}
_ANSWER_CUES = {
    "allowed",
    "amount",
    "capacity",
    "condition",
    "conditions",
    "date",
    "distance",
    "duration",
    "explain",
    "frequency",
    "height",
    "information",
    "info",
    "instruction",
    "instructions",
    "length",
    "limit",
    "limits",
    "maximum",
    "minimum",
    "number",
    "policy",
    "pressure",
    "procedure",
    "prohibited",
    "requirement",
    "requirements",
    "rule",
    "rules",
    "speed",
    "step",
    "steps",
    "temperature",
    "time",
    "timing",
    "value",
    "voltage",
    "weight",
}
_FOLLOWUP_STARTS = (
    "and ",
    "also ",
    "how about ",
    "same for ",
    "what about ",
    "what if ",
)
_REQUEST_VERBS = {
    "clarify",
    "describe",
    "explain",
    "give",
    "list",
    "provide",
    "show",
    "summarize",
    "tell",
}
_CONTEXT_DEPENDENT_WORDS = {
    "above",
    "applicable",
    "detail",
    "details",
    "general",
    "more",
    "previous",
    "procedure",
    "same",
    "then",
}
_FOCUS_STOPWORDS = {
    "a",
    "about",
    "all",
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
    "provide",
    "applicable",
    "general",
    "then",
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

_QUERY_CONTEXT_SYSTEM_PROMPT = """You plan retrieval for official internal metro documents.
Do not answer the user's question.

You may receive recent conversation turns and document-grounded abbreviation hints.
Conversation turns are ONLY for resolving what the current user means (topic, referent,
line, train, system, procedure, comparison target, etc.). Never treat an earlier assistant
answer as factual evidence. Facts must later be re-retrieved from the PDFs.

Abbreviation hints come from the indexed PDFs. You may use a stated expansion only when it
appears in those hints. Never invent or expand an internal abbreviation from general knowledge.
Preserve identifiers, codes, line numbers, train types, equipment names and acronyms.

Return JSON only with these keys:
- contextual_question: a self-contained version of the current question using only context
  actually present in the current question, earlier USER turns, or abbreviation hints;
- search_queries: 1 to 6 semantically equivalent retrieval queries;
- keywords: important exact terms and internal abbreviations;
- intent: one of fact_lookup, definition, procedure, troubleshooting, comparison,
  requirement, list, summary;
- focus_terms: subject terms evidence must discuss;
- context_terms: explicit applicability constraints such as line, mode, rolling stock,
  equipment, procedure, location or internal code;
- search_mode: either references or answer.

Use search_mode=references for a bare concept/keyword lookup where the user primarily wants
where it appears in the corpus. Use search_mode=answer for a question, request for rules/info,
procedure, explanation, comparison, requirement or other synthesized answer.
"""


class QueryPlanner:
    def __init__(self) -> None:
        self._cache: dict[tuple[object, ...], QueryPlan] = {}
        self._cache_lock = threading.Lock()
        self._cache_limit = 512

    def plan(
        self,
        question: str,
        enabled: bool | None = None,
        *,
        conversation_context: list[dict[str, str]] | None = None,
        abbreviation_hints: list[str] | None = None,
        routing_hints: list[str] | None = None,
    ) -> QueryPlan:
        settings = get_settings()
        normalized = " ".join(question.split())
        history = conversation_context or []
        hints = abbreviation_hints or []
        route_hints = routing_hints or []
        fallback_context = _fallback_contextual_question(normalized, history)
        fallback_intent = _infer_intent(fallback_context)
        fallback_mode = _infer_search_mode(normalized, fallback_context)
        fallback_queries = _deterministic_queries(
            normalized,
            fallback_context,
            fallback_intent,
            hints,
            route_hints,
        )[: settings.query_rewrite_max_variants]
        fallback = QueryPlan(
            original_question=normalized,
            rewritten_question=fallback_context,
            contextual_question=fallback_context,
            search_queries=fallback_queries or [normalized],
            keywords=_unique([*_protected_tokens(normalized, hints), *route_hints])[:32],
            intent=fallback_intent,
            response_mode="evidence" if fallback_mode == "references" else "concise",
            search_mode=fallback_mode,
            focus_terms=_extract_focus_terms(fallback_context),
            context_terms=_unique([*_extract_context_terms(fallback_context, hints), *route_hints])[:32],
            abbreviation_hints=hints,
            routing_hints=route_hints,
            used_ai_rewrite=False,
        )

        should_rewrite = settings.query_rewrite_enabled if enabled is None else enabled
        cache_key = (
            normalized.casefold(),
            fallback_context.casefold(),
            tuple(hint.casefold() for hint in hints),
            tuple(hint.casefold() for hint in route_hints),
            bool(should_rewrite),
            settings.query_model,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if not should_rewrite:
            self._cache_put(cache_key, fallback)
            return fallback

        try:
            raw = llm_service.plan(
                _QUERY_CONTEXT_SYSTEM_PROMPT,
                _build_context_prompt(
                    normalized,
                    history,
                    hints,
                    route_hints,
                    settings.query_rewrite_max_variants,
                ),
                max_output_tokens=700,
            )
            payload = _parse_json_object(raw)
            planned = _validate_plan(
                normalized,
                history,
                hints,
                route_hints,
                payload,
                settings.query_rewrite_max_variants,
                fallback,
            )
            self._cache_put(cache_key, planned)
            return planned
        except (LlmConfigurationError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Contextual query planning unavailable; using deterministic plan", exc_info=True)
            self._cache_put(cache_key, fallback)
            return fallback
        except Exception:
            logger.exception("Contextual query planning failed; using deterministic plan")
            self._cache_put(cache_key, fallback)
            return fallback

    def _cache_get(self, key: tuple[object, ...]) -> QueryPlan | None:
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_put(self, key: tuple[object, ...], plan: QueryPlan) -> None:
        with self._cache_lock:
            if len(self._cache) >= self._cache_limit:
                oldest = next(iter(self._cache))
                self._cache.pop(oldest, None)
            self._cache[key] = plan


def _build_context_prompt(
    question: str,
    history: list[dict[str, str]],
    hints: list[str],
    routing_hints: list[str],
    max_variants: int,
) -> str:
    history_lines: list[str] = []
    context_hints: list[str] = []
    for turn in history:
        role = str(turn.get("role", "")).strip().upper()
        content = " ".join(str(turn.get("content", "")).split())
        # Earlier assistant prose is deliberately excluded. It may be wrong or
        # stale. Only prior USER wording plus validated intent/routing metadata
        # can resolve a follow-up.
        if role == "USER" and content:
            history_lines.append(f"USER: {content}")
        context_hint = " ".join(str(turn.get("context_hint", "")).split())
        if context_hint:
            context_hints.append(context_hint)

    return f"""CURRENT USER QUESTION:
{question}

RECENT USER TURN (intent context only, never factual evidence):
{chr(10).join(history_lines) if history_lines else "None"}

PREVIOUS VALIDATED SELF-CONTAINED QUESTION (intent only):
{context_hints[-1] if context_hints else "None"}

PDF-DERIVED DOCUMENT ROUTING HINTS FROM THE PREVIOUS TURN:
{chr(10).join(f"- {hint}" for hint in routing_hints) if routing_hints else "None"}

DOCUMENT-GROUNDED ABBREVIATION/USAGE HINTS:
{chr(10).join(f"- {hint}" for hint in hints) if hints else "None"}

Create at most {max_variants} retrieval queries. The contextual question must remain faithful
to the current user request. If it is a follow-up, make it self-contained using earlier USER
turns and, when useful, the PDF-derived routing hints. A routing hint may identify a document
that must be searched again; it is not itself factual evidence. Keep a complete new topic
isolated from previous turns.
"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Question planner did not return JSON")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Question planner JSON must be an object")
    return parsed


def _validate_plan(
    original: str,
    history: list[dict[str, str]],
    hints: list[str],
    routing_hints: list[str],
    payload: dict[str, Any],
    max_variants: int,
    fallback: QueryPlan,
) -> QueryPlan:
    user_history = [
        str(turn.get("content", ""))
        for turn in history
        if str(turn.get("role", "")).casefold() == "user"
    ]
    resolved_context_hints = [
        str(turn.get("context_hint", ""))
        for turn in history
        if str(turn.get("context_hint", "")).strip()
    ]
    # Only pull earlier user topics into a question when the current turn actually
    # looks like a follow-up/underspecified request. A new bare lookup such as
    # "alcohol" must stay broad even if the previous chat was about Line 8.
    needs_history = needs_conversation_context(original)
    context_history = user_history if needs_history else []
    context_hint_history = resolved_context_hints[-1:] if needs_history else []
    active_routing_hints = routing_hints if needs_history else []
    allowed_parts = [
        original,
        *context_history,
        *context_hint_history,
        *hints,
        *active_routing_hints,
    ]
    allowed_text = " ".join(allowed_parts)
    allowed_terms = {token.casefold() for token in _TERM_PATTERN.findall(allowed_text)}
    for part in allowed_parts:
        for variant in _spelling_variants(part):
            allowed_terms.update(token.casefold() for token in _TERM_PATTERN.findall(variant))

    candidate_context = _clean_string(payload.get("contextual_question"))
    contextual = (
        candidate_context
        if _context_is_safe(candidate_context, allowed_terms, original)
        else fallback.contextual_question
    )

    heuristic_intent = _infer_intent(contextual)
    raw_intent = _clean_string(payload.get("intent")).casefold()
    intent = raw_intent if raw_intent in _VALID_INTENTS else heuristic_intent
    # Explicit procedure/question cues in the current text are more reliable than
    # model classification and should not vary between otherwise identical calls.
    if heuristic_intent != "fact_lookup":
        intent = heuristic_intent

    search_mode = _infer_search_mode(original, contextual)
    protected = _protected_tokens(original, hints)

    raw_queries = payload.get("search_queries")
    model_queries = (
        [_clean_string(item) for item in raw_queries]
        if isinstance(raw_queries, list)
        else []
    )
    model_queries = [
        query
        for query in model_queries
        if query
        and _context_is_safe(query, allowed_terms, original, allow_search_words=True)
        and _contains_protected_tokens(query, protected)
    ]

    deterministic = _deterministic_queries(
        original, contextual, intent, hints, active_routing_hints
    )
    queries = _unique([*deterministic, *model_queries])[:max_variants]

    raw_keywords = payload.get("keywords")
    model_keywords = (
        [_clean_string(item) for item in raw_keywords]
        if isinstance(raw_keywords, list)
        else []
    )
    model_keywords = [
        item for item in model_keywords if _terms_subset(item, allowed_terms)
    ]

    raw_focus = payload.get("focus_terms")
    model_focus = _validated_terms(raw_focus, allowed_terms)
    raw_context = payload.get("context_terms")
    model_context = _validated_terms(raw_context, allowed_terms)

    focus_terms = _unique([*_extract_focus_terms(contextual), *model_focus])[:48]
    context_terms = _unique(
        [*_extract_context_terms(contextual, hints), *active_routing_hints, *model_context]
    )[:32]

    return QueryPlan(
        original_question=original,
        rewritten_question=contextual,
        contextual_question=contextual,
        search_queries=queries or [original],
        keywords=_unique([*protected, *active_routing_hints, *model_keywords])[:32],
        intent=intent,
        response_mode="evidence" if search_mode == "references" else "concise",
        search_mode=search_mode,
        focus_terms=focus_terms,
        context_terms=context_terms,
        abbreviation_hints=hints,
        routing_hints=active_routing_hints,
        used_ai_rewrite=contextual != original or any(q not in deterministic for q in queries),
    )


def _fallback_contextual_question(
    question: str,
    history: list[dict[str, str]],
) -> str:
    if not needs_conversation_context(question):
        return question

    for turn in reversed(history):
        context_hint = " ".join(str(turn.get("context_hint", "")).split())
        if context_hint and context_hint.casefold() != question.casefold():
            return f"{context_hint}. {question}"

    for turn in reversed(history):
        if str(turn.get("role", "")).casefold() != "user":
            continue
        previous = " ".join(str(turn.get("content", "")).split())
        if not previous or previous.casefold() == question.casefold():
            continue
        return f"{previous}. {question}"
    return question



def needs_conversation_context(question: str) -> bool:
    """Return True only when the current turn lacks a self-contained subject.

    Complete new topics must never inherit old line/train/location constraints.
    Subject-less directives such as "provide the applicable procedure" are
    follow-ups even though they do not start with a question word.
    """
    lowered = question.casefold().strip()
    terms = [token.casefold() for token in _TERM_PATTERN.findall(question)]
    term_set = set(terms)
    if lowered.startswith(_FOLLOWUP_STARTS):
        return True
    if {"it", "that", "this", "same", "those", "them", "they", "there", "these"} & term_set:
        return True

    content_terms = [
        term
        for term in terms
        if term not in _FOCUS_STOPWORDS
        and term not in _ANSWER_CUES
        and term not in _REQUEST_VERBS
        and term not in _CONTEXT_DEPENDENT_WORDS
        and term not in {"tell", "regarding"}
    ]

    # Short question forms such as "what are the rules?" have no subject.
    if lowered.startswith(_QUESTION_STARTS) and len(terms) <= 8:
        return not content_terms

    # Imperative follow-ups are common in chat: "provide the procedure",
    # "show applicable details", "give more". Keep them attached to the most
    # recent user topic when no concrete subject remains after removing request
    # boilerplate.
    if terms and terms[0] in _REQUEST_VERBS and len(terms) <= 9:
        return not content_terms

    return False


def _infer_search_mode(original: str, contextual: str) -> str:
    lowered = original.casefold().strip()
    terms = {token.casefold() for token in _TERM_PATTERN.findall(original)}
    if original.rstrip().endswith("?") or lowered.startswith(_QUESTION_STARTS):
        return "answer"
    if terms & _ANSWER_CUES:
        # A fact dimension needs a subject to become a synthesized answer. A lone
        # keyword such as "speed" or "date" is still a broad reference lookup.
        subject_terms = {
            term
            for term in terms
            if term not in _ANSWER_CUES and term not in _FOCUS_STOPWORDS
        }
        if subject_terms:
            return "answer"
    if terms & _REFERENCE_CUES:
        return "references"
    # A bare concept or short noun phrase with no answer dimension is treated as
    # corpus/reference discovery. Fact dimensions such as speed/date/limit are
    # included in _ANSWER_CUES above, so "speed of pilot train on AEL" is an answer.
    contextual_terms = [
        token
        for token in _TERM_PATTERN.findall(contextual)
        if token.casefold() not in _FOCUS_STOPWORDS
    ]
    return "references" if len(contextual_terms) <= 5 else "answer"


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
    if {"required", "requirement", "requirements", "must", "shall", "prerequisite", "rules", "rule"} & terms:
        return "requirement"
    if lowered.startswith("what is ") or lowered.startswith("what does ") or "define" in terms:
        return "definition"
    if {"summarize", "summary", "overview"} & terms:
        return "summary"
    if lowered.startswith("list ") or {"list", "types"} & terms:
        return "list"
    return "fact_lookup"


def _deterministic_queries(
    original: str,
    contextual: str,
    intent: str,
    hints: list[str],
    routing_hints: list[str] | None = None,
) -> list[str]:
    variants = [original]
    if contextual.casefold() != original.casefold():
        variants.append(contextual)

    for value in [contextual, original]:
        variants.extend(_spelling_variants(value))
        variants.extend(_structural_queries(value, intent))

    variants.extend(_abbreviation_queries(contextual, hints))
    for routing_hint in routing_hints or []:
        clean = _clean_string(routing_hint)
        if clean:
            variants.append(f"{contextual} {clean}")
    return _unique(variants)


def _spelling_variants(value: str) -> list[str]:
    variants: list[str] = []
    if re.search(r"\bunder\s+ground\b", value, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bunder\s+ground\b", "underground", value, flags=re.IGNORECASE))
    if re.search(r"\bplatform\s+screen\s+door(s)?\b", value, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bplatform\s+screen\s+door(s)?\b", "PSD", value, flags=re.IGNORECASE))
    return variants


def _structural_queries(value: str, intent: str) -> list[str]:
    normalized = " ".join(value.split())
    lowered = normalized.casefold()
    variants: list[str] = []

    canonical = canonical_phrase(normalized)
    if canonical and canonical.casefold() != normalized.casefold():
        variants.append(canonical)
    number_variant = number_word_variant(normalized)
    if number_variant.casefold() != normalized.casefold():
        variants.append(number_variant)

    if intent == "procedure":
        variants.append(f"{normalized} procedure steps instructions checks prerequisites authorization")
    elif intent == "requirement":
        variants.append(f"{normalized} rules requirements shall must prohibited permitted conditions")
    elif intent == "troubleshooting":
        variants.append(f"{normalized} fault cause check remedy action")
    return _unique(variants)


def _abbreviation_queries(value: str, hints: list[str]) -> list[str]:
    queries: list[str] = []
    lowered = value.casefold()
    for hint in hints:
        match = _HINT_TOKEN_PATTERN.match(hint)
        if not match:
            continue
        token = match.group(1)
        if token.casefold() not in {t.casefold() for t in _TERM_PATTERN.findall(value)}:
            continue
        definition_match = re.search(r"\b=\s*([^—|]+)", hint)
        if definition_match:
            expansion = definition_match.group(1).strip()
            if expansion:
                queries.append(f"{value} {token} {expansion}")
        elif token.casefold() in lowered:
            queries.append(re.sub(rf"\b{re.escape(token)}\b", token, value, flags=re.IGNORECASE))
    return queries


def _protected_tokens(original: str, hints: list[str]) -> list[str]:
    original_terms = {token.casefold() for token in _TERM_PATTERN.findall(original)}
    tokens = list(_UPPER_ACRONYM_PATTERN.findall(original))
    for hint in hints:
        match = _HINT_TOKEN_PATTERN.match(hint)
        if match and match.group(1).casefold() in original_terms:
            tokens.append(match.group(1))
    return _unique(tokens)


def _contains_protected_tokens(value: str, protected: list[str]) -> bool:
    if not protected:
        return True
    present = {token.casefold() for token in _TERM_PATTERN.findall(value)}
    return all(token.casefold() in present for token in protected)


def _extract_focus_terms(value: str) -> list[str]:
    return _unique(
        [
            token
            for token in _TERM_PATTERN.findall(value)
            if len(token) > 1 and token.casefold() not in _FOCUS_STOPWORDS
        ]
    )


def _extract_context_terms(value: str, hints: list[str]) -> list[str]:
    hint_tokens = {
        match.group(1).casefold(): match.group(1)
        for hint in hints
        if (match := _HINT_TOKEN_PATTERN.match(hint))
    }
    codes: list[str] = []
    for token in _TERM_PATTERN.findall(value):
        lowered = token.casefold()
        if (
            any(char.isdigit() for char in token)
            or (len(token) >= 2 and token.upper() == token and any(char.isalpha() for char in token))
            or lowered in hint_tokens
        ):
            codes.append(hint_tokens.get(lowered, token))
    return _unique([*_CONTEXT_PATTERN.findall(value), *codes])


def _context_is_safe(
    value: str,
    allowed_terms: set[str],
    original: str,
    *,
    allow_search_words: bool = False,
) -> bool:
    if not value:
        return False
    if len(value) > max(900, len(original) * 6):
        return False
    terms = {token.casefold() for token in _TERM_PATTERN.findall(value)}
    if allow_search_words:
        allowed_terms = allowed_terms | {
            "action",
            "authorization",
            "check",
            "checks",
            "condition",
            "conditions",
            "instruction",
            "instructions",
            "must",
            "permitted",
            "prerequisite",
            "prerequisites",
            "procedure",
            "prohibited",
            "requirement",
            "requirements",
            "rules",
            "shall",
            "steps",
        }
    return all(term in allowed_terms or term in _FOCUS_STOPWORDS for term in terms)


def _terms_subset(value: str, allowed_terms: set[str]) -> bool:
    terms = {token.casefold() for token in _TERM_PATTERN.findall(value)}
    return bool(terms) and terms.issubset(allowed_terms)


def _validated_terms(value: Any, allowed_terms: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        clean = _clean_string(item)
        if clean and _terms_subset(clean, allowed_terms):
            result.append(clean)
    return _unique(result)


def _clean_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:1200]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = " ".join(str(value).split())
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


query_planner = QueryPlanner()
