from __future__ import annotations

# ruff: noqa: E501

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.config import get_settings
from app.rag.llm import LlmConfigurationError, llm_service

_ALLOWED_INTENTS = {
    "comparison",
    "definition",
    "fact_lookup",
    "list",
    "procedure",
    "requirement",
    "summary",
    "troubleshooting",
}
_ALLOWED_ACTS = {"question", "request", "navigation", "source_paste", "evidence_correction"}
_ALLOWED_ROUTES = {"dedicated_procedure", "authoritative_rule", "structured_lookup", "broad_corpus"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_NEGATIVE_RE = re.compile(
    r"\b(?:do(?:es)?\s+not|doesn't|not\s+(?:state|specify|define|provide|mention|found)|"
    r"no\s+(?:information|amount|procedure|rule)|not\s+available|cannot\s+be\s+found)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SmartInterpretation:
    raw_question: str
    resolved_question: str
    intent: str = "fact_lookup"
    conversation_act: str = "question"
    concepts: tuple[str, ...] = ()
    evidence_needs: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    scope: dict[str, str] = field(default_factory=dict)
    material_ambiguity: bool = False
    ambiguity_note: str = ""
    uses_history: bool = False
    authority_sensitive: bool = False
    route_strategy: str = "broad_corpus"
    corrections: tuple[str, ...] = ()
    ai_used: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceReview:
    sufficient: bool
    missing_evidence: tuple[str, ...] = ()
    retry_queries: tuple[str, ...] = ()
    reason: str = ""
    ai_used: bool = False


_INTERPRET_SYSTEM = """You are the language-understanding and search-planning layer for a CLOSED-BOOK RAG assistant over official internal metro PDFs.

Your job is to understand what the user most likely means BEFORE document search. You are NOT answering the question and you are NOT a source of metro facts.

You MAY use general language ability to:
- silently correct spelling, grammar, inflection and word-order mistakes;
- understand colloquial wording, paraphrases, shorthand and ordinary synonyms;
- infer the user's conversational act and requested output;
- infer semantic concepts that should be searched for;
- use recent USER turns to resolve a genuine follow-up.

You MUST NOT:
- invent a metro procedure, rule, amount, speed, role assignment, document number, revision, line applicability or equipment behavior;
- invent an expansion for an internal acronym. Use an expansion only if supplied in DOCUMENT-GROUNDED TERMINOLOGY HINTS;
- assume two technical states are equivalent merely because they sound similar;
- silently choose between materially different operational scopes. Preserve the ambiguity when it could change the answer.

Treat the user's wording as noisy natural language, not a bag of search keywords. A phrase such as a misspelled or colloquial condition should be converted into a faithful, clear question and several useful semantic search formulations. Do not overfit to the literal words if their intended meaning is clear.

Return JSON only with exactly these keys:
- resolved_question: concise, self-contained statement of the user's most likely intended question;
- intent: one of fact_lookup, definition, procedure, troubleshooting, comparison, requirement, list, summary;
- conversation_act: one of question, request, navigation, source_paste, evidence_correction;
- concepts: 2-12 concise semantic concepts that relevant evidence should discuss;
- evidence_needs: 1-8 specific things the documents must establish to answer well. These are retrieval goals, not facts;
- search_queries: 2-5 diverse document-search queries. Include useful formal/colloquial synonyms and likely heading language, but do not invent internal codes or factual answers;
- scope: object containing only explicitly stated or safely resolved applicability constraints (for example line, rolling_stock, mode, location, person_type, equipment). Unknown values should be omitted;
- material_ambiguity: true only when unresolved ambiguity could materially change the answer;
- ambiguity_note: short explanation, or empty string;
- uses_history: true only if recent USER context is actually needed to understand this turn;
- authority_sensitive: true when current revision/amendment/authoritative version can change the requested answer;
- route_strategy: one of dedicated_procedure, authoritative_rule, structured_lookup, broad_corpus;
- corrections: short list of meaningful spelling/wording corrections made, excluding trivial grammar.

Special conversational behavior:
- If the user pastes a long passage apparently to correct or supply evidence, classify it as source_paste or evidence_correction; do not reinterpret an incidental uppercase word as an acronym-definition request.
- A bare acronym or short internal term should normally be treated as a definition request unless the user explicitly asks to find/search/show corpus occurrences or prior USER context clearly makes it a follow-up.
- If the user asks who is responsible, whether something is necessary/allowed, or what happens under a condition, understand the underlying responsibility/requirement/procedure even when they do not use words like 'procedure' or 'shall'.
- If a request asks for a value tied to a category, include both the category and the value dimension in evidence_needs.
- For an operational scenario, evidence_needs should include applicability/branch conditions when they matter.
"""

_EVIDENCE_REVIEW_SYSTEM = """You are an evidence-coverage critic for a CLOSED-BOOK RAG system.
You do not answer the user's question and you do not add factual knowledge.

Given the resolved question, evidence requirements, and a small set of retrieved PDF excerpts, decide whether the excerpts actually cover the requested answer. Semantic relevance alone is NOT enough.

Mark insufficient when, for example:
- the question asks who is responsible but the excerpts do not assign the responsibility;
- the question asks whether something is required but the excerpts only describe nearby equipment/location;
- the question asks for a current value but the evidence lacks the governing/current schedule or contains conflicting versions without authority;
- the question asks for a procedure but the excerpts contain only generic background or a different failure/scenario;
- important applicability/branch information requested by the interpretation is absent.

If insufficient, propose 1-3 targeted retry queries that search for the missing evidence using concepts or terminology already present in the resolved question, evidence requirements, terminology hints, or retrieved excerpts. Do not invent document codes, factual answers or unsupported acronym expansions.

Return JSON only with:
- sufficient: boolean
- missing_evidence: array of short strings
- retry_queries: array of 0-3 strings
- reason: short string
"""

_VERIFY_SYSTEM = """You are the final grounding editor for a CLOSED-BOOK internal metro PDF assistant.
Use ONLY the supplied source excerpts. The resolved interpretation tells you what the user meant, but it is not factual evidence.

Review the draft for semantic correctness, completeness for the requested intent, scope/applicability, internal consistency, current-authority handling and citation support.

Rules:
- Answer the RESOLVED QUESTION, not accidental spelling or malformed wording in the raw message.
- Never introduce a metro fact that is not supported by a supplied [S#] excerpt.
- Every factual sentence or bullet must cite one or more supplied [S#] labels.
- If the draft says the documents do not state/specify something but a supplied source actually answers it, remove the false negative and give the supported answer.
- Do not mix different line, rolling-stock, mode, person-type or scenario branches.
- If sources contain materially different answers and the evidence does not establish which scope/version controls, state the distinction instead of choosing.
- If an authority/amendment excerpt explicitly supersedes an older value or section, use the current text and do not present the superseded text as current.
- For a vague colloquial category that can map to multiple formal source categories, state the supported conditional match or alternatives rather than silently choosing one.
- Preserve valid source numbers; never create a source number that is not supplied.
- Keep a correct draft unchanged except for improvements needed by these rules.

Return only the final answer text, with citations.
"""


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 100) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _unique(values: Iterable[str], limit: int) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").split()).strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return tuple(result)


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI understanding layer did not return a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("AI understanding payload is not an object")
    return payload


def _string_list(value: object, *, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return _unique((str(item) for item in value if isinstance(item, (str, int, float))), limit)


def _scope(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = re.sub(r"[^a-z0-9_]+", "_", str(raw_key).casefold()).strip("_")[:40]
        val = " ".join(str(raw_value).split())[:160]
        if key and val and val.casefold() not in {"unknown", "none", "null", "n/a"}:
            result[key] = val
        if len(result) >= 12:
            break
    return result


def _fallback(question: str) -> SmartInterpretation:
    normalized = " ".join(question.split())
    act = "source_paste" if len(normalized) > 700 and normalized.count(" ") > 80 else "question"
    return SmartInterpretation(
        raw_question=normalized,
        resolved_question=normalized,
        intent="summary" if act == "source_paste" else "fact_lookup",
        conversation_act=act,
        concepts=_unique(_TOKEN_RE.findall(normalized), 10),
        evidence_needs=("evidence that directly answers the user's request",),
        search_queries=(normalized,),
        route_strategy="broad_corpus",
        ai_used=False,
    )


def _history_text(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "None"
    rows: list[str] = []
    for turn in history[-6:]:
        if str(turn.get("role", "")).casefold() != "user":
            continue
        content = " ".join(str(turn.get("content", "")).split())
        if content:
            rows.append(f"USER: {content[:1400]}")
    return "\n".join(rows) if rows else "None"


def _interpret_prompt(
    question: str,
    history: list[dict[str, str]] | None,
    abbreviation_hints: list[str] | None,
    routing_hints: list[str] | None,
) -> str:
    return f"""CURRENT USER MESSAGE:
{question[:7000]}

RECENT USER CONTEXT (intent resolution only; never factual evidence):
{_history_text(history)}

DOCUMENT-GROUNDED TERMINOLOGY HINTS:
{chr(10).join(f'- {value}' for value in (abbreviation_hints or [])[:20]) or 'None'}

PREVIOUS PDF-DERIVED ROUTING HINTS (navigation only; use only if this is truly a follow-up):
{chr(10).join(f'- {value}' for value in (routing_hints or [])[:12]) or 'None'}

Interpret the current message for robust retrieval. Do not answer it.
"""


def _parse_interpretation(question: str, payload: dict[str, Any]) -> SmartInterpretation:
    fallback = _fallback(question)
    resolved = " ".join(str(payload.get("resolved_question") or fallback.resolved_question).split())[:2000]
    intent = str(payload.get("intent") or fallback.intent).casefold().strip()
    if intent not in _ALLOWED_INTENTS:
        intent = fallback.intent
    act = str(payload.get("conversation_act") or "question").casefold().strip()
    if act not in _ALLOWED_ACTS:
        act = "question"
    route = str(payload.get("route_strategy") or "broad_corpus").casefold().strip()
    if route not in _ALLOWED_ROUTES:
        route = "broad_corpus"

    concepts = _string_list(payload.get("concepts"), limit=12)
    needs = _string_list(payload.get("evidence_needs"), limit=8)
    queries = _string_list(payload.get("search_queries"), limit=_int_env("SMART_RAG_AI_SEARCH_QUERIES", 4, 2, 5))
    # The resolved question is always the first semantic retrieval formulation.
    queries = _unique([resolved, *queries], _int_env("SMART_RAG_AI_SEARCH_QUERIES", 4, 2, 5))
    if not concepts:
        concepts = fallback.concepts
    if not needs:
        needs = fallback.evidence_needs

    return SmartInterpretation(
        raw_question=" ".join(question.split()),
        resolved_question=resolved or fallback.resolved_question,
        intent=intent,
        conversation_act=act,
        concepts=concepts,
        evidence_needs=needs,
        search_queries=queries or fallback.search_queries,
        scope=_scope(payload.get("scope")),
        material_ambiguity=bool(payload.get("material_ambiguity", False)),
        ambiguity_note=" ".join(str(payload.get("ambiguity_note") or "").split())[:500],
        uses_history=bool(payload.get("uses_history", False)),
        authority_sensitive=bool(payload.get("authority_sensitive", False)),
        route_strategy=route,
        corrections=_string_list(payload.get("corrections"), limit=8),
        ai_used=True,
    )


def interpret_user_message(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    abbreviation_hints: list[str] | None = None,
    routing_hints: list[str] | None = None,
) -> SmartInterpretation:
    if not _bool_env("SMART_RAG_AI_INTERPRETATION", True):
        return _fallback(question)
    settings = get_settings()
    try:
        raw = llm_service.generate(
            _INTERPRET_SYSTEM,
            _interpret_prompt(question, history, abbreviation_hints, routing_hints),
            max_output_tokens=_int_env("SMART_RAG_AI_INTERPRET_MAX_TOKENS", 950, 500, 1600),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        return _parse_interpretation(question, _json_object(raw))
    except (LlmConfigurationError, ValueError, TypeError, json.JSONDecodeError):
        return _fallback(question)
    except Exception:
        return _fallback(question)


def relevant_terminology_hints(
    interpretation: SmartInterpretation,
    hints: list[str] | None,
) -> list[str]:
    if not hints:
        return []
    haystack = " ".join(
        [interpretation.raw_question, interpretation.resolved_question, *interpretation.concepts]
    ).casefold()
    tokens = {token.casefold() for token in _TOKEN_RE.findall(haystack)}
    result: list[str] = []
    for hint in hints:
        alias = str(hint).split("=", 1)[0].strip()
        if alias.casefold() in tokens or alias.casefold() in haystack:
            result.append(hint)
    return result[:20]


def should_review_evidence(interpretation: SmartInterpretation) -> bool:
    if not _bool_env("SMART_RAG_AI_EVIDENCE_REVIEW", True):
        return False
    if interpretation.conversation_act == "navigation" or interpretation.intent == "definition":
        return False
    return bool(
        interpretation.authority_sensitive
        or interpretation.material_ambiguity
        or interpretation.intent in {"procedure", "requirement", "troubleshooting", "comparison", "list"}
        or len(interpretation.evidence_needs) >= 2
    )


def _evidence_prompt(interpretation: SmartInterpretation, results: list[object]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(results[:12], 1):
        chunk = getattr(item, "chunk", None)
        if chunk is None:
            continue
        text_value = " ".join(str(getattr(chunk, "text", "")).split())[:900]
        blocks.append(
            f"E{index} | {getattr(chunk, 'filename', '')} | page {getattr(chunk, 'page_number', '')} | "
            f"method={getattr(item, 'method', '')}\n{text_value}"
        )
    return f"""RAW USER MESSAGE:
{interpretation.raw_question}

RESOLVED QUESTION:
{interpretation.resolved_question}

EVIDENCE REQUIREMENTS:
{chr(10).join(f'- {value}' for value in interpretation.evidence_needs)}

MATERIAL AMBIGUITY:
{interpretation.ambiguity_note if interpretation.material_ambiguity else 'None'}

RETRIEVED EXCERPTS:
{chr(10).join(blocks) if blocks else 'None'}

Judge coverage only. Do not answer the question.
"""


def review_retrieved_evidence(
    interpretation: SmartInterpretation,
    results: list[object],
) -> EvidenceReview:
    if not should_review_evidence(interpretation):
        return EvidenceReview(sufficient=True, ai_used=False)
    settings = get_settings()
    try:
        raw = llm_service.generate(
            _EVIDENCE_REVIEW_SYSTEM,
            _evidence_prompt(interpretation, results),
            max_output_tokens=_int_env("SMART_RAG_AI_REVIEW_MAX_TOKENS", 500, 250, 900),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        payload = _json_object(raw)
        return EvidenceReview(
            sufficient=bool(payload.get("sufficient", False)),
            missing_evidence=_string_list(payload.get("missing_evidence"), limit=8),
            retry_queries=_string_list(payload.get("retry_queries"), limit=_int_env("SMART_RAG_AI_RETRY_QUERIES", 2, 1, 3)),
            reason=" ".join(str(payload.get("reason") or "").split())[:600],
            ai_used=True,
        )
    except Exception:
        # Failure of the critic must never block the existing deterministic RAG path.
        return EvidenceReview(sufficient=True, ai_used=False)


def should_verify_answer(interpretation: SmartInterpretation, draft: str) -> bool:
    if not _bool_env("SMART_RAG_AI_ANSWER_VERIFY", True):
        return False
    if interpretation.conversation_act == "navigation" or interpretation.intent == "definition":
        return False
    return bool(
        _NEGATIVE_RE.search(draft or "")
        or interpretation.authority_sensitive
        or interpretation.material_ambiguity
        or interpretation.intent in {"procedure", "requirement", "troubleshooting"}
    )


def verify_answer(
    interpretation: SmartInterpretation,
    draft: str,
    sources: list[object],
) -> str:
    if not draft or not should_verify_answer(interpretation, draft) or not sources:
        return draft
    settings = get_settings()
    source_blocks: list[str] = []
    max_sources = _int_env("SMART_RAG_AI_VERIFY_SOURCES", 24, 8, 40)
    cited = [int(value) for value in re.findall(r"\[S(\d+)\]", draft or "") if value.isdigit()]
    wanted: list[int] = []
    # Always keep the sources the draft actually relied on, then add the strongest
    # nearby evidence so a false negative can be corrected without shipping all 48
    # excerpts through another model call. Source numbers remain the original ones.
    for number in [*cited, *range(1, min(len(sources), max_sources) + 1)]:
        if 1 <= number <= len(sources) and number not in wanted:
            wanted.append(number)
        if len(wanted) >= max_sources:
            break
    for number in wanted:
        source = sources[number - 1]
        result = getattr(source, "result", None)
        chunk = getattr(result, "chunk", None)
        excerpt = str(getattr(source, "excerpt", "") or "")
        if chunk is None:
            continue
        source_blocks.append(
            f"[S{number}] File: {getattr(chunk, 'filename', '')} | Page: {getattr(chunk, 'page_number', '')} | "
            f"Retrieval: {getattr(result, 'method', '')}\n{excerpt[:1900]}"
        )
    prompt = f"""RAW USER MESSAGE:
{interpretation.raw_question}

RESOLVED QUESTION:
{interpretation.resolved_question}

REQUESTED EVIDENCE COVERAGE:
{chr(10).join(f'- {value}' for value in interpretation.evidence_needs)}

SCOPE:
{json.dumps(interpretation.scope, ensure_ascii=False)}

MATERIAL AMBIGUITY:
{interpretation.ambiguity_note if interpretation.material_ambiguity else 'None'}

DRAFT ANSWER:
{draft}

SUPPLIED PDF SOURCES:
{chr(10).join(source_blocks)}

Return the corrected final answer only.
"""
    try:
        corrected = llm_service.generate(
            _VERIFY_SYSTEM,
            prompt,
            max_output_tokens=_int_env("SMART_RAG_AI_VERIFY_MAX_TOKENS", 1800, 700, 2800),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        ).strip()
        return corrected or draft
    except Exception:
        return draft
