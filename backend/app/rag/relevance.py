from __future__ import annotations

import re
from dataclasses import dataclass, replace

from app.config import get_settings
from app.rag.normalization import search_terms
from app.rag.structure import (
    major_section_match_score,
    section_match_score,
    section_path_from_text,
)
from app.rag.types import PrimaryDocumentMatch, QueryPlan, RetrievedChunk, TextChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,80})[\"']")
_SECTION_RE = re.compile(r"^Section path:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_CONTEXT_LINE_RE = re.compile(
    r"^(?:Rolling stock / train context|Procedure context):\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_LINE_SCOPE_RE = re.compile(
    r"\b(?:in\s+|for\s+|applicable(?:\s+to|\s+for)?\s+)?"
    r"lines?\s*[-\u2013\u2014:]?\s*"
    r"(?P<values>(?:\d{1,2}|AEL)(?:\s*(?:,|&|and|/)\s*(?:\d{1,2}|AEL))*)",
    re.IGNORECASE,
)
_DOCUMENT_CODE_RE = re.compile(
    r"\b(?:SC|SM|SOP|JPO|INST(?:RUCTION)?|MRGR)\s*[-_/]?\s*\d+[A-Z]?\b",
    re.IGNORECASE,
)
_ROLE_ACRONYMS = {
    "to", "tc", "occ", "sc", "ra", "eto", "ic", "tpc", "rsc", "sgc",
    "chc", "dcc", "sm",
}

_STOPWORDS = {
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

_INTENT_CUES = {
    "procedure": {
        "action",
        "after",
        "announce",
        "assist",
        "authorization",
        "before",
        "check",
        "confirm",
        "ensure",
        "hand",
        "hold",
        "inform",
        "instruction",
        "isolate",
        "permission",
        "procedure",
        "regulate",
        "remove",
        "report",
        "reset",
        "restore",
        "resume",
        "step",
        "stop",
        "suspend",
        "verify",
        "warning",
    },
    "troubleshooting": {
        "alarm",
        "cause",
        "check",
        "diagnosis",
        "error",
        "failure",
        "fault",
        "remedy",
        "symptom",
        "troubleshooting",
    },
    "comparison": {
        "advantage",
        "compare",
        "difference",
        "disadvantage",
        "versus",
    },
    "requirement": {
        "mandatory",
        "must",
        "permission",
        "prerequisite",
        "prohibited",
        "requirement",
        "required",
        "rule",
        "shall",
    },
    "definition": {"definition", "means", "refers", "term"},
    "summary": {"conclusion", "overview", "scope", "summary"},
    "list": {
        "categories",
        "category",
        "class",
        "classes",
        "carry",
        "carried",
        "contain",
        "contents",
        "equipment",
        "following",
        "include",
        "item",
        "items",
        "keep",
        "kept",
        "kind",
        "kinds",
        "list",
        "mode",
        "modes",
        "namely",
        "possession",
        "required",
        "type",
        "types",
    },
    "fact_lookup": set(),
}


@dataclass(frozen=True, slots=True)
class _ScoredChunk:
    result: RetrievedChunk
    relevance: float
    coverage: float
    anchor_coverage: float
    direct_anchor_coverage: float
    mandatory_anchor_coverage: float
    intent_evidence: float
    section_match: float
    major_section_match: float
    context_key: tuple[str, str]


def select_context_chunks(
    plan: QueryPlan,
    candidates: list[RetrievedChunk],
    max_chunks: int | None = None,
    *,
    preferred_document_ids: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Keep all materially relevant evidence instead of treating top-K as truth."""
    if not candidates:
        return []

    focus_terms = _focus_terms(plan)
    anchor_groups = _anchor_groups(plan)
    mandatory_groups = _mandatory_anchor_groups(plan)
    preferred_ids = preferred_document_ids or set()
    inherited_scopes = _inherited_line_scope_terms(candidates)
    local_anchor_terms = _local_anchor_terms_by_chunk(candidates, inherited_scopes)
    scored = [
        _score_candidate(
            plan,
            candidate,
            focus_terms,
            anchor_groups,
            mandatory_groups,
            preferred_ids,
            inherited_scopes.get(candidate.chunk.chunk_id, set()),
            local_anchor_terms.get(candidate.chunk.chunk_id, set()),
        )
        for candidate in candidates
    ]
    scored.sort(key=_scored_sort_key)

    major_matches = [item for item in scored if item.major_section_match >= 0.9]
    if plan.search_mode == "references" and major_matches:
        scored = major_matches

    best = scored[0].relevance
    if best < 0.08:
        return []

    if plan.search_mode == "references":
        eligible = [
            item
            for item in scored
            if item.coverage > 0
            or item.section_match >= 0.70
            or item.result.keyword_score > 0
            or "corpus-fts" in item.result.method
        ]
    else:
        cutoff = max(0.10, best * 0.32)
        eligible = []
        for item in scored:
            preferred = _is_preferred(item.result.chunk, preferred_ids)
            grounded_alias = "grounded-line-alias" in item.result.method
            has_focus = (
                item.coverage >= 0.25
                or item.section_match >= 0.82
                or not focus_terms
                or grounded_alias
                or (preferred and plan.intent == "procedure" and item.intent_evidence >= 0.05)
            )
            required_anchor_coverage = _required_anchor_coverage(plan, anchor_groups)
            has_anchor = item.anchor_coverage >= required_anchor_coverage or not anchor_groups
            mandatory_ok = (
                not mandatory_groups or item.mandatory_anchor_coverage >= 1.0
            )
            has_intent = item.intent_evidence >= 0.08
            constraint_ok = grounded_alias or (
                mandatory_ok
                and (
                    preferred
                    or item.section_match >= 0.82
                    or (has_anchor if anchor_groups else (has_focus or has_intent))
                )
            )
            if item.relevance >= cutoff and has_focus and constraint_ok:
                eligible.append(item)

    if not eligible:
        top = scored[0]
        if top.coverage >= 0.20 or top.anchor_coverage > 0 or top.section_match >= 0.75:
            eligible = [top]
        else:
            return []

    limit = _selection_limit(plan, max_chunks)
    selected = _preferred_then_diverse_selection(
        eligible,
        limit,
        preferred_ids,
        plan.intent,
    )

    # Pull same-section / adjacent continuations into remaining capacity. A
    # continuation may omit the acronym but contain the next procedural steps.
    selected_ids = {item.result.chunk.chunk_id for item in selected}
    for item in scored:
        if len(selected) >= limit:
            break
        if item.result.chunk.chunk_id in selected_ids:
            continue
        if item.coverage < 0.15 and item.intent_evidence < 0.08:
            continue
        if any(_supports_anchor(item, anchor) for anchor in selected):
            selected.append(item)
            selected_ids.add(item.result.chunk.chunk_id)

    selected.sort(key=_scored_sort_key)
    return [
        replace(
            item.result,
            score=round(item.relevance, 6),
            method=_append_method(item.result.method, "intent-rerank"),
        )
        for item in selected[:limit]
    ]


def _score_candidate(
    plan: QueryPlan,
    result: RetrievedChunk,
    focus_terms: set[str],
    anchor_groups: list[set[str]],
    mandatory_groups: list[set[str]],
    preferred_document_ids: set[str],
    inherited_scope_terms: set[str],
    local_context_terms: set[str],
) -> _ScoredChunk:
    chunk = result.chunk
    text_terms = _tokens(chunk.text)
    exact_text_terms = _tokens(chunk.text, keep_single=True)
    contextual_anchor_terms = exact_text_terms | inherited_scope_terms | local_context_terms
    heading_terms = _tokens(_heading_text(chunk))
    section_path = _context_label(chunk)
    coverage = _coverage(focus_terms, text_terms)
    heading_coverage = _coverage(focus_terms, heading_terms)
    direct_anchor_coverage = _group_coverage(anchor_groups, exact_text_terms)
    anchor_coverage = _group_coverage(anchor_groups, contextual_anchor_terms)
    mandatory_anchor_coverage = _group_coverage(mandatory_groups, contextual_anchor_terms)
    question_for_structure = plan.contextual_question or plan.original_question
    structural_match = section_match_score(question_for_structure, section_path)
    major_structural_match = major_section_match_score(
        question_for_structure,
        section_path,
    )

    cues = _INTENT_CUES.get(plan.intent, set())
    intent_matches = len(cues & text_terms)
    intent_evidence = min(1.0, intent_matches / 3.0) if cues else 0.0
    if plan.intent == "fact_lookup":
        intent_evidence = min(1.0, coverage)

    retrieval = max(0.0, min(1.0, float(result.score)))
    enumeration_bonus = _enumeration_evidence_bonus(plan, chunk.text)
    relevance = (
        retrieval * 0.36
        + coverage * 0.34
        + heading_coverage * 0.12
        + intent_evidence * 0.08
        + anchor_coverage * 0.10
        + structural_match * 0.24
        + enumeration_bonus
    )
    if _is_preferred(chunk, preferred_document_ids):
        relevance += 0.16
    if "grounded-line-alias" in result.method:
        relevance += 0.24
    if _is_low_information_excerpt(chunk.text):
        relevance *= 0.12

    if focus_terms and coverage < 0.12:
        relevance *= 0.55
    if anchor_groups and anchor_coverage == 0 and structural_match < 0.82:
        relevance *= 0.72
    if _is_generic_heading(chunk.text) and coverage < 0.5:
        relevance *= 0.55

    return _ScoredChunk(
        result=result,
        relevance=max(0.0, min(1.0, relevance)),
        coverage=coverage,
        anchor_coverage=anchor_coverage,
        direct_anchor_coverage=direct_anchor_coverage,
        mandatory_anchor_coverage=mandatory_anchor_coverage,
        intent_evidence=intent_evidence,
        section_match=structural_match,
        major_section_match=major_structural_match,
        context_key=(chunk.filename.casefold(), _context_label(chunk).casefold()),
    )


def rank_scenario_documents(
    plan: QueryPlan,
    candidates: list[RetrievedChunk],
    *,
    max_documents: int = 3,
) -> list[PrimaryDocumentMatch]:
    """Infer a dedicated procedure document from local body evidence.

    Filename/opening-title routing can miss a document when the exact scenario is
    stated deep inside the SOP. This second-stage router scores small consecutive
    windows of already retrieved chunks, so a line/applicability heading may live
    in one chunk while VCB/NSCZ actions continue in the next chunks.
    """
    if max_documents <= 0 or not candidates or plan.search_mode != "answer":
        return []
    if plan.intent not in {"procedure", "requirement", "troubleshooting", "summary"}:
        return []

    settings = get_settings()
    focus_terms = _focus_terms(plan)
    anchor_groups = _anchor_groups(plan)
    inherited_scopes = _inherited_line_scope_terms(candidates)
    mandatory_groups = _mandatory_anchor_groups(plan)
    grouped = _group_results_by_document(candidates)
    matches: list[PrimaryDocumentMatch] = []

    for document_key, rows in grouped.items():
        if not rows or not rows[0].chunk.document_id:
            continue
        best_score = 0.0
        window_size = max(2, settings.scenario_document_window_chunks)
        ordered = sorted(rows, key=_chunk_position_key)
        for start in range(len(ordered)):
            window = _consecutive_window(ordered, start, window_size)
            if not window:
                continue
            combined_terms: set[str] = set()
            for item in window:
                combined_terms |= _tokens(item.chunk.text, keep_single=True)
                combined_terms |= inherited_scopes.get(item.chunk.chunk_id, set())
            anchor_coverage = _group_coverage(anchor_groups, combined_terms)
            mandatory_coverage = _group_coverage(mandatory_groups, combined_terms)
            focus_coverage = _coverage(focus_terms, combined_terms)
            if mandatory_groups and mandatory_coverage < 1.0:
                continue
            if anchor_groups and anchor_coverage < _required_anchor_coverage(plan, anchor_groups):
                continue
            if focus_terms and focus_coverage < 0.28:
                continue
            best_retrieval = max(float(item.score) for item in window)
            procedure_terms = _INTENT_CUES.get(plan.intent, set())
            procedure_signal = min(1.0, len(procedure_terms & combined_terms) / 3.0) if procedure_terms else 0.0
            score = (
                anchor_coverage * 0.44
                + focus_coverage * 0.28
                + max(0.0, min(1.0, best_retrieval)) * 0.18
                + procedure_signal * 0.10
            )
            best_score = max(best_score, score)
        if best_score >= 0.54:
            first = rows[0].chunk
            matches.append(
                PrimaryDocumentMatch(
                    document_id=str(first.document_id),
                    filename=first.filename,
                    score=round(min(0.99, best_score), 6),
                    reason="scenario-body",
                )
            )

    matches.sort(key=lambda item: (-item.score, item.filename.casefold(), item.document_id))
    return matches[:max_documents]


def filter_hard_context_candidates(
    plan: QueryPlan,
    candidates: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Keep fallback synthesis inside explicit scenario/equipment constraints.

    This guard is deliberately document-local. If a user asks about Line 1 + VCB
    + NSCZ, a BHS/airport undershoot chunk cannot become the "closest supported"
    answer merely because both mention a stopped train. Documents must satisfy the
    hard groups within a small consecutive evidence window.
    """
    groups = _hard_anchor_groups(plan)
    if not groups or not candidates:
        return candidates
    settings = get_settings()
    inherited_scopes = _inherited_line_scope_terms(candidates)
    mandatory_groups = _mandatory_anchor_groups(plan)
    grouped = _group_results_by_document(candidates)
    allowed_documents: set[str] = set()
    required = _required_anchor_coverage(plan, groups)

    for document_key, rows in grouped.items():
        ordered = sorted(rows, key=_chunk_position_key)
        for start in range(len(ordered)):
            window = _consecutive_window(
                ordered,
                start,
                max(2, settings.scenario_document_window_chunks),
            )
            combined: set[str] = set()
            for item in window:
                combined |= _tokens(item.chunk.text, keep_single=True)
                combined |= inherited_scopes.get(item.chunk.chunk_id, set())
            if mandatory_groups and _group_coverage(mandatory_groups, combined) < 1.0:
                continue
            if _group_coverage(groups, combined) >= required:
                allowed_documents.add(document_key)
                break

    if not allowed_documents:
        return []
    return [item for item in candidates if _document_key(item.chunk) in allowed_documents]


def _required_anchor_coverage(plan: QueryPlan, anchor_groups: list[set[str]]) -> float:
    if not anchor_groups:
        return 0.0
    if plan.intent == "comparison":
        return 1.0 / len(anchor_groups)
    # Operational procedures are commonly split across chunks: a line/applicability
    # heading is followed by action subparagraphs that do not repeat every acronym
    # or actor. Allow one anchor group to be supplied by adjacent local context, but
    # never relax enough for a different scenario/equipment family to qualify.
    if plan.intent in {"procedure", "requirement", "troubleshooting"} and len(anchor_groups) >= 3:
        return max(0.67, (len(anchor_groups) - 1) / len(anchor_groups))
    return 1.0


def _hard_anchor_groups(plan: QueryPlan) -> list[set[str]]:
    question = plan.contextual_question or plan.original_question
    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()

    for value in plan.context_terms:
        lowered = value.casefold()
        if any(char.isdigit() for char in value) or lowered.startswith("line ") or _DOCUMENT_CODE_RE.search(value):
            group = _tokens(value, keep_single=True)
            key = frozenset(group)
            if group and key not in seen:
                groups.append(group)
                seen.add(key)

    for token in _TOKEN_RE.findall(question):
        if len(token) >= 2 and token.upper() == token and any(char.isalpha() for char in token):
            group = _tokens(token, keep_single=True)
            key = frozenset(group)
            if group and key not in seen:
                groups.append(group)
                seen.add(key)

    for quoted in _QUOTED_RE.findall(question):
        group = _tokens(quoted, keep_single=True)
        key = frozenset(group)
        if group and key not in seen:
            groups.append(group)
            seen.add(key)
    return groups


def _mandatory_anchor_groups(plan: QueryPlan) -> list[set[str]]:
    """Return explicit line/equipment/document constraints that may not be dropped."""
    question = plan.contextual_question or plan.original_question
    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()

    for value in plan.context_terms:
        lowered = value.casefold().strip()
        is_line = lowered.startswith("line ") or lowered == "ael"
        is_document = bool(_DOCUMENT_CODE_RE.search(value))
        is_numeric_context = any(char.isdigit() for char in value)
        is_non_role_acronym = (
            len(value) >= 2
            and value.upper() == value
            and any(char.isalpha() for char in value)
            and lowered not in _ROLE_ACRONYMS
        )
        if not (is_line or is_document or is_numeric_context or is_non_role_acronym):
            continue
        group = _tokens(value, keep_single=True)
        key = frozenset(group)
        if group and key not in seen:
            groups.append(group)
            seen.add(key)

    for token in _TOKEN_RE.findall(question):
        lowered = token.casefold()
        if (
            len(token) >= 2
            and token.upper() == token
            and any(char.isalpha() for char in token)
            and lowered not in _ROLE_ACRONYMS
        ):
            group = _tokens(token, keep_single=True)
            key = frozenset(group)
            if group and key not in seen:
                groups.append(group)
                seen.add(key)
    return groups


def _inherited_line_scope_terms(candidates: list[RetrievedChunk]) -> dict[str, set[str]]:
    settings = get_settings()
    max_gap = max(1, settings.applicability_inherit_chunk_window)
    inherited: dict[str, set[str]] = {}
    for _document_key_value, rows in _group_results_by_document(candidates).items():
        ordered = sorted(rows, key=_chunk_position_key)
        active_scope: set[str] = set()
        active_index: int | None = None
        active_major = ""
        for item in ordered:
            chunk = item.chunk
            current_index = chunk.chunk_index if chunk.chunk_index is not None else chunk.page_number
            major = _major_context_label(chunk)
            explicit_scopes = _extract_line_scope_sets(chunk.text)
            if explicit_scopes:
                # Multiple different line groups in one chunk are an index/table or
                # mixed context; do not propagate an ambiguous scope forward.
                unique = {frozenset(scope) for scope in explicit_scopes if scope}
                if len(unique) == 1:
                    active_scope = set(next(iter(unique)))
                    active_index = current_index
                    active_major = major
                else:
                    active_scope = set()
                    active_index = None
                    active_major = major
            elif active_scope and active_index is not None:
                if current_index - active_index > max_gap:
                    active_scope = set()
                    active_index = None
                elif active_major and major and active_major != major:
                    active_scope = set()
                    active_index = None
            if active_scope:
                inherited[chunk.chunk_id] = set(active_scope)
    return inherited


def _local_anchor_terms_by_chunk(
    candidates: list[RetrievedChunk],
    inherited_scopes: dict[str, set[str]],
) -> dict[str, set[str]]:
    settings = get_settings()
    radius = max(0, settings.local_anchor_context_window)
    if radius <= 0:
        return {}
    result: dict[str, set[str]] = {}
    for _document_key_value, rows in _group_results_by_document(candidates).items():
        ordered = sorted(rows, key=_chunk_position_key)
        for index, item in enumerate(ordered):
            current_scope = inherited_scopes.get(item.chunk.chunk_id, set())
            terms: set[str] = set(current_scope)
            for offset in range(max(0, index - radius), min(len(ordered), index + radius + 1)):
                neighbor = ordered[offset]
                if not _locally_compatible(item.chunk, neighbor.chunk, current_scope, inherited_scopes):
                    continue
                terms |= _tokens(neighbor.chunk.text, keep_single=True)
                terms |= inherited_scopes.get(neighbor.chunk.chunk_id, set())
            result[item.chunk.chunk_id] = terms
    return result


def _locally_compatible(
    center: TextChunk,
    neighbor: TextChunk,
    center_scope: set[str],
    scopes: dict[str, set[str]],
) -> bool:
    if _document_key(center) != _document_key(neighbor):
        return False
    center_index = center.chunk_index if center.chunk_index is not None else center.page_number
    neighbor_index = neighbor.chunk_index if neighbor.chunk_index is not None else neighbor.page_number
    if abs(center_index - neighbor_index) > max(2, get_settings().local_anchor_context_window + 1):
        return False
    if _major_context_label(center) and _major_context_label(neighbor):
        if _major_context_label(center) != _major_context_label(neighbor):
            return False
    neighbor_scope = scopes.get(neighbor.chunk_id, set())
    if center_scope and neighbor_scope and center_scope != neighbor_scope:
        return False
    return True


def _extract_line_scope_sets(value: str) -> list[set[str]]:
    scopes: list[set[str]] = []
    for match in _LINE_SCOPE_RE.finditer(value):
        raw = match.group("values")
        values = re.findall(r"AEL|\d{1,2}", raw, flags=re.IGNORECASE)
        if not values:
            continue
        scope: set[str] = {"line"}
        for item in values:
            normalized = item.upper() if item.casefold() == "ael" else str(int(item))
            scope |= _tokens(normalized, keep_single=True)
        scopes.append(scope)
    return scopes


def _group_results_by_document(candidates: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    grouped: dict[str, list[RetrievedChunk]] = {}
    for item in candidates:
        grouped.setdefault(_document_key(item.chunk), []).append(item)
    return grouped


def _chunk_position_key(item: RetrievedChunk) -> tuple[int, int, str]:
    chunk = item.chunk
    return (
        chunk.chunk_index if chunk.chunk_index is not None else chunk.page_number * 1000,
        chunk.page_number,
        chunk.chunk_id,
    )


def _consecutive_window(
    ordered: list[RetrievedChunk],
    start: int,
    max_items: int,
) -> list[RetrievedChunk]:
    if start >= len(ordered):
        return []
    first = ordered[start]
    first_index = first.chunk.chunk_index
    window: list[RetrievedChunk] = []
    for item in ordered[start : start + max_items]:
        if first.chunk.document_id != item.chunk.document_id:
            break
        if first_index is not None and item.chunk.chunk_index is not None:
            if item.chunk.chunk_index - first_index > max_items + 2:
                break
        window.append(item)
    return window


def _major_context_label(chunk: TextChunk) -> str:
    label = _context_label(chunk)
    if not label:
        return ""
    parts = [part.strip() for part in label.split(">") if part.strip()]
    return " > ".join(parts[:2]).casefold()


def _enumeration_evidence_bonus(plan: QueryPlan, text: str) -> float:
    """Boost canonical enumerations for list/taxonomy questions.

    A rule that says "the following ... namely" is stronger list evidence than a
    nearby document that merely contains words such as "types" or "signals".
    The normal focus/constraint gates still apply, so this cannot make an
    unrelated list outrank evidence about the requested subject.
    """
    if plan.intent != "list":
        return 0.0
    lowered = text.casefold()
    if "namely" in lowered and "following" in lowered:
        return 0.16
    if "namely" in lowered:
        return 0.10
    if re.search(r"\bfollowing\b.{0,160}\b(?:are|include|includes|comprise|consist)", lowered, re.DOTALL):
        return 0.08
    return 0.0


def _supports_anchor(candidate: _ScoredChunk, anchor: _ScoredChunk) -> bool:
    left = candidate.result.chunk
    right = anchor.result.chunk
    if left.filename.casefold() != right.filename.casefold():
        return False
    if candidate.context_key[1] and candidate.context_key == anchor.context_key:
        return True
    if (
        left.document_id
        and left.document_id == right.document_id
        and left.chunk_index is not None
        and right.chunk_index is not None
        and abs(left.chunk_index - right.chunk_index) <= 2
    ):
        return True
    return abs(left.page_number - right.page_number) <= 1


def _selection_limit(plan: QueryPlan, requested: int | None) -> int:
    settings = get_settings()
    configured = (
        settings.reference_evidence_chunk_limit
        if plan.search_mode == "references"
        else settings.answer_evidence_chunk_limit
    )
    if requested is None:
        return configured
    return min(configured, max(1, requested))



def _preferred_then_diverse_selection(
    eligible: list[_ScoredChunk],
    limit: int,
    preferred_document_ids: set[str],
    intent: str,
) -> list[_ScoredChunk]:
    if not preferred_document_ids or intent != "procedure":
        return _strong_then_diverse_selection(eligible, limit)

    settings = get_settings()
    preferred = [item for item in eligible if _is_preferred(item.result.chunk, preferred_document_ids)]
    other = [item for item in eligible if not _is_preferred(item.result.chunk, preferred_document_ids)]
    selected = preferred[:limit]
    selected_ids = {item.result.chunk.chunk_id for item in selected}
    remaining = max(0, limit - len(selected))
    supplement_limit = min(remaining, settings.primary_document_supplement_limit)
    if supplement_limit:
        strong_other = [
            item for item in other
            if item.coverage >= 0.45 or item.anchor_coverage > 0 or item.section_match >= 0.82
        ]
        for item in strong_other[:supplement_limit]:
            if item.result.chunk.chunk_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.result.chunk.chunk_id)
    selected.sort(key=_scored_sort_key)
    return selected[:limit]


def _is_preferred(chunk: TextChunk, preferred_document_ids: set[str]) -> bool:
    return bool(chunk.document_id and chunk.document_id in preferred_document_ids)


def _is_low_information_excerpt(value: str) -> bool:
    body = re.sub(
        r"\[PDF CHUNK CONTEXT\].*?\[/PDF CHUNK CONTEXT\]",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if not body:
        return True
    alnum = re.sub(r"[^A-Za-z0-9]+", "", body)
    if len(alnum) < 18:
        return True
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) >= 2 and "|" in lines[0] and re.search(r"-{3,}", lines[1]):
        cells = [
            cell.strip(" *_`|")
            for line in lines[2:]
            for cell in line.split("|")
        ]
        if cells and not any(re.search(r"[A-Za-z0-9]", cell) for cell in cells):
            return True
    return False


def _strong_then_diverse_selection(
    eligible: list[_ScoredChunk],
    limit: int,
) -> list[_ScoredChunk]:
    if limit <= 0:
        return []

    selected: list[_ScoredChunk] = []
    selected_ids: set[str] = set()
    seen_documents: set[str] = set()

    # Keep the strongest procedure/context together first, then broaden document
    # coverage. This avoids displacing a multi-chunk answer merely to show more PDFs.
    strength_slots = min(len(eligible), max(1, int(limit * 0.60)))
    for item in eligible[:strength_slots]:
        selected.append(item)
        selected_ids.add(item.result.chunk.chunk_id)
        seen_documents.add(_document_key(item.result.chunk))

    for item in eligible:
        if len(selected) >= limit:
            break
        if item.result.chunk.chunk_id in selected_ids:
            continue
        document_key = _document_key(item.result.chunk)
        if document_key in seen_documents:
            continue
        selected.append(item)
        selected_ids.add(item.result.chunk.chunk_id)
        seen_documents.add(document_key)

    for item in eligible:
        if len(selected) >= limit:
            break
        if item.result.chunk.chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.result.chunk.chunk_id)

    return selected


def _document_key(chunk: TextChunk) -> str:
    return chunk.document_id or chunk.filename.casefold()


def _focus_terms(plan: QueryPlan) -> set[str]:
    supplied = " ".join([*plan.focus_terms, *plan.context_terms])
    base = supplied if supplied.strip() else (plan.contextual_question or plan.original_question)
    return {term for term in _tokens(base) if term not in _STOPWORDS}


def _anchor_groups(plan: QueryPlan) -> list[set[str]]:
    question = plan.contextual_question or plan.original_question
    values = [*plan.context_terms]
    values.extend(_QUOTED_RE.findall(question))
    values.extend(
        token
        for token in _TOKEN_RE.findall(question)
        if any(char.isdigit() for char in token)
        or (len(token) >= 2 and token.upper() == token and any(char.isalpha() for char in token))
    )
    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for value in values:
        if value.casefold() in _STOPWORDS:
            continue
        group = _tokens(value, keep_single=True)
        key = frozenset(group)
        if group and key not in seen:
            groups.append(group)
            seen.add(key)
    return groups


def _heading_text(chunk: TextChunk) -> str:
    values = [chunk.heading or "", *chunk.section_path]
    section = _SECTION_RE.search(chunk.text[:2500])
    if section:
        values.append(section.group(1))
    values.extend(_CONTEXT_LINE_RE.findall(chunk.text[:2500]))
    return " ".join(values)


def _context_label(chunk: TextChunk) -> str:
    if chunk.section_path:
        return " > ".join(chunk.section_path)
    return section_path_from_text(chunk.text)


def _tokens(value: str, *, keep_single: bool = False) -> set[str]:
    return search_terms(value, keep_single=keep_single)


def _coverage(required: set[str], available: set[str]) -> float:
    if not required:
        return 0.0
    return len(required & available) / len(required)


def _group_coverage(required: list[set[str]], available: set[str]) -> float:
    if not required:
        return 0.0
    matches = sum(group.issubset(available) for group in required)
    return matches / len(required)


def _is_generic_heading(value: str) -> bool:
    first_lines = " ".join(value.splitlines()[:7]).casefold()
    generic = {"contents", "operating manual", "table of contents", "introduction"}
    return any(label in first_lines for label in generic)


def _append_method(method: str, value: str) -> str:
    parts = [part for part in method.split("+") if part]
    if value not in parts:
        parts.append(value)
    return "+".join(parts)


def _scored_sort_key(item: _ScoredChunk) -> tuple[float, str, int, int, str]:
    chunk = item.result.chunk
    return (
        -item.relevance,
        chunk.filename.casefold(),
        chunk.page_number,
        chunk.chunk_index if chunk.chunk_index is not None else -1,
        chunk.chunk_id,
    )
