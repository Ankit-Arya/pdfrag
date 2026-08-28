from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Iterable, Sequence

from app.config import get_settings
from app.rag.llm import llm_service
from app.rag.types import PromptSource, RetrievedChunk

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_PERCENT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_NUMBER_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*(%|percent|km/?h|kmph|kph|m/s|sec(?:ond)?s?|"
    r"min(?:ute)?s?|hours?|days?|months?|years?|cars?|bogies?|doors?|brakes?|units?|items?)\b",
    re.IGNORECASE,
)
_RS_QUERY_RE = re.compile(r"(?<![A-Za-z0-9])RS\s*[-_. ]?\s*([A-Za-z0-9-]{1,12})(?![A-Za-z0-9])", re.IGNORECASE)
_LINE_QUERY_RE = re.compile(r"(?<![A-Za-z0-9])Line\s*[-_. ]?\s*(\d{1,3}[A-Za-z]?)(?![A-Za-z0-9])", re.IGNORECASE)

_PREMIUM_ANSWER_SYSTEM = """You are the final answer writer for a CLOSED-BOOK internal knowledge assistant.
A separate evidence-compiler has already read and organized the retrieved PDF evidence. Your job is now
similar to an expert assistant who already understands the relevant material and is carefully drafting the
reply.

SOURCE OF TRUTH
- Use ONLY facts represented in the EVIDENCE WORKSPACE and the VALID SOURCE BLOCKS supplied here.
- The workspace is an organization layer, not an independent factual source. Every factual statement still
  needs one or more valid [S#] citations from the supplied source blocks.
- Never invent a procedure, speed, threshold, duty, permission, prohibition, acronym expansion, scope,
  conversion, revision relationship, or missing fact.
- Preserve the source's condition-to-action relationship exactly. Never attach a value/action from one row
  or branch to another condition.
- Preserve measurement basis. A percentage/count of brakes, bogies, BICs, cars, doors, passengers, etc. are
  different unless the cited source explicitly establishes equivalence.

PREMIUM DRAFTING
- Answer the user's actual question first. Do not sound like a retrieval report.
- Silently handle ordinary spelling/grammar/colloquial wording.
- Use confident language for supported facts and precise language for unresolved distinctions.
- Do not open with caveats if a useful supported answer exists. Put a material clarification in a short
  final **Note** only when it changes interpretation.
- Do not say "the supplied excerpts", "the retrieved evidence", "the system found", "coverage", "route",
  "required scope", "workspace", or other implementation language.
- Do not create headings for documents/scopes that do not materially contribute to the answer.
- Do not include empty headings or "No applicable evidence" sections unless the user explicitly asks for an
  exhaustive negative inventory.
- Do not expose internal diagnostics or candidate-document lists.

FORMAT
- Direct fact/value: lead with the answer in the first sentence, then only the explanation needed to apply it.
- Procedure: use a clean numbered sequence in operational order; separate prerequisites/warnings only when
  supported and useful.
- Multi-scope/comparison: use a compact Markdown table when it improves comparison; otherwise use concise
  scope headings. Include only materially different/applicable scopes.
- Definition/list: give the definition/list directly and group only genuinely distinct meanings/categories.
- Complex explanatory question: use short descriptive headings, not a wall of bullets.
- Avoid repetitive source-file-name prose. Citations provide provenance; mention the document name only when
  authority/revision/context itself matters.
- Avoid citation spam: cite each factual sentence, bullet, step, or table row at its end, combining citations
  when multiple sources support the same point.
- Never add a generic conclusion that merely repeats the answer.

NEGATIVE CLAIM SAFETY
- A failure to retrieve a fact is not proof that the corpus lacks it.
- Do not make a corpus-wide negative claim unless NEGATIVE CLAIM SAFE is true in the workspace.
- If a point remains unresolved, state the narrow unresolved distinction without claiming the document/corpus
  does not contain it.

Return only the final Markdown answer."""

_PREMIUM_REPAIR_SYSTEM = """You are a precise answer editor for a closed-book PDF assistant.
Repair ONLY the issues listed by the verifier. Preserve supported content and the premium structure.
Use only the supplied evidence workspace and source blocks. Do not introduce new facts. Every factual
sentence/bullet/table row must have valid [S#] citations. Return only the corrected Markdown answer."""

_WORKSPACE_SYSTEM = """You are an evidence compiler for a CLOSED-BOOK assistant over official PDFs.
You do NOT write the final user-facing answer. Read the supplied evidence units and convert them into a
small, accurate internal knowledge map for the user's exact question.

Your most important job is APPLICABILITY, not word matching.

Rules:
1. Judge each source unit against the user's subject, scenario, condition, requested attribute, explicit scope,
   location/mode/person type, and quantitative basis. A source can share many words and still be inapplicable.
2. Reject a rescue/emergency/evacuation/test/depot/maintenance/etc. rule when the user's scenario is different,
   unless the question or evidence makes that scenario applicable. Do not reject merely because context is
   uncertain; mark uncertainty when the evidence could genuinely apply.
3. Keep separate source-defined scopes when they materially change the answer. Do NOT preserve a scope merely
   because retrieval routed it.
4. For tables/procedures, keep each condition and its action/value together. Never combine neighboring branches.
5. Never convert or equate different measurement bases unless a source explicitly provides that relationship.
6. Distinguish governing/defining evidence from incidental mentions, examples, historical/replaced text, and
   related-but-different procedures.
7. If a correction/amendment clearly replaces earlier text, prefer the current rule while retaining the older
   text only when it is needed to explain authority/history. If precedence is unclear, record a conflict.
8. Coverage is about whether the USER'S REQUESTED ATTRIBUTES can be answered, not whether every routed PDF has
   been represented.
9. When a material facet is missing, propose up to 4 targeted semantic searches using terms from the question
   or evidence. Do not invent document codes, line numbers, rolling-stock numbers, or factual answers.
10. A negative statement such as "document X has no such rule" requires unusually strong evidence. Set
    negative_claim_safe=true only when the question explicitly needs a negative/exhaustive conclusion and the
    evidence/search context genuinely supports it.
11. All factual claims in the workspace must list one or more exact supplied source IDs such as S3.
12. Do not include rejected sources in answer_source_ids.

Return JSON only with this shape:
{
  "refined_query_frame": {
    "answer_type": "direct_fact|procedure|comparison|enumeration|definition|explanation",
    "subject": "...",
    "requested_attributes": ["..."],
    "scenario": "...",
    "conditions": ["..."],
    "explicit_scopes": ["..."],
    "quantities_and_units": ["..."],
    "material_ambiguity": "..."
  },
  "answer_mode": "direct_fact|procedure|comparison|enumeration|definition|explanation",
  "coverage": "complete|partial|uncertain",
  "confidence": 0.0,
  "answer_source_ids": ["S1"],
  "claims": [
    {
      "claim": "atomic supported proposition",
      "scope": "source-defined applicability or empty",
      "scenario": "source-defined scenario or empty",
      "condition": "exact condition/threshold or empty",
      "action_or_value": "exact action/value or empty",
      "importance": "primary|supporting|exception|authority",
      "source_ids": ["S1"]
    }
  ],
  "rejected_sources": [
    {"source_id": "S2", "reason": "scenario_mismatch|incidental|duplicate|superseded|wrong_scope|wrong_measurement|other", "detail": "..."}
  ],
  "ambiguities": ["..."],
  "conflicts": ["..."],
  "missing_facets": ["..."],
  "retry_queries": ["..."],
  "negative_claim_safe": false,
  "answer_outline": ["what the final answer should cover, in order"]
}
"""

_VERIFY_SYSTEM = """You are a strict verifier for a closed-book PDF answer.
Do not rewrite the answer. Compare it against the evidence workspace and VALID SOURCE BLOCKS.
Check factual support, condition/action pairing, measurement basis, scenario/scope applicability, omissions of
primary workspace claims, citation validity, and accidental leakage of retrieval/diagnostic language.

Return JSON only:
{
  "supported": true,
  "unsupported_claims": ["..."],
  "missing_key_points": ["..."],
  "citation_issues": ["..."],
  "condition_action_issues": ["..."],
  "formatting_issues": ["..."]
}
"""


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 1000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _unique(values: Iterable[str], limit: int = 100) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean(value)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= limit:
            break
    return output


def _json_object(raw: str) -> dict[str, object]:
    text_value = str(raw or "").strip()
    if not text_value:
        return {}
    try:
        parsed = json.loads(text_value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = _JSON_FENCE_RE.search(text_value)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    start = text_value.find("{")
    end = text_value.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text_value[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    return {}


def build_query_frame(question: str, interpretation: object) -> dict[str, object]:
    """Build a deterministic seed frame; the evidence compiler refines it later."""
    intent = str(getattr(interpretation, "intent", "fact_lookup") or "fact_lookup").casefold()
    if intent in {"procedure", "troubleshooting", "requirement"}:
        answer_type = "procedure"
    elif intent in {"comparison"}:
        answer_type = "comparison"
    elif intent in {"list"}:
        answer_type = "enumeration"
    elif intent in {"definition"}:
        answer_type = "definition"
    elif intent in {"summary", "explanation"}:
        answer_type = "explanation"
    else:
        answer_type = "direct_fact"

    quantities = _unique(
        [f"{value} {unit}" for value, unit in _NUMBER_UNIT_RE.findall(question)]
        + [f"{value}%" for value in _PERCENT_RE.findall(question)],
        12,
    )
    explicit_scopes = _unique(
        [f"RS-{token.upper()}" for token in _RS_QUERY_RE.findall(question)]
        + [f"Line-{token.upper()}" for token in _LINE_QUERY_RE.findall(question)],
        12,
    )
    requested = [str(value) for value in getattr(interpretation, "evidence_needs", ()) if str(value).strip()]
    concepts = [str(value) for value in getattr(interpretation, "concepts", ()) if str(value).strip()]
    scope = getattr(interpretation, "scope", {})
    ambiguity = str(getattr(interpretation, "ambiguity_note", "") or "")
    return {
        "answer_type": answer_type,
        "resolved_question": str(getattr(interpretation, "resolved_question", question) or question),
        "requested_attributes": requested[:10],
        "concepts": concepts[:16],
        "explicit_scopes": explicit_scopes,
        "resolved_scope": scope if isinstance(scope, dict) else {},
        "quantities_and_units": quantities,
        "material_ambiguity": ambiguity,
    }


def _result_priority(item: RetrievedChunk) -> tuple[int, float, int, int]:
    method = str(item.method or "").casefold()
    content_type = str(item.chunk.content_type or "").casefold()
    if "v5.5-procedure-" in method and "-complete" in method:
        structure_priority = 0
    elif content_type.startswith("procedure_"):
        structure_priority = 1
    elif content_type == "table_row":
        structure_priority = 2
    else:
        structure_priority = 3
    return (
        structure_priority,
        -float(item.score),
        int(item.chunk.page_number or 0),
        int(item.chunk.chunk_index or 0),
    )


def select_workspace_sources(
    results: Sequence[RetrievedChunk],
    *,
    limit: int | None = None,
) -> list[PromptSource]:
    """Build a document-balanced evidence set for semantic compilation.

    Routing is deliberately not used as an answer obligation. This function only ensures
    that one repetitive document cannot consume the entire compiler context.
    """
    cap = limit or _int_env("RAG_V6_WORKSPACE_EVIDENCE", 64, 20, 96)
    per_doc_cap = _int_env("RAG_V6_WORKSPACE_PER_DOCUMENT", 10, 2, 24)
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    seen_chunks: set[str] = set()
    for item in results:
        chunk_id = str(item.chunk.chunk_id or "")
        if not chunk_id or chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        key = str(item.chunk.document_id or item.chunk.filename)
        by_doc[key].append(item)
    for values in by_doc.values():
        values.sort(key=_result_priority)

    doc_order = sorted(
        by_doc,
        key=lambda key: (
            -max(float(item.score) for item in by_doc[key]),
            by_doc[key][0].chunk.filename.casefold(),
        ),
    )
    output: list[PromptSource] = []
    for depth in range(per_doc_cap):
        added = False
        for key in doc_order:
            values = by_doc[key]
            if depth >= len(values):
                continue
            item = values[depth]
            output.append(PromptSource(result=item, excerpt=item.chunk.text.strip()))
            added = True
            if len(output) >= cap:
                return output
        if not added:
            break
    return output


def _source_block(index: int, source: PromptSource, *, max_chars: int) -> str:
    chunk = source.result.chunk
    page_end = int(chunk.page_end or chunk.page_number)
    pages = str(chunk.page_number) if page_end == chunk.page_number else f"{chunk.page_number}-{page_end}"
    section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "")
    excerpt = str(source.excerpt or "").strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "\n[excerpt truncated]"
    return (
        f"[S{index}]\n"
        f"File: {chunk.filename}\n"
        f"Pages: {pages}\n"
        f"Content type: {chunk.content_type}\n"
        f"Section: {section or 'Unsectioned'}\n"
        f"Text:\n{excerpt}"
    )


def _valid_source_ids(sources: Sequence[PromptSource]) -> set[str]:
    return {f"S{index}" for index in range(1, len(sources) + 1)}


def _sanitize_workspace(payload: dict[str, object], sources: Sequence[PromptSource]) -> dict[str, object]:
    valid = _valid_source_ids(sources)
    if not payload:
        payload = {}

    source_ids = payload.get("answer_source_ids")
    clean_ids = _unique(
        [str(value).upper() for value in source_ids] if isinstance(source_ids, list) else [],
        len(valid) or 1,
    )
    clean_ids = [value for value in clean_ids if value in valid]

    claims_out: list[dict[str, object]] = []
    raw_claims = payload.get("claims")
    if isinstance(raw_claims, list):
        for raw in raw_claims[:80]:
            if not isinstance(raw, dict):
                continue
            ids = raw.get("source_ids")
            claim_ids = _unique(
                [str(value).upper() for value in ids] if isinstance(ids, list) else [],
                12,
            )
            claim_ids = [value for value in claim_ids if value in valid]
            claim_text = _clean(raw.get("claim"))
            if not claim_text or not claim_ids:
                continue
            claims_out.append({
                "claim": claim_text[:1400],
                "scope": _clean(raw.get("scope"))[:300],
                "scenario": _clean(raw.get("scenario"))[:500],
                "condition": _clean(raw.get("condition"))[:900],
                "action_or_value": _clean(raw.get("action_or_value"))[:1200],
                "importance": _clean(raw.get("importance"))[:40] or "supporting",
                "source_ids": claim_ids,
            })
            clean_ids.extend(claim_ids)

    clean_ids = _unique(clean_ids, len(valid) or 1)
    if not clean_ids:
        clean_ids = [f"S{index}" for index in range(1, min(len(sources), 8) + 1)]

    rejected_out: list[dict[str, str]] = []
    raw_rejected = payload.get("rejected_sources")
    if isinstance(raw_rejected, list):
        for raw in raw_rejected[:80]:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("source_id") or "").upper()
            if source_id not in valid:
                continue
            rejected_out.append({
                "source_id": source_id,
                "reason": _clean(raw.get("reason"))[:80],
                "detail": _clean(raw.get("detail"))[:500],
            })

    coverage = str(payload.get("coverage") or "uncertain").casefold()
    if coverage not in {"complete", "partial", "uncertain"}:
        coverage = "uncertain"
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    answer_mode = str(payload.get("answer_mode") or "explanation").casefold()
    if answer_mode not in {"direct_fact", "procedure", "comparison", "enumeration", "definition", "explanation"}:
        answer_mode = "explanation"

    refined = payload.get("refined_query_frame")
    if not isinstance(refined, dict):
        refined = {}

    return {
        "refined_query_frame": refined,
        "answer_mode": answer_mode,
        "coverage": coverage,
        "confidence": confidence,
        "answer_source_ids": clean_ids,
        "claims": claims_out,
        "rejected_sources": rejected_out,
        "ambiguities": _unique(
            [str(value) for value in payload.get("ambiguities", [])] if isinstance(payload.get("ambiguities"), list) else [],
            16,
        ),
        "conflicts": _unique(
            [str(value) for value in payload.get("conflicts", [])] if isinstance(payload.get("conflicts"), list) else [],
            16,
        ),
        "missing_facets": _unique(
            [str(value) for value in payload.get("missing_facets", [])] if isinstance(payload.get("missing_facets"), list) else [],
            12,
        ),
        "retry_queries": _unique(
            [str(value) for value in payload.get("retry_queries", [])] if isinstance(payload.get("retry_queries"), list) else [],
            4,
        ),
        "negative_claim_safe": bool(payload.get("negative_claim_safe", False)),
        "answer_outline": _unique(
            [str(value) for value in payload.get("answer_outline", [])] if isinstance(payload.get("answer_outline"), list) else [],
            16,
        ),
    }


def compile_evidence_workspace(
    *,
    question: str,
    interpretation: object,
    query_frame: dict[str, object],
    sources: Sequence[PromptSource],
    search_round: int,
) -> dict[str, object]:
    if not sources:
        return _sanitize_workspace({}, sources)
    settings = get_settings()
    excerpt_chars = _int_env("RAG_V6_COMPILER_EXCERPT_CHARS", 3600, 800, 8000)
    blocks = "\n\n".join(
        _source_block(index, source, max_chars=excerpt_chars)
        for index, source in enumerate(sources, 1)
    )
    prompt = f"""ORIGINAL USER QUESTION:
{question}

LANGUAGE-UNDERSTANDING FRAME (not factual evidence):
{json.dumps(query_frame, ensure_ascii=False, indent=2)}

SEARCH ROUND:
{search_round}

SOURCE EVIDENCE UNITS:
{blocks}

Compile the evidence workspace. Apply scenario/scope/condition/measurement matching before deciding what
belongs in the answer. If the user can already be answered well, mark coverage complete instead of requesting
searches merely to represent more documents."""
    try:
        raw = llm_service.generate(
            _WORKSPACE_SYSTEM,
            prompt,
            max_output_tokens=_int_env("RAG_V6_COMPILER_MAX_OUTPUT_TOKENS", 3200, 1200, 6000),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        payload = _json_object(raw)
    except Exception:
        payload = {}
    workspace = _sanitize_workspace(payload, sources)
    if not workspace["claims"]:
        # Fail open to ordinary grounded answering rather than returning nothing if the
        # compiler/model JSON fails. The final writer still receives source-grounded text.
        workspace["coverage"] = "uncertain"
        workspace["answer_source_ids"] = [
            f"S{index}" for index in range(1, min(len(sources), 12) + 1)
        ]
        workspace["answer_outline"] = ["Answer directly from the strongest applicable source evidence."]
    return workspace


def workspace_retry_queries(workspace: dict[str, object]) -> list[str]:
    if str(workspace.get("coverage") or "").casefold() == "complete":
        return []
    raw = workspace.get("retry_queries")
    values = [str(value) for value in raw] if isinstance(raw, list) else []
    return _unique(values, _int_env("RAG_V6_MAX_RETRY_QUERIES", 4, 1, 6))


def workspace_max_rounds() -> int:
    return _int_env("RAG_V6_MAX_WORKSPACE_ROUNDS", 2, 1, 3)


def workspace_evidence_limit(top_k: int | None, query_frame: dict[str, object]) -> int:
    base = int(top_k or _int_env("RAG_V6_WORKSPACE_EVIDENCE", 64, 20, 96))
    answer_type = str(query_frame.get("answer_type") or "direct_fact")
    minimum = 48 if answer_type in {"procedure", "comparison", "enumeration", "explanation"} else 32
    return min(96, max(minimum, base))


def workspace_source_blocks(
    sources: Sequence[PromptSource],
    workspace: dict[str, object],
) -> str:
    allowed = set(
        str(value).upper()
        for value in workspace.get("answer_source_ids", [])
        if str(value).strip()
    )
    max_chars = _int_env("RAG_V6_WRITER_EXCERPT_CHARS", 5200, 1200, 10000)
    blocks: list[str] = []
    for index, source in enumerate(sources, 1):
        source_id = f"S{index}"
        if allowed and source_id not in allowed:
            continue
        blocks.append(_source_block(index, source, max_chars=max_chars))
    return "\n\n".join(blocks)


def premium_answer_prompt(
    *,
    question: str,
    workspace: dict[str, object],
    sources: Sequence[PromptSource],
) -> str:
    source_blocks = workspace_source_blocks(sources, workspace)
    compact_workspace = {
        "refined_query_frame": workspace.get("refined_query_frame", {}),
        "answer_mode": workspace.get("answer_mode"),
        "coverage": workspace.get("coverage"),
        "confidence": workspace.get("confidence"),
        "claims": workspace.get("claims", []),
        "ambiguities": workspace.get("ambiguities", []),
        "conflicts": workspace.get("conflicts", []),
        "missing_facets": workspace.get("missing_facets", []),
        "negative_claim_safe": workspace.get("negative_claim_safe", False),
        "answer_outline": workspace.get("answer_outline", []),
    }
    return f"""USER QUESTION:
{question}

EVIDENCE WORKSPACE:
{json.dumps(compact_workspace, ensure_ascii=False, indent=2)}

NEGATIVE CLAIM SAFE:
{bool(workspace.get('negative_claim_safe', False))}

VALID SOURCE BLOCKS (only these may be cited):
{source_blocks}

Draft the final answer now. Use the workspace to decide what matters and the source blocks to ground every
factual statement. Do not mention the workspace or retrieval process."""


def premium_answer_system() -> str:
    return _PREMIUM_ANSWER_SYSTEM


def verify_workspace_answer(
    *,
    question: str,
    answer: str,
    workspace: dict[str, object],
    sources: Sequence[PromptSource],
) -> dict[str, object]:
    settings = get_settings()
    prompt = f"""USER QUESTION:
{question}

ANSWER TO VERIFY:
{answer}

EVIDENCE WORKSPACE:
{json.dumps(workspace, ensure_ascii=False, indent=2)}

VALID SOURCE BLOCKS:
{workspace_source_blocks(sources, workspace)}

Return the verification JSON only."""
    try:
        raw = llm_service.generate(
            _VERIFY_SYSTEM,
            prompt,
            max_output_tokens=_int_env("RAG_V6_VERIFY_MAX_OUTPUT_TOKENS", 1400, 500, 2600),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        payload = _json_object(raw)
    except Exception:
        payload = {}
    if not payload:
        return {
            "supported": True,
            "unsupported_claims": [],
            "missing_key_points": [],
            "citation_issues": [],
            "condition_action_issues": [],
            "formatting_issues": [],
        }
    return {
        "supported": bool(payload.get("supported", False)),
        "unsupported_claims": _unique([str(v) for v in payload.get("unsupported_claims", [])] if isinstance(payload.get("unsupported_claims"), list) else [], 12),
        "missing_key_points": _unique([str(v) for v in payload.get("missing_key_points", [])] if isinstance(payload.get("missing_key_points"), list) else [], 12),
        "citation_issues": _unique([str(v) for v in payload.get("citation_issues", [])] if isinstance(payload.get("citation_issues"), list) else [], 12),
        "condition_action_issues": _unique([str(v) for v in payload.get("condition_action_issues", [])] if isinstance(payload.get("condition_action_issues"), list) else [], 12),
        "formatting_issues": _unique([str(v) for v in payload.get("formatting_issues", [])] if isinstance(payload.get("formatting_issues"), list) else [], 12),
    }


def verification_requires_repair(verification: dict[str, object]) -> bool:
    if not bool(verification.get("supported", False)):
        return True
    return any(
        bool(verification.get(key))
        for key in (
            "unsupported_claims",
            "missing_key_points",
            "citation_issues",
            "condition_action_issues",
            "formatting_issues",
        )
    )


def premium_repair_prompt(
    *,
    question: str,
    draft: str,
    verification: dict[str, object],
    workspace: dict[str, object],
    sources: Sequence[PromptSource],
) -> str:
    return f"""USER QUESTION:
{question}

CURRENT DRAFT:
{draft}

VERIFIER ISSUES TO FIX:
{json.dumps(verification, ensure_ascii=False, indent=2)}

EVIDENCE WORKSPACE:
{json.dumps(workspace, ensure_ascii=False, indent=2)}

VALID SOURCE BLOCKS:
{workspace_source_blocks(sources, workspace)}

Repair only the listed issues. Preserve the useful direct answer and premium formatting."""


def premium_repair_system() -> str:
    return _PREMIUM_REPAIR_SYSTEM


def workspace_primary_documents(
    sources: Sequence[PromptSource],
    workspace: dict[str, object],
) -> list[str]:
    allowed = set(str(value).upper() for value in workspace.get("answer_source_ids", []))
    output: list[str] = []
    seen: set[str] = set()
    for index, source in enumerate(sources, 1):
        if allowed and f"S{index}" not in allowed:
            continue
        name = source.result.chunk.filename
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())
        output.append(name)
    return output[:40]


def workspace_summary(workspace: dict[str, object]) -> str:
    claims = len(workspace.get("claims", [])) if isinstance(workspace.get("claims"), list) else 0
    rejected = len(workspace.get("rejected_sources", [])) if isinstance(workspace.get("rejected_sources"), list) else 0
    missing = len(workspace.get("missing_facets", [])) if isinstance(workspace.get("missing_facets"), list) else 0
    return (
        f"coverage={workspace.get('coverage', 'uncertain')}; claims={claims}; "
        f"rejected={rejected}; missing={missing}"
    )
