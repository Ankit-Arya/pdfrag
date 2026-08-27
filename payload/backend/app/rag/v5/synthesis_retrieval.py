from __future__ import annotations

import inspect
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from sqlalchemy.orm import Session

from app.config import get_settings
from app.rag.embeddings import embedding_service
from app.rag.llm import llm_service
from app.rag.progress import emit_progress
from app.rag.smart_understanding import SmartInterpretation, review_retrieved_evidence
from app.rag.types import RetrievedChunk
from app.rag.v5.assistant_retrieval import (
    DocumentRoute,
    _clean,
    _expand_sections,
    _heading_candidates,
    _merge_results,
    _scoped_candidates,
    _strict_fts_rows,
    _unique,
    _vector_rows,
    retrieve_assistant_v51,
)
from app.rag.v5.retrieval import retrieve_v5

_SYNTHESIS_CUES_RE = re.compile(
    r"\b(?:dut(?:y|ies)|role(?:s)?|responsibilit(?:y|ies)|should|must|required|requirement(?:s)?|"
    r"responsible|when|whenever|under\s+what\s+conditions?|condition(?:s)?|appl(?:y|ies|icable|icability)|"
    r"exception(?:s)?|circumstance(?:s)?|case(?:s)?|scenario(?:s)?|precaution(?:s)?|all\s+cases|"
    r"what\s+to\s+do|who\s+does\s+what|compare|difference(?:s)?|across\s+documents?)\b",
    re.IGNORECASE,
)
_DIRECT_CUES_RE = re.compile(
    r"\b(?:full\s+form|definition|define|meaning|which\s+page|page\s+number|which\s+rule|rule\s+number|"
    r"what\s+is\s+the\s+(?:exact\s+)?(?:value|amount|number|date|name))\b",
    re.IGNORECASE,
)
_SYNTHESIS_INTENTS = {"list", "procedure", "requirement", "comparison", "summary", "troubleshooting"}

_SYNTHESIS_RERANK_SYSTEM = """You are a cross-document evidence-ranking layer for a CLOSED-BOOK assistant over official internal PDFs.
You DO NOT answer the user's question. Rank only supplied candidate excerpts.

The question has been classified as requiring multi-document synthesis. The complete answer may be distributed across multiple documents or sections.

Rules:
- Keep evidence from every document that materially contributes a governing rule, responsibility, definition, condition, applicability rule, exception, restriction, amendment, or useful supporting context.
- Do not retain a document merely because it mentions the same actor/object or shares generic words.
- Prefer governing/defining provisions over incidental mentions.
- Treat semantically equivalent wording, typos, paraphrases and ordinary synonyms as equivalent where the excerpt clearly covers the concept.
- Preserve scope differences (line, location, mode, equipment, person type, scenario) instead of flattening them.
- Preserve current/amended/superseded signals visible in the evidence.
- For broad/list/procedure questions, keep multiple candidates when needed for completeness.
- Do not invent internal acronym expansions, rules, responsibilities, speeds, conditions, documents or relationships.

Return JSON only:
{"ranking":[{"id":"candidate-id","score":0-100,"contribution":"governing|supporting|definition|applicability|exception|restriction|authority|conflict|incidental"}]}
Omit clearly irrelevant candidates. Keep materially useful candidates from different documents when they contribute distinct parts of the answer.
"""

_SYNTHESIS_COVERAGE_SYSTEM = """You are a cross-document evidence-coverage critic for a CLOSED-BOOK RAG assistant.
You do NOT answer the user's question and you do NOT add factual knowledge.

Decide whether the supplied PDF evidence collectively covers the user's requested synthesis across all materially relevant documents represented in the evidence.

Check:
- every requested evidence dimension;
- whether responsibilities/requirements/conditions are directly established rather than merely mentioned;
- scope and applicability differences;
- exceptions and restrictions when they materially affect the answer;
- authority/currentness when an amendment or conflicting version matters;
- whether different documents are complementary, scope-specific, authority-related, or genuinely unresolved conflicts.

A document is "contributing" only when it adds meaningful evidence. Do not include incidental lexical matches.
A routed/relevant-document candidate is only a retrieval lead, not proof that it must contribute; do not force it into the answer when its excerpts add nothing meaningful.
If important evidence is missing, propose 1-3 targeted retry searches. Use only concepts already present in the question, evidence needs, grounded terminology, or supplied excerpts. Do not invent document codes or factual answers.

Return JSON only with exactly these keys:
- sufficient: boolean
- covered_dimensions: array of short strings
- missing_dimensions: array of short strings
- retry_queries: array of 0-3 search strings
- contributing_documents: array of exact filenames from the supplied evidence
- conflicts: array of objects {type, documents, summary, resolution}; type must be complementary, scope_difference, authority_difference, or unresolved
- reason: short string
"""


@dataclass(frozen=True, slots=True)
class SynthesisPlan:
    answer_strategy: str
    dimensions: tuple[str, ...]
    search_scope: str
    requires_conflict_review: bool


@dataclass(slots=True)
class SynthesisRetrievalBundle:
    results: list[RetrievedChunk]
    search_queries: list[str]
    candidate_count: int
    routed_documents: list[str]
    answer_strategy: str
    synthesis_dimensions: list[str]
    considered_documents: list[str]
    evidence_documents: list[str]
    search_round: int = 1


@dataclass(frozen=True, slots=True)
class SynthesisCoverage:
    sufficient: bool
    missing_evidence: tuple[str, ...] = ()
    retry_queries: tuple[str, ...] = ()
    reason: str = ""
    ai_used: bool = False
    answer_strategy: str = "direct_lookup"
    covered_dimensions: tuple[str, ...] = ()
    uncovered_dimensions: tuple[str, ...] = ()
    contributing_documents: tuple[str, ...] = ()
    conflicts: tuple[dict[str, object], ...] = ()
    evidence_coverage_status: str = "unknown"


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 500) -> int:
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


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _clean(value).casefold()))


def _meaningful_dimensions(interpretation: SmartInterpretation, question: str) -> tuple[str, ...]:
    values: list[str] = [*interpretation.evidence_needs]
    folded = f"{question} {interpretation.resolved_question}".casefold()

    if re.search(r"\b(?:dut(?:y|ies)|role(?:s)?|responsibilit(?:y|ies)|responsible)\b", folded):
        values.extend(("core responsibilities or duties", "responsibility conditions or role boundaries"))
    if re.search(r"\b(?:when|whenever|condition(?:s)?|appl(?:y|ies|icable|icability)|scenario(?:s)?|circumstance(?:s)?)\b", folded):
        values.append("conditions and applicability")
    if re.search(r"\b(?:should|must|required|requirement(?:s)?|necessary|allowed|permitted|prohibited)\b", folded):
        values.append("mandatory or permitted action and responsible actor")
    if re.search(r"\b(?:exception(?:s)?|unless|except|special\s+case|restriction(?:s)?)\b", folded):
        values.append("exceptions, restrictions or special conditions")
    if interpretation.authority_sensitive:
        values.append("current authority, amendment or supersession status")

    if len(values) < 2:
        values.extend(interpretation.concepts[:4])
    return tuple(_unique(values, 8))


def build_synthesis_plan(question: str, interpretation: SmartInterpretation) -> SynthesisPlan:
    folded = f"{question} {interpretation.resolved_question}".strip()
    dimensions = _meaningful_dimensions(interpretation, question)

    explicit_direct = bool(_DIRECT_CUES_RE.search(folded))
    broad_cue = bool(_SYNTHESIS_CUES_RE.search(folded))
    broad_intent = interpretation.intent in _SYNTHESIS_INTENTS
    multi_need = len(interpretation.evidence_needs) >= 3

    direct_definition = interpretation.intent == "definition" and not broad_cue
    direct_navigation = interpretation.conversation_act == "navigation" and interpretation.intent not in {"comparison", "summary", "list"}
    direct_fact = (
        interpretation.intent == "fact_lookup"
        and not broad_cue
        and len(interpretation.evidence_needs) <= 2
        and not interpretation.material_ambiguity
    )

    synthesis = broad_intent or broad_cue or multi_need or interpretation.material_ambiguity
    if explicit_direct and (direct_definition or direct_navigation or direct_fact):
        synthesis = False
    if direct_definition or direct_navigation:
        synthesis = False

    strategy = "multi_document_synthesis" if synthesis else "direct_lookup"
    return SynthesisPlan(
        answer_strategy=strategy,
        dimensions=dimensions,
        search_scope="broad_relevant_corpus" if synthesis else "focused",
        requires_conflict_review=bool(synthesis or interpretation.authority_sensitive or interpretation.intent == "comparison"),
    )


def _dimension_queries(
    interpretation: SmartInterpretation,
    plan: SynthesisPlan,
    extra_queries: Iterable[str] = (),
) -> list[str]:
    values: list[str] = [interpretation.resolved_question, *interpretation.search_queries]
    if plan.answer_strategy == "multi_document_synthesis":
        values.extend(
            f"{interpretation.resolved_question} {dimension}"
            for dimension in plan.dimensions
        )
    values.extend(extra_queries)
    return _unique(values, _int_env("RAG_V52_MAX_QUERY_VARIANTS", 12, 4, 16))


def _discovery_routes(
    db: Session,
    *,
    interpretation: SmartInterpretation,
    plan: SynthesisPlan,
    queries: Sequence[str],
    baseline: Sequence[RetrievedChunk],
) -> list[DocumentRoute]:
    if not queries:
        return []

    scores: defaultdict[str, float] = defaultdict(float)
    strongest: defaultdict[str, float] = defaultdict(float)
    dimension_hits: defaultdict[str, set[int]] = defaultdict(set)
    filenames: dict[str, str] = {}

    for rank, item in enumerate(baseline[:80], 1):
        document_id = item.chunk.document_id
        if not document_id:
            continue
        filenames[document_id] = item.chunk.filename
        scores[document_id] += 0.10 * float(item.score) + 0.14 / (6.0 + rank)
        strongest[document_id] = max(strongest[document_id], float(item.score))
        dimension_hits[document_id].add(0)

    route_queries = list(queries[: _int_env("RAG_V52_DISCOVERY_QUERY_COUNT", 8, 3, 12)])
    try:
        vectors = embedding_service.encode(route_queries)
    except Exception:
        vectors = []

    per_query = _int_env("RAG_V52_DISCOVERY_PER_QUERY", 110, 40, 220)
    for query_index, query in enumerate(route_queries):
        if query_index < len(vectors):
            try:
                for rank, row in enumerate(_vector_rows(db, vectors[query_index].tolist(), per_query), 1):
                    document_id = str(row["document_id"])
                    filename = str(row["filename"])
                    signal = max(0.0, min(1.0, float(row["vector_score"] or 0.0)))
                    filenames[document_id] = filename
                    scores[document_id] += 0.22 / (6.0 + rank) + 0.045 * signal
                    strongest[document_id] = max(strongest[document_id], signal)
                    if signal >= _float_env("RAG_V52_DIMENSION_VECTOR_HIT", 0.42, 0.20, 0.90):
                        dimension_hits[document_id].add(query_index)
            except Exception:
                pass
        try:
            for rank, row in enumerate(_strict_fts_rows(db, query, per_query), 1):
                document_id = str(row["document_id"])
                filename = str(row["filename"])
                signal = max(0.0, min(1.0, float(row["keyword_score"] or 0.0) * 4.5))
                filenames[document_id] = filename
                scores[document_id] += 0.21 / (6.0 + rank) + 0.04 * signal
                strongest[document_id] = max(strongest[document_id], signal)
                if signal >= _float_env("RAG_V52_DIMENSION_KEYWORD_HIT", 0.18, 0.02, 0.90):
                    dimension_hits[document_id].add(query_index)
        except Exception:
            pass

    if not scores:
        return []

    composite: dict[str, float] = {}
    for document_id, raw_score in scores.items():
        coverage_bonus = 0.045 * min(5, len(dimension_hits[document_id]))
        signal_bonus = 0.14 * strongest[document_id]
        composite[document_id] = raw_score + coverage_bonus + signal_bonus

    best = max(composite.values()) or 1.0
    relative = _float_env("RAG_V52_DOCUMENT_RELATIVE_THRESHOLD", 0.18, 0.05, 0.80)
    min_signal = _float_env("RAG_V52_DOCUMENT_MIN_SIGNAL", 0.34, 0.10, 0.90)
    max_docs = _int_env("RAG_V52_MAX_RELEVANT_DOCUMENTS", 24, 6, 32)

    ordered = sorted(
        composite,
        key=lambda document_id: (
            -composite[document_id],
            filenames.get(document_id, "").casefold(),
            document_id,
        ),
    )
    selected: list[str] = []
    for document_id in ordered:
        ratio = composite[document_id] / best
        evidence_hit = strongest[document_id] >= min_signal or len(dimension_hits[document_id]) >= 2
        if ratio >= relative and evidence_hit:
            selected.append(document_id)
        if len(selected) >= max_docs:
            break

    # Do not force a minimum number of documents. Synthesis mode broadens discovery,
    # but a document enters the routed set only when evidence signals make it relevant.
    # If thresholds reject everything, retain only the strongest document as a recovery
    # path rather than inventing artificial document diversity.
    if not selected and ordered:
        selected.append(ordered[0])

    routes: list[DocumentRoute] = []
    seen_filenames: set[str] = set()
    for document_id in selected:
        filename = filenames.get(document_id, document_id)
        filename_key = filename.casefold().strip()
        if filename_key in seen_filenames:
            continue
        seen_filenames.add(filename_key)
        routes.append(
            DocumentRoute(
                document_id=document_id,
                filename=filename,
                score=max(0.0, min(1.0, composite[document_id] / best)),
            )
        )
        if len(routes) >= max_docs:
            break
    return routes


def _balanced_pool(
    candidates: Sequence[RetrievedChunk],
    routes: Sequence[DocumentRoute],
    limit: int,
) -> list[RetrievedChunk]:
    if not candidates:
        return []
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in candidates:
        key = item.chunk.document_id or item.chunk.filename
        by_doc[key].append(item)

    output: list[RetrievedChunk] = []
    seen: set[str] = set()
    per_doc_seed = _int_env("RAG_V52_RERANK_SEEDS_PER_DOCUMENT", 2, 1, 5)
    for route in routes:
        for item in by_doc.get(route.document_id, [])[:per_doc_seed]:
            if item.chunk.chunk_id in seen:
                continue
            seen.add(item.chunk.chunk_id)
            output.append(item)
            if len(output) >= limit:
                return output

    for item in candidates:
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _rerank_prompt(
    interpretation: SmartInterpretation,
    plan: SynthesisPlan,
    candidates: Sequence[RetrievedChunk],
    routed_documents: Sequence[str],
) -> str:
    excerpt_chars = _int_env("RAG_V52_RERANK_EXCERPT_CHARS", 460, 180, 900)
    blocks: list[str] = []
    for item in candidates:
        chunk = item.chunk
        pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "")
        body = re.sub(r"\s+", " ", chunk.text).strip()[:excerpt_chars]
        blocks.append(
            f"ID: {chunk.chunk_id}\n"
            f"File: {chunk.filename}\n"
            f"Pages: {pages}\n"
            f"Section: {section or 'Unsectioned'}\n"
            f"Retrieval score: {item.score:.4f}\n"
            f"Excerpt: {body}"
        )
    return f"""RESOLVED QUESTION:
{interpretation.resolved_question}

SYNTHESIS DIMENSIONS:
{chr(10).join(f'- {item}' for item in plan.dimensions) or '- Direct answer evidence'}

SCOPE:
{interpretation.scope or 'None specified'}

RELEVANT DOCUMENT CANDIDATES:
{chr(10).join(f'- {name}' for name in routed_documents) or 'None'}

CANDIDATES:
{chr(10).join(chr(10) + block for block in blocks)}

Rank evidence for a complete cross-document answer. Do not answer the question.
"""


def _json_object(raw: str) -> dict[str, object]:
    value = raw.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI synthesis layer did not return JSON")
    payload = json.loads(value[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("AI synthesis payload is not an object")
    return payload


def _copy_result(item: RetrievedChunk, *, score: float, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=item.chunk,
        score=max(0.0, min(1.0, score)),
        method=method,
        vector_score=item.vector_score,
        keyword_score=item.keyword_score,
    )


def _synthesis_rerank(
    interpretation: SmartInterpretation,
    plan: SynthesisPlan,
    candidates: Sequence[RetrievedChunk],
    routes: Sequence[DocumentRoute],
) -> list[RetrievedChunk]:
    if not candidates:
        return []
    limit = _int_env("RAG_V52_RERANK_CANDIDATES", 72, 24, 100)
    pool = _balanced_pool(candidates, routes, limit)
    settings = get_settings()
    try:
        raw = llm_service.generate(
            _SYNTHESIS_RERANK_SYSTEM,
            _rerank_prompt(interpretation, plan, pool, [route.filename for route in routes]),
            max_output_tokens=_int_env("RAG_V52_RERANK_MAX_OUTPUT_TOKENS", 1800, 600, 3000),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        payload = _json_object(raw)
        ranking = payload.get("ranking")
        if not isinstance(ranking, list):
            raise ValueError("synthesis reranker ranking is not a list")
    except Exception:
        return list(pool)

    by_id = {item.chunk.chunk_id: item for item in pool}
    ranked: list[RetrievedChunk] = []
    seen: set[str] = set()
    for row in ranking:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("id") or "")
        item = by_id.get(chunk_id)
        if item is None or chunk_id in seen:
            continue
        contribution = str(row.get("contribution") or "supporting").casefold().strip()
        if contribution not in {
            "governing", "supporting", "definition", "applicability", "exception",
            "restriction", "authority", "conflict", "incidental",
        }:
            contribution = "supporting"
        try:
            ai_score = max(0.0, min(100.0, float(row.get("score") or 0.0))) / 100.0
        except (TypeError, ValueError):
            ai_score = 0.0
        # An explicit incidental classification is a negative relevance decision,
        # not a weak positive signal. Mark it reviewed so the fallback below cannot
        # silently re-introduce the same chunk.
        seen.add(chunk_id)
        if contribution == "incidental":
            continue
        combined = min(1.0, 0.70 * ai_score + 0.30 * float(item.score))
        ranked.append(
            _copy_result(
                item,
                score=combined,
                method=f"{item.method}+v5.2-synthesis:{contribution}",
            )
        )

    for item in pool:
        if item.chunk.chunk_id in seen:
            continue
        ranked.append(_copy_result(item, score=max(0.0, item.score - 0.15), method=item.method))
    ranked.sort(key=lambda item: (-item.score, item.chunk.filename.casefold(), item.chunk.page_number))
    return ranked


def _final_diverse(
    candidates: Sequence[RetrievedChunk],
    routes: Sequence[DocumentRoute],
    limit: int,
) -> list[RetrievedChunk]:
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in candidates:
        key = item.chunk.document_id or item.chunk.filename
        by_doc[key].append(item)
    output: list[RetrievedChunk] = []
    seen: set[str] = set()

    # Give each meaningfully routed document a chance to contribute one strong item.
    seed_threshold = _float_env("RAG_V52_FINAL_DOCUMENT_SEED_SCORE", 0.46, 0.20, 0.90)
    for route in routes:
        items = by_doc.get(route.document_id, [])
        if not items or items[0].score < seed_threshold:
            continue
        item = items[0]
        seen.add(item.chunk.chunk_id)
        output.append(item)
        if len(output) >= limit:
            return output

    per_doc: defaultdict[str, int] = defaultdict(int)
    for item in output:
        per_doc[item.chunk.document_id or item.chunk.filename] += 1
    cap = _int_env("RAG_V52_FINAL_PER_DOCUMENT_CAP", 14, 4, 30)
    for item in candidates:
        if item.chunk.chunk_id in seen:
            continue
        key = item.chunk.document_id or item.chunk.filename
        if per_doc[key] >= cap and len(output) >= max(12, limit // 2):
            continue
        seen.add(item.chunk.chunk_id)
        per_doc[key] += 1
        output.append(item)
        if len(output) >= limit:
            break
    return output



def _call_v51(
    db: Session,
    interpretation: SmartInterpretation,
    *,
    original_question: str,
    extra_queries: Iterable[str],
    prior_results: Sequence[RetrievedChunk],
    activity_round: int,
):
    kwargs = {
        "original_question": original_question,
        "extra_queries": extra_queries,
        "prior_results": prior_results,
    }
    try:
        if "activity_round" in inspect.signature(retrieve_assistant_v51).parameters:
            kwargs["activity_round"] = activity_round
    except (TypeError, ValueError):
        pass
    return retrieve_assistant_v51(db, interpretation, **kwargs)

def retrieve_assistant_v52(
    db: Session,
    interpretation: SmartInterpretation,
    *,
    original_question: str,
    extra_queries: Iterable[str] = (),
    prior_results: Sequence[RetrievedChunk] = (),
    activity_round: int = 1,
) -> SynthesisRetrievalBundle:
    plan = build_synthesis_plan(original_question, interpretation)
    emit_progress(
        f"synthesis_policy_{activity_round}",
        "Multi-document synthesis selected" if plan.answer_strategy == "multi_document_synthesis" else "Focused lookup selected",
        (
            "Searching all meaningfully relevant documents before synthesis"
            if plan.answer_strategy == "multi_document_synthesis"
            else "A directly governing source may be sufficient if evidence coverage is complete"
        ),
        actor="ai",
        phase="route",
        status="complete",
        operation_id=f"synthesis-policy-{activity_round}",
        reasoning_summary=(
            f"Answer strategy: {plan.answer_strategy}; evidence dimensions: "
            + "; ".join(plan.dimensions[:6])
        )[:1000],
        metrics={"dimensions": len(plan.dimensions)},
    )

    queries = _dimension_queries(interpretation, plan, extra_queries)

    # Direct lookups keep the complete Assistant v5.1 path. Synthesis queries intentionally
    # bypass the v5.1 AI reranker and use the v5 broad retriever as candidate generation,
    # because v5.2 performs its own document-balanced cross-document reranking below. This
    # avoids paying for two AI rerank calls on the same request.
    if plan.answer_strategy != "multi_document_synthesis":
        baseline = _call_v51(
            db,
            interpretation,
            original_question=original_question,
            extra_queries=extra_queries,
            prior_results=prior_results,
            activity_round=activity_round,
        )
        evidence_docs = _unique((item.chunk.filename for item in baseline.results), 20)
        return SynthesisRetrievalBundle(
            results=list(baseline.results),
            search_queries=_unique([*queries, *baseline.search_queries], 16),
            candidate_count=baseline.candidate_count,
            routed_documents=list(baseline.routed_documents),
            answer_strategy=plan.answer_strategy,
            synthesis_dimensions=list(plan.dimensions),
            considered_documents=list(baseline.routed_documents),
            evidence_documents=evidence_docs,
            search_round=activity_round,
        )

    emit_progress(
        f"synthesis_baseline_{activity_round}",
        "Searching the full corpus for synthesis",
        f"Running broad v5 candidate generation for {len(queries)} evidence formulation(s)",
        actor="search",
        phase="search",
        status="running",
        operation_id=f"synthesis-baseline-{activity_round}",
        metrics={"query_variants": len(queries)},
    )
    baseline_v5 = retrieve_v5(db, interpretation, extra_queries=extra_queries)
    baseline_results = _merge_results([*prior_results, *baseline_v5.results])
    emit_progress(
        f"synthesis_baseline_{activity_round}",
        "Broad synthesis search complete",
        f"Collected {len(baseline_results)} unique broad candidate(s) before document discovery",
        actor="search",
        phase="search",
        status="complete",
        operation_id=f"synthesis-baseline-{activity_round}",
        metrics={"baseline_candidates": baseline_v5.candidate_count, "accumulated_candidates": len(baseline_results)},
    )

    routes = _discovery_routes(
        db,
        interpretation=interpretation,
        plan=plan,
        queries=queries,
        baseline=baseline_results,
    )
    emit_progress(
        f"corpus_discovery_{activity_round}",
        "Relevant-document discovery complete",
        f"{len(routes)} document(s) passed the synthesis relevance threshold",
        actor="search",
        phase="route",
        status="complete",
        operation_id=f"corpus-discovery-{activity_round}",
        metrics={
            "relevant_documents": len(routes),
            "query_dimensions": len(queries),
        },
    )

    if not routes:
        evidence_docs = _unique((item.chunk.filename for item in baseline_results), 20)
        return SynthesisRetrievalBundle(
            results=list(baseline_results),
            search_queries=_unique([*queries, *baseline_v5.search_queries], 16),
            candidate_count=len(baseline_results),
            routed_documents=evidence_docs,
            answer_strategy=plan.answer_strategy,
            synthesis_dimensions=list(plan.dimensions),
            considered_documents=evidence_docs,
            evidence_documents=evidence_docs,
            search_round=activity_round,
        )

    scoped = _scoped_candidates(db, routes, queries)
    headings = _heading_candidates(db, routes, queries, interpretation)
    candidates = _merge_results([
        *baseline_results,
        *headings,
        *scoped,
    ])
    candidate_count = len(candidates)

    reranked = _synthesis_rerank(interpretation, plan, candidates, routes)
    expanded = _expand_sections(db, interpretation, reranked)
    combined = _merge_results([*reranked, *expanded])
    final_limit = _int_env("RAG_V52_FINAL_CANDIDATES", 96, 32, 140)
    final = _final_diverse(combined, routes, final_limit)
    evidence_docs = _unique((item.chunk.filename for item in final), 32)

    emit_progress(
        f"synthesis_retrieval_{activity_round}",
        "Cross-document evidence assembled",
        f"{len(final)} evidence candidate(s) from {len(evidence_docs)} document(s)",
        actor="search",
        phase="search",
        status="complete",
        operation_id=f"synthesis-retrieval-{activity_round}",
        metrics={
            "candidate_chunks": candidate_count,
            "final_candidates": len(final),
            "evidence_documents": len(evidence_docs),
            "relevant_documents": len(routes),
        },
    )

    return SynthesisRetrievalBundle(
        results=final,
        search_queries=_unique([*queries, *baseline_v5.search_queries], 16),
        candidate_count=candidate_count,
        routed_documents=[route.filename for route in routes],
        answer_strategy=plan.answer_strategy,
        synthesis_dimensions=list(plan.dimensions),
        considered_documents=[route.filename for route in routes],
        evidence_documents=evidence_docs,
        search_round=activity_round,
    )


def _coverage_prompt(
    interpretation: SmartInterpretation,
    bundle: SynthesisRetrievalBundle,
) -> str:
    limit = _int_env("RAG_V52_COVERAGE_CANDIDATES", 48, 18, 70)
    # Use a document-balanced pool so a long/repetitive PDF cannot dominate coverage review.
    # Build a document-balanced review sample so a long PDF cannot dominate coverage review.
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in bundle.results:
        by_doc[item.chunk.filename].append(item)
    candidates: list[RetrievedChunk] = []
    seen: set[str] = set()
    for filename in bundle.evidence_documents:
        for item in by_doc.get(filename, [])[:2]:
            if item.chunk.chunk_id not in seen:
                seen.add(item.chunk.chunk_id)
                candidates.append(item)
                if len(candidates) >= limit:
                    break
        if len(candidates) >= limit:
            break
    for item in bundle.results:
        if len(candidates) >= limit:
            break
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        candidates.append(item)

    blocks: list[str] = []
    excerpt_chars = _int_env("RAG_V52_COVERAGE_EXCERPT_CHARS", 620, 220, 1000)
    for index, item in enumerate(candidates, 1):
        chunk = item.chunk
        pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "")
        excerpt = re.sub(r"\s+", " ", chunk.text).strip()[:excerpt_chars]
        blocks.append(
            f"[E{index}] File: {chunk.filename} | Pages: {pages} | Section: {section or 'Unsectioned'}\n"
            f"Retrieval: {item.method}\nExcerpt: {excerpt}"
        )

    return f"""RESOLVED QUESTION:
{interpretation.resolved_question}

ANSWER STRATEGY:
{bundle.answer_strategy}

REQUIRED SYNTHESIS DIMENSIONS:
{chr(10).join(f'- {item}' for item in bundle.synthesis_dimensions) or '- direct answer evidence'}

EXPLICIT SCOPE:
{interpretation.scope or 'None specified'}

RELEVANT DOCUMENT CANDIDATES DISCOVERED BY RETRIEVAL
(routing signals only; a listed document is not automatically a contributor):
{chr(10).join(f'- {name}' for name in bundle.considered_documents) or 'None'}

EVIDENCE:
{chr(10).join(chr(10) + block for block in blocks)}

Review cross-document completeness and conflicts. Do not answer the user.
"""


def _string_tuple(value: object, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_unique((str(item) for item in value if str(item).strip()), limit))


def _conflicts(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    output: list[dict[str, object]] = []
    allowed = {"complementary", "scope_difference", "authority_difference", "unresolved"}
    for item in value:
        if not isinstance(item, dict):
            continue
        conflict_type = str(item.get("type") or "unresolved").casefold().strip()
        if conflict_type not in allowed:
            conflict_type = "unresolved"
        documents = _string_tuple(item.get("documents"), 8)
        summary = _clean(item.get("summary"))[:600]
        resolution = _clean(item.get("resolution"))[:600]
        output.append(
            {
                "type": conflict_type,
                "documents": list(documents),
                "summary": summary,
                "resolution": resolution,
            }
        )
        if len(output) >= 8:
            break
    return tuple(output)


def review_bundle_coverage(
    db: Session,
    interpretation: SmartInterpretation,
    bundle: SynthesisRetrievalBundle,
) -> SynthesisCoverage:
    if bundle.answer_strategy != "multi_document_synthesis":
        review = review_retrieved_evidence(interpretation, bundle.results)
        docs = tuple(_unique((item.chunk.filename for item in bundle.results), 16))
        return SynthesisCoverage(
            sufficient=review.sufficient,
            missing_evidence=review.missing_evidence,
            retry_queries=review.retry_queries,
            reason=review.reason,
            ai_used=review.ai_used,
            answer_strategy=bundle.answer_strategy,
            covered_dimensions=tuple(bundle.synthesis_dimensions) if review.sufficient else (),
            uncovered_dimensions=review.missing_evidence,
            contributing_documents=docs,
            conflicts=(),
            evidence_coverage_status="complete" if review.sufficient else "incomplete",
        )

    settings = get_settings()
    emit_progress(
        f"synthesis_coverage_{bundle.search_round}",
        "Checking cross-document coverage",
        "Comparing responsibilities, conditions, exceptions, scope and authority across relevant evidence",
        actor="verification",
        phase="review",
        status="running",
        operation_id=f"synthesis-coverage-{bundle.search_round}",
        prompt_summary="Check whether all requested synthesis dimensions are supported across materially relevant documents and identify unresolved differences.",
    )
    try:
        raw = llm_service.generate(
            _SYNTHESIS_COVERAGE_SYSTEM,
            _coverage_prompt(interpretation, bundle),
            max_output_tokens=_int_env("RAG_V52_COVERAGE_MAX_OUTPUT_TOKENS", 1600, 600, 2600),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        payload = _json_object(raw)
        sufficient = bool(payload.get("sufficient", False))
        covered = _string_tuple(payload.get("covered_dimensions"), 12)
        missing = _string_tuple(payload.get("missing_dimensions"), 8)
        retry_queries = _string_tuple(payload.get("retry_queries"), 3)
        contributing = _string_tuple(payload.get("contributing_documents"), 24)
        conflicts = _conflicts(payload.get("conflicts"))
        reason = _clean(payload.get("reason"))[:900]
        if not contributing:
            contributing = tuple(bundle.evidence_documents[:16])
        review = SynthesisCoverage(
            sufficient=sufficient,
            missing_evidence=missing,
            retry_queries=retry_queries,
            reason=reason,
            ai_used=True,
            answer_strategy=bundle.answer_strategy,
            covered_dimensions=covered,
            uncovered_dimensions=missing,
            contributing_documents=contributing,
            conflicts=conflicts,
            evidence_coverage_status="complete" if sufficient else "incomplete",
        )
    except Exception:
        fallback = review_retrieved_evidence(interpretation, bundle.results)
        review = SynthesisCoverage(
            sufficient=fallback.sufficient,
            missing_evidence=fallback.missing_evidence,
            retry_queries=fallback.retry_queries,
            reason=fallback.reason,
            ai_used=fallback.ai_used,
            answer_strategy=bundle.answer_strategy,
            covered_dimensions=tuple(bundle.synthesis_dimensions) if fallback.sufficient else (),
            uncovered_dimensions=fallback.missing_evidence,
            contributing_documents=tuple(bundle.evidence_documents[:16]),
            conflicts=(),
            evidence_coverage_status="complete" if fallback.sufficient else "incomplete",
        )

    unresolved = sum(1 for conflict in review.conflicts if conflict.get("type") == "unresolved")
    emit_progress(
        f"synthesis_coverage_{bundle.search_round}",
        "Cross-document coverage complete" if review.sufficient else "Cross-document evidence still incomplete",
        (
            f"{len(review.contributing_documents)} contributing document(s); "
            f"{len(review.covered_dimensions)} dimension(s) covered; "
            f"{len(review.uncovered_dimensions)} missing; {unresolved} unresolved conflict(s)"
        ),
        actor="verification",
        phase="review",
        status="complete" if review.sufficient else "warning",
        operation_id=f"synthesis-coverage-{bundle.search_round}",
        reasoning_summary=(review.reason or "Coverage review completed")[:1000],
        metrics={
            "contributing_documents": len(review.contributing_documents),
            "covered_dimensions": len(review.covered_dimensions),
            "missing_dimensions": len(review.uncovered_dimensions),
            "conflicts": len(review.conflicts),
            "unresolved_conflicts": unresolved,
        },
    )
    return review


def select_results_for_answer(
    results: Sequence[RetrievedChunk],
    review: SynthesisCoverage,
) -> list[RetrievedChunk]:
    if review.answer_strategy != "multi_document_synthesis" or not review.contributing_documents:
        return list(results)
    allowed = {name.casefold() for name in review.contributing_documents}
    for conflict in review.conflicts:
        for document in conflict.get("documents", []):
            allowed.add(str(document).casefold())
    selected = [item for item in results if item.chunk.filename.casefold() in allowed]
    # Do not pad a valid contributing-document set with unrelated chunks merely to
    # hit a numeric quota. If classification unexpectedly produces no matching chunks,
    # fall back to the strongest evidence rather than returning an empty answer context.
    if not selected:
        return list(results[: min(12, len(results))])
    return selected


def coverage_prompt_status(
    base_status: str,
    bundle: SynthesisRetrievalBundle,
    review: SynthesisCoverage,
) -> str:
    if bundle.answer_strategy != "multi_document_synthesis":
        return base_status
    lines = [
        base_status,
        "Answer strategy: multi_document_synthesis",
        "Instruction: synthesize all materially contributing documents; do not force incidental documents into the answer.",
        "Synthesis dimensions: " + ("; ".join(bundle.synthesis_dimensions) or "direct answer evidence"),
        "Contributing documents: " + ("; ".join(review.contributing_documents) or "not yet established"),
    ]
    if review.uncovered_dimensions:
        lines.append("Uncovered dimensions: " + "; ".join(review.uncovered_dimensions))
    if review.conflicts:
        lines.append("Cross-document differences/conflicts identified by evidence review:")
        for conflict in review.conflicts[:6]:
            lines.append(
                f"- {conflict.get('type')}: {conflict.get('summary') or 'difference detected'}; "
                f"resolution: {conflict.get('resolution') or 'not established'}"
            )
    return "\n".join(lines)


def response_synthesis_metadata(
    bundle: SynthesisRetrievalBundle,
    review: SynthesisCoverage,
) -> dict[str, object]:
    return {
        "answer_strategy": bundle.answer_strategy,
        "synthesis_dimensions": list(bundle.synthesis_dimensions),
        "search_scope": "broad_relevant_corpus" if bundle.answer_strategy == "multi_document_synthesis" else "focused",
        "relevant_documents": list(bundle.considered_documents),
        "contributing_documents": list(review.contributing_documents),
        "search_rounds": int(bundle.search_round),
        "evidence_coverage_status": review.evidence_coverage_status,
        "conflicts": [dict(item) for item in review.conflicts],
    }
