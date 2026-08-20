from __future__ import annotations

import logging
import os
import re
from collections import defaultdict

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import AnswerResponse, SourceResult
from app.rag.evidence_format import clean_display_excerpt
from app.rag.guardrails import cited_source_numbers, validate_grounded_answer
from app.rag.llm import llm_service
from app.rag.prompts import NO_ANSWER
from app.rag.progress import emit_progress
from app.rag.service import RagService
from app.rag.smart_understanding import (
    interpret_user_message,
    review_retrieved_evidence,
    verify_answer,
)
from app.rag.types import PromptSource, QueryPlan, RetrievedChunk
from app.rag.v5.ingestion import process_document_v5
from app.rag.v5.retrieval import V5RetrievalBundle, retrieve_v5
from app.rag.v5.terminology import terminology_hints

logger = logging.getLogger(__name__)

_STRUCTURE_RE = re.compile(r"\[PDF STRUCTURE\].*?\[/PDF STRUCTURE\]\s*", re.IGNORECASE | re.DOTALL)


def _display_excerpt(value: str) -> str:
    return clean_display_excerpt(_STRUCTURE_RE.sub("", value).strip())


def _formatted_sources(sources: list[PromptSource], source_numbers: set[int] | None = None) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, 1):
        if source_numbers is not None and index not in source_numbers:
            continue
        chunk = source.result.chunk
        pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "")
        blocks.append(
            f"### S{index} — {chunk.filename} — page {pages}\n"
            + (f"Section: {section}\n\n" if section else "")
            + _display_excerpt(source.excerpt)
        )
    return "\n\n".join(blocks)

_V5_ANSWER_SYSTEM = """You are the answer layer for a CLOSED-BOOK internal metro knowledge assistant.
The language-understanding layer has already resolved the user's likely intended question. Use ONLY
SOURCE evidence supplied in this request for factual metro claims. General language ability may be
used to explain, organize and connect explicitly supported evidence, but never to invent a procedure,
amount, speed, responsibility, acronym expansion, applicability rule or document revision.

USER EXPERIENCE:
- Behave like a capable conversational assistant, not a search-engine audit report.
- Answer the resolved question directly. Silently tolerate spelling, grammar and colloquial wording.
- Lead with the useful supported answer: Yes/No, role, amount, action, definition or procedure.
- Never start with 'the supplied excerpts', 'the retrieved excerpts', or similar retrieval-internal wording.
- If a colloquial category can match multiple formal source categories, state the supported alternatives
  or the missing distinction instead of silently choosing one.

GROUNDING:
- Every factual sentence, bullet or table row must cite one or more existing labels such as [S1].
- Do not create labels that were not supplied.
- Preserve exact numbers, units, conditions, mandatory/optional distinctions, exceptions and sequence.
- Table rows are structured evidence: preserve the relationship between row labels and column values.
- For acronym/full-form questions, preserve the exact expansion stated in the cited PDF source; do not
  paraphrase an official expansion into a plausible synonym.
- A current/substituted/amended provision outranks an explicitly superseded/historical copy for a current
  question. If the evidence does not establish precedence, state the conflict instead of guessing.
- Preserve line, rolling stock, mode, location, person type and scenario applicability.
- A failed retrieval is NOT proof that the corpus lacks a fact. If COVERAGE is insufficient, do not make
  a corpus-wide absence claim. State only that the requested point could not be verified from the
  governing evidence retrieved in this attempt.
- If useful supported evidence exists, answer it even when another requested detail remains unresolved.

For procedure questions, provide an actionable numbered sequence in operational order. For a direct
fact/value/definition, keep the answer concise. Do not dump source excerpts; the UI exposes them separately.
"""

_V5_CITATION_REPAIR_SYSTEM = """Repair citation grounding for a closed-book PDF answer.
Use ONLY the supplied source blocks. Do not add new factual claims. Keep the answer direct and preserve
its meaning, but ensure every factual sentence/bullet has valid [S#] citations and no citation is out of
range. If the draft contains a factual statement unsupported by the sources, remove or qualify it.
Return only the repaired answer."""


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 10000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _source_block(index: int, source: PromptSource) -> str:
    chunk = source.result.chunk
    pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
    section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "")
    return (
        f"[S{index}]\n"
        f"File: {chunk.filename}\n"
        f"Pages: {pages}\n"
        f"Content type: {chunk.content_type}\n"
        f"Section: {section or 'Unsectioned content'}\n"
        f"Retrieval: {source.result.method}\n"
        f"Text:\n{source.excerpt.strip()}\n"
    )


def _answer_prompt(
    *,
    question: str,
    resolved: str,
    interpretation: object,
    sources: list[PromptSource],
    coverage_status: str,
) -> str:
    evidence_needs = getattr(interpretation, "evidence_needs", ())
    scope = getattr(interpretation, "scope", {})
    ambiguity = getattr(interpretation, "ambiguity_note", "")
    return f"""ORIGINAL USER MESSAGE:
{question}

RESOLVED QUESTION (language understanding only; not factual evidence):
{resolved}

REQUESTED EVIDENCE NEEDS:
{chr(10).join(f'- {item}' for item in evidence_needs) if evidence_needs else '- Direct evidence answering the request'}

EXPLICIT/RESOLVED SCOPE:
{scope or 'None specified'}

MATERIAL AMBIGUITY:
{ambiguity or 'None identified'}

COVERAGE STATUS:
{coverage_status}

SOURCES:
{chr(10).join(_source_block(index, source) for index, source in enumerate(sources, 1))}

Answer the resolved question now using only these sources.
"""


def _repair_prompt(draft: str, sources: list[PromptSource]) -> str:
    return f"""DRAFT ANSWER:
{draft}

VALID SOURCES:
{chr(10).join(_source_block(index, source) for index, source in enumerate(sources, 1))}

Repair the draft citations without adding unsupported facts.
"""


def _source_results(answer: str, sources: list[PromptSource]) -> list[SourceResult]:
    used = set(cited_source_numbers(answer, len(sources)))
    output: list[SourceResult] = []
    for index, source in enumerate(sources, 1):
        if index not in used:
            continue
        chunk = source.result.chunk
        pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        output.append(
            SourceResult(
                id=f"S{index}",
                filename=chunk.filename,
                page=chunk.page_number,
                score=round(float(source.result.score), 4),
                excerpt=_display_excerpt(source.excerpt),
                content_type=chunk.content_type,
                retrieval_method=source.result.method,
                pages=pages,
                section=" > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or ""),
            )
        )
    return output


def _evidence_results(sources: list[PromptSource]) -> list[SourceResult]:
    output: list[SourceResult] = []
    for index, source in enumerate(sources, 1):
        chunk = source.result.chunk
        pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        output.append(
            SourceResult(
                id=f"S{index}",
                filename=chunk.filename,
                page=chunk.page_number,
                score=round(float(source.result.score), 4),
                excerpt=_display_excerpt(source.excerpt),
                content_type=chunk.content_type,
                retrieval_method=source.result.method,
                pages=pages,
                section=" > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or ""),
            )
        )
    return output


def _build_plan(question: str, interpretation: object, search_queries: list[str]) -> QueryPlan:
    resolved = str(getattr(interpretation, "resolved_question", question) or question)
    return QueryPlan(
        original_question=question,
        rewritten_question=resolved,
        contextual_question=resolved,
        search_queries=search_queries,
        keywords=list(getattr(interpretation, "concepts", ()))[:40],
        intent=str(getattr(interpretation, "intent", "fact_lookup")),
        response_mode="concise",
        focus_terms=list(getattr(interpretation, "concepts", ()))[:40],
        context_terms=[f"{key} {value}" for key, value in dict(getattr(interpretation, "scope", {})).items()][:32],
        used_ai_rewrite=bool(getattr(interpretation, "ai_used", False)),
        search_mode="references" if getattr(interpretation, "conversation_act", "question") == "navigation" else "answer",
    )


def _document_diverse_sources(results: list[RetrievedChunk], limit: int) -> list[PromptSource]:
    output: list[PromptSource] = []
    seen: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    for item in results:
        if item.chunk.chunk_id in seen:
            continue
        doc = item.chunk.document_id or item.chunk.filename
        if counts[doc] >= 14 and len(output) >= limit // 2:
            continue
        seen.add(item.chunk.chunk_id)
        counts[doc] += 1
        output.append(PromptSource(result=item, excerpt=item.chunk.text.strip()))
        if len(output) >= limit:
            break
    return output


class V5RagService(RagService):
    """Explicit v5 service. It does not rely on the v4 runtime monkeypatch chain."""

    def process_document(self, db: Session, document):  # type: ignore[override,no-untyped-def]
        summary = process_document_v5(db, document, publish_document_state=True)
        logger.info(
            "RAG v5 processed %s: pages=%d chunks=%d tables=%d rows=%d ocr=%d",
            document.filename,
            summary.pages,
            summary.chunks,
            summary.tables,
            summary.table_rows,
            summary.ocr_pages,
        )
        return document

    def _ask_impl(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        rewrite_question: bool | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> AnswerResponse:
        settings = get_settings()
        emit_progress("interpret", "Understanding your question", "Resolving likely intent before document search")
        grounded_terminology = terminology_hints(db, question)
        interpretation = interpret_user_message(
            question,
            history=conversation_context or [],
            abbreviation_hints=grounded_terminology,
            routing_hints=[],
        )
        resolved = interpretation.resolved_question or question
        emit_progress(
            "interpret",
            "Question interpreted",
            f"Intent: {interpretation.intent}; evidence needs: {len(interpretation.evidence_needs)}",
        )

        emit_progress("search", "Searching structured PDF evidence", "Searching sections, prose, table rows and current authority")
        bundle: V5RetrievalBundle = retrieve_v5(db, interpretation)
        review = review_retrieved_evidence(interpretation, bundle.results)
        coverage_status = "sufficient" if review.sufficient else "insufficient_after_review"
        if not review.sufficient and review.retry_queries:
            emit_progress("search_retry", "Filling evidence gaps", "Running one targeted retrieval retry")
            bundle = retrieve_v5(db, interpretation, extra_queries=review.retry_queries)
            final_review = review_retrieved_evidence(interpretation, bundle.results)
            coverage_status = "sufficient_after_retry" if final_review.sufficient else "insufficient_after_retry"

        if not bundle.results:
            plan = _build_plan(question, interpretation, bundle.search_queries)
            return AnswerResponse(
                answer=NO_ANSWER,
                sources=[], evidence=[], formatted_sources="", formatted_evidence="",
                grounded=False, grounding_status="insufficient_evidence",
                interpreted_question=resolved, contextual_question=resolved,
                retrieval_mode=plan.search_mode, candidate_chunks=0, evidence_chunks=0,
                search_queries=bundle.search_queries, answer_policy_version="rag-v5.0.0",
            )

        evidence_limit = min(top_k or _int_env("RAG_V5_FINAL_EVIDENCE", 32, 12, 80), 80)
        prompt_sources = _document_diverse_sources(bundle.results, evidence_limit)
        emit_progress(
            "answer_generation",
            "Writing the grounded answer",
            f"Reviewing {len(prompt_sources)} structure-preserving evidence unit(s)",
        )
        draft = llm_service.generate(
            _V5_ANSWER_SYSTEM,
            _answer_prompt(
                question=question,
                resolved=resolved,
                interpretation=interpretation,
                sources=prompt_sources,
                coverage_status=coverage_status,
            ),
            max_output_tokens=settings.max_output_tokens,
            model=settings.llm_model,
            reasoning_effort=settings.llm_reasoning_effort,
        )
        verified = verify_answer(
            interpretation,
            draft,
            prompt_sources,
            coverage_status=coverage_status,
        )
        answer, grounded = validate_grounded_answer(verified or draft, len(prompt_sources))
        grounding_status = "verified" if grounded else "citation_validation_failed"
        if answer != NO_ANSWER and not grounded:
            emit_progress("citation_repair", "Repairing citation grounding", "Keeping only claims supported by the reviewed PDF evidence")
            repaired = llm_service.generate(
                _V5_CITATION_REPAIR_SYSTEM,
                _repair_prompt(answer, prompt_sources),
                max_output_tokens=settings.max_output_tokens,
                model=settings.query_model,
                reasoning_effort=settings.query_reasoning_effort,
            )
            answer, grounded = validate_grounded_answer(repaired, len(prompt_sources))
            grounding_status = "verified_after_repair" if grounded else "citation_validation_failed"

        used = set(cited_source_numbers(answer, len(prompt_sources)))
        plan = _build_plan(question, interpretation, bundle.search_queries)
        return AnswerResponse(
            answer=answer,
            sources=_source_results(answer, prompt_sources),
            evidence=_evidence_results(prompt_sources),
            formatted_sources=_formatted_sources(prompt_sources, used),
            formatted_evidence=_formatted_sources(prompt_sources),
            grounded=grounded,
            grounding_status=grounding_status,
            interpreted_question=resolved,
            contextual_question=resolved,
            retrieval_mode=plan.search_mode,
            resolved_abbreviations=grounded_terminology,
            routing_hints=[],
            primary_documents=[],
            candidate_chunks=bundle.candidate_count,
            evidence_chunks=len(prompt_sources),
            search_queries=bundle.search_queries,
            answer_policy_version="rag-v5.0.0",
        )
