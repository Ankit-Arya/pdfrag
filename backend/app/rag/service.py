from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db_models import Document, DocumentChunk, DocumentStatus
from app.models import AnswerResponse, SourceResult
from app.rag.chunking import chunk_pages
from app.rag.embeddings import EmbeddingUnavailableError, embedding_service
from app.rag.evidence import build_evidence_answer
from app.rag.guardrails import cited_source_numbers, validate_grounded_answer
from app.rag.pdf import PdfProcessingError, extract_pdf_pages
from app.rag.postgres_store import (
    fetch_neighbor_chunks,
    find_abbreviation_hints,
    scan_matching_chunks,
    search_chunks,
)
from app.rag.prompts import NO_ANSWER
from app.rag.query import query_planner
from app.rag.relevance import select_context_chunks
from app.rag.synthesis import (
    repair_direct_answer,
    repair_hierarchical_answer,
    synthesize_answer,
)
from app.rag.types import PromptSource, QueryPlan, RetrievedChunk

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self) -> None:
        # PDF parsing, OCR and transformer inference are the memory-heavy path.
        # Keep ingestion serialized in this single-worker deployment so multiple
        # admins/batches cannot multiply peak memory usage.
        self._processing_lock = threading.Lock()

    def process_document(self, db: Session, document: Document) -> Document:
        with self._processing_lock:
            return self._process_document_locked(db, document)

    def _process_document_locked(self, db: Session, document: Document) -> Document:
        settings = get_settings()
        document.status = DocumentStatus.processing
        document.error = None
        db.commit()

        try:
            result = extract_pdf_pages(document.content, document.filename)
            chunks = chunk_pages(
                result.blocks,
                chunk_size=settings.chunk_size_chars,
                overlap=settings.chunk_overlap_chars,
            )
            if not chunks:
                raise ValueError("No chunks were created")
            if len(chunks) > settings.max_chunks_per_collection:
                raise ValueError(
                    f"Document produced {len(chunks)} chunks, above MAX_CHUNKS_PER_COLLECTION. "
                    "Increase the limit or split the PDF."
                )

            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
            embedding_batch_size = 64
            for start in range(0, len(chunks), embedding_batch_size):
                batch_chunks = chunks[start : start + embedding_batch_size]
                vectors = embedding_service.encode([chunk.text for chunk in batch_chunks])
                rows = [
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=start + offset,
                        page_number=chunk.page_number,
                        content_type=chunk.content_type,
                        text=chunk.text,
                        embedding=vector.tolist(),
                    )
                    for offset, (chunk, vector) in enumerate(
                        zip(batch_chunks, vectors, strict=True)
                    )
                ]
                db.add_all(rows)
                db.flush()
                del rows, vectors

            document.page_count = result.total_pages
            document.chunk_count = len(chunks)
            document.warnings = result.warnings
            document.status = DocumentStatus.ready
            document.processed_at = datetime.now(UTC)
            db.commit()
            db.refresh(document)
            return document
        except Exception as exc:
            db.rollback()
            refreshed = db.get(Document, document.id)
            if refreshed:
                refreshed.status = DocumentStatus.failed
                if isinstance(exc, (PdfProcessingError, EmbeddingUnavailableError, ValueError)):
                    refreshed.error = str(exc)[:4000]
                else:
                    refreshed.error = "Document processing failed. Check server logs for details."
                db.commit()
            logger.exception("Document processing failed for %s", document.filename)
            raise

    def ask(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        rewrite_question: bool | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> AnswerResponse:
        settings = get_settings()
        history = conversation_context or []
        abbreviation_probe = "\n".join(
            [
                question,
                *[
                    str(turn.get("content", ""))
                    for turn in history
                    if str(turn.get("role", "")).casefold() == "user"
                ],
            ]
        )
        abbreviation_hints = find_abbreviation_hints(db, abbreviation_probe)
        plan = query_planner.plan(
            question,
            enabled=rewrite_question,
            conversation_context=history,
            abbreviation_hints=abbreviation_hints,
        )

        context_limit = (
            settings.reference_evidence_chunk_limit
            if plan.search_mode == "references"
            else (top_k or settings.answer_evidence_chunk_limit)
        )
        # Semantic retrieval remains useful for paraphrases, but is no longer the
        # only gate. Keep its per-query working set bounded because the corpus-wide
        # lexical scan below separately evaluates every ready chunk.
        semantic_limit = min(
            max(context_limit * 2, 128),
            settings.max_retrieval_candidates,
        )
        merged: dict[str, RetrievedChunk] = {}

        vectors = embedding_service.encode(plan.search_queries)
        for query, vector in zip(plan.search_queries, vectors, strict=True):
            for item in search_chunks(db, vector.tolist(), query, semantic_limit):
                _merge_candidate(merged, item)

        corpus_queries = _unique(
            [
                plan.contextual_question or plan.original_question,
                plan.original_question,
                *plan.search_queries[:3],
            ]
        )
        corpus_focus = _unique([*plan.focus_terms, *plan.context_terms, *plan.keywords])
        for item in scan_matching_chunks(
            db,
            corpus_queries,
            focus_terms=corpus_focus,
            reference_mode=plan.search_mode == "references",
            limit=settings.corpus_scan_max_chunks,
        ):
            _merge_candidate(merged, item)

        candidates = sorted(merged.values(), key=_result_sort_key)
        if not candidates:
            return self._empty_answer(plan, abbreviation_hints, candidate_chunks=0)

        relevant = self._select_context_chunks(plan, candidates, context_limit)
        if not relevant:
            return self._empty_answer(
                plan,
                abbreviation_hints,
                candidate_chunks=len(candidates),
            )

        if plan.search_mode == "references":
            return self._reference_answer(
                plan,
                relevant,
                abbreviation_hints,
                candidate_chunks=len(candidates),
            )

        expanded = self._expand_context(db, plan, relevant, context_limit)
        bundle = synthesize_answer(plan, expanded)
        answer, grounded = validate_grounded_answer(bundle.raw_answer, len(bundle.sources))
        grounding_status = "verified" if grounded else "citation_validation_failed"

        if not grounded and answer != NO_ANSWER:
            if bundle.used_hierarchy:
                repaired_raw = repair_hierarchical_answer(
                    plan,
                    answer,
                    bundle.sources,
                    bundle.digests,
                )
                repaired_answer, repaired_grounded = validate_grounded_answer(
                    repaired_raw,
                    len(bundle.sources),
                )
                if repaired_grounded or repaired_answer != NO_ANSWER:
                    answer = repaired_answer
                    grounded = repaired_grounded
                    grounding_status = (
                        "verified_after_repair" if grounded else "citation_validation_failed"
                    )
            else:
                repaired_raw = repair_direct_answer(plan, answer, bundle.sources)
                repaired_answer, repaired_grounded = validate_grounded_answer(
                    repaired_raw,
                    len(bundle.sources),
                )
                if repaired_grounded or repaired_answer != NO_ANSWER:
                    answer = repaired_answer
                    grounded = repaired_grounded
                    grounding_status = (
                        "verified_after_repair" if grounded else "citation_validation_failed"
                    )

        sources = _source_results(answer, bundle.sources)
        evidence = _evidence_results(bundle.sources)
        return AnswerResponse(
            answer=answer,
            sources=sources,
            evidence=evidence,
            grounded=grounded,
            grounding_status=grounding_status,
            interpreted_question=plan.contextual_question or plan.rewritten_question,
            contextual_question=plan.contextual_question or plan.rewritten_question,
            retrieval_mode=plan.search_mode,
            resolved_abbreviations=abbreviation_hints,
            candidate_chunks=len(candidates),
            evidence_chunks=len(bundle.sources),
            search_queries=plan.search_queries,
        )

    def _reference_answer(
        self,
        plan: QueryPlan,
        relevant: list[RetrievedChunk],
        abbreviation_hints: list[str],
        *,
        candidate_chunks: int,
    ) -> AnswerResponse:
        prompt_sources = [PromptSource(result=item, excerpt=item.chunk.text) for item in relevant]
        raw, used_context = build_evidence_answer(
            plan.contextual_question or plan.original_question,
            prompt_sources,
        )
        if not raw or not used_context:
            return self._empty_answer(
                plan,
                abbreviation_hints,
                candidate_chunks=candidate_chunks,
            )
        answer, grounded = validate_grounded_answer(raw, len(used_context))
        return AnswerResponse(
            answer=answer,
            sources=_source_results(answer, used_context),
            evidence=_evidence_results(used_context),
            grounded=grounded,
            grounding_status="verified" if grounded else "citation_validation_failed",
            interpreted_question=plan.contextual_question or plan.rewritten_question,
            contextual_question=plan.contextual_question or plan.rewritten_question,
            retrieval_mode="references",
            resolved_abbreviations=abbreviation_hints,
            candidate_chunks=candidate_chunks,
            evidence_chunks=len(used_context),
            search_queries=plan.search_queries,
        )

    def _empty_answer(
        self,
        plan: QueryPlan,
        abbreviation_hints: list[str],
        *,
        candidate_chunks: int,
    ) -> AnswerResponse:
        return AnswerResponse(
            answer=NO_ANSWER,
            sources=[],
            evidence=[],
            grounded=False,
            grounding_status="insufficient_evidence",
            interpreted_question=plan.contextual_question or plan.rewritten_question,
            contextual_question=plan.contextual_question or plan.rewritten_question,
            retrieval_mode=plan.search_mode,
            resolved_abbreviations=abbreviation_hints,
            candidate_chunks=candidate_chunks,
            evidence_chunks=0,
            search_queries=plan.search_queries,
        )

    def _expand_context(
        self,
        db: Session,
        plan: QueryPlan,
        relevant: list[RetrievedChunk],
        requested_limit: int,
    ) -> list[RetrievedChunk]:
        settings = get_settings()
        seeds = relevant[: min(len(relevant), settings.neighbor_seed_limit)]
        neighbors = fetch_neighbor_chunks(db, seeds, window=settings.neighbor_window)
        merged: dict[str, RetrievedChunk] = {}
        for item in [*relevant, *neighbors]:
            _merge_candidate(merged, item)

        selected = self._select_context_chunks(
            plan,
            list(merged.values()),
            requested_limit,
        )
        return selected or relevant

    @staticmethod
    def _select_context_chunks(
        plan: QueryPlan,
        candidates: list[RetrievedChunk],
        max_chunks: int | None = None,
    ) -> list[RetrievedChunk]:
        return select_context_chunks(plan, candidates, max_chunks=max_chunks)


def _merge_candidate(target: dict[str, RetrievedChunk], item: RetrievedChunk) -> None:
    old = target.get(item.chunk.chunk_id)
    if old is None or item.score > old.score:
        target[item.chunk.chunk_id] = item
    elif old is not None and item.score == old.score and item.method < old.method:
        target[item.chunk.chunk_id] = item


def _source_results(answer: str, context: list[PromptSource]) -> list[SourceResult]:
    used_source_numbers = set(cited_source_numbers(answer, len(context)))
    return [
        SourceResult(
            id=f"S{index}",
            filename=source.result.chunk.filename,
            page=source.result.chunk.page_number,
            score=round(source.result.score, 4),
            excerpt=source.excerpt,
            content_type=source.result.chunk.content_type,
            retrieval_method=source.result.method,
        )
        for index, source in enumerate(context, 1)
        if index in used_source_numbers
    ]


def _evidence_results(context: list[PromptSource]) -> list[SourceResult]:
    return [
        SourceResult(
            id=f"S{index}",
            filename=source.result.chunk.filename,
            page=source.result.chunk.page_number,
            score=round(source.result.score, 4),
            excerpt=source.excerpt,
            content_type=source.result.chunk.content_type,
            retrieval_method=source.result.method,
        )
        for index, source in enumerate(context, 1)
    ]


def _result_sort_key(item: RetrievedChunk) -> tuple[float, str, int, int, str]:
    chunk = item.chunk
    return (
        -float(item.score),
        chunk.filename.casefold(),
        chunk.page_number,
        chunk.chunk_index if chunk.chunk_index is not None else -1,
        chunk.chunk_id,
    )


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


rag_service = RagService()
