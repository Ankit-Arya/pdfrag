from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db_models import Document, DocumentChunk, DocumentStatus
from app.models import AnswerResponse, SourceResult
from app.rag.chunking import chunk_pages
from app.rag.embeddings import embedding_service
from app.rag.guardrails import validate_grounded_answer
from app.rag.llm import llm_service
from app.rag.pdf import extract_pdf_pages
from app.rag.postgres_store import fetch_neighbor_chunks, search_chunks
from app.rag.prompts import NO_ANSWER, build_citation_repair_prompt, build_user_prompt
from app.rag.query import query_planner
from app.rag.relevance import select_context_chunks
from app.rag.types import QueryPlan, RetrievedChunk

logger = logging.getLogger(__name__)


class RagService:
    def process_document(self, db: Session, document: Document) -> Document:
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

            vectors = embedding_service.encode([chunk.text for chunk in chunks])
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                db.add(
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=index,
                        page_number=chunk.page_number,
                        content_type=chunk.content_type,
                        text=chunk.text,
                        embedding=vector.tolist(),
                    )
                )

            document.page_count = result.total_pages
            document.chunk_count = len(chunks)
            document.warnings = result.warnings
            document.status = DocumentStatus.ready
            document.processed_at = datetime.now(UTC)
            db.commit()
            db.refresh(document)
            return document
        except Exception:
            db.rollback()
            refreshed = db.get(Document, document.id)
            if refreshed:
                refreshed.status = DocumentStatus.failed
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
    ) -> AnswerResponse:
        settings = get_settings()
        plan = query_planner.plan(question, enabled=rewrite_question)
        vectors = embedding_service.encode(plan.search_queries)

        requested_limit = top_k or settings.top_k
        retrieval_limit = max(requested_limit * 8, 48)
        merged: dict[str, RetrievedChunk] = {}

        for query, vector in zip(plan.search_queries, vectors, strict=True):
            for item in search_chunks(db, vector.tolist(), query, retrieval_limit):
                old = merged.get(item.chunk.chunk_id)
                if old is None or item.score > old.score:
                    merged[item.chunk.chunk_id] = item

        candidates = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if not candidates:
            return AnswerResponse(
                answer=NO_ANSWER,
                sources=[],
                grounded=False,
                grounding_status="insufficient_evidence",
                interpreted_question=plan.rewritten_question,
                search_queries=plan.search_queries,
            )

        relevant = self._select_context_chunks(plan, candidates, requested_limit)
        if not relevant:
            return AnswerResponse(
                answer=NO_ANSWER,
                sources=[],
                grounded=False,
                grounding_status="insufficient_evidence",
                interpreted_question=plan.rewritten_question,
                search_queries=plan.search_queries,
            )

        expanded = self._expand_context(db, plan, relevant, requested_limit)
        prompt, context = build_user_prompt(
            plan.original_question,
            plan.rewritten_question,
            expanded,
            settings.max_context_chars,
            question_intent=plan.intent,
        )
        if not context:
            return AnswerResponse(
                answer=NO_ANSWER,
                sources=[],
                grounded=False,
                grounding_status="insufficient_evidence",
                interpreted_question=plan.rewritten_question,
                search_queries=plan.search_queries,
            )

        raw = llm_service.answer(prompt)
        answer, grounded = validate_grounded_answer(raw, len(context))
        grounding_status = "verified" if grounded else "citation_validation_failed"

        if not grounded and answer != NO_ANSWER:
            repair_prompt, repair_context = build_citation_repair_prompt(
                plan.original_question,
                plan.rewritten_question,
                answer,
                expanded,
                settings.max_context_chars,
            )
            if repair_context:
                repaired_raw = llm_service.answer(repair_prompt)
                repaired_answer, repaired_grounded = validate_grounded_answer(
                    repaired_raw,
                    len(repair_context),
                )
                if repaired_grounded or repaired_answer != NO_ANSWER:
                    answer = repaired_answer
                    grounded = repaired_grounded
                    grounding_status = "verified_after_repair" if grounded else "citation_validation_failed"
                    context = repair_context

        sources = [
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
        return AnswerResponse(
            answer=answer,
            sources=sources,
            grounded=grounded,
            grounding_status=grounding_status,
            interpreted_question=plan.rewritten_question,
            search_queries=plan.search_queries,
        )

    def _expand_context(
        self,
        db: Session,
        plan: QueryPlan,
        relevant: list[RetrievedChunk],
        requested_limit: int,
    ) -> list[RetrievedChunk]:
        seeds = relevant[: min(len(relevant), requested_limit, 12)]
        neighbors = fetch_neighbor_chunks(db, seeds, window=1)
        merged: dict[str, RetrievedChunk] = {}
        for item in [*seeds, *neighbors]:
            old = merged.get(item.chunk.chunk_id)
            if old is None or item.score > old.score:
                merged[item.chunk.chunk_id] = item

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


rag_service = RagService()
