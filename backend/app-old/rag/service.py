import logging
from collections import Counter

import numpy as np

from app.config import get_settings
from app.models import AnswerResponse, FileSummary, SourceResult
from app.rag.chunking import chunk_pages
from app.rag.embeddings import embedding_service
from app.rag.guardrails import validate_grounded_answer
from app.rag.llm import llm_service
from app.rag.pdf import extract_pdf_pages
from app.rag.prompts import NO_ANSWER, build_citation_repair_prompt, build_user_prompt
from app.rag.store import Collection, collection_store
from app.rag.types import PageText, RetrievedChunk

logger = logging.getLogger(__name__)


class RagService:
    def build_collection(self, uploaded_files: list[tuple[str, bytes]]) -> Collection:
        settings = get_settings()
        all_pages: list[PageText] = []
        per_file_pages: Counter[str] = Counter()

        for filename, data in uploaded_files:
            pages = extract_pdf_pages(data, filename)
            all_pages.extend(pages)
            per_file_pages[filename] = len(pages)
            if len(all_pages) > settings.max_total_pages:
                raise ValueError(
                    f"The collection exceeds the {settings.max_total_pages}-page limit."
                )

        chunks = chunk_pages(
            all_pages,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
        if not chunks:
            raise ValueError("No text chunks could be created from the uploaded PDFs.")

        vectors = embedding_service.encode([chunk.text for chunk in chunks])
        chunks_per_file = Counter(chunk.filename for chunk in chunks)
        files = [
            FileSummary(
                name=filename,
                pages=per_file_pages[filename],
                chunks=chunks_per_file[filename],
            )
            for filename, _ in uploaded_files
        ]
        logger.info(
            "Built RAG collection: files=%s pages=%s chunks=%s",
            len(uploaded_files),
            len(all_pages),
            len(chunks),
        )
        return collection_store.create(chunks, vectors, files, len(all_pages))

    def ask(self, collection_id: str, question: str, top_k: int | None = None) -> AnswerResponse:
        settings = get_settings()
        collection = collection_store.get(collection_id)
        query_vector = embedding_service.encode([question])[0]
        relevant = self._retrieve_relevant_chunks(collection, question, query_vector, top_k)

        if not relevant:
            logger.info("RAG no-answer: retrieval returned no chunks")
            return AnswerResponse(answer=NO_ANSWER, sources=[], grounded=False)

        prompt, context_results = build_user_prompt(
            question, relevant, settings.max_context_chars
        )
        if not context_results:
            logger.info("RAG no-answer: max_context_chars excluded all chunks")
            return AnswerResponse(answer=NO_ANSWER, sources=[], grounded=False)

        raw_answer = llm_service.answer(prompt)
        answer, grounded = validate_grounded_answer(raw_answer, len(context_results))
        sources = self._format_sources(context_results)

        if grounded:
            return AnswerResponse(answer=answer, sources=sources, grounded=True)

        if raw_answer.strip() == NO_ANSWER:
            logger.info("RAG no-answer: LLM reported insufficient PDF evidence")
            return AnswerResponse(answer=NO_ANSWER, sources=sources, grounded=False)

        # One retry for formatting/citation failures. This addresses the common case
        # where the answer is source-grounded but the model missed one citation.
        repair_prompt, repair_context_results = build_citation_repair_prompt(
            question=question,
            previous_answer=raw_answer,
            results=context_results,
            max_context_chars=settings.max_context_chars,
        )
        repaired_raw = llm_service.answer(repair_prompt)
        repaired_answer, repaired_grounded = validate_grounded_answer(
            repaired_raw, len(repair_context_results)
        )
        repaired_sources = self._format_sources(repair_context_results)

        if repaired_grounded:
            logger.info("RAG answer accepted after citation repair")
            return AnswerResponse(
                answer=repaired_answer,
                sources=repaired_sources,
                grounded=True,
            )

        # Development-friendly fallback: do not hide useful retrieved context behind
        # the generic no-answer message. The frontend can still mark grounded=false.
        logger.warning("RAG citation validation failed; returning raw answer with sources")
        return AnswerResponse(
            answer=raw_answer.strip() or NO_ANSWER,
            sources=sources,
            grounded=False,
        )

    def _retrieve_relevant_chunks(
        self,
        collection: Collection,
        question: str,
        query_vector: np.ndarray,
        top_k: int | None,
    ) -> list[RetrievedChunk]:
        settings = get_settings()
        requested_top_k = top_k or settings.top_k
        # A small top_k is a common cause of false no-answer responses. Keep caller
        # overrides, but make default retrieval wider for PDFs.
        effective_top_k = max(int(requested_top_k), 12)
        candidate_k = max(effective_top_k * 3, 24)

        if hasattr(collection, "hybrid_search"):
            retrieved = collection.hybrid_search(
                query=question,
                query_vector=query_vector,
                top_k=effective_top_k,
                vector_k=candidate_k,
                keyword_k=candidate_k,
            )
        else:
            retrieved = collection.search(query_vector, effective_top_k)

        if not retrieved:
            return []

        configured_min_similarity = float(getattr(settings, "min_similarity", 0.0))
        # Cap the threshold to avoid dropping reasonable semantic matches. The LLM
        # still receives only retrieved PDF excerpts and must answer from them.
        effective_min_similarity = min(configured_min_similarity, 0.08)
        relevant = [item for item in retrieved if item.score >= effective_min_similarity]

        if relevant:
            return relevant[:effective_top_k]

        # Last-resort context fallback: when all scores are low, still provide the
        # best chunks to the grounded prompt instead of failing before the LLM sees
        # any PDF text.
        logger.info(
            "RAG retrieval scores below threshold; using best available chunks. "
            "top_score=%s threshold=%s",
            round(float(retrieved[0].score), 4),
            effective_min_similarity,
        )
        return retrieved[: min(effective_top_k, len(retrieved))]

    @staticmethod
    def _format_sources(results: list[RetrievedChunk]) -> list[SourceResult]:
        formatted: list[SourceResult] = []
        for number, result in enumerate(results, start=1):
            formatted.append(
                SourceResult(
                    id=f"S{number}",
                    filename=result.chunk.filename,
                    page=result.chunk.page_number,
                    score=round(result.score, 4),
                    excerpt=result.chunk.text[:500],
                )
            )
        return formatted


rag_service = RagService()
