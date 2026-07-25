import logging
import re
from collections import Counter

import numpy as np

from app.config import get_settings
from app.models import AnswerResponse, FileSummary, SourceResult
from app.rag.chunking import chunk_pages
from app.rag.embeddings import embedding_service
from app.rag.guardrails import grounding_failure_reason, validate_grounded_answer
from app.rag.llm import llm_service
from app.rag.pdf import PdfExtractionResult, extract_pdf_pages
from app.rag.prompts import (
    NO_ANSWER,
    build_citation_repair_prompt,
    build_user_prompt,
)
from app.rag.query import query_planner
from app.rag.store import Collection, collection_store
from app.rag.types import PageText, PromptSource, QueryPlan, RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_LOGGED_ANSWER_CHARS = 4000
_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,9}(?![A-Za-z0-9])")

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "please", "the",
    "these", "this", "to", "what", "when", "where", "which", "who",
    "why", "with",
}


class RagService:
    def build_collection(
        self,
        uploaded_files: list[tuple[str, bytes]],
    ) -> Collection:
        settings = get_settings()
        all_blocks: list[PageText] = []
        files: list[FileSummary] = []
        warnings: list[str] = []
        total_pages = 0
        total_extracted_chars = 0
        extraction_results: list[tuple[str, PdfExtractionResult]] = []

        for filename, data in uploaded_files:
            result = extract_pdf_pages(data, filename)
            extraction_results.append((filename, result))

            total_pages += result.total_pages
            if total_pages > settings.max_total_pages:
                raise ValueError(
                    f"The collection exceeds the {settings.max_total_pages}-page limit."
                )

            total_extracted_chars += sum(len(block.text) for block in result.blocks)
            if total_extracted_chars > settings.max_extracted_chars:
                raise ValueError(
                    f"The collection exceeds the "
                    f"{settings.max_extracted_chars:,}-character extraction limit."
                )

            all_blocks.extend(result.blocks)
            warnings.extend(result.warnings)

        chunks = chunk_pages(
            all_blocks,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
        if not chunks:
            raise ValueError("No text chunks could be created from the uploaded PDFs.")
        if len(chunks) > settings.max_chunks_per_collection:
            raise ValueError(
                f"The collection exceeds the "
                f"{settings.max_chunks_per_collection:,}-chunk limit."
            )

        vectors = embedding_service.encode([chunk.text for chunk in chunks])
        chunks_per_file = Counter(chunk.filename for chunk in chunks)

        for filename, result in extraction_results:
            files.append(
                FileSummary(
                    name=filename,
                    pages=result.total_pages,
                    chunks=chunks_per_file[filename],
                    ocr_pages=len(result.ocr_pages),
                    tables=result.table_count,
                )
            )

        logger.info(
            "Built RAG collection: files=%s pages=%s blocks=%s chunks=%s "
            "ocr_pages=%s tables=%s",
            len(uploaded_files),
            total_pages,
            len(all_blocks),
            len(chunks),
            sum(file.ocr_pages for file in files),
            sum(file.tables for file in files),
        )

        return collection_store.create(
            chunks,
            vectors,
            files,
            total_pages,
            warnings=list(dict.fromkeys(warnings)),
        )

    def ask(
        self,
        collection_id: str,
        question: str,
        top_k: int | None = None,
        rewrite_question: bool | None = None,
    ) -> AnswerResponse:
        settings = get_settings()
        collection = collection_store.get(collection_id)

        plan = query_planner.plan(question, enabled=rewrite_question)
        query_vectors = embedding_service.encode(plan.search_queries)
        retrieved = self._retrieve_relevant_chunks(
            collection,
            plan,
            query_vectors,
            top_k,
        )
        relevant = self._select_context_chunks(plan, retrieved)

        if not relevant:
            logger.info(
                "RAG no-answer: retrieval returned no sufficiently relevant chunks"
            )
            return self._no_answer(plan, status="insufficient_evidence")

        logger.info(
            "RAG context selected: retrieved=%s selected=%s selected_pages=%s",
            len(retrieved),
            len(relevant),
            [
                f"{item.chunk.filename}:p{item.chunk.page_number}"
                for item in relevant
            ],
        )

        prompt, context_sources = build_user_prompt(
            plan.original_question,
            plan.rewritten_question,
            relevant,
            settings.max_context_chars,
        )
        if not context_sources:
            logger.info("RAG no-answer: max_context_chars excluded all chunks")
            return self._no_answer(plan, status="insufficient_evidence")

        raw_answer = llm_service.answer(prompt)
        answer, grounded = validate_grounded_answer(
            raw_answer,
            len(context_sources),
        )
        sources = self._format_sources(context_sources)

        if grounded:
            return AnswerResponse(
                answer=answer,
                sources=sources,
                grounded=True,
                grounding_status="verified",
                interpreted_question=self._interpreted_question(plan),
                search_queries=plan.search_queries,
            )

        initial_reason = grounding_failure_reason(
            raw_answer,
            len(context_sources),
        )
        logger.warning(
            "Initial answer failed grounding validation: "
            "reason=%s source_count=%s answer=%r",
            initial_reason,
            len(context_sources),
            self._for_log(raw_answer),
        )

        repair_prompt, repair_sources = build_citation_repair_prompt(
            original_question=plan.original_question,
            interpreted_question=plan.rewritten_question,
            previous_answer=raw_answer,
            results=[source.result for source in context_sources],
            max_context_chars=settings.max_context_chars,
        )
        repaired_raw = llm_service.answer(repair_prompt)
        repaired_answer, repaired_grounded = validate_grounded_answer(
            repaired_raw,
            len(repair_sources),
        )
        repaired_sources = self._format_sources(repair_sources)

        if repaired_grounded:
            logger.info("RAG answer accepted after citation repair")
            return AnswerResponse(
                answer=repaired_answer,
                sources=repaired_sources,
                grounded=True,
                grounding_status="verified_after_repair",
                interpreted_question=self._interpreted_question(plan),
                search_queries=plan.search_queries,
            )

        repair_reason = grounding_failure_reason(
            repaired_raw,
            len(repair_sources),
        )
        logger.warning(
            "Citation repair failed grounding validation: "
            "reason=%s source_count=%s answer=%r",
            repair_reason,
            len(repair_sources),
            self._for_log(repaired_raw),
        )

        if repaired_raw.strip() and repaired_raw.strip() != NO_ANSWER:
            return AnswerResponse(
                answer=repaired_answer,
                sources=repaired_sources or sources,
                grounded=False,
                grounding_status="citation_validation_failed",
                interpreted_question=self._interpreted_question(plan),
                search_queries=plan.search_queries,
            )

        if raw_answer.strip() and raw_answer.strip() != NO_ANSWER:
            return AnswerResponse(
                answer=answer,
                sources=sources,
                grounded=False,
                grounding_status="citation_validation_failed",
                interpreted_question=self._interpreted_question(plan),
                search_queries=plan.search_queries,
            )

        logger.info("RAG no-answer confirmed after evidence-based retry")
        return AnswerResponse(
            answer=NO_ANSWER,
            sources=repaired_sources or sources,
            grounded=False,
            grounding_status="insufficient_evidence",
            interpreted_question=self._interpreted_question(plan),
            search_queries=plan.search_queries,
        )

    def _retrieve_relevant_chunks(
        self,
        collection: Collection,
        plan: QueryPlan,
        query_vectors: np.ndarray,
        top_k: int | None,
    ) -> list[RetrievedChunk]:
        settings = get_settings()
        effective_top_k = int(top_k) if top_k is not None else settings.top_k
        candidate_k = min(max(effective_top_k * 4, 24), len(collection.chunks))

        retrieved = collection.hybrid_search(
            plan=plan,
            query_vectors=query_vectors,
            top_k=effective_top_k,
            vector_k=candidate_k,
            keyword_k=candidate_k,
        )
        relevant = [
            item for item in retrieved
            if item.score >= settings.min_similarity
        ]

        if retrieved and not relevant:
            logger.info(
                "RAG retrieval scores below threshold: top_score=%s threshold=%s",
                round(float(retrieved[0].score), 4),
                settings.min_similarity,
            )

        return relevant[:effective_top_k]

    @classmethod
    def _select_context_chunks(
        cls,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Remove nearby but question-irrelevant manual chunks."""
        if not retrieved:
            return []

        original = plan.original_question
        acronyms = set(_ACRONYM_PATTERN.findall(original))
        query_terms = cls._significant_terms(original)
        top_score = max(float(item.score) for item in retrieved)
        short_request = len(_WORD_PATTERN.findall(original)) <= 7
        limit = 4 if short_request else 7

        if acronyms:
            anchors = [
                item for item in retrieved
                if cls._contains_any_acronym(item.chunk.text, acronyms)
            ]

            if anchors:
                selected: list[RetrievedChunk] = []

                for item in retrieved:
                    contains_acronym = cls._contains_any_acronym(
                        item.chunk.text,
                        acronyms,
                    )
                    near_anchor = any(
                        item.chunk.filename == anchor.chunk.filename
                        and abs(
                            item.chunk.page_number - anchor.chunk.page_number
                        ) <= 1
                        for anchor in anchors
                    )
                    lexical_overlap = cls._term_overlap(
                        query_terms,
                        item.chunk.text,
                    )

                    if contains_acronym:
                        selected.append(item)
                    elif (
                        near_anchor
                        and float(item.score) >= top_score * 0.68
                        and lexical_overlap >= 1
                    ):
                        selected.append(item)

                selected = cls._unique_chunks(selected)
                return (selected or [retrieved[0]])[:limit]

        selected = []
        for index, item in enumerate(retrieved):
            lexical_overlap = cls._term_overlap(
                query_terms,
                item.chunk.text,
            )
            strong_relative_score = float(item.score) >= top_score * 0.84

            if index == 0 or lexical_overlap >= 1 or strong_relative_score:
                selected.append(item)

        selected = cls._unique_chunks(selected)
        return (selected or [retrieved[0]])[:limit]

    @staticmethod
    def _significant_terms(value: str) -> set[str]:
        return {
            token.casefold()
            for token in _WORD_PATTERN.findall(value)
            if len(token) >= 3
            and token.casefold() not in _STOP_WORDS
        }

    @staticmethod
    def _term_overlap(query_terms: set[str], text: str) -> int:
        if not query_terms:
            return 0
        text_terms = {
            token.casefold()
            for token in _WORD_PATTERN.findall(text)
        }
        return len(query_terms & text_terms)

    @staticmethod
    def _contains_any_acronym(text: str, acronyms: set[str]) -> bool:
        present = set(_ACRONYM_PATTERN.findall(text))
        return bool(present & acronyms)

    @staticmethod
    def _unique_chunks(
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []

        for item in chunks:
            key = item.chunk.chunk_id
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique

    @staticmethod
    def _format_sources(
        sources: list[PromptSource],
    ) -> list[SourceResult]:
        formatted: list[SourceResult] = []

        for number, source in enumerate(sources, start=1):
            result = source.result
            formatted.append(
                SourceResult(
                    id=f"S{number}",
                    filename=result.chunk.filename,
                    page=result.chunk.page_number,
                    score=round(result.score, 4),
                    excerpt=source.excerpt[:1000],
                    content_type=result.chunk.content_type,
                    retrieval_method=result.method,
                )
            )

        return formatted

    @staticmethod
    def _interpreted_question(plan: QueryPlan) -> str | None:
        if plan.rewritten_question == plan.original_question:
            return None
        return plan.rewritten_question

    @staticmethod
    def _for_log(answer: str) -> str:
        normalized = answer.strip()
        if len(normalized) <= _MAX_LOGGED_ANSWER_CHARS:
            return normalized
        return normalized[:_MAX_LOGGED_ANSWER_CHARS] + "...[truncated]"

    @classmethod
    def _no_answer(
        cls,
        plan: QueryPlan,
        status: str,
    ) -> AnswerResponse:
        return AnswerResponse(
            answer=NO_ANSWER,
            sources=[],
            grounded=False,
            grounding_status=status,
            interpreted_question=cls._interpreted_question(plan),
            search_queries=plan.search_queries,
        )


rag_service = RagService()
