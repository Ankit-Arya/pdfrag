from __future__ import annotations

# ruff: noqa: E501

import logging
import os
import re
import time
from dataclasses import replace

from app.rag.scenario_reasoning import (
    compile_scenario,
    current_scenario,
    logical_match_score,
    scenario_logic_value_terms,
    scenario_relevance_question,
    scenario_relaxed_query,
    set_current_scenario,
)
from app.rag.smart_index import index_document
from app.rag.smart_retrieval import (
    fast_corpus_scan,
    fast_search_chunks,
    rule_notes_for_chunk,
)
from app.rag.smart_schema import ensure_smart_schema
from app.rag.terminology import terminology_hints

logger = logging.getLogger(__name__)
_INSTALLED = False


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 5000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def install_smart_rag_patch() -> None:
    """Install an additive runtime patch over the existing pdfrag RAG service."""
    global _INSTALLED
    if _INSTALLED or not _bool_env("SMART_RAG_ENABLED", True):
        return

    try:
        ensure_smart_schema()
    except Exception:
        logger.exception("Smart RAG schema setup failed; continuing with the legacy RAG path")
        return

    import app.rag.query as query_module
    import app.rag.service as service_module
    import app.rag.synthesis as synthesis_module

    original_find_abbreviations = service_module.find_abbreviation_hints
    original_search_chunks = service_module.search_chunks
    original_fetch_primary = service_module.fetch_primary_document_chunks
    original_fetch_referenced = service_module.fetch_referenced_document_chunks
    original_fetch_neighbors = service_module.fetch_neighbor_chunks
    original_select = service_module.select_context_chunks
    original_rank_scenario_documents = service_module.rank_scenario_documents
    original_filter_hard_context = service_module.filter_hard_context_candidates
    original_process_document = service_module.RagService.process_document
    original_plan = query_module.QueryPlanner.plan
    original_source_header = synthesis_module._compact_source_header

    def smart_find_abbreviation_hints(db, text_value, *, max_terms=None, chunks_per_term=None):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        try:
            indexed = terminology_hints(db, text_value)
        except Exception:
            logger.exception("Global terminology lookup failed; using legacy abbreviation scan")
            indexed = []
        if indexed:
            logger.info("smart_rag stage=terminology ms=%.1f hints=%d", (time.perf_counter() - started) * 1000, len(indexed))
            return indexed
        result = original_find_abbreviations(
            db,
            text_value,
            max_terms=max_terms,
            chunks_per_term=chunks_per_term,
        )
        logger.info("smart_rag stage=terminology_legacy ms=%.1f hints=%d", (time.perf_counter() - started) * 1000, len(result))
        return result

    def smart_plan(self, question, enabled=None, *, conversation_context=None, abbreviation_hints=None, routing_hints=None):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        plan = original_plan(
            self,
            question,
            enabled,
            conversation_context=conversation_context,
            abbreviation_hints=abbreviation_hints,
            routing_hints=routing_hints,
        )
        scenario = compile_scenario(
            plan.contextual_question or plan.rewritten_question or plan.original_question,
            abbreviation_hints or [],
        )
        set_current_scenario(scenario)
        max_variants = _int_env("SMART_RAG_QUERY_VARIANTS", 3, 1, 6)
        queries: list[str] = []
        candidate_queries = [scenario.canonical_question]
        if scenario.is_situational:
            candidate_queries.append(scenario_relaxed_query(scenario))
        candidate_queries.extend(plan.search_queries)
        candidate_queries.append(plan.contextual_question or plan.rewritten_question)
        for value in candidate_queries:
            value = " ".join(str(value or "").split())
            if value and value.casefold() not in {item.casefold() for item in queries}:
                queries.append(value)
            if len(queries) >= max_variants:
                break
        logger.info(
            "smart_rag stage=planner ms=%.1f intent=%s situational=%s variants=%d numeric_facts=%d states=%d",
            (time.perf_counter() - started) * 1000,
            plan.intent,
            scenario.is_situational,
            len(queries),
            len(scenario.numeric_facts),
            len(scenario.states) + len(scenario.inferred_states),
        )
        return replace(plan, search_queries=queries or plan.search_queries[:max_variants])

    def smart_search(db, query_vector, query_text, limit):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        result = fast_search_chunks(db, query_vector, query_text, limit)
        fallback_score = _float_env("SMART_RAG_BROAD_FALLBACK_SCORE", 0.12)
        if (
            _bool_env("SMART_RAG_ALLOW_BROAD_FALLBACK", True)
            and (not result or float(result[0].score) < fallback_score)
        ):
            logger.warning(
                "Smart retrieval confidence low (top=%.3f); using bounded legacy fallback",
                float(result[0].score) if result else 0.0,
            )
            result = original_search_chunks(
                db,
                query_vector,
                query_text,
                min(limit, _int_env("SMART_RAG_LEGACY_FALLBACK_LIMIT", 160, 40, 500)),
            )
        logger.info(
            "smart_rag stage=hybrid_retrieval ms=%.1f candidates=%d top=%.3f",
            (time.perf_counter() - started) * 1000,
            len(result),
            float(result[0].score) if result else 0.0,
        )
        return result

    def smart_stemmed_search(db, query_text, limit):  # type: ignore[no-untyped-def]
        # fast_search_chunks already includes English-stemmed GIN FTS.
        return []

    def smart_scan(db, query_texts, *, focus_terms=None, reference_mode=False, limit=10000):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        result = fast_corpus_scan(
            db,
            query_texts,
            focus_terms=focus_terms,
            reference_mode=reference_mode,
            limit=limit,
        )
        logger.info("smart_rag stage=indexed_corpus_fallback ms=%.1f candidates=%d", (time.perf_counter() - started) * 1000, len(result))
        return result

    def capped_primary(db, routes, query_texts, *, chunks_per_document=None):  # type: ignore[no-untyped-def]
        cap = _int_env("SMART_RAG_PRIMARY_CHUNKS_PER_DOCUMENT", 64, 10, 200)
        requested = chunks_per_document or cap
        return original_fetch_primary(db, routes, query_texts, chunks_per_document=min(requested, cap))

    def capped_referenced(db, references, query_texts, *, max_documents=None, chunks_per_document=None):  # type: ignore[no-untyped-def]
        doc_cap = _int_env("SMART_RAG_REFERENCE_DOCUMENTS", 4, 1, 10)
        chunk_cap = _int_env("SMART_RAG_REFERENCE_CHUNKS_PER_DOCUMENT", 48, 10, 160)
        requested_docs = max_documents or doc_cap
        requested_chunks = chunks_per_document or chunk_cap
        return original_fetch_referenced(
            db,
            references,
            query_texts,
            max_documents=min(requested_docs, doc_cap),
            chunks_per_document=min(requested_chunks, chunk_cap),
        )


    def capped_neighbors(db, seeds, window=1):  # type: ignore[no-untyped-def]
        seed_cap = _int_env("SMART_RAG_NEIGHBOR_SEEDS", 60, 8, 180)
        window_cap = _int_env("SMART_RAG_NEIGHBOR_WINDOW", 1, 1, 2)
        bounded_seeds = list(seeds[:seed_cap])
        return original_fetch_neighbors(db, bounded_seeds, window=min(max(0, int(window)), window_cap))

    def relevance_plan(plan):  # type: ignore[no-untyped-def]
        scenario = current_scenario()
        if scenario is None or not scenario.numeric_facts:
            return plan
        removable = scenario_logic_value_terms(scenario)
        if not removable:
            return plan
        focus_terms = [value for value in plan.focus_terms if value.casefold() not in removable]
        context_terms = [value for value in plan.context_terms if value.casefold() not in removable]
        return replace(
            plan,
            contextual_question=scenario_relevance_question(scenario),
            focus_terms=focus_terms,
            context_terms=context_terms,
        )

    def terminology_pairs():  # type: ignore[no-untyped-def]
        scenario = current_scenario()
        if scenario is None:
            return []
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for hint in scenario.terminology:
            if "=" not in hint or "ambiguous corpus meaning" in hint.casefold():
                continue
            left, right = hint.split("=", 1)
            alias = " ".join(left.split()).strip()
            canonical = right.split("—", 1)[0].split("--", 1)[0].strip()
            key = (alias.casefold(), canonical.casefold())
            if alias and canonical and key not in seen:
                seen.add(key)
                pairs.append((alias, canonical))
        return pairs[:12]

    def relevance_enriched_candidates(candidates):  # type: ignore[no-untyped-def]
        pairs = terminology_pairs()
        if not pairs:
            return list(candidates), {item.chunk.chunk_id: item for item in candidates}
        enriched = []
        originals = {}
        for item in candidates:
            originals[item.chunk.chunk_id] = item
            body = item.chunk.text
            body_folded = body.casefold()
            additions: list[str] = []
            for alias, canonical in pairs:
                alias_present = bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", body, re.IGNORECASE))
                canonical_present = canonical.casefold() in body_folded
                if canonical_present and not alias_present:
                    additions.append(alias)
                elif alias_present and not canonical_present:
                    additions.append(canonical)
            if additions:
                synthetic = body + "\n[RETRIEVAL TERMINOLOGY EQUIVALENTS] " + "; ".join(additions[:12])
                enriched.append(replace(item, chunk=replace(item.chunk, text=synthetic)))
            else:
                enriched.append(item)
        return enriched, originals

    def restore_original_chunks(selected, originals):  # type: ignore[no-untyped-def]
        restored = []
        for item in selected:
            original = originals.get(item.chunk.chunk_id)
            if original is None:
                restored.append(item)
            else:
                restored.append(replace(original, score=item.score, method=item.method))
        return restored

    def smart_rank_scenario_documents(plan, candidates, *, max_documents=3):  # type: ignore[no-untyped-def]
        enriched, _ = relevance_enriched_candidates(candidates)
        return original_rank_scenario_documents(relevance_plan(plan), enriched, max_documents=max_documents)

    def smart_filter_hard_context(plan, candidates):  # type: ignore[no-untyped-def]
        enriched, originals = relevance_enriched_candidates(candidates)
        allowed = original_filter_hard_context(relevance_plan(plan), enriched)
        return [originals.get(item.chunk.chunk_id, item) for item in allowed]

    def smart_select(plan, candidates, max_chunks=None, *, preferred_document_ids=None):  # type: ignore[no-untyped-def]
        cap = _int_env("SMART_RAG_FINAL_CONTEXT_CHUNKS", 48, 12, 120)
        scenario = current_scenario()
        adjusted = []
        for item in candidates:
            if scenario is None or "deterministic-rule-" in item.method:
                adjusted.append(item)
                continue
            boost, _ = logical_match_score(scenario, item.chunk.text)
            if boost:
                method = item.method
                method += "+deterministic-rule-match" if boost > 0 else "+deterministic-rule-mismatch"
                adjusted.append(replace(item, score=max(0.0, min(1.0, item.score + boost)), method=method))
            else:
                adjusted.append(item)
        adjusted.sort(key=lambda item: -item.score)
        enriched, originals = relevance_enriched_candidates(adjusted)
        selected = original_select(
            relevance_plan(plan),
            enriched,
            max_chunks=min(max_chunks or cap, cap),
            preferred_document_ids=preferred_document_ids,
        )
        return restore_original_chunks(selected, originals)

    def smart_process_document(self, db, document):  # type: ignore[no-untyped-def]
        result = original_process_document(self, db, document)
        try:
            status = getattr(result.status, "value", str(result.status))
            if status == "ready":
                counts = index_document(db, result.id)
                logger.info("Smart-indexed new document %s: %s", result.id, counts)
        except Exception:
            db.rollback()
            logger.exception("Document processed, but smart derived-index refresh failed for %s", getattr(result, "id", "?"))
        return result

    def smart_source_header(source):  # type: ignore[no-untyped-def]
        header = original_source_header(source)
        notes = rule_notes_for_chunk(source.result.chunk.chunk_id, source.excerpt)
        if notes:
            # This is explicitly marked as derived. It helps the answer model apply
            # e.g. user=4 to a source threshold of >=2 without pretending the PDF said 4.
            header += " | SYSTEM-DERIVED APPLICABILITY (not PDF text): " + " ".join(notes)
        return header

    # Patch module globals actually referenced by RagService._ask_impl.
    service_module.find_abbreviation_hints = smart_find_abbreviation_hints
    service_module.search_chunks = smart_search
    service_module.search_stemmed_chunks = smart_stemmed_search
    service_module.scan_matching_chunks = smart_scan
    service_module.fetch_primary_document_chunks = capped_primary
    service_module.fetch_referenced_document_chunks = capped_referenced
    service_module.fetch_neighbor_chunks = capped_neighbors
    service_module.select_context_chunks = smart_select
    service_module.rank_scenario_documents = smart_rank_scenario_documents
    service_module.filter_hard_context_candidates = smart_filter_hard_context
    service_module.RagService._select_context_chunks = staticmethod(smart_select)
    service_module.RagService.process_document = smart_process_document

    # Planner instance was imported before app lifespan; patch the class method so
    # the existing singleton immediately uses scenario compilation.
    query_module.QueryPlanner.plan = smart_plan

    synthesis_module._compact_source_header = smart_source_header
    if "SYSTEM-DERIVED APPLICABILITY" not in synthesis_module._ANSWER_SYSTEM_PROMPT:
        synthesis_module._ANSWER_SYSTEM_PROMPT += """

SYSTEM-DERIVED APPLICABILITY notes in source headers are not PDF quotations. They are deterministic
comparisons of facts stated by the user against explicit numeric conditions found in the cited PDF
excerpt (for example, user reports 4 affected brakes and the source says 2 or more). You may use a
TRUE derived comparison to decide that the cited rule applies, but cite the underlying PDF source,
state the comparison briefly when material, and never turn an uncertain semantic inference into a fact.
"""

    _INSTALLED = True
    logger.info("Smart RAG runtime patch installed")
