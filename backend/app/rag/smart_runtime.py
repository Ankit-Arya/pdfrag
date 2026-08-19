from __future__ import annotations

# ruff: noqa: E501

import contextvars
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
    definition_evidence_chunks,
    fast_corpus_scan,
    fast_search_chunks,
    retrieval_answerable,
    rule_notes_for_chunk,
    structured_lookup_request,
    value_lookup_request,
)
from app.rag.smart_schema import ensure_smart_schema
from app.rag.smart_understanding import (
    interpret_user_message,
    relevant_terminology_hints,
    review_retrieved_evidence,
    should_review_evidence,
    verify_answer,
)
from app.rag.terminology import (
    definition_for_token,
    definition_request_aliases,
    explicit_definition_hints,
    is_definition_request,
    terminology_hints,
)

logger = logging.getLogger(__name__)
_INSTALLED = False
_CURRENT_ORIGINAL_QUESTION: contextvars.ContextVar[str] = contextvars.ContextVar(
    "pdfrag_smart_original_question", default=""
)
_CURRENT_DEFINITION_ALIASES: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "pdfrag_smart_definition_aliases", default=()
)
_CURRENT_INTERPRETATION: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "pdfrag_smart_interpretation", default=None
)
_CURRENT_AI_EVIDENCE_REVIEWED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pdfrag_smart_ai_evidence_reviewed", default=False
)
_CURRENT_DB: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "pdfrag_smart_db", default=None
)
_CURRENT_COVERAGE_STATUS: contextvars.ContextVar[str] = contextvars.ContextVar(
    "pdfrag_smart_coverage_status", default="not_reviewed"
)


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
    original_ask_impl = service_module.RagService._ask_impl
    original_plan = query_module.QueryPlanner.plan
    original_search_mode = service_module.deterministic_search_mode
    original_needs_conversation_context = service_module.needs_conversation_context
    original_use_primary_backbone = service_module._use_primary_document_backbone
    original_source_header = synthesis_module._compact_source_header
    original_synthesize_answer = service_module.synthesize_answer

    def smart_find_abbreviation_hints(db, text_value, *, max_terms=None, chunks_per_term=None):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        try:
            indexed = terminology_hints(db, text_value)
        except Exception:
            logger.exception("Global terminology lookup failed; trying definition-aware fallback")
            indexed = []
        if indexed:
            logger.info(
                "smart_rag stage=terminology ms=%.1f hints=%d",
                (time.perf_counter() - started) * 1000,
                len(indexed),
            )
            return indexed

        # Definition queries need a different fallback from ordinary abbreviation
        # use. Scan explicitly definition-shaped occurrences so repeated operational
        # uses of an acronym cannot consume the small legacy occurrence limit before
        # an explicit source definition is reached.
        if is_definition_request(text_value):
            try:
                definitions = explicit_definition_hints(db, text_value)
            except Exception:
                logger.exception("Definition-specific abbreviation scan failed")
                definitions = []
            if definitions:
                logger.info(
                    "smart_rag stage=terminology_definition_fallback ms=%.1f hints=%d",
                    (time.perf_counter() - started) * 1000,
                    len(definitions),
                )
                return definitions

        result = original_find_abbreviations(
            db,
            text_value,
            max_terms=max_terms,
            chunks_per_term=chunks_per_term,
        )
        logger.info(
            "smart_rag stage=terminology_legacy ms=%.1f hints=%d",
            (time.perf_counter() - started) * 1000,
            len(result),
        )
        return result

    def smart_plan(self, question, enabled=None, *, conversation_context=None, abbreviation_hints=None, routing_hints=None):  # type: ignore[no-untyped-def]
        """AI-first interpretation, with deterministic legacy planning only as fallback scaffolding.

        The model is allowed to understand noisy natural language and propose search semantics,
        but it is never used as factual metro evidence. All answer facts still come from PDFs.
        """
        started = time.perf_counter()
        raw_hints = abbreviation_hints or []
        raw_routes = routing_hints or []
        interpretation = interpret_user_message(
            question,
            history=conversation_context or [],
            abbreviation_hints=raw_hints,
            routing_hints=raw_routes,
        )
        _CURRENT_INTERPRETATION.set(interpretation)
        _CURRENT_AI_EVIDENCE_REVIEWED.set(False)
        _CURRENT_COVERAGE_STATUS.set("not_reviewed")
        _CURRENT_ORIGINAL_QUESTION.set(question)

        # An unrelated previous turn must not bleed into retrieval merely because we made
        # the full recent USER context available to the interpreter. The AI explicitly
        # decides whether history is needed; only then are route hints retained.
        hints = relevant_terminology_hints(interpretation, raw_hints)
        effective_history = (conversation_context or []) if interpretation.uses_history else []
        effective_routes = raw_routes if interpretation.uses_history else []

        # Build a valid QueryPlan using the repo's deterministic planner, but disable its
        # separate LLM rewrite. The AI interpretation above becomes the single normal
        # language-understanding call, reducing contradictory planners and duplicate cost.
        plan = original_plan(
            self,
            question,
            False,
            conversation_context=effective_history,
            abbreviation_hints=hints,
            routing_hints=effective_routes,
        )

        resolved = interpretation.resolved_question or plan.contextual_question or question
        definition_aliases: tuple[str, ...] = ()
        raw_definition_aliases = definition_request_aliases(question, hints)
        if (
            interpretation.conversation_act in {"question", "request"}
            and (interpretation.intent == "definition" or bool(raw_definition_aliases))
        ):
            aliases = [
                *raw_definition_aliases,
                *definition_request_aliases(resolved, hints),
            ]
            definition_aliases = tuple(dict.fromkeys(value.upper() for value in aliases))
        _CURRENT_DEFINITION_ALIASES.set(definition_aliases)

        intent = interpretation.intent
        if definition_aliases:
            intent = "definition"
        elif structured_lookup_request(resolved) and intent == "fact_lookup":
            intent = "list"

        search_mode = "references" if interpretation.conversation_act == "navigation" else "answer"
        scope_terms = [f"{key} {value}" for key, value in interpretation.scope.items()]
        focus_terms = list(dict.fromkeys([*interpretation.concepts, *plan.focus_terms]))[:48]
        context_terms = list(dict.fromkeys([*scope_terms, *plan.context_terms]))[:32]
        keywords = list(dict.fromkeys([*plan.keywords, *interpretation.concepts]))[:40]

        scenario = compile_scenario(resolved, hints)
        set_current_scenario(scenario)
        max_variants = _int_env("SMART_RAG_QUERY_VARIANTS", 4, 1, 6)
        queries: list[str] = []
        candidates: list[str] = []
        if definition_aliases:
            candidates.extend(definition_aliases)
        # Put the resolved question first, then force one generic authority-oriented
        # formulation when revision/currentness can change the answer. This does not
        # invent a document or value; it simply makes amendment/replacement evidence
        # discoverable before an older cleanly-OCR'd provision can dominate ranking.
        candidates.append(resolved)
        if interpretation.authority_sensitive:
            candidates.append(
                f"{resolved} current amended amendment revised substituted replacement schedule effective"
            )
        candidates.extend(interpretation.search_queries)
        if scenario.is_situational:
            candidates.append(scenario_relaxed_query(scenario))
        # Keep the raw wording only as a late exact-token fallback for identifiers/codes.
        candidates.append(question)
        for value in candidates:
            value = " ".join(str(value or "").split())
            if value and value.casefold() not in {item.casefold() for item in queries}:
                queries.append(value)
            if len(queries) >= max_variants:
                break

        logger.info(
            "smart_rag stage=ai_interpret ms=%.1f ai=%s act=%s intent=%s route=%s history=%s ambiguity=%s variants=%d needs=%d",
            (time.perf_counter() - started) * 1000,
            interpretation.ai_used,
            interpretation.conversation_act,
            intent,
            interpretation.route_strategy,
            interpretation.uses_history,
            interpretation.material_ambiguity,
            len(queries),
            len(interpretation.evidence_needs),
        )
        return replace(
            plan,
            rewritten_question=resolved,
            contextual_question=resolved,
            search_queries=queries or [resolved],
            keywords=keywords,
            intent=intent,
            response_mode="evidence" if search_mode == "references" else "concise",
            search_mode=search_mode,
            focus_terms=focus_terms,
            context_terms=context_terms,
            abbreviation_hints=hints,
            routing_hints=effective_routes,
            used_ai_rewrite=interpretation.ai_used,
        )

    def smart_search(db, query_vector, query_text, limit):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        result = fast_search_chunks(db, query_vector, query_text, limit)

        definition_aliases = _CURRENT_DEFINITION_ALIASES.get()
        if definition_aliases:
            try:
                definition_rows = definition_evidence_chunks(
                    db,
                    _CURRENT_ORIGINAL_QUESTION.get() or query_text,
                    aliases=definition_aliases,
                    limit=min(30, max(len(definition_aliases) * 3, 8, limit // 4)),
                )
            except Exception:
                logger.exception("Definition evidence retrieval failed")
                definition_rows = []
            if definition_rows:
                merged: dict[str, object] = {}
                ordered = [*definition_rows, *result]
                deduped = []
                for item in ordered:
                    if item.chunk.chunk_id in merged:
                        continue
                    merged[item.chunk.chunk_id] = item
                    deduped.append(item)
                    if len(deduped) >= limit:
                        break
                result = deduped

        # v4 deliberately does NOT run the AI evidence critic here. This function is
        # called once per semantic query, before the service has merged other query
        # variants, corpus-wide lexical matches, routed procedures, reference hops and
        # neighbor chunks. Coverage is reviewed later in smart_select against the fully
        # assembled candidate set.

        fallback_score = _float_env("SMART_RAG_BROAD_FALLBACK_SCORE", 0.12)
        answerable = retrieval_answerable(_CURRENT_ORIGINAL_QUESTION.get() or query_text, result)
        needs_fallback = (
            not result
            or float(result[0].score) < fallback_score
            or not answerable
            or (bool(definition_aliases) and not any("terminology-definition" in item.method for item in result))
        )
        if _bool_env("SMART_RAG_ALLOW_BROAD_FALLBACK", True) and needs_fallback:
            logger.warning(
                "Smart retrieval fallback: top=%.3f answerable=%s definition=%s",
                float(result[0].score) if result else 0.0,
                answerable,
                bool(definition_aliases),
            )
            legacy = original_search_chunks(
                db,
                query_vector,
                query_text,
                min(limit, _int_env("SMART_RAG_LEGACY_FALLBACK_LIMIT", 160, 40, 500)),
            )
            # Merge instead of replacing the smart set. Otherwise a fallback can
            # discard the amendment anchor or exact acronym definition that caused
            # the smart path to be more trustworthy than the broad legacy ranking.
            by_id = {item.chunk.chunk_id: item for item in result}
            for item in legacy:
                current = by_id.get(item.chunk.chunk_id)
                if current is None:
                    by_id[item.chunk.chunk_id] = item
                    continue
                # Never discard smart authority/definition/structured metadata just
                # because the legacy scorer assigns the same chunk a larger number.
                merged_method = current.method
                if "legacy-fallback" not in merged_method:
                    merged_method += "+legacy-fallback"
                by_id[item.chunk.chunk_id] = replace(
                    current,
                    score=max(float(current.score), float(item.score)),
                    vector_score=max(float(current.vector_score), float(item.vector_score)),
                    keyword_score=max(float(current.keyword_score), float(item.keyword_score)),
                    method=merged_method,
                )
            result = sorted(
                by_id.values(),
                key=lambda item: (
                    0 if "terminology-definition" in item.method else 1,
                    0 if "current-amendment-context" in item.method else 1,
                    0 if "amendment-authority-anchor" in item.method else 1,
                    -float(item.score),
                    item.chunk.filename.casefold(),
                    item.chunk.page_number,
                ),
            )[:limit]
        logger.info(
            "smart_rag stage=hybrid_retrieval ms=%.1f candidates=%d top=%.3f answerable=%s",
            (time.perf_counter() - started) * 1000,
            len(result),
            float(result[0].score) if result else 0.0,
            retrieval_answerable(_CURRENT_ORIGINAL_QUESTION.get() or query_text, result),
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

    def _postmerge_evidence_review(plan, candidates):  # type: ignore[no-untyped-def]
        """Review coverage only after all normal retrieval arms have been merged.

        v3 reviewed the first per-query result set and then permanently marked the turn
        reviewed. That could miss the governing procedure even when later retrieval arms
        would have found it. v4 reviews the complete candidate set and performs at most
        one bounded semantic+lexical retry using the current request DB session.
        """
        interpretation = _CURRENT_INTERPRETATION.get()
        if (
            interpretation is None
            or _CURRENT_AI_EVIDENCE_REVIEWED.get()
            or not should_review_evidence(interpretation)
        ):
            return list(candidates)

        _CURRENT_AI_EVIDENCE_REVIEWED.set(True)
        ranked = sorted(
            list(candidates),
            key=lambda item: (
                0 if "current-authority-span" in item.method else 1,
                0 if "current-explicit-wording" in item.method else 1,
                0 if "authority-anchor" in item.method else 1,
                0 if "smart-structured-value" in item.method else 1,
                -float(item.score),
                item.chunk.filename.casefold(),
                item.chunk.page_number,
            ),
        )
        review = review_retrieved_evidence(interpretation, ranked)
        logger.info(
            "smart_rag stage=ai_postmerge_evidence_review sufficient=%s ai=%s missing=%d retry=%d reason=%s",
            review.sufficient,
            review.ai_used,
            len(review.missing_evidence),
            len(review.retry_queries),
            review.reason[:240],
        )
        if review.sufficient:
            _CURRENT_COVERAGE_STATUS.set("sufficient")
            return ranked

        db = _CURRENT_DB.get()
        if db is None or not review.retry_queries:
            _CURRENT_COVERAGE_STATUS.set("insufficient_after_review")
            return ranked

        try:
            from app.rag.embeddings import embedding_service

            retry_queries = list(review.retry_queries)[:_int_env("SMART_RAG_AI_RETRY_QUERIES", 2, 1, 3)]
            retry_vectors = embedding_service.encode(retry_queries)
            retry_limit = _int_env("SMART_RAG_AI_RETRY_CANDIDATES", 120, 30, 240)
            retry_items = []
            for retry_query, retry_vector in zip(retry_queries, retry_vectors, strict=True):
                for item in fast_search_chunks(db, retry_vector.tolist(), retry_query, retry_limit):
                    retry_items.append(replace(item, method=item.method + "+ai-postmerge-retry"))

            # Add a bounded corpus-wide lexical arm as well. Semantic similarity is not
            # enough for role assignments, explicit counts, table rows or amendment text.
            focus_terms = list(dict.fromkeys([*interpretation.concepts, *plan.focus_terms, *plan.context_terms]))[:48]
            for item in fast_corpus_scan(
                db,
                retry_queries,
                focus_terms=focus_terms,
                reference_mode=False,
                limit=min(retry_limit * 2, 300),
            ):
                retry_items.append(replace(item, method=item.method + "+ai-postmerge-retry"))

            if retry_items:
                try:
                    neighbor_seeds = sorted(retry_items, key=lambda item: -float(item.score))[:24]
                    for item in original_fetch_neighbors(db, neighbor_seeds, window=1):
                        retry_items.append(replace(item, method=item.method + "+ai-postmerge-neighbor"))
                except Exception:
                    logger.exception("Post-merge AI retry neighbor expansion failed")

            by_id = {item.chunk.chunk_id: item for item in ranked}
            for item in retry_items:
                current = by_id.get(item.chunk.chunk_id)
                if current is None or float(item.score) > float(current.score):
                    by_id[item.chunk.chunk_id] = item
                elif "ai-postmerge" not in current.method:
                    by_id[item.chunk.chunk_id] = replace(current, method=current.method + "+ai-postmerge-retry")

            merged = sorted(
                by_id.values(),
                key=lambda item: (
                    0 if "current-authority-span" in item.method else 1,
                    0 if "current-explicit-wording" in item.method else 1,
                    0 if "authority-anchor" in item.method else 1,
                    0 if "smart-structured-value" in item.method else 1,
                    0 if "ai-postmerge-retry" in item.method else 1,
                    -float(item.score),
                    item.chunk.filename.casefold(),
                    item.chunk.page_number,
                ),
            )

            # A second small critic call is used only after an actual retry. It does not
            # generate another retry, so the loop remains strictly bounded.
            final_review = review_retrieved_evidence(interpretation, merged)
            logger.info(
                "smart_rag stage=ai_postmerge_final_review sufficient=%s ai=%s missing=%d reason=%s",
                final_review.sufficient,
                final_review.ai_used,
                len(final_review.missing_evidence),
                final_review.reason[:240],
            )
            _CURRENT_COVERAGE_STATUS.set(
                "sufficient_after_retry" if final_review.sufficient else "insufficient_after_retry"
            )
            return merged
        except Exception:
            logger.exception("Post-merge AI evidence retry failed; continuing with assembled candidates")
            _CURRENT_COVERAGE_STATUS.set("insufficient_after_retry_error")
            return ranked

    def smart_select(plan, candidates, max_chunks=None, *, preferred_document_ids=None):  # type: ignore[no-untyped-def]
        candidates = _postmerge_evidence_review(plan, candidates)
        cap = _int_env("SMART_RAG_FINAL_CONTEXT_CHUNKS", 48, 12, 120)
        definition_aliases = _CURRENT_DEFINITION_ALIASES.get()
        current_question = _CURRENT_ORIGINAL_QUESTION.get() or plan.original_question
        if definition_aliases:
            effective_cap = min(cap, max(12, len(definition_aliases) * 4))
        else:
            effective_cap = cap
        requested_cap = min(max_chunks or effective_cap, effective_cap)
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

        # Once explicit current-authority evidence is present, do not pass chunks
        # already identified as superseded into synthesis. They remain available in
        # diagnostics/retrieval logs but cannot silently win by cleaner OCR wording.
        has_current_authority = any(
            "current-authority-span" in item.method or "current-explicit-wording" in item.method
            for item in adjusted
        )
        if has_current_authority:
            adjusted = [item for item in adjusted if "superseded-" not in item.method]

        adjusted.sort(
            key=lambda item: (
                0 if "terminology-definition" in item.method else 1,
                0 if "current-authority-span" in item.method else 1,
                0 if "current-explicit-wording" in item.method else 1,
                0 if "authority-anchor" in item.method else 1,
                0 if "structured-value" in item.method else 1,
                0 if "current-amendment-context" in item.method else 1,
                -item.score,
            )
        )
        enriched, originals = relevance_enriched_candidates(adjusted)
        selected = original_select(
            relevance_plan(plan),
            enriched,
            max_chunks=requested_cap,
            preferred_document_ids=preferred_document_ids,
        )
        restored = restore_original_chunks(selected, originals)

        # Some chunks are essential navigation/structure evidence even when their
        # wording has little lexical overlap with a colloquial user question.
        definition_forced = [
            item for item in adjusted
            if definition_aliases and "terminology-definition" in item.method
        ]
        current_authority_forced = [
            item for item in adjusted
            if "current-authority-span" in item.method or "current-explicit-wording" in item.method
        ][:_int_env("SMART_RAG_CURRENT_AUTHORITY_CONTEXT_CHUNKS", 16, 4, 32)]
        authority_forced = [
            item for item in adjusted if "authority-anchor" in item.method
        ][:6]
        structured_forced = []
        if value_lookup_request(current_question):
            structured_forced = [
                item for item in adjusted
                if "smart-structured-value" in item.method and "superseded-" not in item.method
            ][:_int_env("SMART_RAG_STRUCTURED_VALUE_CONTEXT_CHUNKS", 12, 4, 24)]

        forced = [*definition_forced, *current_authority_forced, *authority_forced, *structured_forced]
        merged = []
        seen_ids: set[str] = set()
        for item in [*forced, *restored]:
            if item.chunk.chunk_id in seen_ids:
                continue
            if has_current_authority and "superseded-" in item.method:
                continue
            seen_ids.add(item.chunk.chunk_id)
            merged.append(item)
            if len(merged) >= requested_cap:
                break
        return merged

    def smart_ask_impl(
        self,
        db,
        question,
        top_k=None,
        rewrite_question=None,
        conversation_context=None,
    ):  # type: ignore[no-untyped-def]
        # Make the active request DB session available to the post-merge evidence
        # reviewer without changing the baseline service method signatures. ContextVar
        # keeps concurrent async/threaded requests isolated.
        db_token = _CURRENT_DB.set(db)
        coverage_token = _CURRENT_COVERAGE_STATUS.set("not_reviewed")
        reviewed_token = _CURRENT_AI_EVIDENCE_REVIEWED.set(False)
        try:
            return original_ask_impl(
                self,
                db,
                question,
                top_k,
                rewrite_question,
                conversation_context,
            )
        finally:
            _CURRENT_DB.reset(db_token)
            _CURRENT_COVERAGE_STATUS.reset(coverage_token)
            _CURRENT_AI_EVIDENCE_REVIEWED.reset(reviewed_token)

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

    def smart_synthesize_answer(plan, results, *, primary_document_ids=None, primary_document_names=None):  # type: ignore[no-untyped-def]
        """Use deterministic synthesis only where source text fully determines the answer.

        Acronym/full-form answers are exact strings from original definition chunks;
        the LLM is not allowed to paraphrase them into plausible-but-wrong expansions.
        Other intents continue through the existing grounded synthesis pipeline.
        """
        definition_aliases = _CURRENT_DEFINITION_ALIASES.get()
        current_question = _CURRENT_ORIGINAL_QUESTION.get() or plan.original_question
        filtered_results = list(results)
        if value_lookup_request(current_question) and any(
            "current-authority-span" in item.method or "current-explicit-wording" in item.method
            for item in filtered_results
        ):
            filtered_results = [item for item in filtered_results if "superseded-" not in item.method]

        if definition_aliases and plan.intent == "definition":
            primary_ids = set(primary_document_ids or ())
            primary_names = list(primary_document_names or ())
            sources = synthesis_module._prompt_sources(plan, filtered_results, primary_ids)
            definitions: dict[str, list[tuple[str, int]]] = {}
            for alias in definition_aliases:
                canonical_to_source: dict[str, int] = {}
                canonical_case: dict[str, str] = {}
                for number, source in enumerate(sources, 1):
                    canonical = definition_for_token(source.excerpt, alias)
                    if not canonical:
                        continue
                    key = " ".join(canonical.casefold().split())
                    canonical_to_source.setdefault(key, number)
                    canonical_case.setdefault(key, canonical)
                if canonical_to_source:
                    definitions[alias] = [
                        (canonical_case[key], number)
                        for key, number in canonical_to_source.items()
                    ]

            # Deterministic mode is used only when every requested target has an
            # explicit source definition. Partial coverage falls back to the normal
            # grounded model rather than inventing the missing expansion.
            if len(definitions) == len(definition_aliases):
                lines: list[str] = []
                multi = len(definition_aliases) > 1
                for alias in definition_aliases:
                    values = definitions[alias]
                    if len(values) == 1:
                        canonical, source_number = values[0]
                        sentence = f"{alias} stands for {canonical}. [S{source_number}]"
                    else:
                        alternatives = "; ".join(
                            f"{canonical} [S{source_number}]" for canonical, source_number in values[:3]
                        )
                        sentence = (
                            f"{alias} has more than one explicit expansion in the reviewed PDFs: "
                            f"{alternatives}. The applicable meaning depends on document/scope."
                        )
                    lines.append(f"- {sentence}" if multi else sentence)
                return synthesis_module.SynthesisBundle(
                    raw_answer="\n".join(lines),
                    sources=sources,
                    used_hierarchy=False,
                    primary_document_ids=frozenset(primary_ids),
                    primary_document_names=tuple(primary_names),
                )

        bundle = original_synthesize_answer(
            plan,
            filtered_results,
            primary_document_ids=primary_document_ids,
            primary_document_names=primary_document_names,
        )
        interpretation = _CURRENT_INTERPRETATION.get()
        if interpretation is not None:
            corrected = verify_answer(
                interpretation,
                bundle.raw_answer,
                bundle.sources,
                coverage_status=_CURRENT_COVERAGE_STATUS.get(),
            )
            if corrected and corrected != bundle.raw_answer:
                bundle = replace(bundle, raw_answer=corrected)
        return bundle

    def smart_source_header(source):  # type: ignore[no-untyped-def]
        header = original_source_header(source)
        notes = rule_notes_for_chunk(source.result.chunk.chunk_id, source.excerpt)
        if notes:
            # This is explicitly marked as derived. It helps the answer model apply
            # e.g. user=4 to a source threshold of >=2 without pretending the PDF said 4.
            header += " | SYSTEM-DERIVED APPLICABILITY (not PDF text): " + " ".join(notes)
        method = source.result.method
        if "current-authority-span" in method:
            header += " | CURRENT AUTHORITY SPAN: original PDF text inside an explicitly substituted/replaced section"
        if "current-explicit-wording" in method:
            header += " | CURRENT EXPLICIT WORDING: this text matches replacement wording stated by an authority anchor"
        if "superseded-" in method:
            header += " | SUPERSEDED CANDIDATE: do not present this as current when current authority evidence is supplied"
        if "smart-structured-value" in method:
            header += " | STRUCTURED VALUE EVIDENCE: semantically matched numeric/table row; neighboring chunks may supply its header/unit"
        if "current-amendment-context" in method:
            header += (
                " | SYSTEM-DERIVED AUTHORITY PROXIMITY (not PDF text): this excerpt is within "
                "the configured local chunk window after an explicit amendment/substitution "
                "anchor in the same PDF; use the anchor source itself to establish precedence"
            )
        if "authority-anchor" in method:
            header += " | AUTHORITY ANCHOR: original PDF text explicitly replacing/substituting/omitting prior wording or a section"
        elif "amendment-authority-anchor" in method:
            header += " | AUTHORITY ANCHOR: inspect this PDF text for explicit amendment/substitution wording"
        if "terminology-definition" in method:
            header += " | TERMINOLOGY DEFINITION SOURCE: this original PDF chunk was selected for the requested acronym/full form"
        return header

    def smart_deterministic_search_mode(original, contextual=None, intent=None):  # type: ignore[no-untyped-def]
        interpretation = _CURRENT_INTERPRETATION.get()
        if interpretation is not None and _CURRENT_ORIGINAL_QUESTION.get().strip() == str(original).strip():
            return "references" if interpretation.conversation_act == "navigation" else "answer"
        current_question = _CURRENT_ORIGINAL_QUESTION.get()
        active_aliases = _CURRENT_DEFINITION_ALIASES.get()
        if (current_question and original.strip() == current_question.strip() and active_aliases) or is_definition_request(original):
            return "answer"
        return original_search_mode(original, contextual, intent)

    def smart_needs_conversation_context(question):  # type: ignore[no-untyped-def]
        # Make recent USER turns available to the v4 two-pass interpreter. The first
        # interpretation is always performed without history; only a genuinely
        # context-dependent current message gets a second history-assisted pass.
        if _bool_env("SMART_RAG_AI_INTERPRETATION", True):
            return True
        return original_needs_conversation_context(question)

    def smart_use_primary_backbone(plan):  # type: ignore[no-untyped-def]
        if original_use_primary_backbone(plan):
            return True
        interpretation = _CURRENT_INTERPRETATION.get()
        if interpretation is None or plan.search_mode != "answer":
            return False
        # A routed document is a guaranteed candidate, never the sole corpus search.
        # This lets a precise claims/rule/procedure document contribute evidence even
        # when the user's surface wording looks like a simple fact lookup.
        return interpretation.route_strategy in {
            "dedicated_procedure",
            "authoritative_rule",
            "structured_lookup",
        }

    # Patch module globals actually referenced by RagService._ask_impl.
    service_module.find_abbreviation_hints = smart_find_abbreviation_hints
    service_module.deterministic_search_mode = smart_deterministic_search_mode
    service_module.needs_conversation_context = smart_needs_conversation_context
    service_module._use_primary_document_backbone = smart_use_primary_backbone
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
    service_module.RagService._ask_impl = smart_ask_impl
    service_module.RagService.process_document = smart_process_document
    service_module.synthesize_answer = smart_synthesize_answer

    # Planner instance was imported before app lifespan; patch the class method so
    # the existing singleton immediately uses scenario compilation.
    query_module.deterministic_search_mode = smart_deterministic_search_mode
    query_module.QueryPlanner.plan = smart_plan

    synthesis_module._compact_source_header = smart_source_header
    synthesis_module.synthesize_answer = smart_synthesize_answer
    if "SYSTEM-DERIVED APPLICABILITY" not in synthesis_module._ANSWER_SYSTEM_PROMPT:
        synthesis_module._ANSWER_SYSTEM_PROMPT += """

SYSTEM-DERIVED APPLICABILITY notes in source headers are not PDF quotations. They are deterministic
comparisons of facts stated by the user against explicit numeric conditions found in the cited PDF
excerpt (for example, user reports 4 affected brakes and the source says 2 or more). You may use a
TRUE derived comparison to decide that the cited rule applies, but cite the underlying PDF source,
state the comparison briefly when material, and never turn an uncertain semantic inference into a fact.
"""

    if "SMART-RAG AUTHORITY PRECEDENCE" not in synthesis_module._ANSWER_SYSTEM_PROMPT:
        synthesis_module._ANSWER_SYSTEM_PROMPT += """

SMART-RAG AUTHORITY PRECEDENCE:
- A TERMINOLOGY DEFINITION SOURCE is original PDF evidence. For acronym/full-form questions, answer
  from that definition instead of treating usage-only occurrences as a definition.
- AUTHORITY ANCHOR source text may explicitly state that a rule, amount, provision or schedule is
  amended or substituted. When the PDF explicitly establishes that replacement relationship, use the
  amended/substituted text for the affected subject and do not report the superseded value as current.
- SYSTEM-DERIVED AUTHORITY PROXIMITY is only a navigation hint. It cannot establish precedence by
  itself; confirm precedence from the cited AUTHORITY ANCHOR PDF text. If the evidence does not
  explicitly establish which version controls, state the conflict rather than choosing one.
"""

    if "SMART-RAG EVIDENCE-SET RELIABILITY" not in synthesis_module._ANSWER_SYSTEM_PROMPT:
        synthesis_module._ANSWER_SYSTEM_PROMPT += """

SMART-RAG EVIDENCE-SET RELIABILITY:
- Treat a table row together with its adjacent STRUCTURED VALUE EVIDENCE chunks as one local
  evidence unit when the PDF parser split the row from its heading, unit or column label. Do not
  reject a correct numeric row merely because the word used by the user appears only in a nearby
  heading or because the manual uses a more formal term.
- Semantic retrieval is navigation, not authority. A colloquial user description may route to a
  formal source row, but answer only when the source row is clearly the same concept. If two source
  categories could plausibly fit, state the alternatives or the missing distinction instead of
  silently choosing one.
- A source header marked SUPERSEDED CANDIDATE must never be presented as the current requirement
  when CURRENT AUTHORITY SPAN or CURRENT EXPLICIT WORDING evidence for the same affected subject is
  supplied. Cite the original authority anchor when precedence matters.
- For multi-part requests, answer every supported requested target separately. Do not begin with
  'the excerpts do not define/specify' and then immediately provide the supposedly missing fact.
  If useful evidence exists, state the supported fact directly; mention a limitation only for the
  genuinely unresolved part.
"""

    if "SMART-RAG AI-RESOLVED INTENT" not in synthesis_module._ANSWER_SYSTEM_PROMPT:
        synthesis_module._ANSWER_SYSTEM_PROMPT += """

SMART-RAG AI-RESOLVED INTENT:
- The CONTEXTUAL INTERPRETATION is a language-understanding aid, not factual evidence. Use it to
  understand spelling mistakes, colloquial phrasing, shorthand, follow-ups and what the user is
  actually asking. All metro facts still require cited PDF evidence.
- Answer the user's most likely intended question rather than mechanically echoing malformed wording.
  Silently normalize harmless spelling/grammar. Mention an interpretation only when a material
  ambiguity could change the answer.
- Retrieval similarity is not proof of answerability. A responsibility question requires evidence
  assigning responsibility; a requirement question requires evidence establishing whether it is
  required/allowed; a procedure question requires the applicable operational branch; and a value
  question requires the governing value/category, not merely a nearby number.
- Do not turn a failed first search into a corpus-wide negative claim. Negative answers should be
  used only after the reviewed evidence genuinely fails to satisfy the interpreted request.
- Treat current/amended authority, line, rolling stock, mode, location and person type as part of
  correctness, not optional context.
"""

    if "SMART-RAG ANSWER-FIRST USER STYLE V4" not in synthesis_module._ANSWER_SYSTEM_PROMPT:
        synthesis_module._ANSWER_SYSTEM_PROMPT += """

SMART-RAG ANSWER-FIRST USER STYLE V4:
- The user is asking the knowledge assistant, not inspecting the retrieval engine. Never begin with
  phrases such as 'The supplied excerpts...', 'The retrieved excerpts...', or 'The evidence does not...'
  unless the user explicitly asks about retrieval/evidence quality.
- Start with the most useful supported answer: Yes/No, the responsible role, the current amount, the
  required action, or the applicable procedure. Give source citations immediately with that answer.
- When only part of a multi-part question is unresolved, answer the supported parts first and mention
  the unresolved part afterward in one short sentence.
- A retrieval miss is not proof of corpus absence. Never say the documents do not state/specify/provide
  something unless the system has actually reviewed the governing evidence and the available sources
  establish that absence. Otherwise use a narrow phrase such as 'I could not verify X from the
  retrieved governing evidence' rather than making a corpus-wide claim.
- Do not repeat internal retrieval status, excerpt counts or search limitations in a normal answer.
"""

    _INSTALLED = True
    logger.info("Smart RAG runtime patch installed (AI-first understanding v4; post-merge evidence coverage)")
