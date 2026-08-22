from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path

MARKER = "IMS_LIVE_ACTIVITY_V1"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + ".bak-before-ims-live-activity-v1")
    if not target.exists():
        shutil.copy2(path, target)


def mark_python(source: str) -> str:
    if MARKER in source:
        return source
    return f"# {MARKER}\n" + source


def mark_text(source: str) -> str:
    if MARKER in source:
        return source
    return f"// {MARKER}\n" + source


def patch_models(source: str) -> str:
    if MARKER in source:
        return source
    source = replace_once(
        source,
        '    search_queries: list[str] = Field(default_factory=list)\n    answer_policy_version: str = ""\n',
        '    search_queries: list[str] = Field(default_factory=list)\n'
        '    # Persisted, safe operational trace shown in the collapsible Working panel.\n'
        '    activity: list[dict[str, Any]] = Field(default_factory=list)\n'
        '    answer_policy_version: str = ""\n',
        "AnswerResponse activity field",
    )
    return mark_python(source)


def patch_api(source: str) -> str:
    if MARKER in source:
        return source

    source = replace_once(
        source,
        '            "search_queries": response.search_queries,\n'
        '            "answer_policy_version": response.answer_policy_version,\n'
        '            "request_id": response.request_id,\n',
        '            "search_queries": response.search_queries,\n'
        '            "activity": response.activity,\n'
        '            "answer_policy_version": response.answer_policy_version,\n'
        '            "request_id": response.request_id,\n',
        "persist activity metadata",
    )

    old_callback = '''    loop = asyncio.get_running_loop()\n    queue: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()\n\n    def publish_progress(progress: dict[str, object]) -> None:\n        # rag_service runs in a worker thread. Schedule queue writes safely on the\n        # request event loop rather than touching asyncio.Queue from that thread.\n        loop.call_soon_threadsafe(queue.put_nowait, ("progress", dict(progress)))\n'''
    new_callback = '''    loop = asyncio.get_running_loop()\n    queue: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()\n    activity_trace: list[dict[str, object]] = []\n\n    def publish_progress(progress: dict[str, object]) -> None:\n        # rag_service runs in a worker thread. Both persistence and SSE delivery are\n        # marshalled onto the request event loop so the trace has one deterministic order.\n        event = dict(progress)\n\n        def deliver() -> None:\n            activity_trace.append(event)\n            queue.put_nowait(("progress", event))\n\n        loop.call_soon_threadsafe(deliver)\n'''
    source = replace_once(source, old_callback, new_callback, "stream activity collector")

    old_save = '''            await queue.put(\n                (\n                    "progress",\n                    {\n                        "stage": "save",\n                        "label": "Saving the grounded answer",\n                        "detail": "Recording the answer and its cited evidence in chat history",\n                    },\n                )\n            )\n            response = _finalize_chat_exchange(\n'''
    new_save = '''            # Let any thread-safe progress callbacks queued by the just-finished\n            # worker run before taking the persisted trace snapshot.\n            await asyncio.sleep(0)\n            response.activity = activity_trace[-80:]\n            await queue.put(\n                (\n                    "progress",\n                    {\n                        "stage": "save",\n                        "label": "Saving the grounded answer",\n                        "detail": "Recording the answer and its cited evidence in chat history",\n                        "actor": "backend",\n                        "phase": "save",\n                        "status": "running",\n                        "operation_id": "save",\n                    },\n                )\n            )\n            response = _finalize_chat_exchange(\n'''
    source = replace_once(source, old_save, new_save, "stream activity snapshot")
    return mark_python(source)


def patch_service(source: str) -> str:
    if MARKER in source:
        return source
    if "retrieve_assistant_v51" not in source or "AssistantRetrievalBundle" not in source:
        raise RuntimeError(
            "IMS Assistant v5.1 does not appear to be applied to backend/app/rag/v5/service.py. "
            "Apply the v5.1 assistant retrieval patch first, then run this installer."
        )

    source = replace_once(
        source,
        '        emit_progress("interpret", "Understanding your question", "Resolving likely intent before document search")\n',
        '''        emit_progress(\n            "interpret",\n            "Understanding your question",\n            "Resolving likely intent, typos, abbreviations and requested output",\n            actor="ai",\n            phase="interpret",\n            status="running",\n            operation_id="interpret",\n            prompt_summary="Resolve the user message into a self-contained retrieval question; tolerate typos and paraphrases without inventing metro facts.",\n        )\n''',
        "interpret start activity",
    )

    old_interpreted = '''        emit_progress(\n            "interpret",\n            "Question interpreted",\n            f"Intent: {interpretation.intent}; evidence needs: {len(interpretation.evidence_needs)}",\n        )\n'''
    new_interpreted = '''        emit_progress(\n            "interpret",\n            "Question interpreted",\n            f"Intent: {interpretation.intent}; evidence needs: {len(interpretation.evidence_needs)}; search formulations: {len(interpretation.search_queries)}",\n            actor="ai",\n            phase="interpret",\n            status="complete",\n            operation_id="interpret",\n            reasoning_summary=f"Resolved question: {resolved[:700]}",\n            metrics={\n                "evidence_needs": len(interpretation.evidence_needs),\n                "search_queries": len(interpretation.search_queries),\n                "corrections": len(interpretation.corrections),\n            },\n        )\n'''
    source = replace_once(source, old_interpreted, new_interpreted, "interpret completion activity")

    old_retrieval = '''        emit_progress(\n            "search",\n            "Navigating documents and sections",\n            "Routing across the corpus, matching headings and ranking governing evidence",\n        )\n        max_rounds = _int_env("RAG_V51_MAX_SEARCH_ROUNDS", 3, 1, 3)\n        bundle: AssistantRetrievalBundle = retrieve_assistant_v51(\n            db,\n            interpretation,\n            original_question=question,\n        )\n        review = review_retrieved_evidence(interpretation, bundle.results)\n        coverage_status = "sufficient" if review.sufficient else "insufficient_after_review"\n\n        # Search until the evidence critic is satisfied, within a strict bounded budget.\n        # Each retry accumulates previous good evidence instead of replacing it.\n        for search_round in range(2, max_rounds + 1):\n            if review.sufficient:\n                break\n            retry_queries = list(review.retry_queries) or list(review.missing_evidence)\n            if not retry_queries:\n                break\n            emit_progress(\n                "search_retry",\n                "Filling evidence gaps",\n                f"Targeted evidence search {search_round}/{max_rounds}",\n            )\n            bundle = retrieve_assistant_v51(\n                db,\n                interpretation,\n                original_question=question,\n                extra_queries=retry_queries,\n                prior_results=bundle.results,\n            )\n            review = review_retrieved_evidence(interpretation, bundle.results)\n            coverage_status = (\n                f"sufficient_after_retry_{search_round}"\n                if review.sufficient\n                else f"insufficient_after_retry_{search_round}"\n            )\n'''
    new_retrieval = '''        max_rounds = _int_env("RAG_V51_MAX_SEARCH_ROUNDS", 3, 1, 3)\n        emit_progress(\n            "search_round_1",\n            "Search round 1",\n            "Routing across the corpus, matching headings and ranking governing evidence",\n            actor="search",\n            phase="search",\n            status="running",\n            operation_id="search-round-1",\n            current=1,\n            total=max_rounds,\n            prompt_summary="Search broadly first, then route to likely documents and governing sections using semantic, lexical and structural evidence.",\n        )\n        bundle: AssistantRetrievalBundle = retrieve_assistant_v51(\n            db,\n            interpretation,\n            original_question=question,\n            activity_round=1,\n        )\n        emit_progress(\n            "search_round_1",\n            "Search round 1 complete",\n            f"Collected {bundle.candidate_count} candidate evidence unit(s) across {len(bundle.routed_documents)} routed document(s)",\n            actor="search",\n            phase="search",\n            status="complete",\n            operation_id="search-round-1",\n            current=1,\n            total=max_rounds,\n            metrics={\n                "candidates": bundle.candidate_count,\n                "routed_documents": len(bundle.routed_documents),\n            },\n        )\n        emit_progress(\n            "evidence_review_1",\n            "Checking evidence completeness",\n            "The AI critic is checking whether the retrieved governing material covers every requested point",\n            actor="verification",\n            phase="review",\n            status="running",\n            operation_id="evidence-review-1",\n            prompt_summary="Judge coverage only: identify missing evidence or confirm the retrieved PDF evidence is sufficient. Do not answer the user.",\n        )\n        review = review_retrieved_evidence(interpretation, bundle.results)\n        coverage_status = "sufficient" if review.sufficient else "insufficient_after_review"\n        review_summary = review.reason or (\n            "Evidence covers the requested answer."\n            if review.sufficient\n            else "Missing: " + "; ".join(review.missing_evidence[:3])\n        )\n        emit_progress(\n            "evidence_review_1",\n            "Evidence sufficient" if review.sufficient else "More evidence needed",\n            "Initial evidence review completed",\n            actor="verification",\n            phase="review",\n            status="complete" if review.sufficient else "warning",\n            operation_id="evidence-review-1",\n            reasoning_summary=review_summary[:1000],\n            metrics={\n                "missing_items": len(review.missing_evidence),\n                "retry_queries": len(review.retry_queries),\n            },\n        )\n\n        # Search until the evidence critic is satisfied, within a strict bounded budget.\n        # Each retry accumulates previous good evidence instead of replacing it.\n        for search_round in range(2, max_rounds + 1):\n            if review.sufficient:\n                break\n            retry_queries = list(review.retry_queries) or list(review.missing_evidence)\n            if not retry_queries:\n                break\n            search_operation = f"search-round-{search_round}"\n            review_operation = f"evidence-review-{search_round}"\n            emit_progress(\n                f"search_round_{search_round}",\n                f"Search round {search_round} of {max_rounds}",\n                "Running a targeted search for the evidence the previous review found missing",\n                actor="search",\n                phase="search",\n                status="running",\n                operation_id=search_operation,\n                current=search_round,\n                total=max_rounds,\n                prompt_summary="Search specifically for the missing evidence identified by the coverage critic while retaining useful evidence from earlier rounds.",\n            )\n            bundle = retrieve_assistant_v51(\n                db,\n                interpretation,\n                original_question=question,\n                extra_queries=retry_queries,\n                prior_results=bundle.results,\n                activity_round=search_round,\n            )\n            emit_progress(\n                f"search_round_{search_round}",\n                f"Search round {search_round} complete",\n                f"Candidate pool now contains {bundle.candidate_count} evidence unit(s)",\n                actor="search",\n                phase="search",\n                status="complete",\n                operation_id=search_operation,\n                current=search_round,\n                total=max_rounds,\n                metrics={\n                    "candidates": bundle.candidate_count,\n                    "routed_documents": len(bundle.routed_documents),\n                },\n            )\n            emit_progress(\n                f"evidence_review_{search_round}",\n                "Rechecking evidence completeness",\n                f"Reviewing accumulated evidence after search round {search_round}",\n                actor="verification",\n                phase="review",\n                status="running",\n                operation_id=review_operation,\n                prompt_summary="Judge whether the accumulated PDF evidence now covers all requested points; identify only genuinely missing evidence.",\n            )\n            review = review_retrieved_evidence(interpretation, bundle.results)\n            coverage_status = (\n                f"sufficient_after_retry_{search_round}"\n                if review.sufficient\n                else f"insufficient_after_retry_{search_round}"\n            )\n            review_summary = review.reason or (\n                "Evidence covers the requested answer."\n                if review.sufficient\n                else "Missing: " + "; ".join(review.missing_evidence[:3])\n            )\n            emit_progress(\n                f"evidence_review_{search_round}",\n                "Evidence sufficient" if review.sufficient else "Evidence still incomplete",\n                f"Coverage review after search round {search_round}",\n                actor="verification",\n                phase="review",\n                status="complete" if review.sufficient else "warning",\n                operation_id=review_operation,\n                reasoning_summary=review_summary[:1000],\n                metrics={\n                    "missing_items": len(review.missing_evidence),\n                    "retry_queries": len(review.retry_queries),\n                },\n            )\n'''
    source = replace_once(source, old_retrieval, new_retrieval, "assistant retrieval live activity")

    old_answer_progress = '''        emit_progress(\n            "answer_generation",\n            "Writing the grounded answer",\n            f"Reviewing {len(prompt_sources)} structure-preserving evidence unit(s)",\n        )\n'''
    new_answer_progress = '''        emit_progress(\n            "answer_generation",\n            "Writing the grounded answer",\n            f"Using {len(prompt_sources)} reviewed evidence unit(s) from {len({source.result.chunk.filename for source in prompt_sources})} document(s)",\n            actor="ai",\n            phase="answer",\n            status="running",\n            operation_id="answer-generation",\n            prompt_summary="Answer the resolved question using only the supplied PDF evidence; preserve conditions, numbers and citations and do not invent metro facts.",\n            metrics={"evidence_units": len(prompt_sources)},\n        )\n'''
    source = replace_once(source, old_answer_progress, new_answer_progress, "answer generation activity")

    source = replace_once(
        source,
        '''        verified = verify_answer(\n''',
        '''        emit_progress(\n            "answer_generation",\n            "Draft answer prepared",\n            "The grounded draft is ready for factual and citation verification",\n            actor="ai",\n            phase="answer",\n            status="complete",\n            operation_id="answer-generation",\n        )\n        emit_progress(\n            "verification",\n            "Verifying facts and citations",\n            "Checking semantic correctness, completeness, scope, authority and citation support",\n            actor="verification",\n            phase="verify",\n            status="running",\n            operation_id="verification",\n            prompt_summary="Verify the draft only against supplied PDF sources; remove unsupported claims and preserve valid citations.",\n        )\n        verified = verify_answer(\n''',
        "verification start activity",
    )

    source = replace_once(
        source,
        '        grounding_status = "verified" if grounded else "citation_validation_failed"\n',
        '''        grounding_status = "verified" if grounded else "citation_validation_failed"\n        emit_progress(\n            "verification",\n            "Grounding verified" if grounded else "Citation validation needs repair",\n            "Final grounding check completed",\n            actor="verification",\n            phase="verify",\n            status="complete" if grounded else "warning",\n            operation_id="verification",\n            reasoning_summary=(\n                "Every retained factual claim has valid PDF citation support."\n                if grounded\n                else "The draft needs citation repair before it can be returned."\n            ),\n        )\n''',
        "verification completion activity",
    )

    source = replace_once(
        source,
        '            emit_progress("citation_repair", "Repairing citation grounding", "Keeping only claims supported by the reviewed PDF evidence")\n',
        '''            emit_progress(\n                "citation_repair",\n                "Repairing citation grounding",\n                "Keeping only claims supported by the reviewed PDF evidence",\n                actor="verification",\n                phase="verify",\n                status="running",\n                operation_id="citation-repair",\n                prompt_summary="Repair citations without adding new factual claims; remove or qualify anything unsupported by the supplied sources.",\n            )\n''',
        "citation repair activity",
    )

    source = replace_once(
        source,
        '''        used = set(cited_source_numbers(answer, len(prompt_sources)))\n''',
        '''        used = set(cited_source_numbers(answer, len(prompt_sources)))\n        emit_progress(\n            "answer_ready",\n            "Answer ready",\n            f"Grounded answer prepared with {len(used)} cited source(s)",\n            actor="verification",\n            phase="verify",\n            status="complete",\n            operation_id="answer-ready",\n            metrics={\n                "cited_sources": len(used),\n                "evidence_units": len(prompt_sources),\n                "grounded": grounded,\n            },\n        )\n''',
        "answer ready activity",
    )
    return mark_python(source)


def patch_assistant_retrieval(source: str) -> str:
    if MARKER in source:
        return source
    if "def retrieve_assistant_v51(" not in source:
        raise RuntimeError("assistant_retrieval.py does not contain retrieve_assistant_v51")

    source = replace_once(
        source,
        "from app.rag.llm import llm_service\n",
        "from app.rag.llm import llm_service\nfrom app.rag.progress import emit_progress\n",
        "assistant progress import",
    )

    new_function = r'''def retrieve_assistant_v51(
    db: Session,
    interpretation: SmartInterpretation,
    *,
    original_question: str,
    extra_queries: Iterable[str] = (),
    prior_results: Sequence[RetrievedChunk] = (),
    activity_round: int = 1,
) -> AssistantRetrievalBundle:
    queries = _query_variants(interpretation, extra_queries)
    suffix = max(1, int(activity_round))

    emit_progress(
        f"baseline_retrieval_{suffix}",
        "Searching the full corpus",
        f"Running {len(queries)} semantic/lexical search formulation(s) before document routing",
        actor="search",
        phase="search",
        status="running",
        operation_id=f"baseline-retrieval-{suffix}",
        metrics={"query_variants": len(queries)},
    )
    baseline_bundle = retrieve_v5(db, interpretation, extra_queries=extra_queries)
    emit_progress(
        f"baseline_retrieval_{suffix}",
        "Broad corpus search complete",
        f"Initial retrieval produced {baseline_bundle.candidate_count} candidate(s)",
        actor="search",
        phase="search",
        status="complete",
        operation_id=f"baseline-retrieval-{suffix}",
        metrics={"baseline_candidates": baseline_bundle.candidate_count},
    )

    if not _bool_env("RAG_V51_ASSISTANT_ENABLED", True):
        combined = _merge_results([*prior_results, *baseline_bundle.results])
        emit_progress(
            f"assistant_routing_{suffix}",
            "Assistant routing disabled",
            "Using the existing broad v5 retrieval order",
            actor="backend",
            phase="route",
            status="warning",
            operation_id=f"assistant-routing-{suffix}",
            metrics={"candidates": len(combined)},
        )
        return AssistantRetrievalBundle(
            results=combined[: _int_env("RAG_V51_FINAL_CANDIDATES", 64, 24, 120)],
            search_queries=_unique([*queries, *baseline_bundle.search_queries], 16),
            candidate_count=len(combined),
            routed_documents=[],
        )

    emit_progress(
        f"document_routing_{suffix}",
        "Identifying likely governing documents",
        "Combining filename hints, broad evidence, vector similarity and strict lexical matches",
        actor="search",
        phase="route",
        status="running",
        operation_id=f"document-routing-{suffix}",
        prompt_summary="Route to likely documents without assuming the user knows the source; document hints are preferences, not hard constraints.",
    )
    routes = _route_documents(
        db,
        question=original_question,
        queries=queries,
        baseline=baseline_bundle.results,
    )
    routed_names = [route.filename for route in routes]
    emit_progress(
        f"document_routing_{suffix}",
        "Document routing complete",
        "; ".join(routed_names[:5]) if routed_names else "No single document dominated; broad retrieval remains active",
        actor="search",
        phase="route",
        status="complete",
        operation_id=f"document-routing-{suffix}",
        metrics={"routed_documents": len(routes)},
    )

    emit_progress(
        f"structural_navigation_{suffix}",
        "Searching headings and governing sections",
        f"Inspecting structural headings within {len(routes)} routed document(s)",
        actor="search",
        phase="search",
        status="running",
        operation_id=f"structural-navigation-{suffix}",
    )
    headings = _heading_candidates(db, routes, queries, interpretation)
    emit_progress(
        f"structural_navigation_{suffix}",
        "Structural navigation complete",
        f"Found {len(headings)} heading/section candidate(s)",
        actor="search",
        phase="search",
        status="complete",
        operation_id=f"structural-navigation-{suffix}",
        metrics={"heading_matches": len(headings)},
    )
    if headings and headings[0].score >= 0.68:
        top_heading = headings[0].chunk
        emit_progress(
            f"strong_structural_match_{suffix}",
            "Strong structural match found",
            "A heading/section closely matches the requested subject",
            actor="search",
            phase="search",
            status="complete",
            operation_id=f"strong-structural-match-{suffix}",
            document=top_heading.filename,
            page=top_heading.page_number,
            heading=top_heading.heading or (" > ".join(top_heading.section_path) if top_heading.section_path else ""),
            metrics={"score": round(float(headings[0].score), 3)},
        )

    emit_progress(
        f"scoped_retrieval_{suffix}",
        "Searching inside routed documents",
        "Running semantic and strict lexical retrieval inside the strongest document candidates",
        actor="search",
        phase="search",
        status="running",
        operation_id=f"scoped-retrieval-{suffix}",
    )
    scoped = _scoped_candidates(db, routes, queries)
    emit_progress(
        f"scoped_retrieval_{suffix}",
        "Scoped document search complete",
        f"Collected {len(scoped)} routed-document candidate(s)",
        actor="search",
        phase="search",
        status="complete",
        operation_id=f"scoped-retrieval-{suffix}",
        metrics={"scoped_candidates": len(scoped)},
    )

    candidates = _merge_results([
        *prior_results,
        *baseline_bundle.results,
        *headings,
        *scoped,
    ])
    candidate_count = len(candidates)
    rerank_limit = min(candidate_count, _int_env("RAG_V51_RERANK_CANDIDATES", 48, 12, 80))
    emit_progress(
        f"ai_rerank_{suffix}",
        "AI ranking candidate evidence",
        f"Comparing the strongest {rerank_limit} of {candidate_count} candidate(s) for governing relevance",
        actor="ai",
        phase="rerank",
        status="running",
        operation_id=f"ai-rerank-{suffix}",
        prompt_summary="Rank candidate excerpts for directness, governing relevance and completeness. Prefer defining rules/sections over incidental mentions; do not answer the question.",
        metrics={"candidate_pool": candidate_count, "rerank_pool": rerank_limit},
    )
    reranked = _ai_rerank(interpretation, candidates, routed_names)
    ai_ranked = sum(1 for item in reranked[:rerank_limit] if "v5.1-ai-rerank" in item.method)
    top = reranked[0].chunk if reranked else None
    emit_progress(
        f"ai_rerank_{suffix}",
        "AI evidence ranking complete" if ai_ranked else "Deterministic evidence ranking used",
        f"{ai_ranked} candidate(s) received explicit AI reranking" if ai_ranked else "AI reranking was unavailable or disabled; deterministic ranking was preserved",
        actor="ai",
        phase="rerank",
        status="complete" if ai_ranked else "warning",
        operation_id=f"ai-rerank-{suffix}",
        reasoning_summary=(
            f"Top evidence is {top.filename}, page {top.page_number}, section {top.heading or (' > '.join(top.section_path) if top.section_path else 'unsectioned')}."
            if top is not None else "No evidence survived candidate ranking."
        ),
        document=top.filename if top is not None else None,
        page=top.page_number if top is not None else None,
        heading=(top.heading or (" > ".join(top.section_path) if top.section_path else "")) if top is not None else None,
        metrics={"ai_ranked": ai_ranked, "candidate_pool": candidate_count},
    )

    emit_progress(
        f"section_expansion_{suffix}",
        "Expanding governing sections",
        "Reading surrounding chunks from the strongest parent sections so lists and procedures are not truncated",
        actor="search",
        phase="search",
        status="running",
        operation_id=f"section-expansion-{suffix}",
    )
    expanded = _expand_sections(db, interpretation, reranked)
    emit_progress(
        f"section_expansion_{suffix}",
        "Section expansion complete",
        f"Added {len(expanded)} surrounding section evidence unit(s)",
        actor="search",
        phase="search",
        status="complete",
        operation_id=f"section-expansion-{suffix}",
        metrics={"expanded_evidence": len(expanded)},
    )

    final = _merge_results([*reranked, *expanded])
    final_limit = _int_env("RAG_V51_FINAL_CANDIDATES", 64, 24, 120)
    emit_progress(
        f"retrieval_ready_{suffix}",
        "Evidence candidates ready",
        f"Passing the strongest {min(len(final), final_limit)} evidence unit(s) to completeness review",
        actor="backend",
        phase="search",
        status="complete",
        operation_id=f"retrieval-ready-{suffix}",
        metrics={"final_candidates": min(len(final), final_limit)},
    )
    return AssistantRetrievalBundle(
        results=final[:final_limit],
        search_queries=_unique([*queries, *baseline_bundle.search_queries], 16),
        candidate_count=candidate_count,
        routed_documents=routed_names,
    )
'''

    pattern = re.compile(r"\ndef retrieve_assistant_v51\([\s\S]*\Z")
    match = pattern.search(source)
    if not match:
        raise RuntimeError("Could not locate retrieve_assistant_v51 function tail")
    source = source[: match.start()] + "\n" + new_function.strip() + "\n"
    return mark_python(source)


def patch_api_ts(source: str) -> str:
    if MARKER in source:
        return source
    source = replace_once(
        source,
        '''  search_queries: string[]\n  /** Backend answer-routing policy fingerprint for deployment diagnostics. */\n''',
        '''  search_queries: string[]\n  /** Safe operational/AI activity trace used by the collapsible Working panel. */\n  activity?: ChatProgressEvent[]\n  /** Backend answer-routing policy fingerprint for deployment diagnostics. */\n''',
        "AnswerResponse activity TS",
    )

    old_progress = '''export interface ChatProgressEvent {\n  stage: string\n  label: string\n  detail?: string\n  current?: number\n  total?: number\n  timestamp?: number\n}\n'''
    new_progress = '''export interface ChatProgressEvent {\n  stage: string\n  label: string\n  detail?: string\n  current?: number\n  total?: number\n  timestamp?: number\n  actor?: 'ai' | 'backend' | 'search' | 'verification' | string\n  phase?: string\n  status?: 'running' | 'complete' | 'completed' | 'warning' | 'error' | string\n  operation_id?: string\n  sequence?: number\n  duration_ms?: number\n  total_elapsed_ms?: number\n  prompt_summary?: string\n  reasoning_summary?: string\n  document?: string\n  page?: number\n  heading?: string\n  metrics?: Record<string, string | number | boolean | null>\n}\n'''
    source = replace_once(source, old_progress, new_progress, "structured progress TS")
    return mark_text(source)


def patch_app(source: str) -> str:
    if MARKER in source:
        return source

    source = replace_once(
        source,
        '''    search_queries: Array.isArray(metadata.search_queries)\n      ? (metadata.search_queries as string[])\n      : [],\n    answer_policy_version:\n''',
        '''    search_queries: Array.isArray(metadata.search_queries)\n      ? (metadata.search_queries as string[])\n      : [],\n    activity: Array.isArray(metadata.activity)\n      ? (metadata.activity as ChatProgressEvent[])\n      : [],\n    answer_policy_version:\n''',
        "reload saved activity",
    )

    old_update = '''function updateChatProgress(event: ChatProgressEvent): void {\n  // Keep one live row per backend stage. When a stage resumes after a temporary\n  // rate-limit wait, move it back to the end so the newest backend activity is\n  // always the active item in the UI.\n  chatProgress.value = [\n    ...chatProgress.value.filter((item) => item.stage !== event.stage),\n    event,\n  ].slice(-8)\n}\n'''
    new_update = '''function updateChatProgress(event: ChatProgressEvent): void {\n  // Keep one row per logical operation while retaining distinct search rounds and\n  // AI/backend phases. A completion event replaces its matching running event.\n  const key = event.operation_id || event.stage\n  const previous = chatProgress.value.findIndex(\n    (item) => (item.operation_id || item.stage) === key,\n  )\n  const next = [...chatProgress.value]\n  if (previous >= 0) next.splice(previous, 1, event)\n  else next.push(event)\n  chatProgress.value = next.slice(-48)\n}\n'''
    source = replace_once(source, old_update, new_update, "live operation progress state")

    old_finish = '''    // The backend work panel is temporary. Replace it with the final grounded\n    // answer instead of leaving a synthetic progress transcript in chat history.\n    chatProgress.value = []\n    messages.value.push({\n'''
    new_finish = '''    // Persist the server-provided safe activity trace with the assistant answer so\n    // the Working panel collapses after completion but can be reopened later.\n    if (!response.activity?.length && chatProgress.value.length) {\n      response.activity = [...chatProgress.value]\n    }\n    chatProgress.value = []\n    messages.value.push({\n'''
    source = replace_once(source, old_finish, new_finish, "persist completed activity in UI")
    return mark_text(source)


def patch_chat_panel(source: str) -> str:
    if MARKER in source:
        return source

    source = replace_once(
        source,
        "import { renderMarkdown } from '../utils/markdown'\n",
        "import { renderMarkdown } from '../utils/markdown'\nimport ActivityTrace from './ActivityTrace.vue'\n",
        "ActivityTrace import",
    )

    old_helpers = '''const visibleProgress = computed(() => props.progress.slice(-6))\nconst currentProgress = computed(() => props.progress[props.progress.length - 1] ?? null)\n\nfunction progressPercent(event: ChatProgressEvent): number | null {\n  if (typeof event.current !== 'number' || typeof event.total !== 'number' || event.total <= 0) {\n    return null\n  }\n  return Math.max(0, Math.min(100, Math.round((event.current / event.total) * 100)))\n}\n\nfunction isActiveProgress(event: ChatProgressEvent): boolean {\n  return currentProgress.value?.stage === event.stage\n}\n\n'''
    source = replace_once(source, old_helpers, "", "legacy work progress helpers")

    # The ChatPanel used optional chaining on `evidence` before this patch was
    # packaged (`message.response?.evidence?.length`). Older builds used
    # `message.response?.evidence.length`. Match either form so the installer
    # remains compatible with both and can safely resume a partially applied run.
    evidence_pattern = re.compile(
        r'          <details\n            v-if="message\.response\?\.evidence(?:\?\.|\.)length"\n'
    )
    evidence_matches = list(evidence_pattern.finditer(source))
    if len(evidence_matches) != 1:
        raise RuntimeError(
            f"completed Working panel: expected exactly one evidence panel anchor, found {len(evidence_matches)}"
        )
    activity_insert = '''          <ActivityTrace
            v-if="message.role === 'assistant' && message.response?.activity?.length"
            :events="message.response.activity"
          />

'''
    match = evidence_matches[0]
    source = source[: match.start()] + activity_insert + source[match.start() :]

    pattern = re.compile(
        r'''          <section class="work-progress" aria-live="polite" aria-label="Answer preparation progress">[\s\S]*?          </section>\n          <button class="cancel-link"'''
    )
    replacement = '''          <ActivityTrace :events="props.progress" live />\n          <button class="cancel-link"'''
    source, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"live Working panel: expected one legacy progress section, found {count}")
    return mark_text(source)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add rich ChatGPT-style live operational activity to IMS Assistant v5.1."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    payload = Path(__file__).resolve().parent / "payload"

    progress = repo / "backend/app/rag/progress.py"
    models = repo / "backend/app/models.py"
    api = repo / "backend/app/api.py"
    service = repo / "backend/app/rag/v5/service.py"
    assistant = repo / "backend/app/rag/v5/assistant_retrieval.py"
    api_ts = repo / "frontend/src/services/api.ts"
    app_vue = repo / "frontend/src/App.vue"
    chat_panel = repo / "frontend/src/components/ChatPanel.vue"
    activity_component = repo / "frontend/src/components/ActivityTrace.vue"
    test_target = repo / "backend/tests/test_live_activity_progress.py"

    required = [progress, models, api, service, assistant, api_ts, app_vue, chat_panel]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Required repository file(s) not found:\n" + "\n".join(missing))

    if "retrieve_assistant_v51" not in service.read_text(encoding="utf-8", errors="ignore"):
        raise SystemExit(
            "IMS Assistant v5.1 is not active in backend/app/rag/v5/service.py. "
            "Apply the v5.1 assistant patch before this live-activity patch."
        )

    payload_progress = payload / "backend/app/rag/progress.py"
    payload_component = payload / "frontend/src/components/ActivityTrace.vue"
    payload_test = payload / "backend/tests/test_live_activity_progress.py"
    for source in (payload_progress, payload_component, payload_test):
        if not source.exists():
            raise SystemExit(f"Patch payload missing: {source}")

    # Idempotent shortcut.
    if all(MARKER in path.read_text(encoding="utf-8", errors="ignore") for path in (models, api, service, assistant, api_ts, app_vue, chat_panel)) and activity_component.exists():
        print("IMS live activity patch is already applied.")
        return 0

    for path in required:
        backup(path)
    if activity_component.exists():
        backup(activity_component)

    progress.write_text(payload_progress.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    activity_component.parent.mkdir(parents=True, exist_ok=True)
    activity_component.write_text("<!-- IMS_LIVE_ACTIVITY_V1 -->\n" + payload_component.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    test_target.parent.mkdir(parents=True, exist_ok=True)
    test_target.write_text(payload_test.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    models.write_text(patch_models(models.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    api.write_text(patch_api(api.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    service.write_text(patch_service(service.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    assistant.write_text(patch_assistant_retrieval(assistant.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    api_ts.write_text(patch_api_ts(api_ts.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    app_vue.write_text(patch_app(app_vue.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    chat_panel.write_text(patch_chat_panel(chat_panel.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")

    for path in (progress, models, api, service, assistant, test_target):
        py_compile.compile(str(path), doraise=True)

    print("Applied IMS live activity patch.")
    print("Adds:")
    print(" - structured actor/phase/status progress events")
    print(" - real routing/search/rerank/section counts")
    print(" - strong-match filename/page/heading events")
    print(" - search-round and evidence-completeness events")
    print(" - elapsed operation + total working time")
    print(" - persistent, collapsible Working panel")
    print(" - safe AI task/reasoning summaries (no hidden chain-of-thought/raw system prompts)")
    print("No database migration and no PDF reprocessing are required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
