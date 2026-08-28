from __future__ import annotations

import argparse
import json
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_RAG_V6_EVIDENCE_WORKSPACE_PILOT"
V55_MARKER = "IMS_RAG_V55_PROCEDURE_INTEGRITY"
V551_MARKER = "IMS_RAG_V551_VIRTUAL_CHUNK_ID_GUARD"
BACKUP_SUFFIX = ".bak-before-ims-v6-evidence-workspace-pilot"


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not target.exists():
        shutil.copy2(path, target)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def insert_before_function(source: str, name: str, addition: str, label: str) -> str:
    anchor = f"\ndef {name}("
    index = source.find(anchor)
    if index < 0:
        raise RuntimeError(f"{label}: function {name} not found")
    return source[:index] + "\n" + addition.rstrip() + "\n" + source[index:]


def replace_class_method_to_eof(source: str, method_name: str, replacement: str) -> str:
    anchor = f"    def {method_name}("
    start = source.find(anchor)
    if start < 0:
        raise RuntimeError(f"Class method {method_name} not found")
    # V5RagService._ask_impl is intentionally the final method in service.py in the
    # supported v5.3-v5.5 runtime. Replacing to EOF cleanly bypasses the accumulated
    # answer-policy repair chain while preserving ingestion/process_document above it.
    return source[:start] + replacement.rstrip() + "\n"


def patch_synthesis_retrieval(source: str) -> str:
    if MARKER in source:
        return source
    if V55_MARKER not in source:
        raise RuntimeError(
            "backend/app/rag/v5/synthesis_retrieval.py is not the v5.5 procedure-integrity runtime. "
            "Apply v5.5 first."
        )

    applicability_helpers = r'''_V6_APPLICABILITY_GATE_SYSTEM = """You are a retrieval applicability gate for a CLOSED-BOOK PDF assistant.
You do NOT answer the question. You decide which already-reranked evidence units are safe seeds for
STRUCTURAL EXPANSION (whole table/section retrieval).

Why this gate exists: expanding the wrong seed can turn one loosely related excerpt into a large,
authoritative-looking but irrelevant table/section. Therefore be stricter than ordinary retrieval.

Keep a candidate when it directly matches or can genuinely govern the user's subject + scenario + condition
+ requested action/value + explicit scope. Reject candidates that merely share words but belong to a materially
different scenario (for example rescue movement vs ordinary fault handling, evacuation vs routine operation,
maintenance/test/depot vs revenue operation) unless the user's question makes that scenario applicable.

Rules:
- Do not reject merely because wording differs or a source uses a formal synonym.
- Preserve source-defined measurement bases; do not assume that different units/populations are equivalent.
- A candidate may be useful final context even if it is not safe to EXPAND. This decision controls expansion only.
- Prefer governing, applicability, restriction, exception and authority evidence as expansion seeds.
- When uncertain, keep the candidate rather than inventing a mismatch.

Return JSON only:
{"keep_ids":["chunk-id"],"reject_ids":["chunk-id"],"reason":"short summary"}
"""


def _v6_applicable_structure_seeds(
    question: str,
    interpretation: SmartInterpretation,
    candidates: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Select safe seeds for table/section expansion without changing final evidence.

    The original reranked candidates remain in the final pool. Only expansion is gated,
    so an over-strict classifier cannot erase source evidence from the answer pipeline.
    """
    values = list(candidates)
    if not values or os.getenv("RAG_V6_APPLICABILITY_GATE", "1").strip().casefold() in {"0", "false", "no", "off"}:
        return values

    limit = _int_env("RAG_V6_APPLICABILITY_CANDIDATES", 48, 12, 80)
    excerpt_chars = _int_env("RAG_V6_APPLICABILITY_EXCERPT_CHARS", 700, 240, 1400)
    pool = values[:limit]
    blocks: list[str] = []
    for index, item in enumerate(pool, 1):
        chunk = item.chunk
        pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        excerpt = re.sub(r"\s+", " ", chunk.text).strip()[:excerpt_chars]
        blocks.append(
            f"[{index}] id={chunk.chunk_id}\n"
            f"File: {chunk.filename} | Pages: {pages} | Section: {chunk.heading or 'Unsectioned'}\n"
            f"Retrieval role/method: {item.method}\n"
            f"Excerpt: {excerpt}"
        )

    prompt = f"""USER QUESTION:
{question}

RESOLVED QUESTION:
{interpretation.resolved_question}

EXPLICIT/RESOLVED SCOPE:
{interpretation.scope or 'None specified'}

CANDIDATES TO CLASSIFY FOR STRUCTURAL EXPANSION:
{chr(10).join(chr(10) + block for block in blocks)}

Return only the JSON keep/reject decision."""
    try:
        settings = get_settings()
        raw = llm_service.generate(
            _V6_APPLICABILITY_GATE_SYSTEM,
            prompt,
            max_output_tokens=_int_env("RAG_V6_APPLICABILITY_MAX_OUTPUT_TOKENS", 1100, 400, 2200),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        payload = _json_object(raw)
        raw_keep = payload.get("keep_ids")
        keep = {
            str(value)
            for value in raw_keep
            if str(value).strip()
        } if isinstance(raw_keep, list) else set()
    except Exception:
        return values

    if not keep:
        # Fail open. Expansion quality must improve when the gate works, but a malformed
        # model response must not delete all structural retrieval.
        return values
    selected = [item for item in pool if item.chunk.chunk_id in keep]
    if not selected:
        return values
    return selected
'''
    source = insert_before_function(
        source,
        "retrieve_assistant_v52",
        applicability_helpers,
        "v6 pre-expansion applicability gate",
    )

    old_block = '''    reranked = _synthesis_rerank(interpretation, plan, candidates, routes)
    expanded = _expand_sections(db, interpretation, reranked)
    combined = _merge_results([*reranked, *expanded])
    completeness = completeness_policy(original_question, interpretation)
    if completeness == "cross_scope_procedure":
        combined = expand_cross_scope_procedure_evidence(
            db,
            question=original_question,
            interpretation=interpretation,
            results=combined,
            limit=_int_env("RAG_V55_PROCEDURE_EVIDENCE", 180, 64, 280),
        )
'''
    new_block = '''    reranked = _synthesis_rerank(interpretation, plan, candidates, routes)
    completeness = completeness_policy(original_question, interpretation)

    # v6: applicability is checked BEFORE structural expansion. The full reranked set
    # remains available as evidence, but only scenario-compatible seeds are allowed to
    # pull an entire table/section into context.
    structure_seeds = list(reranked)
    if completeness == "cross_scope_procedure":
        structure_seeds = _v6_applicable_structure_seeds(
            original_question,
            interpretation,
            reranked,
        )

    expanded = _expand_sections(db, interpretation, structure_seeds)
    combined = _merge_results([*reranked, *expanded])
    if completeness == "cross_scope_procedure":
        procedure_base = _merge_results([*structure_seeds, *expanded])
        procedure_expanded = expand_cross_scope_procedure_evidence(
            db,
            question=original_question,
            interpretation=interpretation,
            results=procedure_base,
            limit=_int_env("RAG_V55_PROCEDURE_EVIDENCE", 180, 64, 280),
        )
        combined = _merge_results([*reranked, *procedure_expanded])
'''
    source = replace_once(
        source,
        old_block,
        new_block,
        "v6 applicability-before-expansion pipeline",
    )
    return f"# {MARKER}\n" + source


def patch_service(source: str) -> str:
    if MARKER in source:
        return source
    if V55_MARKER not in source:
        raise RuntimeError(
            "backend/app/rag/v5/service.py is not the v5.5 procedure-integrity runtime. Apply v5.5 first."
        )

    ask_impl = r'''    def _ask_impl(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        rewrite_question: bool | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> AnswerResponse:
        """v6 pilot: retrieve broadly, compile an evidence workspace, then draft.

        This intentionally bypasses the accumulated v5.3-v5.5 required-scope answer
        repair chain. Retrieval diagnostics remain retrieval concerns; the final writer
        sees only a question frame, evidence claims, ambiguities/conflicts and source text.
        """
        from app.rag.v5.synthesis_retrieval import retrieve_assistant_v52
        from app.rag.v6.evidence_workspace import (
            build_query_frame,
            compile_evidence_workspace,
            premium_answer_prompt,
            premium_answer_system,
            premium_repair_prompt,
            premium_repair_system,
            select_workspace_sources,
            verification_requires_repair,
            verify_workspace_answer,
            workspace_evidence_limit,
            workspace_max_rounds,
            workspace_primary_documents,
            workspace_retry_queries,
            workspace_summary,
        )

        settings = get_settings()
        emit_progress(
            "interpret",
            "Understanding your question",
            "Resolving intent, scope, conditions and the information needed for a complete answer",
        )
        try:
            from app.rag.v5.retrieval_completeness import complete_terminology_hints
            grounded_terminology = complete_terminology_hints(db, question)
        except Exception:
            grounded_terminology = terminology_hints(db, question)

        interpretation = interpret_user_message(
            question,
            history=conversation_context or [],
            abbreviation_hints=grounded_terminology,
            routing_hints=[],
        )
        resolved = interpretation.resolved_question or question
        query_frame = build_query_frame(question, interpretation)
        emit_progress(
            "interpret",
            "Question understood",
            f"Intent: {interpretation.intent}; requested evidence dimensions: {len(getattr(interpretation, 'evidence_needs', ())) }",
        )

        max_rounds = workspace_max_rounds()
        search_round = 1
        prior_results: list[RetrievedChunk] = []
        extra_queries: list[str] = []
        all_search_queries: list[str] = []
        candidate_chunks = 0
        bundle = None
        prompt_sources: list[PromptSource] = []
        workspace: dict[str, object] = {}

        while search_round <= max_rounds:
            emit_progress(
                f"search_{search_round}",
                "Searching the knowledge base" if search_round == 1 else "Filling evidence gaps",
                (
                    "Finding governing evidence across relevant documents"
                    if search_round == 1
                    else "Running targeted searches based on what is still missing from the evidence map"
                ),
            )
            bundle = retrieve_assistant_v52(
                db,
                interpretation,
                original_question=question,
                extra_queries=extra_queries,
                prior_results=prior_results,
                activity_round=search_round,
            )
            candidate_chunks = max(candidate_chunks, int(getattr(bundle, "candidate_count", 0) or 0))
            all_search_queries = list(dict.fromkeys([
                *all_search_queries,
                *[str(value) for value in getattr(bundle, "search_queries", []) if str(value).strip()],
                *extra_queries,
            ]))

            if not bundle.results:
                break

            evidence_limit = workspace_evidence_limit(top_k, query_frame)
            prompt_sources = select_workspace_sources(bundle.results, limit=evidence_limit)
            emit_progress(
                f"evidence_compile_{search_round}",
                "Organizing the evidence",
                f"Checking applicability and reconstructing a clean knowledge map from {len(prompt_sources)} evidence unit(s)",
            )
            workspace = compile_evidence_workspace(
                question=question,
                interpretation=interpretation,
                query_frame=query_frame,
                sources=prompt_sources,
                search_round=search_round,
            )
            logger.info("RAG v6 evidence workspace round %d: %s", search_round, workspace_summary(workspace))

            retry_queries = workspace_retry_queries(workspace)
            if not retry_queries or search_round >= max_rounds:
                break

            prior_results = list(bundle.results)
            extra_queries = retry_queries
            search_round += 1

        if bundle is None or not getattr(bundle, "results", None):
            plan = _build_plan(question, interpretation, all_search_queries)
            return AnswerResponse(
                answer=NO_ANSWER,
                sources=[],
                evidence=[],
                formatted_sources="",
                formatted_evidence="",
                grounded=False,
                grounding_status="insufficient_evidence",
                interpreted_question=resolved,
                contextual_question=resolved,
                retrieval_mode=plan.search_mode,
                resolved_abbreviations=grounded_terminology,
                candidate_chunks=candidate_chunks,
                evidence_chunks=0,
                search_queries=all_search_queries,
                answer_policy_version="rag-v6-evidence-workspace-pilot",
            )

        if not prompt_sources:
            evidence_limit = workspace_evidence_limit(top_k, query_frame)
            prompt_sources = select_workspace_sources(bundle.results, limit=evidence_limit)
            workspace = compile_evidence_workspace(
                question=question,
                interpretation=interpretation,
                query_frame=query_frame,
                sources=prompt_sources,
                search_round=search_round,
            )

        emit_progress(
            "answer_generation",
            "Drafting your answer",
            "Writing from the organized evidence rather than from raw retrieval results",
        )
        draft = llm_service.generate(
            premium_answer_system(),
            premium_answer_prompt(
                question=question,
                workspace=workspace,
                sources=prompt_sources,
            ),
            max_output_tokens=settings.max_output_tokens,
            model=settings.llm_model,
            reasoning_effort=settings.llm_reasoning_effort,
        )

        emit_progress(
            "answer_verification",
            "Checking the draft",
            "Verifying facts, conditions, units, applicability, completeness and citations",
        )
        verification = verify_workspace_answer(
            question=question,
            answer=draft,
            workspace=workspace,
            sources=prompt_sources,
        )
        candidate_answer = draft
        if verification_requires_repair(verification):
            emit_progress(
                "answer_precision_repair",
                "Refining the draft",
                "Correcting only the specific support, condition, citation or formatting issues found by verification",
            )
            candidate_answer = llm_service.generate(
                premium_repair_system(),
                premium_repair_prompt(
                    question=question,
                    draft=draft,
                    verification=verification,
                    workspace=workspace,
                    sources=prompt_sources,
                ),
                max_output_tokens=settings.max_output_tokens,
                model=settings.query_model,
                reasoning_effort=settings.query_reasoning_effort,
            )

        answer, grounded = validate_grounded_answer(candidate_answer, len(prompt_sources))
        grounding_status = "v6_workspace_verified" if grounded else "citation_validation_failed"
        if answer != NO_ANSWER and not grounded:
            emit_progress(
                "citation_repair",
                "Repairing citations",
                "Fixing citation labels without changing the supported answer",
            )
            repaired = llm_service.generate(
                _V5_CITATION_REPAIR_SYSTEM,
                _repair_prompt(answer, prompt_sources),
                max_output_tokens=settings.max_output_tokens,
                model=settings.query_model,
                reasoning_effort=settings.query_reasoning_effort,
            )
            answer, grounded = validate_grounded_answer(repaired, len(prompt_sources))
            grounding_status = "v6_workspace_verified_after_citation_repair" if grounded else "citation_validation_failed"

        used = set(cited_source_numbers(answer, len(prompt_sources)))
        plan = _build_plan(question, interpretation, all_search_queries)
        primary_documents = workspace_primary_documents(prompt_sources, workspace)
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
            primary_documents=primary_documents,
            candidate_chunks=candidate_chunks,
            evidence_chunks=len(prompt_sources),
            search_queries=all_search_queries,
            answer_policy_version="rag-v6-evidence-workspace-pilot",
        )
'''
    source = replace_class_method_to_eof(source, "_ask_impl", ask_impl)
    return f"# {MARKER}\n" + source


def prepare(repo: Path, package: Path) -> tuple[dict[Path, str], list[tuple[Path, Path]]]:
    service = repo / "backend/app/rag/v5/service.py"
    synthesis = repo / "backend/app/rag/v5/synthesis_retrieval.py"
    assistant = repo / "backend/app/rag/v5/assistant_retrieval.py"
    for path in (service, synthesis, assistant):
        if not path.exists():
            raise SystemExit(f"Required runtime file missing: {path}")

    assistant_text = assistant.read_text(encoding="utf-8-sig")
    if V551_MARKER not in assistant_text and "def _physical_uuid_chunk_ids(" not in assistant_text:
        raise RuntimeError(
            "v5.5.1 virtual chunk-id guard is not present in assistant_retrieval.py. "
            "Apply the v5.5.1 hotfix before v6 so synthetic procedure evidence cannot be cast to uuid[]."
        )

    transforms: dict[Path, object] = {
        service: patch_service,
        synthesis: patch_synthesis_retrieval,
    }

    # Optional payload mirrors are transformed only when they are already v5.5-compatible.
    optional = [
        (repo / "payload/backend/app/rag/v5/service.py", patch_service),
        (repo / "payload/backend/app/rag/v5/synthesis_retrieval.py", patch_synthesis_retrieval),
    ]
    for path, fn in optional:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8-sig")
        if V55_MARKER in current:
            transforms[path] = fn

    transformed: dict[Path, str] = {}
    for path, fn in transforms.items():
        transformed[path] = fn(path.read_text(encoding="utf-8-sig"))  # type: ignore[operator]

    copies = [
        (package / "backend/app/rag/v6/__init__.py", repo / "backend/app/rag/v6/__init__.py"),
        (package / "backend/app/rag/v6/evidence_workspace.py", repo / "backend/app/rag/v6/evidence_workspace.py"),
        (package / "backend/tests/test_v6_evidence_workspace.py", repo / "backend/tests/test_v6_evidence_workspace.py"),
    ]
    return transformed, copies


def validate(transformed: dict[Path, str], copies: list[tuple[Path, Path]]) -> None:
    for path, content in transformed.items():
        if path.suffix == ".py":
            compile(content, str(path), "exec")
    for src, dst in copies:
        if dst.suffix == ".py":
            compile(src.read_text(encoding="utf-8"), str(dst), "exec")


def write(
    transformed: dict[Path, str],
    copies: list[tuple[Path, Path]],
    repo: Path,
) -> None:
    for path, content in transformed.items():
        current = path.read_text(encoding="utf-8-sig")
        if current == content:
            print(f"[already compatible] {path.relative_to(repo)}")
            continue
        backup(path)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"[patched] {path.relative_to(repo)}")

    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            print(f"[already copied] {dst.relative_to(repo)}")
            continue
        if dst.exists():
            backup(dst)
        shutil.copy2(src, dst)
        print(f"[copied] {dst.relative_to(repo)}")

    snapshot = repo / "pdfrag-v6-replacement-files"
    for path in transformed:
        relative = path.relative_to(repo)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for _src, dst in copies:
        relative = dst.relative_to(repo)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, target)
    manifest = {
        "version": "rag-v6-evidence-workspace-pilot",
        "baseline": "v5.5 procedure-integrity + v5.5.1 virtual chunk-id guard",
        "replacement_files": [
            str(path.relative_to(repo)).replace("\\", "/") for path in transformed
        ] + [str(dst.relative_to(repo)).replace("\\", "/") for _src, dst in copies],
        "notes": [
            "No database migration is required.",
            "No embedding rebuild or PDF reprocessing is required for the first test.",
            "Applicability is checked before procedure structure expansion.",
            "The final writer receives an evidence workspace, not route/required-scope diagnostics.",
            "One workspace-guided targeted retrieval retry is enabled by default for unresolved facets.",
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[snapshot] {snapshot.relative_to(repo)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply IMS RAG v6 evidence-workspace/premium-drafting pilot over v5.5.1."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    parser.add_argument(
        "--check",
        action="store_true",
        help="preflight transform/compile without writing repository files",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    package = Path(__file__).resolve().parent
    transformed, copies = prepare(repo, package)
    validate(transformed, copies)
    print("[preflight] v6 transformed/copied Python compiles in memory")
    if args.check:
        print("[check only] no repository files were changed")
        return 0

    write(transformed, copies, repo)
    for path in [
        repo / "backend/app/rag/v5/service.py",
        repo / "backend/app/rag/v5/synthesis_retrieval.py",
        repo / "backend/app/rag/v6/evidence_workspace.py",
        repo / "backend/tests/test_v6_evidence_workspace.py",
    ]:
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied IMS RAG v6 evidence-workspace pilot.")
    print(" - query results are compiled into evidence-level claims before writing")
    print(" - scenario/applicability is checked before whole table/section expansion")
    print(" - noisy routed documents can be rejected without becoming answer headings")
    print(" - missing facets can trigger one targeted semantic retrieval retry by default")
    print(" - the writer sees clean claims + source blocks, not retrieval diagnostics")
    print(" - a verifier checks support, condition/action pairing, units, applicability and formatting")
    print(" - final answers use premium answer-first Markdown formatting")
    print()
    print("No DB migration, embedding rebuild, OCR rerun or full PDF reprocessing is required for first testing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
