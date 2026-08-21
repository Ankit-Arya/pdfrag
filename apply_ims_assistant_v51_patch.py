from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_ASSISTANT_V51_PATCH"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + ".bak-before-ims-assistant-v51")
    if not target.exists():
        shutil.copy2(path, target)


def patch_service(source: str) -> str:
    if MARKER in source:
        return source

    source = replace_once(
        source,
        "from app.rag.v5.ingestion import process_document_v5\n"
        "from app.rag.v5.retrieval import V5RetrievalBundle, retrieve_v5\n"
        "from app.rag.v5.terminology import terminology_hints\n",
        "from app.rag.v5.ingestion import process_document_v5\n"
        "from app.rag.v5.assistant_retrieval import (\n"
        "    AssistantRetrievalBundle,\n"
        "    assistant_terminology_hints,\n"
        "    retrieve_assistant_v51,\n"
        ")\n",
        "service imports",
    )

    old_sources = '''def _document_diverse_sources(results: list[RetrievedChunk], limit: int) -> list[PromptSource]:
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
'''
    new_sources = '''def _assistant_sources(
    results: list[RetrievedChunk],
    limit: int,
    interpretation: object,
) -> list[PromptSource]:
    """Keep document diversity without truncating a governing multi-section answer."""
    output: list[PromptSource] = []
    seen: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    intent = str(getattr(interpretation, "intent", "fact_lookup"))
    conversation_act = str(getattr(interpretation, "conversation_act", "question"))
    section_heavy = intent in {"list", "procedure", "summary", "requirement", "troubleshooting", "comparison"} or conversation_act == "navigation"
    per_document_cap = 24 if section_heavy else 16
    diversity_gate = max(8, (limit * 3) // 4) if section_heavy else max(6, limit // 2)
    for item in results:
        if item.chunk.chunk_id in seen:
            continue
        doc = item.chunk.document_id or item.chunk.filename
        if counts[doc] >= per_document_cap and len(output) >= diversity_gate:
            continue
        seen.add(item.chunk.chunk_id)
        counts[doc] += 1
        output.append(PromptSource(result=item, excerpt=item.chunk.text.strip()))
        if len(output) >= limit:
            break
    return output
'''
    source = replace_once(source, old_sources, new_sources, "assistant source selection")

    source = replace_once(
        source,
        "        grounded_terminology = terminology_hints(db, question)\n",
        "        # IMS_ASSISTANT_V51_PATCH\n"
        "        grounded_terminology = assistant_terminology_hints(db, question)\n",
        "grounded terminology",
    )

    old_retrieval = '''        emit_progress("search", "Searching structured PDF evidence", "Searching sections, prose, table rows and current authority")
        bundle: V5RetrievalBundle = retrieve_v5(db, interpretation)
        review = review_retrieved_evidence(interpretation, bundle.results)
        coverage_status = "sufficient" if review.sufficient else "insufficient_after_review"
        if not review.sufficient and review.retry_queries:
            emit_progress("search_retry", "Filling evidence gaps", "Running one targeted retrieval retry")
            bundle = retrieve_v5(db, interpretation, extra_queries=review.retry_queries)
            final_review = review_retrieved_evidence(interpretation, bundle.results)
            coverage_status = "sufficient_after_retry" if final_review.sufficient else "insufficient_after_retry"
'''
    new_retrieval = '''        emit_progress(
            "search",
            "Navigating documents and sections",
            "Routing across the corpus, matching headings and ranking governing evidence",
        )
        max_rounds = _int_env("RAG_V51_MAX_SEARCH_ROUNDS", 3, 1, 3)
        bundle: AssistantRetrievalBundle = retrieve_assistant_v51(
            db,
            interpretation,
            original_question=question,
        )
        review = review_retrieved_evidence(interpretation, bundle.results)
        coverage_status = "sufficient" if review.sufficient else "insufficient_after_review"

        # Search until the evidence critic is satisfied, within a strict bounded budget.
        # Each retry accumulates previous good evidence instead of replacing it.
        for search_round in range(2, max_rounds + 1):
            if review.sufficient:
                break
            retry_queries = list(review.retry_queries) or list(review.missing_evidence)
            if not retry_queries:
                break
            emit_progress(
                "search_retry",
                "Filling evidence gaps",
                f"Targeted evidence search {search_round}/{max_rounds}",
            )
            bundle = retrieve_assistant_v51(
                db,
                interpretation,
                original_question=question,
                extra_queries=retry_queries,
                prior_results=bundle.results,
            )
            review = review_retrieved_evidence(interpretation, bundle.results)
            coverage_status = (
                f"sufficient_after_retry_{search_round}"
                if review.sufficient
                else f"insufficient_after_retry_{search_round}"
            )
'''
    source = replace_once(source, old_retrieval, new_retrieval, "assistant retrieval loop")

    source = replace_once(
        source,
        "        prompt_sources = _document_diverse_sources(bundle.results, evidence_limit)\n",
        "        prompt_sources = _assistant_sources(bundle.results, evidence_limit, interpretation)\n",
        "assistant evidence selection",
    )

    source = source.replace('answer_policy_version="rag-v5.0.0"', 'answer_policy_version="rag-v5.1-assistant"')
    source = replace_once(
        source,
        "            primary_documents=[],\n",
        "            primary_documents=bundle.routed_documents[:8],\n",
        "routed document diagnostics",
    )
    return source


def patch_understanding(source: str) -> str:
    if "IMS_ASSISTANT_V51_SEARCH_GUIDANCE" in source:
        return source
    anchor = "Return JSON only with exactly these keys:\n"
    guidance = '''SEARCH QUALITY (IMS_ASSISTANT_V51_SEARCH_GUIDANCE):
- Assume the user may not know which document contains the answer. Do not invent or force a document when none is stated.
- If the user names or hints at a document/family, preserve that hint as a routing preference, but still express the underlying semantic question so downstream search can recover when the hint is incomplete or mistaken.
- Correct likely typos and generate meaning-preserving variants even when the PDF uses different wording.
- Include formal policy/manual heading formulations that could express the same intent (for example a user's everyday wording may correspond to a heading phrased as responsibilities, requirements, prohibition, procedure, provision, definition, precautions, failures, duties or conditions). These are linguistic variants only, never factual claims.
- Search queries should diversify across: corrected natural wording, likely formal heading wording, actor/object wording, and the requested output dimension (who/what/when/how much/page/rule/steps/etc.).
- A request asking which page, rule, section, chapter or location contains a subject is a navigation request even when it also asks for a short explanation.
- Prefer semantic equivalence over exact token copying, but preserve technical scope words and do not invent internal acronym expansions.

'''
    return replace_once(source, anchor, guidance + anchor, "smart understanding guidance")


def patch_compose(source: str) -> str:
    if "RAG_V51_AI_RERANK_ENABLED" in source:
        return source
    anchor = "      RAG_V5_LEGACY_CHUNK_MIRROR: ${RAG_V5_LEGACY_CHUNK_MIRROR:-1}\n"
    addition = anchor + '''      # Assistant-style query-time retrieval. No document reprocessing is required.
      RAG_V51_ASSISTANT_ENABLED: ${RAG_V51_ASSISTANT_ENABLED:-1}
      RAG_V51_AI_RERANK_ENABLED: ${RAG_V51_AI_RERANK_ENABLED:-1}
      RAG_V51_MAX_SEARCH_ROUNDS: ${RAG_V51_MAX_SEARCH_ROUNDS:-3}
      RAG_V51_ROUTE_DOCUMENTS: ${RAG_V51_ROUTE_DOCUMENTS:-8}
      RAG_V51_ROUTE_PER_QUERY: ${RAG_V51_ROUTE_PER_QUERY:-80}
      RAG_V51_SCOPED_PER_ARM: ${RAG_V51_SCOPED_PER_ARM:-56}
      RAG_V51_RERANK_CANDIDATES: ${RAG_V51_RERANK_CANDIDATES:-48}
      RAG_V51_FINAL_CANDIDATES: ${RAG_V51_FINAL_CANDIDATES:-64}
      RAG_V51_SECTION_EXPANSION_ENABLED: ${RAG_V51_SECTION_EXPANSION_ENABLED:-1}
      RAG_V51_SECTION_SEEDS: ${RAG_V51_SECTION_SEEDS:-10}
      RAG_V51_MAX_SECTION_CHUNKS: ${RAG_V51_MAX_SECTION_CHUNKS:-12}
'''
    return replace_once(source, anchor, addition, "compose v5.1 environment")


def patch_env_merge(source: str) -> str:
    if '"RAG_V51_AI_RERANK_ENABLED"' in source:
        return source
    anchor = '    "RAG_V5_LEGACY_CHUNK_MIRROR" = "1"\n'
    addition = anchor + '''    "RAG_V51_ASSISTANT_ENABLED" = "1"
    "RAG_V51_AI_RERANK_ENABLED" = "1"
    "RAG_V51_MAX_SEARCH_ROUNDS" = "3"
    "RAG_V51_ROUTE_DOCUMENTS" = "8"
    "RAG_V51_ROUTE_PER_QUERY" = "80"
    "RAG_V51_SCOPED_PER_ARM" = "56"
    "RAG_V51_RERANK_CANDIDATES" = "48"
    "RAG_V51_FINAL_CANDIDATES" = "64"
    "RAG_V51_SECTION_EXPANSION_ENABLED" = "1"
    "RAG_V51_SECTION_SEEDS" = "10"
    "RAG_V51_MAX_SECTION_CHUNKS" = "12"
'''
    return replace_once(source, anchor, addition, "env merge v5.1 settings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install IMS assistant-style RAG v5.1 retrieval over existing v5 generations.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    payload = Path(__file__).resolve().parent / "payload"
    service = repo / "backend/app/rag/v5/service.py"
    understanding = repo / "backend/app/rag/smart_understanding.py"
    compose = repo / "docker-compose.v5.yml"
    env_merge = repo / "merge-v5-env.ps1"
    assistant_target = repo / "backend/app/rag/v5/assistant_retrieval.py"
    debug_target = repo / "backend/app/rag/v5/assistant_debug.py"
    test_target = repo / "backend/tests/test_v5_assistant_retrieval.py"

    required = [service, understanding, compose, env_merge]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Required repository file(s) not found:\n" + "\n".join(missing))

    payload_files = {
        assistant_target: payload / "backend/app/rag/v5/assistant_retrieval.py",
        debug_target: payload / "backend/app/rag/v5/assistant_debug.py",
        test_target: payload / "backend/tests/test_v5_assistant_retrieval.py",
    }
    for source in payload_files.values():
        if not source.exists():
            raise SystemExit(f"Patch payload missing: {source}")

    for path in required:
        backup(path)

    # Copy additive files first; existing non-patch files are backed up rather than silently overwritten.
    for target, source in payload_files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and MARKER not in target.read_text(encoding="utf-8", errors="ignore"):
            backup(target)
        text_value = source.read_text(encoding="utf-8")
        if target.name == "assistant_retrieval.py" and MARKER not in text_value:
            text_value = f"# {MARKER}\n" + text_value
        elif target.name == "assistant_debug.py" and MARKER not in text_value:
            text_value = f"# {MARKER}\n" + text_value
        target.write_text(text_value, encoding="utf-8", newline="\n")

    service.write_text(patch_service(service.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    understanding.write_text(patch_understanding(understanding.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    compose.write_text(patch_compose(compose.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    env_merge.write_text(patch_env_merge(env_merge.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")

    for path in (assistant_target, debug_target, service, understanding, test_target):
        py_compile.compile(str(path), doraise=True)

    print("Applied IMS assistant-style RAG v5.1 patch.")
    print("Changed:")
    print(" - backend/app/rag/v5/service.py")
    print(" - backend/app/rag/smart_understanding.py")
    print(" - docker-compose.v5.yml")
    print(" - merge-v5-env.ps1")
    print("Added:")
    print(" - backend/app/rag/v5/assistant_retrieval.py")
    print(" - backend/app/rag/v5/assistant_debug.py")
    print(" - backend/tests/test_v5_assistant_retrieval.py")
    print("No database migration and no PDF reprocessing are required.")
    print("Processing version remains rag-v5.0.0; answer policy becomes rag-v5.1-assistant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
