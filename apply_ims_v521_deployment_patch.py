from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_V521_DEPLOYMENT_COMPLETENESS_DIAGNOSTICS"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def replace_between(source: str, start: str, end: str, replacement: str, label: str) -> str:
    start_index = source.find(start)
    if start_index < 0:
        raise RuntimeError(f"{label}: start anchor not found")
    end_index = source.find(end, start_index + len(start))
    if end_index < 0:
        raise RuntimeError(f"{label}: end anchor not found")
    return source[:start_index] + replacement + source[end_index:]


def class_block(source: str, class_name: str) -> tuple[int, int, str]:
    start = source.find(f"class {class_name}(")
    if start < 0:
        start = source.find(f"class {class_name}:")
    if start < 0:
        raise RuntimeError(f"{class_name} class not found")
    end = source.find("\n\nclass ", start + 1)
    if end < 0:
        end = len(source)
    return start, end, source[start:end]


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + ".bak-before-ims-v521-deployment")
    if not target.exists():
        shutil.copy2(path, target)


def patch_models(source: str) -> str:
    if "retrieval_diagnostics: list[dict[str, Any]]" in source:
        return source
    start, end, block = class_block(source, "AnswerResponse")
    anchor = "    conflicts: list[dict[str, Any]] = Field(default_factory=list)\n"
    if anchor not in block:
        raise RuntimeError("AnswerResponse v5.2 synthesis fields not found. Apply v5.2 before v5.2.1.")
    addition = (
        "    retrieval_diagnostics: list[dict[str, Any]] = Field(default_factory=list)\n"
        "    retrieval_diagnostic_summary: dict[str, Any] = Field(default_factory=dict)\n"
    )
    block = block.replace(anchor, anchor + addition, 1)
    return source[:start] + block + source[end:]


def patch_api(source: str) -> str:
    source = source.replace(
        "    response.answer_policy_version = ANSWER_POLICY_VERSION\n",
        "    response.answer_policy_version = response.answer_policy_version or ANSWER_POLICY_VERSION\n",
        1,
    )
    if '"retrieval_diagnostics": response.retrieval_diagnostics' not in source:
        start = source.find("        message_metadata={")
        end = source.find("\n        },\n    )\n    db.add(assistant_message)", start)
        if start < 0 or end < 0:
            raise RuntimeError("assistant message_metadata block not found")
        block = source[start:end]
        anchor = '            "conflicts": response.conflicts,\n'
        if anchor not in block:
            raise RuntimeError("v5.2 conflicts metadata field not found in assistant message metadata")
        block = block.replace(
            anchor,
            anchor
            + '            "retrieval_diagnostics": response.retrieval_diagnostics,\n'
            + '            "retrieval_diagnostic_summary": response.retrieval_diagnostic_summary,\n',
            1,
        )
        source = source[:start] + block + source[end:]

    if '"retrieval_diagnostic_summary": response.retrieval_diagnostic_summary' not in source[source.find("add_audit_event("):]:
        audit_start = source.find("    add_audit_event(\n", source.find("def _finalize_chat_exchange"))
        audit_end = source.find("\n    )\n    chat_session.updated_at", audit_start)
        if audit_start >= 0 and audit_end >= 0:
            block = source[audit_start:audit_end]
            anchor = '            "answer_policy_version": response.answer_policy_version,\n'
            if anchor in block:
                block = block.replace(
                    anchor,
                    anchor + '            "retrieval_diagnostic_summary": response.retrieval_diagnostic_summary,\n',
                    1,
                )
                source = source[:audit_start] + block + source[audit_end:]
    return source


def patch_smart_index(source: str) -> str:
    if "IMS_V521_CASE_TOLERANT_ALIAS" in source:
        return source
    old1 = r'''_LONG_THEN_ABBR_RE = re.compile(rf"(?P<long>{_LONG_FORM})\s*\(\s*(?P<abbr>[A-Z][A-Z0-9]{{1,9}})\s*\)")'''
    new1 = r'''# IMS_V521_CASE_TOLERANT_ALIAS: tolerate table/OCR case degradation; initials still validate the definition.
_LONG_THEN_ABBR_RE = re.compile(rf"(?P<long>{_LONG_FORM})\s*\(\s*(?P<abbr>[A-Za-z][A-Za-z0-9]{{1,9}})\s*\)")'''
    source = replace_once(source, old1, new1, "smart-index long-form(alias) pattern")
    old2 = r'''_ABBR_THEN_LONG_PAREN_RE = re.compile(rf"\b(?P<abbr>[A-Z][A-Z0-9]{{1,9}})\s*\(\s*(?P<long>{_LONG_FORM})\s*\)")'''
    new2 = r'''_ABBR_THEN_LONG_PAREN_RE = re.compile(rf"\b(?P<abbr>[A-Za-z][A-Za-z0-9]{{1,9}})\s*\(\s*(?P<long>{_LONG_FORM})\s*\)")'''
    return replace_once(source, old2, new2, "smart-index alias(long-form) pattern")


def patch_ingestion(source: str) -> str:
    if "IMS_V521_TABLE_ALIAS_CASE" in source:
        return source
    old = '''            if not re.fullmatch(r"[A-Z][A-Z0-9/-]{1,9}", alias):
                continue
            if len(canonical) < 4 or not _terminology_initials_match(alias, canonical):
                continue
'''
    new = '''            # IMS_V521_TABLE_ALIAS_CASE: table extraction can turn BIC into Bic.
            # Accept the case variant only when the canonical initials prove the alias.
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9/-]{1,9}", alias):
                continue
            alias = alias.upper()
            if len(canonical) < 4 or not _terminology_initials_match(alias, canonical):
                continue
'''
    return replace_once(source, old, new, "v5 table-cell terminology alias")

def patch_service(source: str) -> str:
    if "from app.rag.v5.retrieval_completeness import complete_terminology_hints" not in source:
        anchor = "from app.rag.v5.ingestion import process_document_v5\n"
        source = replace_once(
            source,
            anchor,
            anchor + "from app.rag.v5.retrieval_completeness import complete_terminology_hints\n",
            "v5 service completeness import",
        )

    variants = [
        "        grounded_terminology = assistant_terminology_hints(db, question)\n",
        "        grounded_terminology = terminology_hints(db, question)\n",
    ]
    if "grounded_terminology = complete_terminology_hints(db, question)" not in source:
        for old in variants:
            if old in source:
                source = source.replace(old, "        grounded_terminology = complete_terminology_hints(db, question)\n", 1)
                break
        else:
            raise RuntimeError("grounded terminology assignment not found in v5 service")

    multi_alias_rule = (
        "- For an unscoped acronym/full-form/meaning question, if sources establish multiple distinct expansions,\n"
        "  list every materially supported meaning and identify its equipment/document context; never collapse\n"
        "  one alias to a single global canonical meaning.\n"
    )
    if multi_alias_rule not in source:
        anchor = (
            "- For acronym/full-form questions, preserve the exact expansion stated in the cited PDF source; do not\n"
            "  paraphrase an official expansion into a plausible synonym.\n"
        )
        source = replace_once(source, anchor, anchor + multi_alias_rule, "multi-meaning acronym answer rule")

    source = source.replace("rag-v5.2-synthesis", "rag-v5.2.1-completeness")
    return source


def patch_synthesis(source: str) -> str:
    if "from dataclasses import dataclass, field" not in source:
        source = replace_once(source, "from dataclasses import dataclass\n", "from dataclasses import dataclass, field\n", "synthesis dataclass import")

    if "from app.rag.v5.retrieval_completeness import (" not in source:
        anchor = "from app.rag.v5.retrieval import retrieve_v5\n"
        addition = '''from app.rag.v5.retrieval_completeness import (
    direct_retrieval_diagnostics,
    exact_alias_results,
    finalize_retrieval_diagnostics,
    response_diagnostics,
    unified_corpus_discovery,
)
'''
        source = replace_once(source, anchor, anchor + addition, "synthesis completeness imports")

    if "retrieval_diagnostics: list[dict[str, object]]" not in source:
        start, end, block = class_block(source, "SynthesisRetrievalBundle")
        anchor = "    search_round: int = 1\n"
        if anchor not in block:
            raise RuntimeError("SynthesisRetrievalBundle search_round field not found")
        block = block.replace(
            anchor,
            anchor
            + "    retrieval_diagnostics: list[dict[str, object]] = field(default_factory=list)\n"
            + "    retrieval_diagnostic_summary: dict[str, object] = field(default_factory=dict)\n",
            1,
        )
        source = source[:start] + block + source[end:]

    scenario_rule = (
        "- For a condition-to-action/value question, retain a candidate only when it connects the requested\n"
        "  condition/scenario to the requested action, value, responsibility or consequence. A generic document\n"
        "  mentioning the same speed/value/actor without that causal link is incidental.\n"
    )
    if scenario_rule not in source:
        anchor = "- Prefer governing/defining provisions over incidental mentions.\n"
        source = replace_once(source, anchor, anchor + scenario_rule, "synthesis scenario-binding rule")

    direct_start = "    # Direct lookups keep the complete Assistant v5.1 path. Synthesis queries intentionally\n"
    synthesis_start = "    emit_progress(\n        f\"synthesis_baseline_{activity_round}\",\n"
    if "exhaustive source-grounded alias lookup first" not in source:
        replacement = '''    # Direct lookups stay focused. Exact acronym/full-form requests are handled by an
    # exhaustive source-grounded alias lookup first, avoiding unnecessary broad semantic corpus passes.
    if plan.answer_strategy != "multi_document_synthesis":
        exact_results = exact_alias_results(db, original_question, limit=80)
        if interpretation.intent == "definition" and exact_results:
            exact_docs = _unique((item.chunk.filename for item in exact_results), 40)
            diagnostics, diagnostic_summary = direct_retrieval_diagnostics(
                db,
                results=exact_results,
                routed_documents=exact_docs,
            )
            return SynthesisRetrievalBundle(
                results=list(exact_results),
                search_queries=list(queries),
                candidate_count=len(exact_results),
                routed_documents=exact_docs,
                answer_strategy=plan.answer_strategy,
                synthesis_dimensions=list(plan.dimensions),
                considered_documents=exact_docs,
                evidence_documents=exact_docs,
                search_round=activity_round,
                retrieval_diagnostics=diagnostics,
                retrieval_diagnostic_summary=diagnostic_summary,
            )

        baseline = _call_v51(
            db,
            interpretation,
            original_question=original_question,
            extra_queries=extra_queries,
            prior_results=prior_results,
            activity_round=activity_round,
        )
        direct_results = _merge_results([*exact_results, *baseline.results]) if exact_results else list(baseline.results)
        evidence_docs = _unique((item.chunk.filename for item in direct_results), 20)
        diagnostics, diagnostic_summary = direct_retrieval_diagnostics(
            db,
            results=direct_results,
            routed_documents=list(baseline.routed_documents),
        )
        return SynthesisRetrievalBundle(
            results=direct_results,
            search_queries=_unique([*queries, *baseline.search_queries], 16),
            candidate_count=baseline.candidate_count + len(exact_results),
            routed_documents=list(baseline.routed_documents),
            answer_strategy=plan.answer_strategy,
            synthesis_dimensions=list(plan.dimensions),
            considered_documents=list(baseline.routed_documents),
            evidence_documents=evidence_docs,
            search_round=activity_round,
            retrieval_diagnostics=diagnostics,
            retrieval_diagnostic_summary=diagnostic_summary,
        )

'''
        source = replace_between(source, direct_start, synthesis_start, replacement, "direct-lookup strategy block")

    if "Unified corpus discovery" not in source:
        start = "    emit_progress(\n        f\"synthesis_baseline_{activity_round}\",\n"
        end = "    if not routes:\n"
        replacement = '''    emit_progress(
        f"synthesis_discovery_{activity_round}",
        "Unified corpus discovery",
        f"Running one document-aware corpus stage for {len(queries)} evidence formulation(s)",
        actor="search",
        phase="search",
        status="running",
        operation_id=f"synthesis-discovery-{activity_round}",
        metrics={"query_variants": len(queries), "corpus_discovery_stages": 1},
    )
    discovery = unified_corpus_discovery(
        db,
        queries=queries,
        prior_results=prior_results,
    )
    baseline_results = _merge_results([*prior_results, *discovery.candidates])
    routes = discovery.routes
    emit_progress(
        f"synthesis_discovery_{activity_round}",
        "Unified corpus discovery complete",
        f"{len(routes)} document(s) routed from {discovery.summary.get('documents_with_signal', 0)} document(s) with retrieval signal",
        actor="search",
        phase="route",
        status="complete",
        operation_id=f"synthesis-discovery-{activity_round}",
        metrics={
            "eligible_documents": discovery.summary.get("eligible_documents", 0),
            "documents_with_signal": discovery.summary.get("documents_with_signal", 0),
            "relevant_documents": len(routes),
            "corpus_discovery_stages": 1,
            "query_variants": discovery.summary.get("query_variants", len(queries)),
        },
    )

'''
        source = replace_between(source, start, end, replacement, "unified synthesis discovery block")

    source = source.replace("_unique([*queries, *baseline_v5.search_queries], 16)", "_unique([*queries], 16)")

    no_route_start = source.find("    if not routes:\n")
    scoped_anchor = "    scoped = _scoped_candidates"
    no_route_end = source.find(scoped_anchor, no_route_start)
    if no_route_start < 0 or no_route_end < 0:
        raise RuntimeError("no-route/scoped synthesis boundaries not found")
    no_route_block = source[no_route_start:no_route_end]
    if "retrieval_diagnostics=" not in no_route_block:
        no_route_replacement = '''    if not routes:
        evidence_docs = _unique((item.chunk.filename for item in baseline_results), 20)
        return SynthesisRetrievalBundle(
            results=list(baseline_results),
            search_queries=_unique([*queries], 16),
            candidate_count=discovery.candidate_count,
            routed_documents=evidence_docs,
            answer_strategy=plan.answer_strategy,
            synthesis_dimensions=list(plan.dimensions),
            considered_documents=evidence_docs,
            evidence_documents=evidence_docs,
            search_round=activity_round,
            retrieval_diagnostics=list(discovery.diagnostics),
            retrieval_diagnostic_summary=dict(discovery.summary),
        )

'''
        source = source[:no_route_start] + no_route_replacement + source[no_route_end:]

    evidence_anchor = "    evidence_docs = _unique((item.chunk.filename for item in final), 32)\n"
    if "finalize_retrieval_diagnostics(" not in source:
        source = replace_once(
            source,
            evidence_anchor,
            evidence_anchor
            + "    retrieval_diagnostics = finalize_retrieval_diagnostics(\n"
            + "        discovery.diagnostics, routes=routes, final_results=final\n"
            + "    )\n",
            "final synthesis diagnostics",
        )

    final_return_search_start = source.find("    retrieval_diagnostics = finalize_retrieval_diagnostics(")
    if final_return_search_start < 0:
        raise RuntimeError("final synthesis diagnostics assignment not found")
    final_return_start = source.find("return SynthesisRetrievalBundle(", final_return_search_start)
    if final_return_start >= 4 and source[final_return_start-4:final_return_start] == "    ":
        final_return_start -= 4
    final_return_end = source.find("\n\n\ndef _coverage_prompt", final_return_start)
    if final_return_start < 0 or final_return_end < 0:
        raise RuntimeError("final SynthesisRetrievalBundle return boundaries not found")
    final_return_block = source[final_return_start:final_return_end]
    if "retrieval_diagnostics=retrieval_diagnostics" not in final_return_block:
        final_return_replacement = '''    return SynthesisRetrievalBundle(
        results=final,
        search_queries=_unique([*queries], 16),
        candidate_count=candidate_count,
        routed_documents=[route.filename for route in routes],
        answer_strategy=plan.answer_strategy,
        synthesis_dimensions=list(plan.dimensions),
        considered_documents=[route.filename for route in routes],
        evidence_documents=evidence_docs,
        search_round=activity_round,
        retrieval_diagnostics=retrieval_diagnostics,
        retrieval_diagnostic_summary=dict(discovery.summary),
    )'''
        source = source[:final_return_start] + final_return_replacement + source[final_return_end:]

    if "if not review.sufficient:" not in source[source.find("def select_results_for_answer"):source.find("def coverage_prompt_status")]:
        anchor = '''    if review.answer_strategy != "multi_document_synthesis" or not review.contributing_documents:
        return list(results)
'''
        source = replace_once(
            source,
            anchor,
            anchor + '''    if not review.sufficient:
        # An incomplete coverage critic is not authoritative enough to hard-prune
        # strong evidence from other routed documents.
        return list(results)
''',
            "incomplete coverage non-destructive selection",
        )

    response_start = "def response_synthesis_metadata(\n"
    if "response_diagnostics(" not in source[source.find(response_start):]:
        index = source.find(response_start)
        if index < 0:
            raise RuntimeError("response_synthesis_metadata function not found")
        new_function = '''def response_synthesis_metadata(
    bundle: SynthesisRetrievalBundle,
    review: SynthesisCoverage,
) -> dict[str, object]:
    diagnostics, diagnostic_summary = response_diagnostics(
        bundle.retrieval_diagnostics,
        bundle.retrieval_diagnostic_summary,
        review.contributing_documents,
    )
    return {
        "answer_strategy": bundle.answer_strategy,
        "synthesis_dimensions": list(bundle.synthesis_dimensions),
        "search_scope": "broad_relevant_corpus" if bundle.answer_strategy == "multi_document_synthesis" else "focused",
        "relevant_documents": list(bundle.considered_documents),
        "contributing_documents": list(review.contributing_documents),
        "search_rounds": int(bundle.search_round),
        "evidence_coverage_status": review.evidence_coverage_status,
        "conflicts": [dict(item) for item in review.conflicts],
        "retrieval_diagnostics": diagnostics,
        "retrieval_diagnostic_summary": diagnostic_summary,
    }
'''
        source = source[:index] + new_function

    return source

def patch_api_ts(source: str) -> str:
    if "export interface RetrievalDocumentDiagnostic" not in source:
        anchor = "export interface AnswerResponse {\n"
        addition = '''export interface RetrievalDocumentDiagnostic {
  document_id?: string
  filename: string
  discovery_score: number
  vector_score: number
  keyword_score: number
  dimension_hits: number
  signals: string[]
  routed: boolean
  deep_searched: boolean
  rerank_role?: string
  final_evidence: boolean
  contributing: boolean
  decision: string
  reason: string
  best_page?: number | null
  best_heading?: string
}

export interface RetrievalDiagnosticSummary {
  eligible_documents?: number
  documents_with_signal?: number
  routed_documents?: number
  deep_searched_documents?: number
  final_evidence_documents?: number
  contributing_documents?: number
  no_signal_documents?: number
  corpus_discovery_stages?: number
  query_variants?: number
  diagnostics_truncated?: boolean
  diagnostic_documents_included?: number
}

'''
        source = replace_once(source, anchor, addition + anchor, "frontend retrieval diagnostic types")

    if "retrieval_diagnostics?: RetrievalDocumentDiagnostic[]" not in source:
        anchor = "  conflicts?: ConflictSummary[]\n"
        if anchor not in source:
            raise RuntimeError("v5.2 frontend conflicts field not found in AnswerResponse")
        source = replace_once(
            source,
            anchor,
            anchor
            + "  retrieval_diagnostics?: RetrievalDocumentDiagnostic[]\n"
            + "  retrieval_diagnostic_summary?: RetrievalDiagnosticSummary\n",
            "frontend AnswerResponse diagnostics fields",
        )
    return source


def patch_app(source: str) -> str:
    if "retrieval_diagnostics:" in source:
        return source
    anchor = '''    conflicts: Array.isArray(metadata.conflicts)
      ? (metadata.conflicts as NonNullable<AnswerResponse['conflicts']>)
      : [],
'''
    if anchor not in source:
        raise RuntimeError("v5.2 stored conflicts metadata mapping not found in App.vue")
    addition = '''    retrieval_diagnostics: Array.isArray(metadata.retrieval_diagnostics)
      ? (metadata.retrieval_diagnostics as NonNullable<AnswerResponse['retrieval_diagnostics']>)
      : [],
    retrieval_diagnostic_summary:
      metadata.retrieval_diagnostic_summary && typeof metadata.retrieval_diagnostic_summary === 'object'
        ? (metadata.retrieval_diagnostic_summary as NonNullable<AnswerResponse['retrieval_diagnostic_summary']>)
        : {},
'''
    return replace_once(source, anchor, anchor + addition, "stored retrieval diagnostic metadata mapping")


def patch_inspector(source: str) -> str:
    if "RetrievalDiagnostics" not in source:
        index = source.find("const props = defineProps")
        if index < 0:
            raise RuntimeError("AnswerInspector props block not found")
        source = source[:index] + "import RetrievalDiagnostics from './RetrievalDiagnostics.vue'\n\n" + source[index:]

    source = source.replace(
        "const tab = ref<'sources' | 'plan' | 'details'>('sources')",
        "const tab = ref<'sources' | 'plan' | 'details' | 'diagnostics'>('sources')",
        1,
    )

    if ">Diagnostics</button>" not in source:
        anchor = '<button :class="{ active: tab === \'details\' }" @click="tab = \'details\'">Details</button>\n'
        source = replace_once(
            source,
            anchor,
            anchor + '<button :class="{ active: tab === \'diagnostics\' }" @click="tab = \'diagnostics\'">Diagnostics</button>\n',
            "AnswerInspector diagnostics tab",
        )

    old = '      <template v-else>\n        <section class="inspector-card">\n'
    if 'v-else-if="tab === \'details\'"' not in source:
        new = '      <template v-else-if="tab === \'details\'">\n        <section class="inspector-card">\n'
        source = replace_once(source, old, new, "AnswerInspector details branch")

    if '<RetrievalDiagnostics v-else' not in source:
        anchor = '      </template>\n    </div>\n  </aside>\n'
        source = replace_once(
            source,
            anchor,
            '      </template>\n\n      <RetrievalDiagnostics v-else :response="response" />\n    </div>\n  </aside>\n',
            "AnswerInspector diagnostics component",
        )

    source = source.replace("grid-template-columns: repeat(3, 1fr);", "grid-template-columns: repeat(4, 1fr);", 1)
    return source


def prepare(repo: Path, payload: Path) -> tuple[dict[Path, str], list[tuple[Path, Path]]]:
    targets: list[tuple[Path, callable]] = [
        (repo / "backend/app/models.py", patch_models),
        (repo / "backend/app/api.py", patch_api),
        (repo / "backend/app/rag/smart_index.py", patch_smart_index),
        (repo / "backend/app/rag/v5/ingestion.py", patch_ingestion),
        (repo / "backend/app/rag/v5/service.py", patch_service),
        (repo / "backend/app/rag/v5/synthesis_retrieval.py", patch_synthesis),
        (repo / "frontend/src/services/api.ts", patch_api_ts),
        (repo / "frontend/src/App.vue", patch_app),
        (repo / "frontend/src/components/AnswerInspector.vue", patch_inspector),
    ]
    optional: list[tuple[Path, callable]] = [
        (repo / "payload/backend/app/rag/v5/synthesis_retrieval.py", patch_synthesis),
        (repo / "payload/frontend/src/components/AnswerInspector.vue", patch_inspector),
    ]
    missing = [str(path) for path, _ in targets if not path.exists()]
    if missing:
        raise SystemExit("Required v5.2 repository file(s) not found:\n" + "\n".join(missing))

    transformed: dict[Path, str] = {}
    for path, transform in [*targets, *((pair for pair in optional if pair[0].exists()))]:
        original = path.read_text(encoding="utf-8-sig")
        transformed[path] = transform(original)

    copies = [
        (payload / "backend/app/rag/v5/retrieval_completeness.py", repo / "backend/app/rag/v5/retrieval_completeness.py"),
        (payload / "backend/tests/test_v521_retrieval_completeness.py", repo / "backend/tests/test_v521_retrieval_completeness.py"),
        (payload / "backend/app/rag/v5/query_trace.py", repo / "backend/app/rag/v5/query_trace.py"),
        (payload / "frontend/src/components/RetrievalDiagnostics.vue", repo / "frontend/src/components/RetrievalDiagnostics.vue"),
        (payload / "backend/app/rag/v5/retrieval_completeness.py", repo / "payload/backend/app/rag/v5/retrieval_completeness.py"),
        (payload / "frontend/src/components/RetrievalDiagnostics.vue", repo / "payload/frontend/src/components/RetrievalDiagnostics.vue"),
    ]
    for src, _ in copies:
        if not src.exists():
            raise SystemExit(f"Patch payload file missing: {src}")
    return transformed, copies


def validate_python(repo: Path, transformed: dict[Path, str], copies: list[tuple[Path, Path]]) -> None:
    python_paths = {
        repo / "backend/app/models.py",
        repo / "backend/app/api.py",
        repo / "backend/app/rag/smart_index.py",
        repo / "backend/app/rag/v5/ingestion.py",
        repo / "backend/app/rag/v5/service.py",
        repo / "backend/app/rag/v5/synthesis_retrieval.py",
    }
    for path in python_paths:
        if path in transformed:
            compile(transformed[path], str(path), "exec")
    for src, dst in copies:
        if dst.suffix == ".py":
            compile(src.read_text(encoding="utf-8"), str(dst), "exec")


def write_changes(transformed: dict[Path, str], copies: list[tuple[Path, Path]]) -> None:
    for path, content in transformed.items():
        current = path.read_text(encoding="utf-8-sig")
        if current == content:
            print(f"[already compatible] {path}")
            continue
        backup(path)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"[patched] {path}")
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            print(f"[already copied] {dst}")
            continue
        if dst.exists():
            backup(dst)
        shutil.copy2(src, dst)
        print(f"[copied] {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply deployable IMS RAG v5.2.1 completeness + diagnostics patch.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    here = Path(__file__).resolve().parent
    payload = here / "payload"

    transformed, copies = prepare(repo, payload)
    validate_python(repo, transformed, copies)
    print("[preflight] all backend Python transforms compile before any file is written")
    write_changes(transformed, copies)

    compile_targets = [
        repo / "backend/app/models.py",
        repo / "backend/app/api.py",
        repo / "backend/app/rag/smart_index.py",
        repo / "backend/app/rag/v5/ingestion.py",
        repo / "backend/app/rag/v5/retrieval_completeness.py",
        repo / "backend/app/rag/v5/synthesis_retrieval.py",
        repo / "backend/app/rag/v5/service.py",
        repo / "backend/tests/test_v521_retrieval_completeness.py",
        repo / "backend/app/rag/v5/query_trace.py",
    ]
    for path in compile_targets:
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied IMS RAG v5.2.1 deployment patch.")
    print(" - one unified corpus-discovery stage for synthesis")
    print(" - deep search only inside routed documents")
    print(" - exact alias/full-form lookup before semantic retrieval")
    print(" - case-tolerant source-grounded acronym recovery")
    print(" - incomplete coverage cannot hard-prune strong routed evidence")
    print(" - deterministic PDF-by-PDF retrieval diagnostics in the UI")
    print(" - no private chain-of-thought or hidden prompts exposed")
    print()
    print("No DB migration, PDF reprocessing, OCR rerun, chunk rebuild, or embedding rebuild is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
