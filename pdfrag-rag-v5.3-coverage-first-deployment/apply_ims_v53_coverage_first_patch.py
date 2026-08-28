from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_RAG_V53_COVERAGE_FIRST"
BACKUP_SUFFIX = ".bak-before-ims-v53-coverage-first"


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not target.exists():
        shutil.copy2(path, target)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def replace_function(source: str, name: str, replacement: str, next_name: str) -> str:
    start = source.find(f"def {name}(")
    if start < 0:
        raise RuntimeError(f"Function {name} not found")
    end = source.find(f"\ndef {next_name}(", start + 1)
    if end < 0:
        raise RuntimeError(f"Function boundary {next_name} after {name} not found")
    return source[:start] + replacement.rstrip() + "\n\n" + source[end + 1:]


def patch_synthesis(source: str) -> str:
    if MARKER in source:
        return source
    if "v5.2.1" not in source and "retrieval_diagnostics" not in source:
        raise RuntimeError("Expected the v5.2.1 synthesis/diagnostics layer before v5.3")

    cue_anchor = r'''    r"what\s+to\s+do|who\s+does\s+what|compare|difference(?:s)?|across\s+documents?)\b",'''
    cue_replacement = r'''    r"what\s+to\s+do|what\s+(?:will\s+|would\s+)?happen\s+if|what\s+if|how\s+to|who\s+does\s+what|compare|difference(?:s)?|across\s+documents?)\b",'''
    if cue_anchor in source:
        source = source.replace(cue_anchor, cue_replacement, 1)

    import_anchor = '''    direct_retrieval_diagnostics,
    exact_alias_results,
    finalize_retrieval_diagnostics,
    response_diagnostics,
    unified_corpus_discovery,
)
'''
    import_replacement = '''    direct_retrieval_diagnostics,
    ensure_routed_document_evidence,
    exact_alias_results,
    finalize_retrieval_diagnostics,
    enrich_retrieval_summary,
    required_coverage_documents,
    response_diagnostics,
    unified_corpus_discovery,
)
'''
    source = replace_once(source, import_anchor, import_replacement, "v5.3 synthesis coverage imports")

    rerank_anchor = "- Prefer governing/defining provisions over incidental mentions.\n"
    rerank_rule = (
        "- For conditional/procedural questions, equivalent governing evidence in a different rolling-stock or line scope is NOT incidental merely because another scope states the same rule. Keep at least one governing candidate per applicable RS/Line so the answer can show each source separately.\n"
    )
    if rerank_rule not in source:
        source = replace_once(source, rerank_anchor, rerank_anchor + rerank_rule, "per-scope rerank rule")

    coverage_anchor = "A document is \"contributing\" only when it adds meaningful evidence. Do not include incidental lexical matches.\n"
    coverage_rule = (
        "For conditional/procedural questions, a governing copy for a distinct rolling-stock or line scope is contributing evidence even when its instruction is identical to another scope, because it independently establishes applicability and source provenance.\n"
    )
    if coverage_rule not in source:
        source = replace_once(source, coverage_anchor, coverage_anchor + coverage_rule, "per-scope coverage rule")

    source = source.replace("exact_alias_results(db, original_question, limit=80)", "exact_alias_results(db, original_question, limit=120)")
    source = source.replace("_unique((item.chunk.filename for item in exact_results), 40)", "_unique((item.chunk.filename for item in exact_results), 120)")
    source = source.replace("evidence_docs = _unique((item.chunk.filename for item in final), 32)", "evidence_docs = _unique((item.chunk.filename for item in final), 80)")
    source = source.replace('contributing = _string_tuple(payload.get("contributing_documents"), 24)', 'contributing = _string_tuple(payload.get("contributing_documents"), 80)')
    source = source.replace(
        "            diagnostics, diagnostic_summary = direct_retrieval_diagnostics(\n                db,\n                results=exact_results,\n                routed_documents=exact_docs,\n            )",
        "            diagnostics, diagnostic_summary = direct_retrieval_diagnostics(\n                db,\n                results=exact_results,\n                routed_documents=exact_docs,\n                question=original_question,\n            )",
        1,
    )
    source = source.replace(
        "        diagnostics, diagnostic_summary = direct_retrieval_diagnostics(\n            db,\n            results=direct_results,\n            routed_documents=list(baseline.routed_documents),\n        )",
        "        diagnostics, diagnostic_summary = direct_retrieval_diagnostics(\n            db,\n            results=direct_results,\n            routed_documents=list(baseline.routed_documents),\n            question=original_question,\n        )",
        1,
    )

    old_call = '''    discovery = unified_corpus_discovery(
        db,
        queries=queries,
        prior_results=prior_results,
    )
'''
    new_call = '''    discovery = unified_corpus_discovery(
        db,
        queries=queries,
        prior_results=prior_results,
        question=original_question,
        interpretation=interpretation,
    )
'''
    source = replace_once(source, old_call, new_call, "scope-aware unified discovery call")
    source = source.replace('"Unified corpus discovery"', '"Coverage-aware corpus discovery"')
    source = source.replace('"Unified corpus discovery complete"', '"Coverage-aware corpus discovery complete"')

    final_anchor = '''    final = _final_diverse(combined, routes, final_limit)
'''
    final_replacement = '''    final = _final_diverse(combined, routes, final_limit)
    final = ensure_routed_document_evidence(final, combined, routes, final_limit)
'''
    source = replace_once(source, final_anchor, final_replacement, "routed-document final evidence preservation")

    diag_anchor = '''    retrieval_diagnostics = finalize_retrieval_diagnostics(
        discovery.diagnostics, routes=routes, final_results=final
    )
'''
    diag_replacement = '''    retrieval_diagnostics = finalize_retrieval_diagnostics(
        discovery.diagnostics, routes=routes, final_results=final
    )
    diagnostic_summary = enrich_retrieval_summary(
        dict(discovery.summary), retrieval_diagnostics
    )
'''
    source = replace_once(source, diag_anchor, diag_replacement, "scope coverage diagnostic summary")
    old_summary_tail = "retrieval_diagnostic_summary=dict(discovery.summary),\n    )\n\n\ndef _coverage_prompt"
    if old_summary_tail in source:
        source = source.replace(
            old_summary_tail,
            "retrieval_diagnostic_summary=diagnostic_summary,\n    )\n\n\ndef _coverage_prompt",
            1,
        )
    else:
        raise RuntimeError("final synthesis diagnostic summary anchor not found")

    select_replacement = '''def select_results_for_answer(
    results: Sequence[RetrievedChunk],
    review: SynthesisCoverage,
) -> list[RetrievedChunk]:
    # v5.3: ranking/rerank already removed incidental evidence. Preserve the complete
    # final cross-document evidence set so identical rules from different applicable
    # RS/Line scopes remain available to the answer layer.
    return list(results)
'''
    source = replace_function(source, "select_results_for_answer", select_replacement, "coverage_prompt_status")

    coverage_status_replacement = '''def coverage_prompt_status(
    base_status: str,
    bundle: SynthesisRetrievalBundle,
    review: SynthesisCoverage,
) -> str:
    summary = bundle.retrieval_diagnostic_summary or {}
    definition_docs = [str(value) for value in summary.get("definition_documents", [])]
    required_scope_docs = [str(value) for value in summary.get("required_scope_documents", [])]
    if bundle.answer_strategy != "multi_document_synthesis" and not definition_docs:
        return base_status

    lines = [base_status, f"Answer strategy: {bundle.answer_strategy}"]
    if definition_docs:
        lines.append("Definition enumeration: list every distinct supported meaning and show every supplied PDF/page source; do not collapse same meanings across PDFs.")
        lines.append("Definition source documents: " + "; ".join(definition_docs))
    if bundle.answer_strategy == "multi_document_synthesis":
        lines.append("Instruction: answer every applicable RS/Line scope represented by governing evidence; identical procedures in different scopes must still be shown separately with source citations.")
        lines.append("Synthesis dimensions: " + ("; ".join(bundle.synthesis_dimensions) or "direct answer evidence"))
        if required_scope_docs:
            lines.append("Required RS/Line source documents for answer presentation: " + "; ".join(required_scope_docs))
        lines.append("Coverage-review contributing documents: " + ("; ".join(review.contributing_documents) or "not yet established"))
    if review.uncovered_dimensions:
        lines.append("Uncovered dimensions: " + "; ".join(review.uncovered_dimensions))
    if review.conflicts:
        lines.append("Cross-document differences/conflicts identified by evidence review:")
        for conflict in review.conflicts[:6]:
            lines.append(
                f"- {conflict.get('type')}: {conflict.get('summary') or 'difference detected'}; "
                f"resolution: {conflict.get('resolution') or 'not established'}"
            )
    return "\\n".join(lines)
'''
    source = replace_function(source, "coverage_prompt_status", coverage_status_replacement, "response_synthesis_metadata")

    response_replacement = '''def response_synthesis_metadata(
    bundle: SynthesisRetrievalBundle,
    review: SynthesisCoverage,
) -> dict[str, object]:
    summary = enrich_retrieval_summary(
        bundle.retrieval_diagnostic_summary, bundle.retrieval_diagnostics
    )
    required = [str(value) for value in summary.get("required_answer_documents", [])]
    definition_docs = [str(value) for value in summary.get("definition_documents", [])]
    effective_contributing = _unique(
        [*review.contributing_documents, *required, *definition_docs], 120
    )
    diagnostics, diagnostic_summary = response_diagnostics(
        bundle.retrieval_diagnostics, summary, effective_contributing
    )
    search_scope = (
        "definition_enumeration"
        if definition_docs
        else ("broad_relevant_corpus" if bundle.answer_strategy == "multi_document_synthesis" else "focused")
    )
    return {
        "answer_strategy": bundle.answer_strategy,
        "synthesis_dimensions": list(bundle.synthesis_dimensions),
        "search_scope": search_scope,
        "relevant_documents": list(bundle.considered_documents),
        "contributing_documents": effective_contributing,
        "search_rounds": int(bundle.search_round),
        "evidence_coverage_status": review.evidence_coverage_status,
        "conflicts": [dict(item) for item in review.conflicts],
        "retrieval_diagnostics": diagnostics,
        "retrieval_diagnostic_summary": diagnostic_summary,
    }
'''
    start = source.find("def response_synthesis_metadata(")
    if start < 0:
        raise RuntimeError("response_synthesis_metadata not found")
    source = source[:start] + response_replacement.rstrip() + "\n"
    return "# IMS_RAG_V53_COVERAGE_FIRST\n" + source


def patch_service(source: str) -> str:
    if MARKER in source:
        return source
    if "rag-v5.2.1-completeness" not in source:
        raise RuntimeError("Expected active v5.2.1 service before applying v5.3")

    ux_anchor = "- If a colloquial category can match multiple formal source categories, state the supported alternatives\n  or the missing distinction instead of silently choosing one.\n"
    ux_add = '''- ANSWER-FIRST POLICY: when supplied evidence materially answers the user's operational intent, start with that supported action/value/procedure. Do not lead with absence, uncertainty or a wording correction.
- If the user's wording differs from the source's formal condition (for example percentage of brakes vs percentage/number of bogies), answer the closest clearly applicable source-defined condition first. Put the terminology/scope clarification in one concise final Note unless the distinction changes the answer materially.
- A negative opening such as "cannot be verified", "the documents do not state", or "no direct mention" is allowed only when no supplied evidence materially answers the request.
'''
    source = replace_once(source, ux_anchor, ux_anchor + ux_add, "answer-first user experience policy")

    acronym_anchor = '''- For an unscoped acronym/full-form/meaning question, if sources establish multiple distinct expansions,
  list every materially supported meaning and identify its equipment/document context; never collapse
  one alias to a single global canonical meaning.
'''
    acronym_add = '''- DEFINITION ENUMERATION: group the answer by each distinct source-defined meaning. Under each meaning, list every supplied PDF and page/section that independently supports that meaning. If the same meaning appears in RS-1, RS-3 and RS-10, show all three source locations rather than citing only one representative PDF.
'''
    source = replace_once(source, acronym_anchor, acronym_anchor + acronym_add, "definition enumeration answer policy")

    multi_anchor = '- Cite the specific source(s) supporting every important synthesized point; multi-source citations are encouraged when several documents jointly support a statement.\n'
    multi_add = '''- CROSS-SCOPE PROCEDURE ENUMERATION: for "what happens if", "what to do if", procedure, troubleshooting, responsibility or similar conditional questions, use a separate heading for every applicable rolling-stock/line source represented in the supplied governing evidence. Show the PDF name and cite the relevant page/section under that heading.
- If RS-1, RS-3 and RS-10 prescribe the same action, still show RS-1, RS-3 and RS-10 separately; after those headings you may state that their cited procedures are equivalent. Never source-collapse identical rules across scopes.
- If the user names one RS/Line without saying "only", prioritize that scope first but still report other applicable RS/Line procedures found by the coverage search. If the user explicitly says only/just/solely/specifically that scope, respect the restriction.
'''
    source = replace_once(source, multi_anchor, multi_anchor + multi_add, "cross-scope answer presentation policy")

    assistant_sources = '''def _assistant_sources(
    results: list[RetrievedChunk],
    limit: int,
    interpretation: object,
) -> list[PromptSource]:
    """Seed one evidence unit per document, then fill by ranked order."""
    if not results or limit <= 0:
        return []
    by_doc: dict[str, list[RetrievedChunk]] = {}
    doc_order: list[str] = []
    for item in results:
        doc = item.chunk.document_id or item.chunk.filename
        if doc not in by_doc:
            by_doc[doc] = []
            doc_order.append(doc)
        by_doc[doc].append(item)

    output: list[PromptSource] = []
    seen: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    for doc in doc_order:
        item = by_doc[doc][0]
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        counts[doc] += 1
        output.append(PromptSource(result=item, excerpt=item.chunk.text.strip()))
        if len(output) >= limit:
            return output

    intent = str(getattr(interpretation, "intent", "fact_lookup"))
    conversation_act = str(getattr(interpretation, "conversation_act", "question"))
    section_heavy = intent in {"list", "procedure", "summary", "requirement", "troubleshooting", "comparison"} or conversation_act == "navigation"
    per_document_cap = 24 if section_heavy else 16
    for item in results:
        if item.chunk.chunk_id in seen:
            continue
        doc = item.chunk.document_id or item.chunk.filename
        if counts[doc] >= per_document_cap:
            continue
        seen.add(item.chunk.chunk_id)
        counts[doc] += 1
        output.append(PromptSource(result=item, excerpt=item.chunk.text.strip()))
        if len(output) >= limit:
            break
    return output
'''
    start = source.find("def _assistant_sources(")
    end = source.find("\n\nclass V5RagService", start)
    if start < 0 or end < 0:
        raise RuntimeError("_assistant_sources/class boundary not found")
    source = source[:start] + assistant_sources.rstrip() + source[end:]

    old_limit = '''        base_evidence_limit = min(top_k or _int_env("RAG_V5_FINAL_EVIDENCE", 32, 12, 80), 80)
        evidence_limit = (
            min(80, max(base_evidence_limit, _int_env("RAG_V52_SYNTHESIS_EVIDENCE", 48, 24, 80)))
            if bundle.answer_strategy == "multi_document_synthesis"
            else base_evidence_limit
        )
'''
    new_limit = '''        base_evidence_limit = min(top_k or _int_env("RAG_V5_FINAL_EVIDENCE", 32, 12, 80), 80)
        diagnostic_summary = bundle.retrieval_diagnostic_summary or {}
        enumeration_mode = bool(
            diagnostic_summary.get("definition_enumeration")
            or diagnostic_summary.get("coverage_mode")
        )
        evidence_limit = (
            min(80, max(base_evidence_limit, _int_env("RAG_V53_ENUMERATION_EVIDENCE", 64, 32, 80)))
            if enumeration_mode
            else (
                min(80, max(base_evidence_limit, _int_env("RAG_V52_SYNTHESIS_EVIDENCE", 48, 24, 80)))
                if bundle.answer_strategy == "multi_document_synthesis"
                else base_evidence_limit
            )
        )
'''
    source = replace_once(source, old_limit, new_limit, "v5.3 enumeration evidence budget")

    repair_anchor = '''_V5_CITATION_REPAIR_SYSTEM = """Repair citation grounding for a closed-book PDF answer.
Use ONLY the supplied source blocks. Do not add new factual claims. Keep the answer direct and preserve
its meaning, but ensure every factual sentence/bullet has valid [S#] citations and no citation is out of
range. If the draft contains a factual statement unsupported by the sources, remove or qualify it.
Return only the repaired answer."""
'''
    repair_add = r'''

_V53_NEGATIVE_OPENING_RE = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:for\b.{0,100}\b)?(?:the\s+)?(?:documents?|sources?|evidence)?\s*"
    r"(?:do(?:es)?\s+not\s+state|doesn't\s+state|do\s+not\s+state|cannot\s+be\s+verified|"
    r"could\s+not\s+be\s+verified|not\s+verified|no\s+direct\s+mention|not\s+explicitly\s+stated)",
    re.IGNORECASE | re.DOTALL,
)

_V53_ANSWER_POLICY_REPAIR_SYSTEM = """Rewrite a grounded draft to satisfy the IMS answer-presentation contract.
Use ONLY the supplied source blocks and claims already supportable from them. Do not invent facts.

Rules:
- Put the useful supported answer/action/value first.
- Do not open with a missing-information caveat when a materially applicable source rule is present.
- If the user's wording differs from the formal source condition, answer the closest clearly applicable source condition first and put one concise clarification Note at the end.
- For acronym/full-form/meaning requests, include every distinct supported meaning and every required PDF/page source location supplied below; do not collapse identical meanings across PDFs.
- For cross-scope conditional/procedural requests, use separate RS/Line/PDF headings for every required document supplied below, even when the procedures are identical.
- Preserve all valid [S#] citations and ensure every factual statement is cited.
Return only the rewritten answer."""


def _summary_list(bundle: object, key: str) -> list[object]:
    summary = dict(getattr(bundle, "retrieval_diagnostic_summary", {}) or {})
    value = summary.get(key, [])
    return list(value) if isinstance(value, list) else []


def _required_answer_documents(bundle: object) -> list[str]:
    values = [
        *_summary_list(bundle, "definition_documents"),
        *_summary_list(bundle, "required_scope_documents"),
        *_summary_list(bundle, "required_answer_documents"),
    ]
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean.casefold() in seen:
            continue
        seen.add(clean.casefold())
        output.append(clean)
    return output[:120]


def _required_scope_labels(bundle: object) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in _summary_list(bundle, "required_scope_labels"):
        clean = str(value or "").strip()
        if clean and clean.casefold() not in seen:
            seen.add(clean.casefold())
            output.append(clean)
    return output[:120]


def _required_definition_meanings(bundle: object) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in _summary_list(bundle, "definition_meanings"):
        clean = str(value or "").strip()
        if clean and clean.casefold() not in seen:
            seen.add(clean.casefold())
            output.append(clean)
    return output[:120]


def _required_definition_locations(bundle: object) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in _summary_list(bundle, "definition_locations"):
        if not isinstance(raw, dict):
            continue
        filename = str(raw.get("filename") or "").strip()
        meaning = str(raw.get("meaning") or "").strip()
        if not filename or not meaning:
            continue
        try:
            page_start = int(raw.get("page_start") or 0)
            page_end = int(raw.get("page_end") or page_start)
        except (TypeError, ValueError):
            continue
        output.append({
            "alias": str(raw.get("alias") or "").strip(),
            "meaning": meaning,
            "filename": filename,
            "page_start": page_start,
            "page_end": page_end,
            "heading": str(raw.get("heading") or "").strip(),
        })
        if len(output) >= 240:
            break
    return output


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _source_matches_location(source: PromptSource, location: dict[str, object]) -> bool:
    chunk = source.result.chunk
    if chunk.filename.casefold() != str(location.get("filename") or "").casefold():
        return False
    start = int(location.get("page_start") or 0)
    end = int(location.get("page_end") or start)
    chunk_start = int(chunk.page_number or 0)
    chunk_end = int(chunk.page_end or chunk_start)
    return start <= chunk_end and end >= chunk_start


def _cited_document_names(answer: str, sources: list[PromptSource]) -> set[str]:
    used = set(cited_source_numbers(answer, len(sources)))
    return {
        source.result.chunk.filename.casefold()
        for index, source in enumerate(sources, 1)
        if index in used
    }


def _needs_v53_answer_policy_repair(
    answer: str,
    sources: list[PromptSource],
    required_documents: list[str],
    required_scope_labels: list[str],
    required_meanings: list[str],
    required_locations: list[dict[str, object]],
) -> bool:
    if not answer or answer == NO_ANSWER or not sources:
        return False
    if _V53_NEGATIVE_OPENING_RE.search(answer[:700]):
        return True

    available_docs = {source.result.chunk.filename.casefold() for source in sources}
    cited_docs = _cited_document_names(answer, sources)
    for filename in required_documents:
        folded = filename.casefold()
        if folded in available_docs and folded not in cited_docs:
            return True

    normalized_answer = _normalized_words(answer)
    for label in required_scope_labels:
        if _normalized_words(label) not in normalized_answer:
            return True

    available_locations = [
        location
        for location in required_locations
        if any(_source_matches_location(source, location) for source in sources)
    ]
    available_meanings = {
        str(location.get("meaning") or "").strip().casefold()
        for location in available_locations
        if str(location.get("meaning") or "").strip()
    }
    for meaning in required_meanings:
        if available_meanings and meaning.casefold() not in available_meanings:
            continue
        if _normalized_words(meaning) not in normalized_answer:
            return True

    used = set(cited_source_numbers(answer, len(sources)))
    cited_sources = [source for index, source in enumerate(sources, 1) if index in used]
    for location in available_locations:
        if not any(_source_matches_location(source, location) for source in cited_sources):
            return True
    return False


def _v53_answer_policy_repair_prompt(
    draft: str,
    sources: list[PromptSource],
    required_documents: list[str],
    required_scope_labels: list[str],
    required_meanings: list[str],
    required_locations: list[dict[str, object]],
) -> str:
    location_lines = []
    for item in required_locations:
        pages = str(item.get("page_start") or "?")
        if item.get("page_end") and item.get("page_end") != item.get("page_start"):
            pages += f"-{item.get('page_end')}"
        location_lines.append(
            f"- {item.get('meaning')} — {item.get('filename')} — page {pages}"
        )
    return f"""DRAFT ANSWER:
{draft}

REQUIRED SOURCE DOCUMENTS THAT MUST BE REPRESENTED WHEN THEIR SUPPLIED EVIDENCE APPLIES:
{chr(10).join(f'- {item}' for item in required_documents) if required_documents else '- None beyond ordinary grounding'}

REQUIRED RS/LINE LABELS THAT SHOULD APPEAR AS SEPARATE ANSWER SCOPE HEADINGS:
{chr(10).join(f'- {item}' for item in required_scope_labels) if required_scope_labels else '- None'}

REQUIRED DISTINCT DEFINITION MEANINGS:
{chr(10).join(f'- {item}' for item in required_meanings) if required_meanings else '- None'}

REQUIRED DEFINITION SOURCE LOCATIONS WHEN PRESENT IN VALID SOURCES:
{chr(10).join(location_lines) if location_lines else '- None'}

VALID SOURCES:
{chr(10).join(_source_block(index, source) for index, source in enumerate(sources, 1))}

Rewrite the draft under the answer-first and source-enumeration rules without adding unsupported claims.
"""
'''
    source = replace_once(source, repair_anchor, repair_anchor + repair_add, "answer policy repair helpers")

    old_verify = '''        verified = verify_answer(
            interpretation,
            draft,
            prompt_sources,
            coverage_status=coverage_status,
        )
        answer, grounded = validate_grounded_answer(verified or draft, len(prompt_sources))
'''
    new_verify = '''        verified = verify_answer(
            interpretation,
            draft,
            prompt_sources,
            coverage_status=coverage_status,
        )
        candidate_answer = verified or draft
        required_answer_documents = _required_answer_documents(bundle)
        required_scope_labels = _required_scope_labels(bundle)
        required_definition_meanings = _required_definition_meanings(bundle)
        required_definition_locations = _required_definition_locations(bundle)
        if _needs_v53_answer_policy_repair(
            candidate_answer,
            prompt_sources,
            required_answer_documents,
            required_scope_labels,
            required_definition_meanings,
            required_definition_locations,
        ):
            emit_progress(
                "answer_policy_repair",
                "Improving answer presentation",
                "Putting the supported answer first and preserving required RS/Line/PDF source coverage",
                actor="verification",
                phase="verify",
                status="running",
                operation_id="answer-policy-repair",
                prompt_summary="Reorganize only supported claims: answer first, keep source-specific headings, and move wording clarification to the end.",
            )
            candidate_answer = llm_service.generate(
                _V53_ANSWER_POLICY_REPAIR_SYSTEM,
                _v53_answer_policy_repair_prompt(
                    candidate_answer,
                    prompt_sources,
                    required_answer_documents,
                    required_scope_labels,
                    required_definition_meanings,
                    required_definition_locations,
                ),
                max_output_tokens=settings.max_output_tokens,
                model=settings.query_model,
                reasoning_effort=settings.query_reasoning_effort,
            )
            policy_repaired = candidate_answer
            policy_verified = verify_answer(
                interpretation,
                policy_repaired,
                prompt_sources,
                coverage_status=coverage_status,
            ) or policy_repaired
            # A semantic verifier must not undo a presentation repair by dropping an
            # otherwise grounded required RS/Line/PDF source. Prefer the repaired draft
            # when the verifier reintroduces the same deterministic policy violation.
            candidate_answer = (
                policy_repaired
                if _needs_v53_answer_policy_repair(
                    policy_verified,
                    prompt_sources,
                    required_answer_documents,
                    required_scope_labels,
                    required_definition_meanings,
                    required_definition_locations,
                )
                and not _needs_v53_answer_policy_repair(
                    policy_repaired,
                    prompt_sources,
                    required_answer_documents,
                    required_scope_labels,
                    required_definition_meanings,
                    required_definition_locations,
                )
                else policy_verified
            )
        answer, grounded = validate_grounded_answer(candidate_answer, len(prompt_sources))
'''
    source = replace_once(source, old_verify, new_verify, "conditional answer-policy repair integration")

    citation_tail = '''            answer, grounded = validate_grounded_answer(repaired, len(prompt_sources))
            grounding_status = "verified_after_repair" if grounded else "citation_validation_failed"

        used = set(cited_source_numbers(answer, len(prompt_sources)))
'''
    final_policy = '''            answer, grounded = validate_grounded_answer(repaired, len(prompt_sources))
            grounding_status = "verified_after_repair" if grounded else "citation_validation_failed"

        # Citation repair may rewrite prose, but it must not collapse source-enumeration
        # coverage. Reapply the v5.3 presentation contract once at the end only when
        # the deterministic policy check still detects a violation.
        if answer != NO_ANSWER and _needs_v53_answer_policy_repair(
            answer,
            prompt_sources,
            required_answer_documents,
            required_scope_labels,
            required_definition_meanings,
            required_definition_locations,
        ):
            emit_progress(
                "answer_policy_final",
                "Preserving source coverage",
                "Restoring answer-first presentation and required RS/Line/PDF headings after citation repair",
                actor="verification",
                phase="verify",
                status="running",
                operation_id="answer-policy-final",
                prompt_summary="Preserve grounded claims while restoring required source-specific answer coverage.",
            )
            final_policy_repaired = llm_service.generate(
                _V53_ANSWER_POLICY_REPAIR_SYSTEM,
                _v53_answer_policy_repair_prompt(
                    answer,
                    prompt_sources,
                    required_answer_documents,
                    required_scope_labels,
                    required_definition_meanings,
                    required_definition_locations,
                ),
                max_output_tokens=settings.max_output_tokens,
                model=settings.query_model,
                reasoning_effort=settings.query_reasoning_effort,
            )
            final_policy_verified = verify_answer(
                interpretation,
                final_policy_repaired,
                prompt_sources,
                coverage_status=coverage_status,
            ) or final_policy_repaired
            final_candidate = (
                final_policy_repaired
                if _needs_v53_answer_policy_repair(
                    final_policy_verified,
                    prompt_sources,
                    required_answer_documents,
                    required_scope_labels,
                    required_definition_meanings,
                    required_definition_locations,
                )
                and not _needs_v53_answer_policy_repair(
                    final_policy_repaired,
                    prompt_sources,
                    required_answer_documents,
                    required_scope_labels,
                    required_definition_meanings,
                    required_definition_locations,
                )
                else final_policy_verified
            )
            answer, grounded = validate_grounded_answer(
                final_candidate, len(prompt_sources)
            )
            grounding_status = (
                "verified_after_policy_repair"
                if grounded
                else "citation_validation_failed"
            )

        used = set(cited_source_numbers(answer, len(prompt_sources)))
'''
    source = replace_once(source, citation_tail, final_policy, "final source-coverage answer repair")
    source = source.replace("rag-v5.2.1-completeness", "rag-v5.3-coverage-first")
    return "# IMS_RAG_V53_COVERAGE_FIRST\n" + source


def patch_api_ts(source: str) -> str:
    if "scope_label?: string" not in source:
        anchor = "  best_heading?: string\n}\n"
        addition = '''  scope_type?: 'rs' | 'line' | 'common' | string
  scope_label?: string
  scope_pinned?: boolean
  explicit_scope?: boolean
  route_rank?: number | null
  global_route_cap?: number | null
'''
        source = replace_once(source, anchor, "  best_heading?: string\n" + addition + "}\n", "diagnostic scope fields")
    if "scope_count_routed?: number" not in source:
        anchor = "  diagnostic_documents_included?: number\n}\n"
        addition = '''  coverage_mode?: boolean
  explicit_scopes?: string[]
  scope_count_with_signal?: number
  scope_count_routed?: number
  scope_promoted_documents?: number
  scope_final_evidence_documents?: number
  global_route_cap?: number
  required_scope_documents?: string[]
  required_scope_labels?: string[]
  required_answer_documents?: string[]
  definition_enumeration?: boolean
  definition_documents?: string[]
  definition_meanings?: string[]
  definition_inventory?: Array<{
    alias: string
    meaning: string
    sources: Array<{ filename: string; page_start: number; page_end: number; heading?: string }>
  }>
  definition_locations?: Array<{
    alias: string
    meaning: string
    filename: string
    page_start: number
    page_end: number
    heading?: string
  }>
  definition_source_count?: number
'''
        source = replace_once(source, anchor, "  diagnostic_documents_included?: number\n" + addition + "}\n", "diagnostic summary coverage fields")
    return source


def prepare(repo: Path, payload: Path):
    required = [
        repo / "backend/app/rag/v5/service.py",
        repo / "backend/app/rag/v5/synthesis_retrieval.py",
        repo / "backend/app/rag/v5/retrieval_completeness.py",
        repo / "frontend/src/services/api.ts",
        repo / "frontend/src/components/RetrievalDiagnostics.vue",
        payload / "backend/app/rag/v5/retrieval_completeness.py",
        payload / "backend/tests/test_v53_coverage_first.py",
        payload / "frontend/src/components/RetrievalDiagnostics.vue",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Required v5.2.1 file(s) missing:\n" + "\n".join(missing))

    transforms = {
        repo / "backend/app/rag/v5/service.py": patch_service,
        repo / "backend/app/rag/v5/synthesis_retrieval.py": patch_synthesis,
        repo / "frontend/src/services/api.ts": patch_api_ts,
    }
    payload_synthesis = repo / "payload/backend/app/rag/v5/synthesis_retrieval.py"
    if payload_synthesis.exists():
        transforms[payload_synthesis] = patch_synthesis

    transformed: dict[Path, str] = {}
    for path, fn in transforms.items():
        transformed[path] = fn(path.read_text(encoding="utf-8-sig"))

    copies = [
        (payload / "backend/app/rag/v5/retrieval_completeness.py", repo / "backend/app/rag/v5/retrieval_completeness.py"),
        (payload / "backend/app/rag/v5/retrieval_completeness.py", repo / "payload/backend/app/rag/v5/retrieval_completeness.py"),
        (payload / "backend/tests/test_v53_coverage_first.py", repo / "backend/tests/test_v53_coverage_first.py"),
        (payload / "frontend/src/components/RetrievalDiagnostics.vue", repo / "frontend/src/components/RetrievalDiagnostics.vue"),
        (payload / "frontend/src/components/RetrievalDiagnostics.vue", repo / "payload/frontend/src/components/RetrievalDiagnostics.vue"),
    ]
    return transformed, copies


def validate(transformed: dict[Path, str], copies: list[tuple[Path, Path]]) -> None:
    for path, content in transformed.items():
        if path.suffix == ".py":
            compile(content, str(path), "exec")
    for src, dst in copies:
        if dst.suffix == ".py":
            compile(src.read_text(encoding="utf-8"), str(dst), "exec")


def write(transformed: dict[Path, str], copies: list[tuple[Path, Path]]) -> None:
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
    parser = argparse.ArgumentParser(description="Apply IMS RAG v5.3 coverage-first deployment patch.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    payload = Path(__file__).resolve().parent / "payload"

    transformed, copies = prepare(repo, payload)
    validate(transformed, copies)
    print("[preflight] v5.3 transforms compile before any repository file is written")
    write(transformed, copies)

    for path in [
        repo / "backend/app/rag/v5/service.py",
        repo / "backend/app/rag/v5/synthesis_retrieval.py",
        repo / "backend/app/rag/v5/retrieval_completeness.py",
        repo / "backend/tests/test_v53_coverage_first.py",
    ]:
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied IMS RAG v5.3 coverage-first deployment patch.")
    print(" - definition/full-form queries enumerate all distinct source-grounded meanings")
    print(" - same definition in multiple PDFs remains visible with each PDF/page source")
    print(" - conditional/procedure queries retain every relevant RS/Line scope")
    print(" - explicit RS/Line scopes are pinned and cannot be lost to the ordinary route cap")
    print(" - same procedure in RS-1/RS-3/RS-10 remains separately source-visible")
    print(" - one final evidence seed per routed document survives the final global candidate limit")
    print(" - answer-first policy moves wording/scope clarification to a concise final note")
    print(" - conditional answer-policy repair runs only when the draft violates these rules")
    print(" - diagnostics show scope label, pinning and route-rank/cap information")
    print()
    print("No DB migration, PDF reprocessing, OCR rerun, chunk rebuild, or embedding rebuild is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
