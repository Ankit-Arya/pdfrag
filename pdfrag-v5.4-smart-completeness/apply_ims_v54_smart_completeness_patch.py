from __future__ import annotations

import argparse
import json
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_RAG_V54_SMART_COMPLETENESS"
V53_MARKER = "IMS_RAG_V53_COVERAGE_FIRST"
BACKUP_SUFFIX = ".bak-before-ims-v54-smart-completeness"


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
    return source[:start] + replacement.rstrip() + "\n\n" + source[end + 1 :]


def patch_retrieval_completeness(source: str) -> str:
    if MARKER in source:
        return source
    if V53_MARKER not in source:
        raise RuntimeError(
            "backend/app/rag/v5/retrieval_completeness.py is not the v5.3 coverage-first version. "
            "Apply the existing v5.3 deployment patch first."
        )

    constants = r'''

# v5.4 query-completeness classifier. This is intentionally domain-generic: it decides
# HOW complete retrieval must be, never WHAT the answer is. No line, rolling-stock,
# document, equipment, acronym, value or procedure is hard-coded.
_ENTITY_ENUMERATION_RE = re.compile(
    r"\b(?:members?|membership|composition|types?|categories?|classes?|kinds?|components?|"
    r"designations?|items?|stages?|levels?|modes?|functions?|dut(?:y|ies)|roles?|"
    r"responsibilit(?:y|ies)|list\s+(?:all\s+|the\s+)?|who\s+are|what\s+are)\b",
    re.IGNORECASE,
)
_OPERATIONAL_COMPARISON_RE = re.compile(
    r"\b(?:if|when|failure|failed|fault|procedure|steps?|action|requirement|required|"
    r"allowed|permitted|prohibited|restriction|exception|condition|applicab(?:le|ility))\b",
    re.IGNORECASE,
)
'''
    source = replace_once(
        source,
        "\n\n@dataclass(slots=True)\nclass _DocumentState:",
        constants + "\n\n@dataclass(slots=True)\nclass _DocumentState:",
        "v5.4 completeness constants",
    )

    policy_functions = r'''def _explicit_definition_request(question: str) -> bool:
    value = _clean(question)
    return bool(
        _ALIAS_EXPLICIT_RE.search(value)
        or _WHAT_IS_ALIAS_RE.search(value)
        or _ALIAS_STANDS_RE.search(value)
    )


def completeness_policy(question: str, interpretation: object | None = None) -> str:
    """Classify the retrieval completeness contract for the current question.

    Values:
    - definition_enumeration: enumerate every explicit source-defined meaning/location.
    - entity_enumeration: complete governing lists/tables/sections, not RS/Line fan-out.
    - cross_scope_procedure: retain every materially applicable RS/Line procedure scope.
    - direct_lookup: focused fact/value/navigation retrieval.

    The classifier only controls retrieval behavior. It contains no domain facts.
    """
    value = _clean(question)
    intent = str(getattr(interpretation, "intent", "") or "").casefold()

    # Conditional semantics must outrank a generic model classification such as "list".
    if _CONDITIONAL_PROCEDURE_RE.search(value):
        return "cross_scope_procedure"
    if intent in {"procedure", "troubleshooting", "requirement"}:
        return "cross_scope_procedure"

    # A comparison becomes cross-scope only when the user actually names RS/Line scope
    # or compares operational conditions/actions. Ordinary conceptual comparisons do not
    # need one candidate from every RS/Line filename in the corpus.
    if intent == "comparison" and (
        explicit_scope_labels(value) or _OPERATIONAL_COMPARISON_RE.search(value)
    ):
        return "cross_scope_procedure"

    if intent == "definition" or _explicit_definition_request(value):
        return "definition_enumeration"

    if intent == "list" or _ENTITY_ENUMERATION_RE.search(value):
        return "entity_enumeration"

    return "direct_lookup"


def is_entity_enumeration_query(question: str, interpretation: object | None = None) -> bool:
    return completeness_policy(question, interpretation) == "entity_enumeration"


def is_conditional_procedure_query(question: str, interpretation: object | None = None) -> bool:
    return completeness_policy(question, interpretation) == "cross_scope_procedure"
'''
    source = replace_function(
        source,
        "is_conditional_procedure_query",
        policy_functions,
        "_active_documents",
    )

    old_mode = (
        "    coverage_mode = is_conditional_procedure_query(question, interpretation) "
        "if coverage_mode is None else bool(coverage_mode)\n"
    )
    new_mode = (
        "    policy = completeness_policy(question, interpretation)\n"
        "    coverage_mode = (policy == \"cross_scope_procedure\") if coverage_mode is None else bool(coverage_mode)\n"
    )
    source = replace_once(source, old_mode, new_mode, "v5.4 policy-aware discovery mode")

    # There are two summary dictionaries in this function (empty and completed). Make
    # the active policy visible to diagnostics without changing existing field meanings.
    source = source.replace(
        '        "coverage_mode": coverage_mode,\n',
        '        "coverage_mode": coverage_mode,\n        "coverage_policy": policy,\n',
    )

    enumeration_helpers = r'''

def _enum_seed_role(method: str) -> str:
    match = re.search(r"v5\.2-synthesis:([a-z_]+)", str(method or ""), re.IGNORECASE)
    return match.group(1).casefold() if match else ""


def _headerless_first_row_recovery(item: RetrievedChunk) -> RetrievedChunk | None:
    """Recover a first data row that pdfplumber accidentally promoted to column names.

    This recovery is deliberately conservative. It only fires when the first extracted
    "column name" begins with an integer N and the first stored table row begins with N+1.
    All recovered cell text is copied verbatim from the source chunk's PDF STRUCTURE
    metadata; no domain label or value is invented.
    """
    chunk = item.chunk
    if chunk.content_type != "table_row":
        return None
    match = re.search(
        r"^Columns:\s*(?P<columns>.+?)\s*$",
        chunk.text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    columns = [_clean(part) for part in match.group("columns").split("|")]
    if len(columns) < 2 or not re.fullmatch(r"\(?\d{1,3}[.)]?\)?", columns[0]):
        return None
    first_number_match = re.search(r"\d{1,3}", columns[0])
    if not first_number_match:
        return None
    first_number = int(first_number_match.group(0))

    body = _STRUCTURE_RE.sub("", chunk.text)
    row_serial_match = re.search(
        r"Table\s+row\s+\d+\s*:\s*[^:|]{0,80}:\s*(?P<serial>\d{1,3})(?:\s*\||\s*$)",
        body,
        flags=re.IGNORECASE,
    )
    if not row_serial_match or int(row_serial_match.group("serial")) != first_number + 1:
        return None

    # Do not reinterpret an obvious semantic header as a first data row.
    header_cues = {
        "no", "no.", "s no", "sl no", "serial", "item", "description", "category",
        "type", "status", "value", "unit", "remarks", "remark", "action", "role",
        "responsibility", "designation", "name",
    }
    cue_hits = sum(1 for cell in columns[1:] if _clean(cell).casefold() in header_cues)
    if cue_hits >= max(1, len(columns[1:]) // 2 + len(columns[1:]) % 2):
        return None

    pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
    section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "Unsectioned content")
    text_value = (
        "[PDF STRUCTURE]\n"
        f"File: {chunk.filename}\n"
        f"Pages: {pages}\n"
        f"Section path: {section}\n"
        "Content type: table_row\n"
        f"Heading: {chunk.heading or section}\n"
        "[/PDF STRUCTURE]\n\n"
        "Table row recovered from source table metadata (the PDF extractor classified "
        "the first data row as column names): " + " | ".join(columns)
    )
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=f"v54-headerless:{chunk.chunk_id}",
            filename=chunk.filename,
            page_number=chunk.page_number,
            page_end=chunk.page_end,
            text=text_value,
            content_type="table_row",
            section_path=chunk.section_path,
            heading=chunk.heading,
            document_id=chunk.document_id,
            chunk_index=max(0, int(chunk.chunk_index or 0) - 1),
        ),
        score=min(1.0, max(float(item.score), 0.90)),
        method=str(item.method) + "+v5.4-headerless-first-row",
        vector_score=float(item.vector_score),
        keyword_score=max(float(item.keyword_score), 0.90),
    )


def _enumeration_rows_for_seed(
    db: Session,
    seed: RetrievedChunk,
    *,
    row_limit: int,
) -> list[RetrievedChunk]:
    chunk = seed.chunk
    if not chunk.document_id:
        return []
    parent_key = ""
    # TextChunk intentionally stays lightweight and does not expose v5 parent_key.
    # Resolve the structural parent from the active v5 row by chunk id.
    if chunk.chunk_id and not str(chunk.chunk_id).startswith("v54-"):
        try:
            parent_key = str(db.execute(text("""
                SELECT c.parent_key
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r
                  ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                WHERE c.id=:chunk_id
                LIMIT 1
            """), {"chunk_id": chunk.chunk_id}).scalar_one_or_none() or "")
        except Exception:
            parent_key = ""
    start_page = max(1, int(chunk.page_number or 1) - 1)
    end_page = int(chunk.page_end or chunk.page_number or 1) + 1

    # Same-parent expansion reconstructs a complete logical table/section. Page-local
    # table expansion is a fallback for legacy chunks whose table was attached to the
    # wrong preceding heading during ingestion.
    rows = db.execute(text("""
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
               c.content_type, c.parent_key, c.section_path, c.heading,
               c.authority_status, c.text, d.filename
        FROM rag_v5_chunks c
        JOIN rag_v5_processing_runs r
          ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
        JOIN documents d
          ON d.id=c.document_id AND d.status='ready'
        WHERE c.document_id=:document_id
          AND (
              (:parent_key <> '' AND c.parent_key=:parent_key)
              OR (
                  c.content_type='table_row'
                  AND c.page_number BETWEEN :start_page AND :end_page
              )
          )
        ORDER BY c.page_number, c.chunk_index
        LIMIT :row_limit
    """), {
        "document_id": chunk.document_id,
        "parent_key": parent_key,
        "start_page": start_page,
        "end_page": end_page,
        "row_limit": row_limit,
    }).mappings()

    score = min(1.0, max(0.45, float(seed.score) * 0.97))
    output: list[RetrievedChunk] = []
    for row in rows:
        try:
            output.append(_row_to_result(
                row,
                score=score,
                method="v5.4-entity-enumeration-sibling",
                keyword=max(0.35, float(seed.keyword_score)),
            ))
        except Exception:
            # A malformed optional sibling should never break the user's whole query.
            continue
    return output


def expand_entity_enumeration_evidence(
    db: Session,
    *,
    question: str,
    interpretation: object | None,
    results: Sequence[RetrievedChunk],
    limit: int = 96,
) -> list[RetrievedChunk]:
    """Expand list/entity evidence by complete local semantic units.

    This is intentionally different from cross-RS/Line coverage. A membership/type/
    component/category query becomes complete by reconstructing governing lists/tables/
    sections in documents that already have a relevance signal, not by promoting one
    arbitrary document from every RS/Line filename family.
    """
    if completeness_policy(question, interpretation) != "entity_enumeration" or not results:
        return list(results)

    seed_limit = _int_env("RAG_V54_ENUMERATION_SEEDS", 18, 4, 48)
    per_seed = _int_env("RAG_V54_ENUMERATION_ROWS_PER_SEED", 80, 12, 240)
    min_score = _float_env("RAG_V54_ENUMERATION_SEED_SCORE", 0.24, 0.02, 0.90)
    useful_roles = {"governing", "definition", "applicability", "exception", "restriction", "authority"}

    ordered = sorted(
        results,
        key=lambda item: (-float(item.score), item.chunk.filename.casefold(), item.chunk.page_number),
    )
    seeds: list[RetrievedChunk] = []
    seen_units: set[tuple[str, str, int]] = set()
    for index, item in enumerate(ordered):
        role = _enum_seed_role(item.method)
        if index > 0 and float(item.score) < min_score and role not in useful_roles:
            continue
        chunk = item.chunk
        key = (
            str(chunk.document_id or chunk.filename),
            str(getattr(chunk, "parent_key", "") or ""),
            int(chunk.page_number or 0),
        )
        if key in seen_units:
            continue
        seen_units.add(key)
        seeds.append(item)
        if len(seeds) >= seed_limit:
            break

    expanded: list[RetrievedChunk] = list(results)
    for seed in seeds:
        try:
            expanded.extend(_enumeration_rows_for_seed(db, seed, row_limit=per_seed))
        except Exception:
            continue

    merged = _merge_results(expanded)

    # Recover source-grounded first rows lost by the legacy header assumption. One
    # recovery per physical table is enough; dedupe by document/page/column signature.
    recoveries: list[RetrievedChunk] = []
    recovery_keys: set[tuple[str, int, str]] = set()
    for item in merged:
        recovered = _headerless_first_row_recovery(item)
        if recovered is None:
            continue
        signature = _clean(_STRUCTURE_RE.sub("", recovered.chunk.text)).casefold()
        key = (recovered.chunk.document_id or recovered.chunk.filename, recovered.chunk.page_number, signature)
        if key in recovery_keys:
            continue
        recovery_keys.add(key)
        recoveries.append(recovered)

    # Preserve original ranked evidence first, then structural siblings/recoveries. The
    # later contributor-validation stage decides which documents may reach the answer.
    return _merge_results([*merged, *recoveries])[: max(12, int(limit))]
'''
    source = replace_once(
        source,
        "\n\ndef ensure_routed_document_evidence(\n",
        enumeration_helpers + "\n\ndef ensure_routed_document_evidence(\n",
        "v5.4 entity enumeration expansion",
    )

    return f"# {MARKER}\n" + source


def patch_synthesis_retrieval(source: str) -> str:
    if MARKER in source:
        return source
    if V53_MARKER not in source:
        raise RuntimeError(
            "backend/app/rag/v5/synthesis_retrieval.py is not the v5.3 coverage-first version."
        )

    old_import = '''    direct_retrieval_diagnostics,
    ensure_routed_document_evidence,
    exact_alias_results,
'''
    new_import = '''    completeness_policy,
    direct_retrieval_diagnostics,
    ensure_routed_document_evidence,
    exact_alias_results,
    expand_entity_enumeration_evidence,
'''
    source = replace_once(source, old_import, new_import, "v5.4 synthesis imports")

    rerank_anchor = "- Prefer governing/defining provisions over incidental mentions.\n"
    rerank_rule = (
        "- For entity/list enumeration questions (members, types, categories, components, roles, duties and similar), "
        "a document contributes only when it contains a defining/governing list, table, section or an authoritative qualifier. "
        "A document that merely mentions the requested entity is incidental and must not be retained just to increase document coverage.\n"
    )
    if rerank_rule not in source:
        source = replace_once(source, rerank_anchor, rerank_anchor + rerank_rule, "entity-enumeration rerank rule")

    coverage_anchor = 'A document is "contributing" only when it adds meaningful evidence. Do not include incidental lexical matches.\n'
    coverage_rule = (
        "For entity/list enumeration questions, a contributing document must actually define, enumerate, tabulate, assign or authoritatively qualify the requested items. "
        "A usage-only mention is not a contributor and must remain diagnostics-only.\n"
    )
    if coverage_rule not in source:
        source = replace_once(source, coverage_anchor, coverage_anchor + coverage_rule, "entity-enumeration coverage rule")

    old_final = '''    final = _final_diverse(combined, routes, final_limit)
    final = ensure_routed_document_evidence(final, combined, routes, final_limit)
'''
    new_final = '''    final = _final_diverse(combined, routes, final_limit)
    completeness = completeness_policy(original_question, interpretation)
    # One-per-route preservation is valuable only for genuine cross-scope operational
    # coverage. It is harmful for ordinary lists because a weak RS/Line route then gets
    # promoted into answer context even when it only mentions the subject.
    if completeness == "cross_scope_procedure":
        final = ensure_routed_document_evidence(final, combined, routes, final_limit)
    elif completeness == "entity_enumeration":
        final = expand_entity_enumeration_evidence(
            db,
            question=original_question,
            interpretation=interpretation,
            results=final,
            limit=max(final_limit, _int_env("RAG_V54_ENUMERATION_EVIDENCE", 96, 32, 160)),
        )
'''
    source = replace_once(source, old_final, new_final, "policy-aware final evidence preservation")

    select_replacement = r'''def select_results_for_answer(
    results: Sequence[RetrievedChunk],
    review: SynthesisCoverage,
) -> list[RetrievedChunk]:
    """Return only evidence from validated contributors for multi-document answers.

    v5.3 intentionally returned every routed result because it assumed reranking had
    already removed incidental documents. At corpus scale that assumption is too strong:
    a weak RS/Line route can survive and then become a mandatory answer heading. v5.4
    restores a contributor gate while preserving independently governing scope evidence.
    """
    values = list(results)
    if review.answer_strategy != "multi_document_synthesis" or not values:
        return values

    allowed = {name.casefold() for name in review.contributing_documents if str(name).strip()}
    for conflict in review.conflicts:
        for document in conflict.get("documents", []):
            if str(document).strip():
                allowed.add(str(document).casefold())

    # Deterministic defense-in-depth: a governing/definition/applicability/exception/
    # restriction/authority candidate is allowed even if the coverage critic omitted its
    # filename. Supporting/incidental candidates never become mandatory by themselves.
    strong_roles = {"governing", "definition", "applicability", "exception", "restriction", "authority", "conflict"}
    for item in values:
        match = re.search(r"v5\.2-synthesis:([a-z_]+)", str(item.method or ""), re.IGNORECASE)
        if match and match.group(1).casefold() in strong_roles:
            allowed.add(item.chunk.filename.casefold())

    if not allowed:
        # Coverage critic failure should degrade to a small strongest set, not a giant
        # cross-corpus dump and not an empty answer.
        return values[: min(16, len(values))]

    selected = [item for item in values if item.chunk.filename.casefold() in allowed]
    return selected or values[: min(16, len(values))]
'''
    source = replace_function(source, "select_results_for_answer", select_replacement, "coverage_prompt_status")

    coverage_status_replacement = r'''def coverage_prompt_status(
    base_status: str,
    bundle: SynthesisRetrievalBundle,
    review: SynthesisCoverage,
) -> str:
    summary = bundle.retrieval_diagnostic_summary or {}
    definition_docs = [str(value) for value in summary.get("definition_documents", [])]
    required_scope_docs = [str(value) for value in summary.get("required_scope_documents", [])]
    completeness = str(summary.get("coverage_policy") or "direct_lookup")
    if bundle.answer_strategy != "multi_document_synthesis" and not definition_docs:
        return base_status

    lines = [base_status, f"Answer strategy: {bundle.answer_strategy}", f"Completeness policy: {completeness}"]
    if definition_docs:
        lines.append("Definition enumeration: list every distinct supported meaning and show every supplied PDF/page source; do not collapse same meanings across PDFs.")
        lines.append("Definition source documents: " + "; ".join(definition_docs))

    if bundle.answer_strategy == "multi_document_synthesis":
        if completeness == "cross_scope_procedure":
            lines.append("Instruction: answer every materially applicable RS/Line scope represented by governing evidence; identical procedures in different scopes must still be shown separately with source citations.")
            if required_scope_docs:
                lines.append("Required RS/Line source documents for answer presentation: " + "; ".join(required_scope_docs))
        elif completeness == "entity_enumeration":
            lines.append("Instruction: enumerate the complete supported governing list/table/section. Include another document only when it independently defines, enumerates, assigns or authoritatively qualifies the requested items; usage-only mentions must not become answer sections.")
        else:
            lines.append("Instruction: synthesize the materially contributing governing evidence only; do not pad the answer with routed documents that merely mention the topic.")

        lines.append("Synthesis dimensions: " + ("; ".join(bundle.synthesis_dimensions) or "direct answer evidence"))
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
    return "\n".join(lines)
'''
    source = replace_function(source, "coverage_prompt_status", coverage_status_replacement, "response_synthesis_metadata")

    return f"# {MARKER}\n" + source


def patch_service(source: str) -> str:
    if MARKER in source:
        return source
    if V53_MARKER not in source:
        raise RuntimeError("backend/app/rag/v5/service.py is not the v5.3 coverage-first version.")

    anchor = '''- For acronym/full-form questions, preserve the exact expansion stated in the cited PDF source; do not
  paraphrase an official expansion into a plausible synonym.
'''
    addition = '''- For entity/list enumeration questions (members, types, categories, components, roles, duties and similar), enumerate every explicit item present in the supplied governing list/table/section. Do not create an RS/Line/PDF subsection merely because a source mentions the requested entity; another document belongs in the answer only when it independently defines, enumerates, assigns or materially qualifies the requested items.
- A source block marked as a v5.4 headerless-table recovery contains only text already present in PDF table metadata. Treat it as source-grounded recovery of the first data row, not as a model-created fact.
'''
    source = replace_once(source, anchor, anchor + addition, "v5.4 list answer contract")

    repair_anchor = "- For cross-scope conditional/procedural requests, use separate RS/Line/PDF headings for every required document supplied below, even when the procedures are identical.\n"
    repair_add = "- For entity/list enumeration, prefer a complete governing list/table/section and remove source sections that merely mention the entity without defining or enumerating the requested items.\n"
    if repair_anchor in source and repair_add not in source:
        source = source.replace(repair_anchor, repair_anchor + repair_add, 1)

    source = source.replace("rag-v5.3-coverage-first", "rag-v5.4-smart-completeness")
    return f"# {MARKER}\n" + source


def patch_layout(source: str) -> str:
    if MARKER in source:
        return source
    if "def _plumber_tables(" not in source:
        raise RuntimeError("backend/app/rag/v5/layout.py does not contain the expected v5 table parser.")

    helper = r'''

def _plumber_first_row_is_header(rows: list[list[str]]) -> bool:
    """Return False only when the first row is confidently a data row.

    Legacy v5 assumed rows[0] was always a header. Many official PDF tables are
    headerless and start `1 | <role> | <designation>`, so the first real item was lost.
    This conservative detector keeps legacy behavior unless it sees a serial sequence.
    """
    if len(rows) < 2 or not rows[0]:
        return True
    first = [_clean(cell) for cell in rows[0]]
    following = [[_clean(cell) for cell in row] for row in rows[1:4] if row]
    if not first or not following:
        return True

    def serial_number(value: str) -> int | None:
        match = re.fullmatch(r"\(?\s*(\d{1,3})\s*[.)]?\s*", _clean(value))
        return int(match.group(1)) if match else None

    first_serial = serial_number(first[0])
    next_serials = [serial_number(row[0]) for row in following if row]
    next_serials = [value for value in next_serials if value is not None]
    if first_serial is not None and next_serials:
        # N followed by N+1 is very strong evidence that row 0 is data, not a header.
        if next_serials[0] == first_serial + 1:
            return False
        if len(next_serials) >= 2 and next_serials[:2] == [first_serial + 1, first_serial + 2]:
            return False

    # If the first row has semantic header vocabulary and subsequent first cells are
    # serials, retain normal header behavior.
    first_folded = " ".join(first).casefold()
    if any(cue in first_folded for cue in _TABLE_HEADER_CUES):
        return True
    if first_serial is None and sum(value is not None for value in next_serials) >= 1:
        return True

    # Preserve existing behavior when uncertain.
    return True
'''
    source = replace_once(
        source,
        "\n\ndef _plumber_tables(plumber_page: Any, page_number: int) -> list[V5Table]:\n",
        helper + "\n\ndef _plumber_tables(plumber_page: Any, page_number: int) -> list[V5Table]:\n",
        "headerless table helper",
    )

    old_rows = '''            header = rows[0]
            columns = [cell or f"Column {index + 1}" for index, cell in enumerate(header)]
            data_rows = rows[1:] if len(rows) > 2 else rows
'''
    new_rows = '''            has_header = _plumber_first_row_is_header(rows)
            if has_header:
                header = rows[0]
                columns = [cell or f"Column {index + 1}" for index, cell in enumerate(header)]
                data_rows = rows[1:] if len(rows) > 2 else rows
            else:
                # Headerless source table: preserve the first real data row. Generic
                # column labels avoid inventing semantics not printed in the PDF.
                columns = [f"Column {index + 1}" for index in range(len(rows[0]))]
                data_rows = rows
'''
    source = replace_once(source, old_rows, new_rows, "preserve headerless first table row")
    return f"# {MARKER}\n" + source


def prepare(repo: Path, package: Path) -> tuple[dict[Path, str], list[tuple[Path, Path]]]:
    files = {
        repo / "backend/app/rag/v5/retrieval_completeness.py": patch_retrieval_completeness,
        repo / "backend/app/rag/v5/synthesis_retrieval.py": patch_synthesis_retrieval,
        repo / "backend/app/rag/v5/service.py": patch_service,
        repo / "backend/app/rag/v5/layout.py": patch_layout,
    }
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise SystemExit("Required runtime file(s) missing:\n" + "\n".join(missing))

    transformed: dict[Path, str] = {}
    for path, fn in files.items():
        transformed[path] = fn(path.read_text(encoding="utf-8-sig"))

    copies = [
        (package / "backend/tests/test_v54_smart_completeness.py", repo / "backend/tests/test_v54_smart_completeness.py"),
    ]
    return transformed, copies


def validate(transformed: dict[Path, str], copies: list[tuple[Path, Path]]) -> None:
    for path, content in transformed.items():
        if path.suffix == ".py":
            compile(content, str(path), "exec")
    for src, dst in copies:
        if dst.suffix == ".py":
            compile(src.read_text(encoding="utf-8"), str(dst), "exec")


def write(transformed: dict[Path, str], copies: list[tuple[Path, Path]], repo: Path) -> None:
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

    # Export exact post-patch replacement files so operators can inspect/copy them to a
    # second host without rerunning the transform script.
    snapshot = repo / "pdfrag-v5.4-replacement-files"
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
        "version": "rag-v5.4-smart-completeness",
        "replacement_files": [str(path.relative_to(repo)).replace("\\", "/") for path in transformed]
        + [str(dst.relative_to(repo)).replace("\\", "/") for _src, dst in copies],
        "notes": [
            "No database migration is required.",
            "Immediate query fixes use existing v5 chunks.",
            "layout.py fixes future/reprocessed headerless tables; reprocessing can be staged after acceptance testing.",
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[snapshot] {snapshot.relative_to(repo)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply IMS RAG v5.4 smart-completeness patch over v5.3.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    parser.add_argument("--check", action="store_true", help="preflight transforms/compile without writing")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    package = Path(__file__).resolve().parent
    transformed, copies = prepare(repo, package)
    validate(transformed, copies)
    print("[preflight] all transformed Python files compile in memory")
    if args.check:
        print("[check only] no repository files were changed")
        return 0

    write(transformed, copies, repo)
    for path in [
        repo / "backend/app/rag/v5/retrieval_completeness.py",
        repo / "backend/app/rag/v5/synthesis_retrieval.py",
        repo / "backend/app/rag/v5/service.py",
        repo / "backend/app/rag/v5/layout.py",
        repo / "backend/tests/test_v54_smart_completeness.py",
    ]:
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied IMS RAG v5.4 smart-completeness patch.")
    print(" - list/entity queries reconstruct complete governing tables/sections")
    print(" - list queries no longer trigger blanket RS/Line coverage promotion")
    print(" - operational procedure queries still preserve every applicable RS/Line scope")
    print(" - routed documents must pass contributor/governing-evidence validation before answer use")
    print(" - weak routed documents remain diagnostics-only instead of becoming forced answer headings")
    print(" - legacy headerless-table first rows can be recovered at query time")
    print(" - ingestion now preserves headerless first rows for future/reprocessed PDFs")
    print(" - answer-first and multi-source definition behavior from v5.3 is retained")
    print()
    print("No database migration or embedding rebuild is required.")
    print("Full PDF reprocessing is NOT required for the first retrieval test.")
    print("Reprocess documents later only if you want the table-ingestion correction persisted in stored chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
