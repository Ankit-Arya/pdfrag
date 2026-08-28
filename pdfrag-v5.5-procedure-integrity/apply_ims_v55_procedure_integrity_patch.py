from __future__ import annotations

import argparse
import json
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_RAG_V55_PROCEDURE_INTEGRITY"
V54_MARKER = "IMS_RAG_V54_SMART_COMPLETENESS"
BACKUP_SUFFIX = ".bak-before-ims-v55-procedure-integrity"


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


def insert_before_function(source: str, name: str, addition: str, label: str) -> str:
    anchor = f"\ndef {name}("
    index = source.find(anchor)
    if index < 0:
        raise RuntimeError(f"{label}: function {name} not found")
    return source[:index] + "\n" + addition.rstrip() + "\n" + source[index:]


def patch_retrieval_completeness(source: str) -> str:
    if MARKER in source:
        return source
    if V54_MARKER not in source:
        raise RuntimeError(
            "backend/app/rag/v5/retrieval_completeness.py is not the v5.4 smart-completeness version. "
            "Apply v5.4 first."
        )

    # -------------------------------------------------------------------------
    # 1) Multi-label, title/content-aware document scope profiles.
    # -------------------------------------------------------------------------
    scope_block = r'''# v5.5 scope extraction supports both numeric scopes and source-defined codes/named
# lines from document identity pages. It deliberately avoids arbitrary body-text scope
# mining: a rescue/reference paragraph inside one manual must not relabel that manual.
_RS_IDENTITY_SCOPE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?i:RS)\s*[-_. ]?\s*(?P<value>(?:\d{1,3}[A-Za-z]?|[A-Z][A-Z0-9-]{1,11}))(?![A-Za-z0-9])"
)
_NAMED_LINE_IDENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<name>[A-Z][A-Za-z0-9&/-]{2,30}(?:\s+[A-Z][A-Za-z0-9&/-]{2,30}){0,2})\s+(?i:Line)(?![A-Za-z0-9])"
)
_RS_QUERY_SCOPE_RE = re.compile(
    r"(?<![A-Za-z0-9])RS\s*[-_. ]?\s*(?P<value>(?:\d{1,3}[A-Za-z]?|[A-Za-z][A-Za-z0-9-]{1,11}))(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_NAMED_LINE_QUERY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<name>[A-Za-z][A-Za-z0-9&/-]{2,30}(?:\s+[A-Za-z][A-Za-z0-9&/-]{2,30}){0,2})\s+Line(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_GENERIC_NAMED_LINE_WORDS = {
    "main", "running", "stabling", "traffic", "test", "train", "revenue",
    "siding", "depot", "platform", "terminal", "track", "power", "brake",
}
_STRONG_ANSWER_ROLES = {
    "governing", "definition", "applicability", "exception", "restriction",
    "authority", "conflict",
}


def _scope_type_for_label(label: str) -> str:
    value = _clean(label)
    if value.upper().startswith("RS-"):
        return "rs"
    if value.lower().startswith("line-") or value.lower().endswith(" line"):
        return "line"
    return "common"


def scope_labels_from_text(value: str) -> list[str]:
    """Extract all plausible source identity scope labels from one title-like string."""
    text_value = str(value or "")
    labels: list[str] = []

    for match in _RS_IDENTITY_SCOPE_RE.finditer(text_value):
        raw = match.group("value")
        if raw[:1].isdigit():
            token = _canonical_scope_value(raw)
        else:
            token = raw.upper()
        labels.append(f"RS-{token}")

    for match in _LINE_SCOPE_RE.finditer(text_value):
        labels.append(f"Line-{_canonical_scope_value(match.group('value'))}")

    for match in _NAMED_LINE_IDENTITY_RE.finditer(text_value):
        name = _clean(match.group("name"))
        if not name or name.casefold() in _GENERIC_NAMED_LINE_WORDS:
            continue
        # A named line is kept in natural source order (e.g. "Airport Line") rather
        # than rewritten to a fabricated internal code.
        labels.append(f"{name} Line")

    return _unique(labels, 24)


def scope_from_text(value: str) -> tuple[str, str]:
    """Backward-compatible primary scope while v5.5 stores all labels separately."""
    labels = scope_labels_from_text(value)
    if not labels:
        return "common", "Common/Other"
    primary = labels[0]
    return _scope_type_for_label(primary), primary


def _document_scope_profiles(
    db: Session,
    filenames: dict[str, str],
) -> dict[str, list[str]]:
    """Build multi-label scope profiles from filename + first identity pages in one query.

    Only the first few pages are scanned, which captures cover/subject pages while
    avoiding false scope labels from later cross-references and rescue procedures.
    """
    if not filenames:
        return {}
    output: dict[str, list[str]] = {
        doc_id: scope_labels_from_text(filename)
        for doc_id, filename in filenames.items()
    }
    pages = _int_env("RAG_V55_SCOPE_PROFILE_PAGES", 4, 1, 8)
    try:
        rows = db.execute(text("""
            SELECT c.document_id, c.page_number, c.chunk_index, c.text
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r
              ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            WHERE c.document_id = ANY(CAST(:document_ids AS uuid[]))
              AND c.page_number <= :pages
            ORDER BY c.document_id, c.page_number, c.chunk_index
        """), {"document_ids": list(filenames), "pages": pages}).mappings()
    except Exception:
        return output

    for row in rows:
        doc_id = str(row["document_id"])
        output[doc_id] = _unique(
            [*output.get(doc_id, []), *scope_labels_from_text(str(row["text"]))],
            24,
        )
    return output


def explicit_scope_labels(question: str) -> list[str]:
    # Query scope parsing is case-insensitive; identity-page parsing above stays more
    # conservative to avoid turning ordinary body prose into document identity.
    value = str(question or "")
    labels = list(scope_labels_from_text(value))
    for match in _RS_QUERY_SCOPE_RE.finditer(value):
        raw = match.group("value")
        token = _canonical_scope_value(raw) if raw[:1].isdigit() else raw.upper()
        labels.append(f"RS-{token}")
    for match in _LINE_SCOPE_RE.finditer(value):
        labels.append(f"Line-{_canonical_scope_value(match.group('value'))}")
    for match in _NAMED_LINE_QUERY_RE.finditer(value):
        name = _clean(match.group("name"))
        words = [word.casefold() for word in name.split()]
        if words and all(word in _GENERIC_NAMED_LINE_WORDS for word in words):
            continue
        labels.append(" ".join(word.capitalize() for word in name.split()) + " Line")
    return _unique(labels, 24)
'''
    source = replace_function(source, "scope_from_text", scope_block, "explicit_scope_only")

    # Add content-aware scope profiles to unified discovery before explicit-scope pinning.
    scope_profile_anchor = '''    # Explicitly named scopes are resolved against the active document census even if
    # the first global vector/FTS arms did not surface them. This prevents a route cap
    # or crowd-out from excluding the user's named rolling stock/line before deep search.
    forced_explicit_ids: set[str] = set()
'''
    scope_profile_add = '''    # v5.5: one document can legitimately carry multiple identity scopes, and a
    # filename may omit the real rolling-stock/line name. Resolve source identity from
    # the cover/subject pages once per discovery request.
    active_rows = _active_documents(db) if explicit_scopes else []
    profile_filenames = {doc_id: state.filename for doc_id, state in states.items()}
    if explicit_scopes:
        profile_filenames.update({row["document_id"]: row["filename"] for row in active_rows})
    scope_profiles = _document_scope_profiles(db, profile_filenames)

'''
    source = replace_once(source, scope_profile_anchor, scope_profile_add + scope_profile_anchor, "v5.5 scope profile discovery")

    old_explicit = '''    if explicit_scopes:
        explicit_set = {label.casefold() for label in explicit_scopes}
        for row in _active_documents(db):
            scope_type, scope_label = scope_from_text(row["filename"])
            if scope_label.casefold() not in explicit_set:
                continue
            doc_id = row["document_id"]
            state = _state_for(states, doc_id, row["filename"])
            state.methods.add("explicit_scope")
            forced_explicit_ids.add(doc_id)
'''
    new_explicit = '''    if explicit_scopes:
        explicit_set = {label.casefold() for label in explicit_scopes}
        for row in active_rows:
            doc_id = row["document_id"]
            labels = scope_profiles.get(doc_id) or scope_labels_from_text(row["filename"])
            if not any(label.casefold() in explicit_set for label in labels):
                continue
            state = _state_for(states, doc_id, row["filename"])
            state.methods.add("explicit_scope")
            forced_explicit_ids.add(doc_id)
'''
    source = replace_once(source, old_explicit, new_explicit, "v5.5 explicit multi-scope pinning")

    old_group = '''        for doc_id in ordered_ids:
            scope_type, scope_label = scope_from_text(states[doc_id].filename)
            if scope_type not in {"rs", "line"}:
                continue
            strongest = max(states[doc_id].vector_score, states[doc_id].keyword_score)
            score = scored[doc_id]
            has_signal = strongest >= scope_min_signal or bool(states[doc_id].query_hits)
            if has_signal:
                scope_signal_labels.add(scope_label)
            if doc_id in forced_explicit_ids or (score >= scope_min_score and has_signal):
                grouped[scope_label].append(doc_id)
'''
    new_group = '''        for doc_id in ordered_ids:
            labels = scope_profiles.get(doc_id) or scope_labels_from_text(states[doc_id].filename)
            strongest = max(states[doc_id].vector_score, states[doc_id].keyword_score)
            score = scored[doc_id]
            has_signal = strongest >= scope_min_signal or bool(states[doc_id].query_hits)
            for scope_label in labels:
                if _scope_type_for_label(scope_label) not in {"rs", "line"}:
                    continue
                if has_signal:
                    scope_signal_labels.add(scope_label)
                if doc_id in forced_explicit_ids or (score >= scope_min_score and has_signal):
                    grouped[scope_label].append(doc_id)
'''
    source = replace_once(source, old_group, new_group, "v5.5 multi-label coverage grouping")

    old_diag_scope = '''        scope_type, scope_label = scope_from_text(state.filename)
        explicit_pinned = doc_id in forced_explicit_ids
'''
    new_diag_scope = '''        scope_labels = scope_profiles.get(doc_id) or scope_labels_from_text(state.filename)
        if scope_labels:
            scope_label = scope_labels[0]
            scope_type = _scope_type_for_label(scope_label)
        else:
            scope_type, scope_label = "common", "Common/Other"
        explicit_pinned = doc_id in forced_explicit_ids
'''
    source = replace_once(source, old_diag_scope, new_diag_scope, "v5.5 diagnostic scope profile")
    source = replace_once(
        source,
        '            "scope_label": scope_label,\n            "scope_pinned": bool(explicit_pinned or scope_pinned),\n',
        '            "scope_label": scope_label,\n            "scope_labels": list(scope_labels),\n            "scope_pinned": bool(explicit_pinned or scope_pinned),\n',
        "v5.5 diagnostic multi-scope labels",
    )

    # -------------------------------------------------------------------------
    # 2) Balanced deep retrieval: top-N semantic/lexical evidence PER ROUTED DOC.
    #    This removes the global top-K crowd-out failure without N*Q DB calls.
    # -------------------------------------------------------------------------
    balanced_helpers = r'''
def _per_document_vector_rows_scoped(
    db: Session,
    query_vector: list[float],
    document_ids: Sequence[str],
    *,
    per_document: int,
    max_rows: int,
) -> list[object]:
    if not document_ids:
        return []
    return list(db.execute(text("""
        WITH ranked AS (
            SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                   c.content_type, c.parent_key, c.section_path, c.heading,
                   c.authority_status, c.text, d.filename,
                   GREATEST(0.0, 1 - (c.embedding <=> CAST(:embedding AS vector))) AS vector_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.document_id
                       ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id
                   ) AS doc_rank
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r
              ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            JOIN documents d ON d.id=c.document_id AND d.status='ready'
            WHERE c.document_id = ANY(CAST(:document_ids AS uuid[]))
        )
        SELECT id, document_id, chunk_index, page_number, page_end, content_type,
               parent_key, section_path, heading, authority_status, text, filename, vector_score
        FROM ranked
        WHERE doc_rank <= :per_document
        ORDER BY vector_score DESC, lower(filename), page_number, chunk_index
        LIMIT :max_rows
    """), {
        "embedding": str(query_vector),
        "document_ids": list(document_ids),
        "per_document": per_document,
        "max_rows": max_rows,
    }).mappings())


def _per_document_fts_rows_scoped(
    db: Session,
    query: str,
    document_ids: Sequence[str],
    *,
    per_document: int,
    max_rows: int,
) -> list[object]:
    if not document_ids:
        return []
    return list(db.execute(text("""
        WITH q AS (
            SELECT websearch_to_tsquery('simple', :query) AS simple_q,
                   websearch_to_tsquery('english', :query) AS english_q
        ), hits AS (
            SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                   c.content_type, c.parent_key, c.section_path, c.heading,
                   c.authority_status, c.text, d.filename,
                   GREATEST(
                       ts_rank_cd(to_tsvector('simple', c.text), q.simple_q),
                       ts_rank_cd(to_tsvector('english', c.text), q.english_q) * 0.9
                   ) AS keyword_score
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r
              ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            JOIN documents d ON d.id=c.document_id AND d.status='ready'
            CROSS JOIN q
            WHERE c.document_id = ANY(CAST(:document_ids AS uuid[]))
              AND (
                  to_tsvector('simple', c.text) @@ q.simple_q
                  OR to_tsvector('english', c.text) @@ q.english_q
              )
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY document_id
                ORDER BY keyword_score DESC, page_number, chunk_index
            ) AS doc_rank
            FROM hits
        )
        SELECT id, document_id, chunk_index, page_number, page_end, content_type,
               parent_key, section_path, heading, authority_status, text, filename, keyword_score
        FROM ranked
        WHERE doc_rank <= :per_document
        ORDER BY keyword_score DESC, lower(filename), page_number, chunk_index
        LIMIT :max_rows
    """), {
        "query": query,
        "document_ids": list(document_ids),
        "per_document": per_document,
        "max_rows": max_rows,
    }).mappings())


def balanced_routed_candidates(
    db: Session,
    routes: Sequence[DocumentRoute],
    queries: Sequence[str],
) -> list[RetrievedChunk]:
    """Return a bounded deep-search pool with candidate slots per routed document.

    Existing `_scoped_candidates` uses a global top-K over all routed document IDs. A
    long/highly lexical PDF can therefore consume the pool and starve another already
    routed PDF. v5.5 performs partitioned ranking in SQL, so each route gets its own
    semantic/lexical slots while the number of DB round-trips stays O(query variants),
    not O(documents * variants).
    """
    if not routes or not queries:
        return []
    document_ids = [route.document_id for route in routes]
    route_scores = {route.document_id: float(route.score) for route in routes}
    query_limit = _int_env("RAG_V55_BALANCED_QUERY_COUNT", 4, 1, 8)
    per_document = _int_env("RAG_V55_BALANCED_PER_DOCUMENT", 6, 2, 16)
    selected_queries = [_clean(value) for value in queries[:query_limit] if _clean(value)]
    if not selected_queries:
        return []

    output: list[RetrievedChunk] = []
    try:
        vectors = embedding_service.encode(selected_queries)
    except Exception:
        vectors = []

    max_rows = min(5000, max(64, len(document_ids) * per_document))
    for query_index, query in enumerate(selected_queries):
        if query_index < len(vectors):
            try:
                rows = _per_document_vector_rows_scoped(
                    db,
                    vectors[query_index].tolist(),
                    document_ids,
                    per_document=per_document,
                    max_rows=max_rows,
                )
            except Exception:
                rows = []
            for row in rows:
                signal = max(0.0, min(1.0, float(row["vector_score"] or 0.0)))
                route_score = route_scores.get(str(row["document_id"]), 0.0)
                score = min(1.0, 0.70 * signal + 0.20 * route_score + 0.10)
                output.append(_row_to_result(
                    row,
                    score=score,
                    method=f"v5.5-balanced-vector-{query_index}",
                    vector=signal,
                ))

        try:
            rows = _per_document_fts_rows_scoped(
                db,
                query,
                document_ids,
                per_document=per_document,
                max_rows=max_rows,
            )
        except Exception:
            rows = []
        for row in rows:
            signal = min(1.0, max(0.0, float(row["keyword_score"] or 0.0)) * 4.5)
            route_score = route_scores.get(str(row["document_id"]), 0.0)
            score = min(1.0, 0.70 * signal + 0.20 * route_score + 0.10)
            output.append(_row_to_result(
                row,
                score=score,
                method=f"v5.5-balanced-fts-{query_index}",
                keyword=signal,
            ))
    return _merge_results(output)
'''
    source = insert_before_function(source, "_state_for", balanced_helpers, "v5.5 balanced route retrieval")

    # -------------------------------------------------------------------------
    # 3) Procedure structure expansion + source-grounded table aggregates.
    # -------------------------------------------------------------------------
    procedure_helpers = r'''
def _evidence_role(method: str) -> str:
    matches = re.findall(r"v5\.2-synthesis:([a-z_]+)", str(method or ""), re.IGNORECASE)
    for value in matches:
        role = value.casefold()
        if role in _STRONG_ANSWER_ROLES:
            return role
    return matches[-1].casefold() if matches else ""


def _procedure_seed_record(db: Session, chunk_id: str) -> dict[str, object] | None:
    if not chunk_id or str(chunk_id).startswith(("v54-", "v55-")):
        return None
    try:
        row = db.execute(text("""
            SELECT c.id, c.document_id, c.table_id, c.parent_key,
                   c.page_number, c.page_end, c.content_type
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r
              ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            WHERE c.id=CAST(:chunk_id AS uuid)
            LIMIT 1
        """), {"chunk_id": chunk_id}).mappings().first()
    except Exception:
        return None
    return dict(row) if row else None


def _procedure_structure_rows(
    db: Session,
    seed: RetrievedChunk,
    record: dict[str, object],
) -> tuple[str, list[object]]:
    document_id = str(record.get("document_id") or seed.chunk.document_id or "")
    table_id = str(record.get("table_id") or "")
    parent_key = str(record.get("parent_key") or "")
    page = int(record.get("page_number") or seed.chunk.page_number or 1)
    if not document_id:
        return "none", []

    common_select = """
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
               c.content_type, c.parent_key, c.section_path, c.heading,
               c.authority_status, c.text, d.filename
        FROM rag_v5_chunks c
        JOIN rag_v5_processing_runs r
          ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
        JOIN documents d ON d.id=c.document_id AND d.status='ready'
    """

    if table_id:
        limit = _int_env("RAG_V55_PROCEDURE_TABLE_ROWS", 200, 16, 500)
        try:
            rows = list(db.execute(text(common_select + """
                WHERE c.table_id=CAST(:table_id AS uuid)
                ORDER BY c.table_row_index NULLS LAST, c.chunk_index
                LIMIT :limit
            """), {"table_id": table_id, "limit": limit}).mappings())
        except Exception:
            rows = []
        if rows:
            return "table", rows

    if parent_key:
        limit = _int_env("RAG_V55_PROCEDURE_SECTION_CHUNKS", 48, 8, 120)
        try:
            rows = list(db.execute(text(common_select + """
                WHERE c.document_id=CAST(:document_id AS uuid)
                  AND c.parent_key=:parent_key
                ORDER BY c.chunk_index
                LIMIT :limit
            """), {
                "document_id": document_id,
                "parent_key": parent_key,
                "limit": limit,
            }).mappings())
        except Exception:
            rows = []
        if rows:
            return "section", rows

    # Legacy fallback: a useful table row can have a bad/missing parent heading. Keep
    # the fallback local to adjacent pages so unrelated tables are never swept in.
    limit = _int_env("RAG_V55_PROCEDURE_PAGE_ROWS", 48, 8, 120)
    try:
        rows = list(db.execute(text(common_select + """
            WHERE c.document_id=CAST(:document_id AS uuid)
              AND c.content_type='table_row'
              AND c.page_number BETWEEN :start_page AND :end_page
            ORDER BY c.page_number, c.chunk_index
            LIMIT :limit
        """), {
            "document_id": document_id,
            "start_page": max(1, page - 1),
            "end_page": page + 1,
            "limit": limit,
        }).mappings())
    except Exception:
        rows = []
    return ("page_table", rows) if rows else ("none", [])


def _procedure_aggregate_chunks(
    seed: RetrievedChunk,
    rows: Sequence[RetrievedChunk],
    *,
    structure_kind: str,
    role: str,
) -> list[RetrievedChunk]:
    """Pack a complete governing structure without changing source facts.

    Each branch remains explicitly delimited. This prevents the answer model from
    pairing a condition in one table row with the action from a neighboring row.
    Long structures are split only at row/chunk boundaries.
    """
    if not rows:
        return []
    max_chars = _int_env("RAG_V55_PROCEDURE_AGGREGATE_CHARS", 18000, 4000, 40000)
    groups: list[list[RetrievedChunk]] = []
    current: list[RetrievedChunk] = []
    chars = 0
    for item in rows:
        body = _STRUCTURE_RE.sub("", item.chunk.text).strip()
        cost = len(body) + 120
        if current and chars + cost > max_chars:
            groups.append(current)
            current = []
            chars = 0
        current.append(item)
        chars += cost
    if current:
        groups.append(current)

    output: list[RetrievedChunk] = []
    safe_role = role if role in _STRONG_ANSWER_ROLES else "supporting"
    for group_index, group in enumerate(groups, 1):
        page_start = min(item.chunk.page_number for item in group)
        page_end = max(int(item.chunk.page_end or item.chunk.page_number) for item in group)
        branch_blocks: list[str] = []
        for branch_index, item in enumerate(group, 1):
            body = _STRUCTURE_RE.sub("", item.chunk.text).strip()
            branch_blocks.append(
                f"BRANCH {branch_index} (source page {item.chunk.page_number}):\n{body}"
            )
        text_value = (
            "[PDF STRUCTURE]\n"
            f"File: {seed.chunk.filename}\n"
            f"Pages: {page_start}-{page_end}\n"
            f"Section path: {' > '.join(seed.chunk.section_path) if seed.chunk.section_path else (seed.chunk.heading or 'Unsectioned content')}\n"
            f"Content type: procedure_{structure_kind}\n"
            f"Heading: {seed.chunk.heading or (seed.chunk.section_path[-1] if seed.chunk.section_path else '')}\n"
            "[/PDF STRUCTURE]\n\n"
            "[SOURCE-GROUNDED PROCEDURE STRUCTURE]\n"
            "Rows/branches below are kept in source order. Each branch is independent: "
            "pair its condition only with the action written in that same branch.\n\n"
            + "\n\n".join(branch_blocks)
        )
        score = min(1.0, max(0.88 if safe_role != "supporting" else 0.70, float(seed.score)))
        output.append(RetrievedChunk(
            chunk=TextChunk(
                chunk_id=f"v55-procedure:{seed.chunk.chunk_id}:{structure_kind}:{group_index}",
                filename=seed.chunk.filename,
                page_number=page_start,
                page_end=page_end,
                text=text_value,
                content_type=f"procedure_{structure_kind}",
                section_path=seed.chunk.section_path,
                heading=seed.chunk.heading,
                document_id=seed.chunk.document_id,
                chunk_index=seed.chunk.chunk_index,
            ),
            score=score,
            method=f"v5.5-procedure-{structure_kind}-complete+v5.2-synthesis:{safe_role}",
            vector_score=float(seed.vector_score),
            keyword_score=float(seed.keyword_score),
        ))
    return output


def expand_cross_scope_procedure_evidence(
    db: Session,
    *,
    question: str,
    interpretation: object | None,
    results: Sequence[RetrievedChunk],
    limit: int = 180,
) -> list[RetrievedChunk]:
    """Expand governing procedural seeds to complete table/section branch structures."""
    if completeness_policy(question, interpretation) != "cross_scope_procedure" or not results:
        return list(results)

    seed_limit = _int_env("RAG_V55_PROCEDURE_SEEDS", 36, 6, 80)
    per_doc_seed = _int_env("RAG_V55_PROCEDURE_SEEDS_PER_DOCUMENT", 3, 1, 6)
    fallback_score = _float_env("RAG_V55_PROCEDURE_TABLE_SEED_SCORE", 0.58, 0.20, 0.95)
    ordered = sorted(
        results,
        key=lambda item: (-float(item.score), item.chunk.filename.casefold(), item.chunk.page_number),
    )
    doc_counts: defaultdict[str, int] = defaultdict(int)
    seen_units: set[tuple[str, str, str]] = set()
    additions: list[RetrievedChunk] = []

    for item in ordered:
        role = _evidence_role(item.method)
        is_strong = role in _STRONG_ANSWER_ROLES
        if not is_strong and not (
            item.chunk.content_type == "table_row" and float(item.score) >= fallback_score
        ):
            continue
        doc_key = item.chunk.document_id or item.chunk.filename
        if doc_counts[doc_key] >= per_doc_seed:
            continue
        record = _procedure_seed_record(db, item.chunk.chunk_id)
        if not record:
            continue
        table_id = str(record.get("table_id") or "")
        parent_key = str(record.get("parent_key") or "")
        unit_key = (doc_key, table_id, parent_key if not table_id else "")
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)
        doc_counts[doc_key] += 1

        kind, raw_rows = _procedure_structure_rows(db, item, record)
        if not raw_rows:
            continue
        inherited_role = role if role in _STRONG_ANSWER_ROLES else "supporting"
        structure_rows: list[RetrievedChunk] = []
        for position, row in enumerate(raw_rows):
            try:
                structure_rows.append(_row_to_result(
                    row,
                    score=max(0.62, float(item.score) - 0.004 * position),
                    method=f"v5.5-procedure-{kind}-sibling+v5.2-synthesis:{inherited_role}",
                    vector=float(item.vector_score),
                    keyword=float(item.keyword_score),
                ))
            except Exception:
                continue
        additions.extend(structure_rows)
        additions.extend(_procedure_aggregate_chunks(
            item,
            structure_rows,
            structure_kind=kind,
            role=inherited_role,
        ))
        if len(seen_units) >= seed_limit:
            break

    return _merge_results([*results, *additions])[: max(32, int(limit))]
'''
    source = insert_before_function(source, "ensure_routed_document_evidence", procedure_helpers, "v5.5 procedure structure expansion")

    # -------------------------------------------------------------------------
    # 4) Required-answer boundary: contributing + strong governing role only.
    # -------------------------------------------------------------------------
    required_replacement = r'''def _diagnostic_scope_labels(item: dict[str, object]) -> list[str]:
    raw = item.get("scope_labels")
    if isinstance(raw, list):
        values = [str(value) for value in raw if str(value).strip()]
        if values:
            return _unique(values, 24)
    label = _clean(item.get("scope_label"))
    return [label] if label and label != "Common/Other" else []


def _diagnostic_strong_role(item: dict[str, object]) -> bool:
    return str(item.get("rerank_role") or "").casefold() in _STRONG_ANSWER_ROLES


def required_coverage_documents(
    diagnostics: Sequence[dict[str, object]],
    *,
    include_definitions: bool = True,
) -> list[str]:
    """Documents that the answer policy may REQUIRE, not merely inspect.

    v5.3/v5.4 used `final_evidence` as the boundary. That made a routed review sample
    mandatory even when the coverage critic explicitly rejected the document. v5.5
    requires contributor validation plus a governing/definition/applicability/etc role.
    """
    values: list[str] = []
    for item in diagnostics:
        filename = _clean(item.get("filename"))
        if not filename:
            continue
        if include_definitions and item.get("decision") == "EXACT_ALIAS_DEFINITION":
            values.append(filename)
            continue
        if not bool(item.get("contributing")) or not bool(item.get("final_evidence")):
            continue
        if not _diagnostic_strong_role(item):
            continue
        values.append(filename)
    return _unique(values, 120)


def enrich_retrieval_summary(
    summary: dict[str, object],
    diagnostics: Sequence[dict[str, object]],
) -> dict[str, object]:
    updated = dict(summary)
    required = required_coverage_documents(diagnostics)
    required_folded = {value.casefold() for value in required}
    updated["required_scope_documents"] = [
        value for value in required
        if any(
            str(item.get("filename") or "").casefold() == value.casefold()
            and any(_scope_type_for_label(label) in {"rs", "line"} for label in _diagnostic_scope_labels(item))
            for item in diagnostics
        )
    ]
    updated["required_answer_documents"] = required
    updated["required_scope_labels"] = _unique(
        (
            label
            for item in diagnostics
            if str(item.get("filename") or "").casefold() in required_folded
            for label in _diagnostic_scope_labels(item)
            if _scope_type_for_label(label) in {"rs", "line"}
        ),
        120,
    )
    updated["scope_final_evidence_documents"] = len(updated["required_scope_documents"])
    updated["answer_eligible_documents"] = len(required)
    return updated
'''
    source = replace_function(source, "required_coverage_documents", required_replacement, "response_diagnostics")

    response_replacement = r'''def response_diagnostics(
    diagnostics: Sequence[dict[str, object]],
    summary: dict[str, object],
    contributing_documents: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    contributing = {name.casefold() for name in contributing_documents}
    output: list[dict[str, object]] = []
    for raw in diagnostics:
        item = dict(raw)
        filename = str(item.get("filename") or "")
        is_contributing = filename.casefold() in contributing
        item["contributing"] = is_contributing
        exact_definition = item.get("decision") == "EXACT_ALIAS_DEFINITION"
        answer_eligible = bool(
            exact_definition
            or (
                is_contributing
                and bool(item.get("final_evidence"))
                and _diagnostic_strong_role(item)
            )
        )
        item["answer_eligible"] = answer_eligible
        if is_contributing and answer_eligible:
            item["decision"] = "CONTRIBUTING_GOVERNING_EVIDENCE"
            item["reason"] = "Coverage review validated this document and its final evidence is governing/defining/applicable to the answer."
        elif is_contributing:
            item["decision"] = "CONTRIBUTING_REVIEW_ONLY"
            item["reason"] = "Coverage review found context here, but no strong governing final-evidence role makes it a mandatory answer scope."
        elif bool(item.get("deep_searched")):
            item["decision"] = "REVIEWED_NON_CONTRIBUTOR"
            item["reason"] = "The document was deep-searched for coverage but was not validated as a contributing answer source."
        output.append(item)

    updated = enrich_retrieval_summary(dict(summary), output)
    updated["deep_searched_documents"] = sum(1 for item in output if bool(item.get("deep_searched")))
    updated["final_evidence_documents"] = sum(1 for item in output if bool(item.get("final_evidence")))
    updated["contributing_documents"] = sum(1 for item in output if bool(item.get("contributing")))
    updated["answer_eligible_documents"] = sum(1 for item in output if bool(item.get("answer_eligible")))
    return output, updated
'''
    start = source.find("def response_diagnostics(")
    if start < 0:
        raise RuntimeError("response_diagnostics not found")
    # response_diagnostics is the last function in the v5.4 module.
    source = source[:start] + response_replacement.rstrip() + "\n"

    return f"# {MARKER}\n" + source


def patch_synthesis_retrieval(source: str) -> str:
    if MARKER in source:
        return source
    if V54_MARKER not in source:
        raise RuntimeError(
            "backend/app/rag/v5/synthesis_retrieval.py is not the v5.4 smart-completeness version."
        )

    old_import = '''    completeness_policy,
    direct_retrieval_diagnostics,
    ensure_routed_document_evidence,
    exact_alias_results,
    expand_entity_enumeration_evidence,
'''
    new_import = '''    balanced_routed_candidates,
    completeness_policy,
    direct_retrieval_diagnostics,
    ensure_routed_document_evidence,
    exact_alias_results,
    expand_cross_scope_procedure_evidence,
    expand_entity_enumeration_evidence,
'''
    source = replace_once(source, old_import, new_import, "v5.5 synthesis retrieval imports")

    # Review-only evidence is separate from answer candidates. Existing constructor calls
    # remain source compatible because the field has a default.
    source = replace_once(
        source,
        "    search_round: int = 1\n",
        "    search_round: int = 1\n    review_results: list[RetrievedChunk] | None = None\n",
        "v5.5 review evidence field",
    )

    # Numeric threshold/unit integrity belongs in evidence selection, not just final prose.
    rerank_anchor = "- Preserve scope differences (line, location, mode, equipment, person type, scenario) instead of flattening them.\n"
    rerank_add = (
        "- For tables/conditional ladders, keep each condition and its action from the SAME source row/branch. Never attach an action/value from a neighboring branch to a different threshold.\n"
        "- Measurement bases are not interchangeable. A percentage/count of one source-defined unit must not be converted to another unit unless the supplied evidence explicitly defines that equivalence.\n"
    )
    if rerank_add not in source:
        source = replace_once(source, rerank_anchor, rerank_anchor + rerank_add, "v5.5 condition-action rerank contract")

    coverage_anchor = "- whether different documents are complementary, scope-specific, authority-related, or genuinely unresolved conflicts.\n"
    coverage_add = (
        "- for a threshold/table procedure, whether the visible governing branch ladder is complete enough to pair every retained condition with its own action;\n"
        "- whether numeric/percentage/count conditions use the same source-defined measurement basis as the user's wording; do not infer a conversion that is not in evidence.\n"
    )
    if coverage_add not in source:
        source = replace_once(source, coverage_anchor, coverage_anchor + coverage_add, "v5.5 branch completeness coverage contract")

    # Add a generic synthesis dimension for threshold-bearing questions.
    dimension_anchor = '''    if interpretation.authority_sensitive:
        values.append("current authority, amendment or supersession status")
'''
    dimension_add = '''    if re.search(r"(?:\\b\\d+(?:\\.\\d+)?\\s*%|\\b(?:up\\s+to|more\\s+than|less\\s+than|at\\s+least|at\\s+most|between)\\b)", folded):
        values.append("condition quantity/unit and same-branch threshold-to-action pairing")
'''
    if dimension_add not in source:
        source = replace_once(source, dimension_anchor, dimension_add + dimension_anchor, "v5.5 threshold dimension")

    # The old global scoped arm remains a useful signal. Add a partitioned per-document
    # arm so every already-routed PDF gets its own deep-search slots.
    old_candidates = '''    scoped = _scoped_candidates(db, routes, queries)
    headings = _heading_candidates(db, routes, queries, interpretation)
    candidates = _merge_results([
        *baseline_results,
        *headings,
        *scoped,
    ])
    candidate_count = len(candidates)

    reranked = _synthesis_rerank(interpretation, plan, candidates, routes)
    expanded = _expand_sections(db, interpretation, reranked)
    combined = _merge_results([*reranked, *expanded])
'''
    new_candidates = '''    scoped = _scoped_candidates(db, routes, queries)
    balanced_scoped = balanced_routed_candidates(db, routes, queries)
    headings = _heading_candidates(db, routes, queries, interpretation)
    candidates = _merge_results([
        *baseline_results,
        *headings,
        *scoped,
        *balanced_scoped,
    ])
    candidate_count = len(candidates)

    reranked = _synthesis_rerank(interpretation, plan, candidates, routes)
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
    source = replace_once(source, old_candidates, new_candidates, "v5.5 balanced + structural procedure retrieval")

    final_helpers = r'''
def _v55_method_role(method: str) -> str:
    matches = re.findall(r"v5\.2-synthesis:([a-z_]+)", str(method or ""), re.IGNORECASE)
    return matches[-1].casefold() if matches else ""


def _v55_procedure_final(
    candidates: Sequence[RetrievedChunk],
    routes: Sequence[DocumentRoute],
    limit: int,
) -> list[RetrievedChunk]:
    """Seed strong governing evidence per route, never arbitrary route evidence."""
    strong = {"governing", "definition", "applicability", "exception", "restriction", "authority", "conflict"}
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in candidates:
        by_doc[item.chunk.document_id or item.chunk.filename].append(item)
    for values in by_doc.values():
        values.sort(key=lambda item: -float(item.score))

    output: list[RetrievedChunk] = []
    seen: set[str] = set()
    for route in routes:
        eligible = [
            item for item in by_doc.get(route.document_id, [])
            if _v55_method_role(item.method) in strong
        ]
        if not eligible:
            continue
        # Prefer the complete procedure aggregate when present; otherwise top strong row.
        eligible.sort(key=lambda item: (
            0 if "v5.5-procedure-" in item.method and "-complete" in item.method else 1,
            -float(item.score),
            item.chunk.page_number,
        ))
        item = eligible[0]
        seen.add(item.chunk.chunk_id)
        output.append(item)
        if len(output) >= limit:
            return output

    for item in _final_diverse(candidates, routes, limit):
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        output.append(item)
        if len(output) >= limit:
            break
    return output
'''
    source = insert_before_function(source, "retrieve_assistant_v52", final_helpers, "v5.5 strong procedure final selection")

    old_final = '''    final_limit = _int_env("RAG_V52_FINAL_CANDIDATES", 96, 32, 140)
    final = _final_diverse(combined, routes, final_limit)
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
    new_final = '''    final_limit = _int_env("RAG_V52_FINAL_CANDIDATES", 96, 32, 140)
    if completeness == "cross_scope_procedure":
        # Answer candidates contain only naturally/strongly surviving evidence. Route
        # coverage samples are kept separately for the critic below, so a weak route can
        # never become a mandatory answer heading just because it was inspected.
        final = _v55_procedure_final(combined, routes, final_limit)
        review_results = ensure_routed_document_evidence(
            final,
            combined,
            routes,
            _int_env("RAG_V55_REVIEW_CANDIDATES", 120, 48, 220),
        )
    else:
        final = _final_diverse(combined, routes, final_limit)
        review_results = list(final)
        if completeness == "entity_enumeration":
            final = expand_entity_enumeration_evidence(
                db,
                question=original_question,
                interpretation=interpretation,
                results=final,
                limit=max(final_limit, _int_env("RAG_V54_ENUMERATION_EVIDENCE", 96, 32, 160)),
            )
            review_results = list(final)
'''
    source = replace_once(source, old_final, new_final, "v5.5 separate review and answer evidence")

    # Attach review pool only to the final multi-document constructor.
    source = replace_once(
        source,
        '''    return SynthesisRetrievalBundle(
        results=final,
''',
        '''    return SynthesisRetrievalBundle(
        results=final,
        review_results=review_results,
''',
        "v5.5 final bundle review evidence",
    )

    # Coverage critic must inspect at least one candidate from every routed document
    # before any document receives a second slot. This is separate from answer evidence.
    old_coverage_pool = '''    # Use a document-balanced pool so a long/repetitive PDF cannot dominate coverage review.
    # Build a document-balanced review sample so a long PDF cannot dominate coverage review.
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in bundle.results:
        by_doc[item.chunk.filename].append(item)
    candidates: list[RetrievedChunk] = []
    seen: set[str] = set()
    for filename in bundle.evidence_documents:
        for item in by_doc.get(filename, [])[:2]:
            if item.chunk.chunk_id not in seen:
                seen.add(item.chunk.chunk_id)
                candidates.append(item)
                if len(candidates) >= limit:
                    break
        if len(candidates) >= limit:
            break
    for item in bundle.results:
        if len(candidates) >= limit:
            break
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        candidates.append(item)
'''
    new_coverage_pool = '''    review_source = bundle.review_results or bundle.results
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in review_source:
        by_doc[item.chunk.filename].append(item)
    for values in by_doc.values():
        values.sort(key=lambda item: -float(item.score))

    candidates: list[RetrievedChunk] = []
    seen: set[str] = set()
    route_order = list(bundle.considered_documents or bundle.routed_documents or bundle.evidence_documents)

    # First pass: one review candidate per routed document.
    for filename in route_order:
        values = by_doc.get(filename, [])
        if not values:
            continue
        item = values[0]
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        candidates.append(item)
        if len(candidates) >= limit:
            break

    # Second pass: give routed documents a second candidate when capacity remains.
    if len(candidates) < limit:
        for filename in route_order:
            for item in by_doc.get(filename, [])[1:2]:
                if item.chunk.chunk_id in seen:
                    continue
                seen.add(item.chunk.chunk_id)
                candidates.append(item)
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break

    # Final fill: strongest remaining review evidence.
    for item in review_source:
        if len(candidates) >= limit:
            break
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        candidates.append(item)
'''
    source = replace_once(source, old_coverage_pool, new_coverage_pool, "v5.5 route-complete coverage review pool")

    source = replace_once(
        source,
        "    return review\n\n\ndef select_results_for_answer(\n",
        "    # v5.5: materialize coverage-critic contributor decisions into bundle diagnostics/summary before answer prompting.\n"
        "    if getattr(bundle, \"retrieval_diagnostics\", None):\n"
        "        updated_diagnostics, updated_summary = response_diagnostics(\n"
        "            bundle.retrieval_diagnostics,\n"
        "            bundle.retrieval_diagnostic_summary or {},\n"
        "            review.contributing_documents,\n"
        "        )\n"
        "        bundle.retrieval_diagnostics = updated_diagnostics\n"
        "        bundle.retrieval_diagnostic_summary = updated_summary\n"
        "    return review\n\n\ndef select_results_for_answer(\n",
        "v5.5 post-coverage contributor materialization",
    )

    select_replacement = r'''def select_results_for_answer(
    results: Sequence[RetrievedChunk],
    review: SynthesisCoverage,
) -> list[RetrievedChunk]:
    """Use contributor validation as the normal answer boundary.

    Strong reranker roles are a fallback only when the coverage critic was unavailable
    or returned no contributor set. When a successful critic rejects a route, that route
    stays diagnostics/review-only even if an earlier reranker was over-optimistic.
    """
    values = list(results)
    if review.answer_strategy != "multi_document_synthesis" or not values:
        return values

    allowed = {name.casefold() for name in review.contributing_documents if str(name).strip()}
    for conflict in review.conflicts:
        for document in conflict.get("documents", []):
            if str(document).strip():
                allowed.add(str(document).casefold())

    if not allowed:
        strong_roles = {"governing", "definition", "applicability", "exception", "restriction", "authority", "conflict"}
        for item in values:
            role = _v55_method_role(item.method)
            if role in strong_roles:
                allowed.add(item.chunk.filename.casefold())

    if not allowed:
        return values[: min(16, len(values))]
    selected = [item for item in values if item.chunk.filename.casefold() in allowed]
    return selected or values[: min(16, len(values))]
'''
    source = replace_function(source, "select_results_for_answer", select_replacement, "coverage_prompt_status")

    return f"# {MARKER}\n" + source


def patch_service(source: str) -> str:
    if MARKER in source:
        return source
    if V54_MARKER not in source:
        raise RuntimeError("backend/app/rag/v5/service.py is not the v5.4 smart-completeness version.")

    # Main answer contract: keep threshold/action associations and measurement bases.
    anchor = '''- A source block marked as a v5.4 headerless-table recovery contains only text already present in PDF table metadata. Treat it as source-grounded recovery of the first data row, not as a model-created fact.
'''
    addition = '''- PROCEDURE TABLE INTEGRITY: when evidence contains multiple threshold/condition rows, pair each condition only with the action/value from the same row or explicitly linked continuation. Never move a speed, limit, permission, prohibition or action from a neighboring branch onto another threshold.
- MEASUREMENT-BASIS INTEGRITY: do not silently equate different source-defined units or populations (for example a percentage of one component type versus a percentage/count of another). Give the closest applicable source rule first, then state the unresolved unit/scope distinction briefly when it matters.
- ROUTING IS NOT ANSWER ELIGIBILITY: a PDF/RS/Line may be searched for coverage and still be absent from the answer. Do not create a "No applicable evidence" heading for a routed/review-only scope unless the user explicitly asked for a negative inventory of every searched scope.
- A source block marked `SOURCE-GROUNDED PROCEDURE STRUCTURE` is a query-time grouping of original source rows. Treat each labelled BRANCH independently and preserve its condition-to-action pairing.
'''
    source = replace_once(source, anchor, anchor + addition, "v5.5 procedure answer integrity rules")

    repair_anchor = "- For cross-scope conditional/procedural requests, use separate RS/Line/PDF headings for every required document supplied below, even when the procedures are identical.\n"
    repair_add = "- Never create an RS/Line/PDF heading merely because that scope was routed or reviewed; headings are required only for documents listed in REQUIRED SOURCE DOCUMENTS / REQUIRED RS/LINE LABELS below.\n- For threshold tables, keep each condition and action/value from the same source branch and preserve source-defined units without inferred conversions.\n"
    if repair_add not in source:
        source = replace_once(source, repair_anchor, repair_anchor + repair_add, "v5.5 repair boundary rules")

    source = source.replace("rag-v5.4-smart-completeness", "rag-v5.5-procedure-integrity")
    return f"# {MARKER}\n" + source


def prepare(repo: Path, package: Path) -> tuple[dict[Path, str], list[tuple[Path, Path]]]:
    transforms = {
        repo / "backend/app/rag/v5/retrieval_completeness.py": patch_retrieval_completeness,
        repo / "backend/app/rag/v5/synthesis_retrieval.py": patch_synthesis_retrieval,
        repo / "backend/app/rag/v5/service.py": patch_service,
    }
    # Optional payload mirrors are patched only when they already carry the v5.4
    # runtime marker. Some deployments keep older payload templates beside the active
    # backend; an optional stale template must never make the runtime patch fail.
    optional_payloads = [
        (repo / "payload/backend/app/rag/v5/retrieval_completeness.py", patch_retrieval_completeness),
        (repo / "payload/backend/app/rag/v5/synthesis_retrieval.py", patch_synthesis_retrieval),
        (repo / "payload/backend/app/rag/v5/service.py", patch_service),
    ]
    for path, fn in optional_payloads:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8-sig")
        if V54_MARKER in current:
            transforms[path] = fn

    missing = [str(path) for path in list(transforms)[:3] if not path.exists()]
    if missing:
        raise SystemExit("Required v5.4 runtime file(s) missing:\n" + "\n".join(missing))

    transformed: dict[Path, str] = {}
    for path, fn in transforms.items():
        transformed[path] = fn(path.read_text(encoding="utf-8-sig"))

    copies = [
        (package / "backend/tests/test_v55_procedure_integrity.py", repo / "backend/tests/test_v55_procedure_integrity.py"),
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

    snapshot = repo / "pdfrag-v5.5-replacement-files"
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
        "version": "rag-v5.5-procedure-integrity",
        "base_required": "rag-v5.4-smart-completeness",
        "replacement_files": [str(path.relative_to(repo)).replace("\\", "/") for path in transformed]
        + [str(dst.relative_to(repo)).replace("\\", "/") for _src, dst in copies],
        "database_migration": False,
        "embedding_rebuild": False,
        "full_reprocessing_required_for_first_test": False,
        "contracts": [
            "routed/review evidence is separate from answer evidence",
            "required scopes come only from validated contributing governing evidence",
            "deep retrieval is balanced per routed document",
            "governing procedure tables/sections are expanded as complete structures",
            "condition/action branches and measurement bases are preserved",
            "scope identity supports multiple labels and cover-page source codes/named lines",
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[snapshot] {snapshot.relative_to(repo)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply IMS RAG v5.5 procedure-integrity patch over v5.4."
    )
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
        repo / "backend/tests/test_v55_procedure_integrity.py",
    ]:
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied IMS RAG v5.5 procedure-integrity patch.")
    print(" - per-routed-document semantic + lexical deep retrieval prevents cross-document crowd-out")
    print(" - review-only routed evidence can no longer force empty RS/Line/PDF answer headings")
    print(" - required answer scopes are contributor-validated governing evidence only")
    print(" - governing procedure tables/sections are expanded before coverage/answering")
    print(" - complete source rows are grouped without mixing condition/action branches")
    print(" - source-defined measurement bases are preserved; no unsupported percentage/count conversion")
    print(" - document scope identity supports multiple RS/Line labels and cover-page codes/named lines")
    print(" - v5.4 definition/entity completeness and headerless-table protections are retained")
    print()
    print("No database migration or embedding rebuild is required.")
    print("Do not reprocess the full corpus for the first test; this patch works against current v5 chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
