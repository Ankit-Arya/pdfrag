from __future__ import annotations

import re

from rapidfuzz import fuzz
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.rag.normalization import canonical_phrase, search_terms
from app.rag.structure import section_match_score, section_path_from_text
from app.rag.types import PrimaryDocumentMatch, RetrievedChunk, TextChunk

_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_/-]{1,}")
_ABBREVIATION_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9/-]{1,7}\b")
_CONTEXT_BLOCK_RE = re.compile(r"\[PDF CHUNK CONTEXT\].*?\[/PDF CHUNK CONTEXT\]", re.IGNORECASE | re.DOTALL)
_DOCUMENT_REFERENCE_RE = re.compile(
    r"\b(SC|SM|SOP|JPO|INST(?:RUCTION)?|MRGR)\s*[-_ ]?\s*(\d{1,4}[A-Z]?)\b",
    re.IGNORECASE,
)


def search_chunks(
    db: Session,
    query_vector: list[float],
    query_text: str,
    limit: int,
) -> list[RetrievedChunk]:
    """Hybrid pgvector + full-text retrieval with deterministic tie ordering."""
    settings = get_settings()
    candidate_limit = max(limit * 2, 40)
    per_document_limit = settings.retrieval_chunks_per_document
    sql = text(
        """
        WITH q AS (
          SELECT CAST(:embedding AS vector) AS embedding,
                 plainto_tsquery('simple', :query) AS tsq,
                 to_tsquery('simple', :any_query) AS tsq_any,
                 CAST(:normalized_section_query AS text) AS normalized_section_query
        ),
        ready_documents AS (
          SELECT id, filename
          FROM documents
          WHERE status = 'ready'
        ),
        global_vector_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM document_chunks c
          JOIN ready_documents d ON d.id = c.document_id
          CROSS JOIN q
          ORDER BY c.embedding <=> q.embedding, lower(d.filename), c.page_number, c.chunk_index, c.id
          LIMIT :candidate_limit
        ),
        global_keyword_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM document_chunks c
          JOIN ready_documents d ON d.id = c.document_id
          CROSS JOIN q
          WHERE to_tsvector('simple', c.text) @@ q.tsq_any
          ORDER BY ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) DESC,
                   lower(d.filename), c.page_number, c.chunk_index, c.id
          LIMIT :candidate_limit
        ),
        per_document_vector_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM ready_documents d
          CROSS JOIN q
          CROSS JOIN LATERAL (
            SELECT candidate.*
            FROM document_chunks candidate
            WHERE candidate.document_id = d.id
            ORDER BY candidate.embedding <=> q.embedding, candidate.chunk_index, candidate.id
            LIMIT :per_document_limit
          ) c
        ),
        per_document_keyword_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM ready_documents d
          CROSS JOIN q
          CROSS JOIN LATERAL (
            SELECT candidate.*
            FROM document_chunks candidate
            WHERE candidate.document_id = d.id
              AND to_tsvector('simple', candidate.text) @@ q.tsq_any
            ORDER BY ts_rank_cd(to_tsvector('simple', candidate.text), q.tsq_any) DESC,
                     candidate.chunk_index, candidate.id
            LIMIT :per_document_limit
          ) c
        ),
        section_path_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM document_chunks c
          JOIN ready_documents d ON d.id = c.document_id
          CROSS JOIN q
          WHERE length(q.normalized_section_query) >= 3
            AND btrim(regexp_replace(
                  lower(split_part(split_part(c.text, 'Section path:', 2), E'\n', 1)),
                  '[^a-z0-9]+', ' ', 'g'
                )) LIKE '%' || q.normalized_section_query || '%'
          ORDER BY lower(d.filename), c.page_number, c.chunk_index, c.id
          LIMIT :candidate_limit
        ),
        candidates AS (
          SELECT * FROM global_vector_candidates
          UNION
          SELECT * FROM global_keyword_candidates
          UNION
          SELECT * FROM per_document_vector_candidates
          UNION
          SELECT * FROM per_document_keyword_candidates
          UNION
          SELECT * FROM section_path_candidates
        )
        SELECT * FROM candidates
        ORDER BY lower(filename), page_number, chunk_index, id
        """
    )
    rows = db.execute(
        sql,
        {
            "embedding": str(query_vector),
            "query": query_text,
            "any_query": _keyword_or_query(query_text),
            "normalized_section_query": _normalize_phrase(query_text),
            "candidate_limit": candidate_limit,
            "per_document_limit": per_document_limit,
        },
    ).mappings()

    query_terms = _terms(query_text)
    results: list[RetrievedChunk] = []
    for row in rows:
        vector_score = float(row["vector_score"] or 0.0)
        keyword_score = min(float(row["keyword_score"] or 0.0) * 4.0, 1.0)
        lexical_score = _lexical_overlap(query_terms, row["text"])
        phrase_score = _phrase_match(query_text, row["text"])
        section_score = _section_phrase_match(query_text, row["text"])
        blended_score = min(
            1.0,
            max(
                0.0,
                vector_score * 0.36
                + keyword_score * 0.24
                + lexical_score * 0.18
                + phrase_score * 0.10
                + section_score * 0.12,
            ),
        )
        score = max(blended_score, 0.78 + section_score * 0.18) if section_score else blended_score
        results.append(
            _row_to_result(
                row,
                score=score,
                method=_method(
                    vector_score,
                    keyword_score,
                    lexical_score,
                    phrase_score,
                    section_score,
                ),
                vector_score=vector_score,
                keyword_score=keyword_score,
            )
        )

    results.sort(key=_result_sort_key)
    return _document_diverse_results(results, settings.max_retrieval_candidates)



def search_stemmed_chunks(
    db: Session,
    query_text: str,
    limit: int,
) -> list[RetrievedChunk]:
    """Complement exact/simple FTS with PostgreSQL English stemming.

    The exact path remains authoritative for acronyms/codes. This path is only a
    recall supplement for ordinary English morphology (obstruct/obstruction,
    demonstrate/demonstration, etc.) and is re-ranked by the normal relevance layer.
    """
    settings = get_settings()
    if not settings.stemmed_search_enabled:
        return []
    query_terms = _terms(query_text)
    if not query_terms:
        return []
    max_rows = min(max(40, limit), settings.stemmed_search_max_chunks)
    sql = text(
        """
        WITH q AS (SELECT to_tsquery('english', :any_query) AS tsq_any)
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
               c.text, d.filename,
               ts_rank_cd(to_tsvector('english', c.text), q.tsq_any) AS keyword_score
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN q
        WHERE d.status = 'ready'
          AND to_tsvector('english', c.text) @@ q.tsq_any
        ORDER BY keyword_score DESC, lower(d.filename), c.page_number, c.chunk_index, c.id
        LIMIT :max_rows
        """
    )
    rows = db.execute(
        sql,
        {"any_query": _keyword_or_query_from_terms(query_terms), "max_rows": max_rows},
    ).mappings()
    results: list[RetrievedChunk] = []
    for row in rows:
        keyword_score = min(float(row["keyword_score"] or 0.0) * 5.0, 1.0)
        lexical_score = _lexical_overlap(query_terms, str(row["text"]))
        score = min(0.93, 0.36 + keyword_score * 0.38 + lexical_score * 0.20)
        results.append(
            _row_to_result(
                row,
                score=score,
                method="english-fts",
                vector_score=0.0,
                keyword_score=keyword_score,
            )
        )
    results.sort(key=_result_sort_key)
    return _deduplicate_results(results, max_rows)


def find_primary_documents(
    db: Session,
    question_text: str,
    *,
    max_documents: int | None = None,
) -> list[PrimaryDocumentMatch]:
    """Find dedicated SOP/instruction documents whose subject matches the question.

    Filename/title matching is combined with the first two indexed chunks so generic
    filenames can still route correctly. This is deliberately a *routing* signal;
    answer facts continue to come from retrieved PDF chunks.
    """
    settings = get_settings()
    if not settings.primary_document_routing_enabled:
        return []
    limit = max_documents or settings.primary_document_max_documents
    query = _route_phrase(question_text)
    query_terms = _route_terms(question_text)
    if len(query_terms) < 2 or not query:
        return []

    rows = db.execute(
        text(
            """
            SELECT d.id, d.filename,
                   COALESCE((
                     SELECT string_agg(x.text, E'\n' ORDER BY x.chunk_index)
                     FROM (
                       SELECT c.text, c.chunk_index
                       FROM document_chunks c
                       WHERE c.document_id = d.id
                       ORDER BY c.chunk_index
                       LIMIT 2
                     ) x
                   ), '') AS opening_text
            FROM documents d
            WHERE d.status = 'ready'
            ORDER BY lower(d.filename), d.id
            """
        )
    ).mappings()

    matches: list[PrimaryDocumentMatch] = []
    for row in rows:
        filename = str(row["filename"])
        if re.search(r"(?:index|catalog|master[ _-]*list)", filename, re.I):
            continue
        opening = _strip_chunk_context(str(row["opening_text"] or ""))[:2400]
        filename_phrase = _route_phrase(filename)
        opening_phrase = _route_phrase(opening)
        filename_terms = _route_terms(filename)
        opening_terms = _route_terms(opening)

        file_shared = len(query_terms & filename_terms)
        open_shared = len(query_terms & opening_terms)
        file_coverage = file_shared / max(len(query_terms), 1)
        open_coverage = open_shared / max(len(query_terms), 1)
        file_fuzzy = max(
            fuzz.WRatio(query, filename_phrase),
            fuzz.token_set_ratio(query, filename_phrase),
        ) / 100.0
        opening_fuzzy = (
            max(
                fuzz.WRatio(query, opening_phrase),
                fuzz.partial_ratio(query, opening_phrase),
            ) / 100.0
            if opening_phrase
            else 0.0
        )
        filename_score = file_fuzzy * 0.68 + file_coverage * 0.32
        opening_score = opening_fuzzy * 0.58 + open_coverage * 0.42
        score = max(filename_score, opening_score)

        # Dedicated-document routing should be precise. Require at least 60% of
        # the question's meaningful subject terms to occur in the filename/opening
        # text after light morphology normalization. This prevents every document
        # containing generic phrases such as "train movement" from becoming primary.
        shared = max(file_shared, open_shared)
        coverage = max(file_coverage, open_coverage)
        if score < settings.primary_document_match_threshold:
            continue
        if shared < 2 or coverage < 0.60:
            continue
        matches.append(
            PrimaryDocumentMatch(
                document_id=str(row["id"]),
                filename=filename,
                score=round(score, 6),
                reason="filename" if filename_score >= opening_score else "opening-subject",
            )
        )

    matches.sort(key=lambda item: (-item.score, item.filename.casefold(), item.document_id))
    return matches[:limit]


def fetch_primary_document_chunks(
    db: Session,
    routes: list[PrimaryDocumentMatch],
    query_texts: list[str],
    *,
    chunks_per_document: int | None = None,
) -> list[RetrievedChunk]:
    """Read relevant sections from strongly matched dedicated documents."""
    if not routes:
        return []
    settings = get_settings()
    chunk_limit = chunks_per_document or settings.primary_document_chunks_per_document
    query_terms = _terms(" ".join(query_texts))
    any_query = _keyword_or_query_from_terms(query_terms)
    sql = text(
        """
        WITH q AS (
          SELECT to_tsquery('simple', :any_query) AS simple_any,
                 to_tsquery('english', :any_query) AS english_any
        )
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
               c.text, d.filename,
               GREATEST(
                 ts_rank_cd(to_tsvector('simple', c.text), q.simple_any),
                 ts_rank_cd(to_tsvector('english', c.text), q.english_any) * 0.9
               ) AS keyword_score,
               CASE WHEN
                 to_tsvector('simple', c.text) @@ q.simple_any
                 OR to_tsvector('english', c.text) @@ q.english_any
               THEN 1 ELSE 0 END AS matched
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN q
        WHERE d.status = 'ready' AND c.document_id::text = :document_id
        ORDER BY matched DESC, keyword_score DESC, c.chunk_index, c.id
        LIMIT :chunk_limit
        """
    )
    results: list[RetrievedChunk] = []
    for route in routes:
        rows = db.execute(
            sql,
            {
                "any_query": any_query,
                "document_id": route.document_id,
                "chunk_limit": chunk_limit,
            },
        ).mappings()
        for row in rows:
            body_terms = _terms(_strip_chunk_context(str(row["text"]))[:16000])
            coverage = len(query_terms & body_terms) / max(len(query_terms), 1) if query_terms else 0.0
            keyword_score = min(float(row["keyword_score"] or 0.0) * 5.0, 1.0)
            matched = bool(row["matched"])
            score = min(
                0.99,
                0.50 + route.score * 0.22 + (0.14 if matched else 0.0)
                + coverage * 0.10 + keyword_score * 0.06,
            )
            results.append(
                _row_to_result(
                    row,
                    score=score,
                    method=f"primary-document:{route.reason}",
                    vector_score=0.0,
                    keyword_score=keyword_score,
                )
            )
    results.sort(key=_result_sort_key)
    return _deduplicate_results(results, len(routes) * chunk_limit)


def scan_matching_chunks(
    db: Session,
    query_texts: list[str],
    *,
    focus_terms: list[str] | None = None,
    reference_mode: bool = False,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Corpus-wide lexical pass over every ready chunk.

    PostgreSQL evaluates the full ready corpus against an OR full-text query. The
    returned matches are then checked for literal/normalized term coverage. This
    complements top-K semantic retrieval so a directly matching operational rule
    cannot disappear merely because another chunk has a slightly better vector.
    """
    settings = get_settings()
    max_rows = limit or settings.corpus_scan_max_chunks
    source_text = " ".join(focus_terms or []) if focus_terms else " ".join(query_texts)
    query_terms = _terms(source_text)
    if not query_terms:
        return []

    sql = text(
        """
        WITH q AS (
          SELECT to_tsquery('simple', :any_query) AS tsq_any
        )
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
               c.text, d.filename,
               ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) AS keyword_score
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN q
        WHERE d.status = 'ready'
          AND to_tsvector('simple', c.text) @@ q.tsq_any
        ORDER BY keyword_score DESC, lower(d.filename), c.page_number, c.chunk_index, c.id
        LIMIT :max_rows
        """
    )
    rows = db.execute(
        sql,
        {
            "any_query": _keyword_or_query_from_terms(query_terms),
            "max_rows": max_rows,
        },
    ).mappings()

    required_matches = 1 if reference_mode or len(query_terms) <= 2 else 2
    results: list[RetrievedChunk] = []
    for row in rows:
        text_terms = _terms(str(row["text"])[:12000])
        matches = len(query_terms & text_terms)
        if matches < required_matches:
            continue
        coverage = matches / max(len(query_terms), 1)
        keyword_score = min(float(row["keyword_score"] or 0.0) * 5.0, 1.0)
        phrase = max((_phrase_match(query, str(row["text"])) for query in query_texts), default=0.0)
        score = min(1.0, 0.30 + coverage * 0.48 + keyword_score * 0.14 + phrase * 0.08)
        results.append(
            _row_to_result(
                row,
                score=score,
                method="corpus-fts" if not phrase else "corpus-fts+exact-phrase",
                vector_score=0.0,
                keyword_score=keyword_score,
            )
        )

    results.sort(key=_result_sort_key)
    return _deduplicate_results(results, max_rows)


def discover_document_references(
    candidates: list[RetrievedChunk],
    question_text: str,
    *,
    max_references: int = 8,
) -> list[str]:
    """Rank procedure/document codes from query-relevant index lines.

    Exact two-term overlap was too brittle: ``pilot speed AEL`` could not follow an
    index row saying ``SC-04 ... Movement of Pilot Train``. We now combine overlap,
    fuzzy line similarity and the parent candidate score, then follow a small ranked
    set. Wrong extra references are harmless because target-document chunks still
    pass the normal relevance filter before synthesis.
    """
    if max_references <= 0:
        return []
    query_terms = _route_terms(question_text)
    if not query_terms:
        return []
    reference_generic = {
        "speed", "limit", "date", "time", "value", "amount", "number",
        "rule", "require", "procedur", "step", "action", "information",
    }
    subject_terms = {term for term in query_terms if term not in reference_generic}
    if not subject_terms:
        subject_terms = query_terms
    required_subject_coverage = 0.60 if len(subject_terms) >= 3 else 0.50

    scored: dict[str, float] = {}
    # Explicit codes in the user's own question always win.
    for match in _DOCUMENT_REFERENCE_RE.finditer(question_text):
        reference = _canonical_document_reference(match.group(1), match.group(2))
        scored[reference] = 10.0

    for candidate in candidates[:350]:
        body = _strip_chunk_context(candidate.chunk.text)
        for raw_line in body.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            code_matches = list(_DOCUMENT_REFERENCE_RE.finditer(line))
            if not code_matches:
                continue
            line_terms = _route_terms(line)
            overlap = len(query_terms & line_terms)
            subject_overlap = len(subject_terms & line_terms)
            subject_coverage = subject_overlap / max(len(subject_terms), 1)
            fuzzy_score = fuzz.WRatio(_route_phrase(question_text), _route_phrase(line)) / 100.0
            if subject_overlap == 0:
                continue
            if subject_coverage < required_subject_coverage:
                if len(subject_terms) > 2 or fuzzy_score < 0.72:
                    continue
            overlap_score = overlap / max(min(len(query_terms), 4), 1)
            line_score = (
                subject_coverage * 0.50
                + overlap_score * 0.16
                + fuzzy_score * 0.18
                + min(candidate.score, 1.0) * 0.16
            )
            for match in code_matches:
                reference = _canonical_document_reference(match.group(1), match.group(2))
                scored[reference] = max(scored.get(reference, 0.0), line_score)

    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0].casefold()))
    return [reference for reference, _score in ordered[:max_references]]

def fetch_referenced_document_chunks(
    db: Session,
    references: list[str],
    query_texts: list[str],
    *,
    max_documents: int | None = None,
    chunks_per_document: int | None = None,
) -> list[RetrievedChunk]:
    """Resolve PDF-derived codes to ready documents and retrieve from them.

    This is a second retrieval hop, not a citation shortcut: the index/catalog row
    only identifies which PDF to inspect. Facts still come from the actual target
    document chunks returned here and are re-ranked/synthesized normally.
    """
    settings = get_settings()
    if not settings.reference_hop_enabled or not references:
        return []
    document_limit = max_documents or settings.reference_hop_max_documents
    per_document_limit = chunks_per_document or settings.reference_hop_chunks_per_document
    if document_limit <= 0 or per_document_limit <= 0:
        return []

    ready_documents = list(
        db.execute(
            text(
                """
                SELECT id, filename
                FROM documents
                WHERE status = 'ready'
                ORDER BY lower(filename), id
                """
            )
        ).mappings()
    )
    selected_documents: list[tuple[str, str, str]] = []
    selected_ids: set[str] = set()
    for reference in references:
        normalized_reference = _normalize_document_reference(reference)
        matches: list[tuple[int, str, str]] = []
        for row in ready_documents:
            filename = str(row["filename"])
            normalized_filename = _normalize_document_reference(filename)
            if normalized_reference and normalized_reference in normalized_filename:
                # Prefer the actual procedure over an index/catalog file when both
                # contain the same code in their filename.
                index_penalty = 1 if re.search(r"\\b(?:index|catalog|master)\\b", filename, re.I) else 0
                matches.append((index_penalty, str(row["id"]), filename))
        ordered_matches = sorted(matches, key=lambda item: (item[0], item[2].casefold()))
        non_index_matches = [item for item in ordered_matches if item[0] == 0]
        for _, document_id, filename in (non_index_matches or ordered_matches):
            if document_id in selected_ids:
                continue
            selected_ids.add(document_id)
            selected_documents.append((document_id, filename, reference))
            if len(selected_documents) >= document_limit:
                break
        if len(selected_documents) >= document_limit:
            break

    if not selected_documents:
        return []

    query_terms = _terms(" ".join(query_texts))
    any_query = _keyword_or_query_from_terms(query_terms)
    sql = text(
        """
        WITH q AS (SELECT to_tsquery('simple', :any_query) AS tsq_any)
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
               c.text, d.filename,
               ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) AS keyword_score,
               CASE WHEN to_tsvector('simple', c.text) @@ q.tsq_any THEN 1 ELSE 0 END AS matched
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN q
        WHERE d.status = 'ready'
          AND c.document_id::text = :document_id
        ORDER BY matched DESC, keyword_score DESC, c.chunk_index, c.id
        LIMIT :chunk_limit
        """
    )

    results: list[RetrievedChunk] = []
    for document_id, _filename, _reference in selected_documents:
        rows = db.execute(
            sql,
            {
                "any_query": any_query,
                "document_id": document_id,
                "chunk_limit": per_document_limit,
            },
        ).mappings()
        for row in rows:
            body_terms = _terms(_strip_chunk_context(str(row["text"]))[:16000])
            coverage = (
                len(query_terms & body_terms) / max(len(query_terms), 1)
                if query_terms
                else 0.0
            )
            keyword_score = min(float(row["keyword_score"] or 0.0) * 5.0, 1.0)
            score = min(0.98, 0.62 + coverage * 0.28 + keyword_score * 0.08)
            results.append(
                _row_to_result(
                    row,
                    score=score,
                    method="reference-hop",
                    vector_score=0.0,
                    keyword_score=keyword_score,
                )
            )

    results.sort(key=_result_sort_key)
    return _deduplicate_results(results, document_limit * per_document_limit)


def _canonical_document_reference(prefix: str, number: str) -> str:
    clean_prefix = re.sub(r"[^A-Za-z]", "", prefix).upper()
    if clean_prefix.startswith("INSTRUCTION"):
        clean_prefix = "INST"
    return f"{clean_prefix}-{number.upper()}"


def _normalize_document_reference(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _strip_chunk_context(value: str) -> str:
    return _CONTEXT_BLOCK_RE.sub("", value).strip()


def find_abbreviation_hints(
    db: Session,
    text_value: str,
    *,
    max_terms: int | None = None,
    chunks_per_term: int | None = None,
) -> list[str]:
    """Resolve short internal tokens only from ready PDF text.

    The planner receives these hints before it rewrites a query. This prevents a
    general-purpose model from guessing what SC, PSD, OCC, UTO, etc. mean.
    """
    settings = get_settings()
    term_limit = settings.abbreviation_scan_terms if max_terms is None else max_terms
    chunk_limit = (
        settings.abbreviation_scan_chunks_per_term
        if chunks_per_term is None
        else chunks_per_term
    )
    if term_limit <= 0:
        return []

    candidates = _abbreviation_candidates(text_value)[:term_limit]
    hints: list[str] = []
    seen: set[str] = set()

    sql = text(
        """
        SELECT c.text, c.page_number, d.filename
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready'
          AND c.text ~* :pattern
        ORDER BY lower(d.filename), c.page_number, c.chunk_index
        LIMIT :chunk_limit
        """
    )

    for raw_token in candidates:
        token = raw_token.upper()
        pattern = rf"(^|[^A-Za-z0-9]){re.escape(raw_token)}([^A-Za-z0-9]|$)"
        rows = list(
            db.execute(
                sql,
                {"pattern": pattern, "chunk_limit": chunk_limit},
            ).mappings()
        )
        if not rows:
            continue

        expansions: list[tuple[str, str, int]] = []
        for row in rows:
            expansion = _definition_for_token(str(row["text"]), raw_token)
            if expansion:
                expansions.append((expansion, str(row["filename"]), int(row["page_number"])))

        if expansions:
            for expansion, filename, page in expansions[:3]:
                hint = f"{token} = {expansion} — {filename} p.{page}"
                key = hint.casefold()
                if key not in seen:
                    seen.add(key)
                    hints.append(hint)
            continue

        # No explicit expansion was found. Still tell the planner that this short
        # token is genuinely present in the internal corpus, without guessing it.
        for row in rows[:2]:
            excerpt = _token_excerpt(str(row["text"]), raw_token)
            hint = f"{token} — internal document usage in {row['filename']} p.{row['page_number']}: {excerpt}"
            key = hint.casefold()
            if key not in seen:
                seen.add(key)
                hints.append(hint)

    return hints


def fetch_neighbor_chunks(
    db: Session,
    seeds: list[RetrievedChunk],
    window: int = 1,
) -> list[RetrievedChunk]:
    if window <= 0 or not seeds:
        return []

    pairs = [
        (seed.chunk.document_id, seed.chunk.chunk_index)
        for seed in seeds
        if seed.chunk.document_id and seed.chunk.chunk_index is not None
    ]
    if not pairs:
        return []

    values_sql = ",".join(f"(:doc_{index}, :idx_{index})" for index, _ in enumerate(pairs))
    params: dict[str, object] = {"window": window}
    for index, (document_id, chunk_index) in enumerate(pairs):
        params[f"doc_{index}"] = document_id
        params[f"idx_{index}"] = chunk_index

    sql = text(
        f"""
        WITH seeds(document_id, chunk_index) AS (VALUES {values_sql})
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
               c.text, d.filename,
               MIN(ABS(c.chunk_index - seeds.chunk_index)) AS distance
        FROM seeds
        JOIN document_chunks c
          ON c.document_id::text = seeds.document_id
         AND c.chunk_index BETWEEN seeds.chunk_index - :window AND seeds.chunk_index + :window
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready'
        GROUP BY c.id, c.document_id, c.chunk_index, c.page_number,
                 c.content_type, c.text, d.filename
        ORDER BY lower(d.filename), c.page_number, c.chunk_index, c.id
        """
    )
    rows = db.execute(sql, params).mappings()
    neighbors: list[RetrievedChunk] = []
    for row in rows:
        distance = int(row["distance"] or 0)
        neighbors.append(
            _row_to_result(
                row,
                score=max(0.01, 0.70 - distance * 0.15),
                method="neighbor-context" if distance else "direct-context",
                vector_score=0.0,
                keyword_score=0.0,
            )
        )
    neighbors.sort(key=_result_sort_key)
    return neighbors


def _row_to_result(
    row: object,
    *,
    score: float,
    method: str,
    vector_score: float,
    keyword_score: float,
) -> RetrievedChunk:
    mapping = row
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=str(mapping["id"]),  # type: ignore[index]
            filename=str(mapping["filename"]),  # type: ignore[index]
            page_number=int(mapping["page_number"]),  # type: ignore[index]
            text=str(mapping["text"]),  # type: ignore[index]
            content_type=str(mapping["content_type"]),  # type: ignore[index]
            document_id=str(mapping["document_id"]),  # type: ignore[index]
            chunk_index=int(mapping["chunk_index"]),  # type: ignore[index]
        ),
        score=score,
        method=method,
        vector_score=vector_score,
        keyword_score=keyword_score,
    )


def _method(
    vector_score: float,
    keyword_score: float,
    lexical_score: float,
    phrase_score: float,
    section_score: float,
) -> str:
    if section_score >= 0.88:
        return "section-heading"
    if phrase_score:
        return "pgvector+exact-phrase"
    if keyword_score >= 0.08 and keyword_score >= vector_score * 0.5:
        return "pgvector+fts"
    if lexical_score >= 0.35:
        return "pgvector+lexical"
    return "pgvector"


def _terms(value: str) -> set[str]:
    return {
        term
        for term in search_terms(value, keep_single=True)
        if term not in _STOPWORDS and (len(term) > 1 or term.isdigit())
    }


def _lexical_overlap(query_terms: set[str], text_value: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = _terms(text_value[:12000])
    if not text_terms:
        return 0.0
    return min(1.0, len(query_terms & text_terms) / max(len(query_terms), 1))


def _phrase_match(query: str, text_value: str) -> float:
    normalized_query = _normalize_phrase(query)
    if len(normalized_query) < 4:
        return 0.0
    return 1.0 if normalized_query in _normalize_phrase(text_value[:12000]) else 0.0


def _section_phrase_match(query: str, text_value: str) -> float:
    return section_match_score(query, section_path_from_text(text_value))


def _normalize_phrase(value: str) -> str:
    return canonical_phrase(value)


def _keyword_or_query(value: str) -> str:
    return _keyword_or_query_from_terms(_terms(value))


def _keyword_or_query_from_terms(terms: set[str]) -> str:
    safe_terms = sorted(term for term in terms if re.fullmatch(r"[a-z0-9]+", term))
    return " | ".join(term if len(term) <= 3 or term.isdigit() else f"{term}:*" for term in safe_terms) or "pdfrag_no_match"



_ROUTE_NOISE = {
    "the", "and", "for", "with", "from", "this", "that", "what", "when",
    "where", "which", "how", "are", "is", "in", "on", "of", "to", "a",
    "an", "or", "by", "as", "be", "do", "does", "can", "could", "should",
    "please", "tell", "give", "provide", "show", "explain", "find", "if",
    "document", "documents", "docs", "information", "info",
}



def _route_terms(value: str) -> set[str]:
    return {
        _route_stem(term)
        for term in _terms(value)
        if term not in _ROUTE_NOISE and len(term) > 1
    }


def _route_stem(term: str) -> str:
    token = term.casefold()
    if len(token) > 7 and token.endswith("ation"):
        return token[:-5]
    if len(token) > 6 and token.endswith("tion"):
        return token[:-3]
    if len(token) > 6 and token.endswith("ment"):
        return token[:-4]
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed") and not token.endswith("eed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    return token.rstrip("e") if len(token) > 4 and token.endswith("e") else token


def _route_phrase(value: str) -> str:
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", value)
        if token.casefold() not in _ROUTE_NOISE
    ]
    return " ".join(tokens[:80])


def _document_diverse_results(
    results: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    if limit <= 0:
        return []

    deduplicated = _deduplicate_results(results, max(limit * 2, limit))
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    seen_documents: set[str] = set()

    # Reserve roughly half the slots for absolute strongest results so a single
    # multi-chunk procedure is not displaced merely for document diversity.
    strength_slots = max(1, limit // 2)
    for item in deduplicated[:strength_slots]:
        selected.append(item)
        selected_ids.add(item.chunk.chunk_id)
        seen_documents.add(item.chunk.document_id or item.chunk.filename.casefold())

    for item in deduplicated:
        if len(selected) >= limit:
            break
        if item.chunk.chunk_id in selected_ids:
            continue
        document_key = item.chunk.document_id or item.chunk.filename.casefold()
        if document_key in seen_documents:
            continue
        selected.append(item)
        selected_ids.add(item.chunk.chunk_id)
        seen_documents.add(document_key)

    for item in deduplicated:
        if len(selected) >= limit:
            break
        if item.chunk.chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.chunk.chunk_id)

    selected.sort(key=_result_sort_key)
    return selected


def _deduplicate_results(results: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    seen: set[str] = set()
    for item in results:
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _result_sort_key(item: RetrievedChunk) -> tuple[float, str, int, int, str]:
    return (
        -float(item.score),
        item.chunk.filename.casefold(),
        item.chunk.page_number,
        item.chunk.chunk_index if item.chunk.chunk_index is not None else -1,
        item.chunk.chunk_id,
    )


def _abbreviation_candidates(value: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for token in _ABBREVIATION_TOKEN_RE.findall(value):
        lowered = token.casefold()
        if lowered in _ABBREVIATION_STOPWORDS:
            continue
        looks_internal = (
            (token.upper() == token and any(char.isalpha() for char in token))
            or (token.isalpha() and 2 <= len(token) <= 4)
        )
        if not looks_internal or lowered in seen:
            continue
        seen.add(lowered)
        candidates.append(token)
    return candidates


def _definition_for_token(text_value: str, token: str) -> str:
    escaped = re.escape(token)
    word = r"[A-Za-z][A-Za-z0-9/&.'-]*"
    before = re.compile(
        rf"((?:{word}[ \t]+){{1,7}}{word})[ \t]*\([ \t]*{escaped}[ \t]*\)",
        re.IGNORECASE,
    )
    after = re.compile(
        rf"(?<![A-Za-z0-9]){escaped}[ \t]*\([ \t]*((?:{word}[ \t]+){{0,7}}{word})[ \t]*\)",
        re.IGNORECASE,
    )
    for pattern in (before, after):
        match = pattern.search(text_value[:12000])
        if match:
            expansion = re.sub(r"\s+", " ", match.group(1)).strip(" -,:;")
            if 3 <= len(expansion) <= 100 and expansion.casefold() != token.casefold():
                return expansion
    return ""


def _token_excerpt(text_value: str, token: str) -> str:
    match = re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text_value)
    if not match:
        return "term is present"
    start = max(0, match.start() - 100)
    end = min(len(text_value), match.end() + 160)
    excerpt = re.sub(r"\s+", " ", text_value[start:end]).strip()
    return excerpt[:320]


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "when",
    "where",
    "which",
    "how",
    "are",
    "is",
    "in",
    "on",
    "of",
    "to",
    "a",
    "an",
    "or",
    "by",
    "as",
    "be",
    "do",
    "does",
    "can",
    "could",
    "should",
}

_ABBREVIATION_STOPWORDS = _STOPWORDS | {
    "all",
    "any",
    "get",
    "give",
    "info",
    "rule",
    "same",
    "show",
    "tell",
    "user",
}
