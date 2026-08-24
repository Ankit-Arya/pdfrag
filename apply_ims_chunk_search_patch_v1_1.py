from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path

MARKER = "IMS_CHUNK_SEARCH_V1"


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + ".bak-before-chunk-search-v1")
    if not target.exists():
        shutil.copy2(path, target)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def mark_text(source: str) -> str:
    if MARKER in source:
        return source
    return f"<!-- {MARKER} -->\n" + source


def mark_ts(source: str) -> str:
    if MARKER in source:
        return source
    return f"// {MARKER}\n" + source


def mark_python(source: str) -> str:
    if MARKER in source:
        return source
    lines = source.splitlines()
    insert_at = 0
    if lines and lines[0].startswith("from __future__"):
        insert_at = 1
    lines.insert(insert_at, f"# {MARKER}")
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def patch_models(source: str) -> str:
    if "class ChunkSearchResult(BaseModel):" in source:
        return source

    match = re.search(r"^class HealthResponse\(BaseModel\):", source, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("chunk search models: HealthResponse anchor not found")
    addition = r'''

class ChunkSearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int
    page_end: int
    pages: str
    chunk_index: int
    content_type: str
    heading: str = ""
    section: str = ""
    authority_status: str = "unknown"
    score: float
    retrieval_method: str
    text: str


class ChunkSearchResponse(BaseModel):
    query: str
    mode: Literal["hybrid", "keyword", "semantic"]
    returned: int
    results: list[ChunkSearchResult]
'''
    source = source[: match.start()] + addition.lstrip("\n") + "\n\n" + source[match.start():]
    return mark_python(source)


def patch_api(source: str) -> str:
    if '@router.get("/search/chunks"' in source:
        return source

    if not re.search(r"^import re$", source, flags=re.MULTILINE):
        if "import logging\n" in source:
            source = source.replace("import logging\n", "import logging\nimport re\n", 1)
        else:
            source = "import re\n" + source

    if "from sqlalchemy import select, text, update" not in source:
        if "from sqlalchemy import select, update" in source:
            source = source.replace(
                "from sqlalchemy import select, update",
                "from sqlalchemy import select, text, update",
                1,
            )
        elif re.search(r"from sqlalchemy import [^\n]+", source):
            match = re.search(r"from sqlalchemy import ([^\n]+)", source)
            assert match
            names = [item.strip() for item in match.group(1).split(",")]
            if "text" not in names:
                names.append("text")
            source = source[: match.start()] + "from sqlalchemy import " + ", ".join(names) + source[match.end():]
        else:
            raise RuntimeError("SQLAlchemy import anchor not found")

    if "    ChunkSearchResponse,\n" not in source:
        source = replace_once(
            source,
            "    ChatSessionOut,\n",
            "    ChatSessionOut,\n    ChunkSearchResponse,\n    ChunkSearchResult,\n",
            "chunk search model imports",
        )

    endpoint_match = re.search(
        r'^@router\.get\("/documents", response_model=list\[DocumentOut\]\)',
        source,
        flags=re.MULTILINE,
    )
    if not endpoint_match:
        raise RuntimeError("chunk search endpoint: /documents anchor not found")
    endpoint = r'''

_PDF_STRUCTURE_RE = re.compile(
    r"\[PDF STRUCTURE\].*?\[/PDF STRUCTURE\]\s*",
    re.IGNORECASE | re.DOTALL,
)


def _search_display_text(value: object) -> str:
    return _PDF_STRUCTURE_RE.sub("", str(value or "")).strip()


def _chunk_search_filters(
    *,
    document_id: uuid.UUID | None,
    content_type: str | None,
) -> tuple[str, dict[str, object]]:
    clauses: list[str] = []
    params: dict[str, object] = {}
    if document_id is not None:
        clauses.append("c.document_id = :document_id")
        params["document_id"] = document_id
    if content_type:
        clauses.append("c.content_type = :content_type")
        params["content_type"] = content_type
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _keyword_chunk_search(
    db: Session,
    *,
    query: str,
    limit: int,
    document_id: uuid.UUID | None,
    content_type: str | None,
) -> list[dict[str, object]]:
    filters, params = _chunk_search_filters(
        document_id=document_id,
        content_type=content_type,
    )
    params.update({"query": query, "limit": limit})
    rows = db.execute(
        text(
            f"""
            WITH q AS (
                SELECT websearch_to_tsquery('simple', :query) AS simple_q,
                       websearch_to_tsquery('english', :query) AS english_q
            )
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.filename,
                c.page_number,
                c.page_end,
                c.chunk_index,
                c.content_type,
                c.heading,
                c.section_path,
                c.authority_status,
                c.text,
                GREATEST(
                    ts_rank_cd(to_tsvector('simple', c.text), q.simple_q),
                    ts_rank_cd(to_tsvector('english', c.text), q.english_q) * 0.9
                ) AS raw_score
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r
              ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            JOIN documents d
              ON d.id=c.document_id AND d.status='ready'
            CROSS JOIN q
            WHERE (
                to_tsvector('simple', c.text) @@ q.simple_q
                OR to_tsvector('english', c.text) @@ q.english_q
            )
            {filters}
            ORDER BY raw_score DESC, lower(d.filename), c.page_number, c.chunk_index
            LIMIT :limit
            """
        ),
        params,
    ).mappings()
    return [dict(row) for row in rows]


def _semantic_chunk_search(
    db: Session,
    *,
    query: str,
    limit: int,
    document_id: uuid.UUID | None,
    content_type: str | None,
) -> list[dict[str, object]]:
    vector = embedding_service.encode([query])[0].tolist()
    filters, params = _chunk_search_filters(
        document_id=document_id,
        content_type=content_type,
    )
    params.update({"embedding": str(vector), "limit": limit})
    rows = db.execute(
        text(
            f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.filename,
                c.page_number,
                c.page_end,
                c.chunk_index,
                c.content_type,
                c.heading,
                c.section_path,
                c.authority_status,
                c.text,
                GREATEST(
                    0.0,
                    1 - (c.embedding <=> CAST(:embedding AS vector))
                ) AS raw_score
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r
              ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            JOIN documents d
              ON d.id=c.document_id AND d.status='ready'
            WHERE 1=1
            {filters}
            ORDER BY c.embedding <=> CAST(:embedding AS vector),
                     lower(d.filename), c.page_number, c.chunk_index
            LIMIT :limit
            """
        ),
        params,
    ).mappings()
    return [dict(row) for row in rows]


def _keyword_score(value: object) -> float:
    raw = max(0.0, float(value or 0.0))
    return raw / (raw + 0.18) if raw else 0.0


def _chunk_search_result(
    row: dict[str, object],
    *,
    score: float,
    method: str,
) -> ChunkSearchResult:
    page = int(row["page_number"])
    page_end = int(row.get("page_end") or page)
    section_path = row.get("section_path")
    if not isinstance(section_path, list):
        section_path = []
    heading = str(row.get("heading") or "")
    section = " > ".join(str(item) for item in section_path if str(item).strip()) or heading
    return ChunkSearchResult(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        filename=str(row["filename"]),
        page=page,
        page_end=page_end,
        pages=str(page) if page == page_end else f"{page}-{page_end}",
        chunk_index=int(row["chunk_index"]),
        content_type=str(row["content_type"]),
        heading=heading,
        section=section,
        authority_status=str(row.get("authority_status") or "unknown"),
        score=round(max(0.0, min(1.0, score)), 4),
        retrieval_method=method,
        text=_search_display_text(row.get("text")),
    )


@router.get("/search/chunks", response_model=ChunkSearchResponse)
def search_chunks(
    q: str = Query(min_length=2, max_length=500),
    mode: str = Query(default="hybrid"),
    document_id: uuid.UUID | None = None,
    content_type: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> ChunkSearchResponse:
    """Directly search stored active v5 chunks without chat or answer generation."""
    clean_query = " ".join(q.split())
    clean_mode = mode.strip().casefold()
    if clean_mode not in {"hybrid", "keyword", "semantic"}:
        raise HTTPException(400, "Search mode must be hybrid, keyword, or semantic")

    allowed_types = {"text", "list", "table_row", "figure"}
    clean_type = (content_type or "").strip().casefold() or None
    if clean_type is not None and clean_type not in allowed_types:
        raise HTTPException(400, "Unsupported chunk content type")

    pool = min(400, max(limit * 4, 80))
    keyword_rows: list[dict[str, object]] = []
    semantic_rows: list[dict[str, object]] = []

    if clean_mode in {"hybrid", "keyword"}:
        keyword_rows = _keyword_chunk_search(
            db,
            query=clean_query,
            limit=pool,
            document_id=document_id,
            content_type=clean_type,
        )

    if clean_mode in {"hybrid", "semantic"}:
        try:
            semantic_rows = _semantic_chunk_search(
                db,
                query=clean_query,
                limit=pool,
                document_id=document_id,
                content_type=clean_type,
            )
        except EmbeddingUnavailableError as exc:
            if clean_mode == "semantic":
                raise HTTPException(503, str(exc)) from exc
            semantic_rows = []

    merged: dict[str, dict[str, object]] = {}

    for rank, row in enumerate(keyword_rows, 1):
        chunk_id = str(row["chunk_id"])
        entry = merged.setdefault(
            chunk_id,
            {"row": row, "keyword": 0.0, "semantic": 0.0, "keyword_rank": 0, "semantic_rank": 0},
        )
        entry["keyword"] = max(float(entry["keyword"]), _keyword_score(row.get("raw_score")))
        entry["keyword_rank"] = rank

    for rank, row in enumerate(semantic_rows, 1):
        chunk_id = str(row["chunk_id"])
        entry = merged.setdefault(
            chunk_id,
            {"row": row, "keyword": 0.0, "semantic": 0.0, "keyword_rank": 0, "semantic_rank": 0},
        )
        entry["semantic"] = max(0.0, min(1.0, float(row.get("raw_score") or 0.0)))
        entry["semantic_rank"] = rank

    ranked: list[tuple[float, str, dict[str, object]]] = []
    for entry in merged.values():
        keyword_score = float(entry["keyword"])
        semantic_score = float(entry["semantic"])
        if clean_mode == "keyword":
            final_score = keyword_score
            method = "keyword"
        elif clean_mode == "semantic":
            final_score = semantic_score
            method = "semantic"
        else:
            strongest = max(keyword_score, semantic_score)
            support = min(keyword_score, semantic_score)
            agreement = 0.04 if keyword_score > 0 and semantic_score > 0 else 0.0
            final_score = min(1.0, 0.82 * strongest + 0.18 * support + agreement)
            method = (
                "hybrid"
                if keyword_score > 0 and semantic_score > 0
                else ("keyword" if keyword_score > 0 else "semantic")
            )
        ranked.append((final_score, method, entry))

    ranked.sort(
        key=lambda item: (
            -item[0],
            str(item[2]["row"]["filename"]).casefold(),
            int(item[2]["row"]["page_number"]),
            int(item[2]["row"]["chunk_index"]),
        )
    )

    results = [
        _chunk_search_result(entry["row"], score=score, method=method)
        for score, method, entry in ranked[:limit]
    ]
    return ChunkSearchResponse(
        query=clean_query,
        mode=clean_mode,
        returned=len(results),
        results=results,
    )
'''
    source = source[: endpoint_match.start()] + endpoint.lstrip("\n") + "\n\n" + source[endpoint_match.start():]
    return mark_python(source)


def patch_api_ts(source: str) -> str:
    if "export interface ChunkSearchResult" not in source:
        anchor = "\n\nexport interface AdminUser {"
        types = r'''

export type ChunkSearchMode = 'hybrid' | 'keyword' | 'semantic'

export interface ChunkSearchResult {
  chunk_id: string
  document_id: string
  filename: string
  page: number
  page_end: number
  pages: string
  chunk_index: number
  content_type: string
  heading: string
  section: string
  authority_status: string
  score: number
  retrieval_method: string
  text: string
}

export interface ChunkSearchResponse {
  query: string
  mode: ChunkSearchMode
  returned: number
  results: ChunkSearchResult[]
}

export interface ChunkSearchRequest {
  query: string
  mode?: ChunkSearchMode
  documentId?: string | null
  contentType?: string | null
  limit?: number
}
'''
        source = replace_once(source, anchor, types + anchor, "chunk search TS types")

    if "export async function searchChunks(" not in source:
        anchor = "\n\nexport async function listDocuments(): Promise<DocumentRecord[]> {"
        function = r'''

export async function searchChunks(
  request: ChunkSearchRequest,
): Promise<ChunkSearchResponse> {
  const params = new URLSearchParams()
  params.set('q', request.query)
  params.set('mode', request.mode || 'hybrid')
  if (request.documentId) params.set('document_id', request.documentId)
  if (request.contentType) params.set('content_type', request.contentType)
  params.set('limit', String(request.limit ?? 30))
  return apiRequest<ChunkSearchResponse>(`/api/search/chunks?${params.toString()}`)
}
'''
        source = replace_once(source, anchor, function + anchor, "chunk search API function")
    return mark_ts(source)


def patch_app(source: str) -> str:
    if "import ChunkSearchPanel from './components/ChunkSearchPanel.vue'" not in source:
        source = replace_once(
            source,
            "import ChatPanel from './components/ChatPanel.vue'\n",
            "import ChatPanel from './components/ChatPanel.vue'\nimport ChunkSearchPanel from './components/ChunkSearchPanel.vue'\n",
            "ChunkSearchPanel import",
        )

    if "type ViewName = 'chat' | 'search' | 'admin' | 'account'" not in source:
        source = replace_once(
            source,
            "type ViewName = 'chat' | 'admin' | 'account'",
            "type ViewName = 'chat' | 'search' | 'admin' | 'account'",
            "App ViewName search",
        )

    if "<ChunkSearchPanel" not in source:
        anchor = '''    <AdminPanel
      v-else-if="view === 'admin' && user.role === 'admin'"
'''
        addition = '''    <ChunkSearchPanel
      v-else-if="view === 'search'"
      :knowledge="knowledge"
    />

'''
        source = replace_once(source, anchor, addition + anchor, "ChunkSearchPanel view")
    return mark_text(source)


def patch_upload_panel(source: str) -> str:
    if "type ViewName = 'chat' | 'search' | 'admin' | 'account'" not in source:
        source = replace_once(
            source,
            "type ViewName = 'chat' | 'admin' | 'account'",
            "type ViewName = 'chat' | 'search' | 'admin' | 'account'",
            "sidebar ViewName search",
        )

    if "@click=\"emit('navigate', 'search')\"" not in source:
        anchor = '''      <button
        v-if="user.role === 'admin'"
'''
        addition = '''      <button :class="{ active: view === 'search' }" @click="emit('navigate', 'search')">
        <span aria-hidden="true">S</span>
        Search chunks
      </button>
'''
        source = replace_once(source, anchor, addition + anchor, "sidebar chunk search navigation")
    return mark_text(source)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a separate authenticated v5 chunk-search workspace to IMS."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    payload = Path(__file__).resolve().parent / "payload"

    models = repo / "backend/app/models.py"
    api = repo / "backend/app/api.py"
    api_ts = repo / "frontend/src/services/api.ts"
    app = repo / "frontend/src/App.vue"
    sidebar = repo / "frontend/src/components/UploadPanel.vue"
    panel = repo / "frontend/src/components/ChunkSearchPanel.vue"
    payload_panel = payload / "frontend/src/components/ChunkSearchPanel.vue"

    required = [models, api, api_ts, app, sidebar]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Required repository file(s) not found:\n" + "\n".join(missing))
    if not payload_panel.exists():
        raise SystemExit(f"Patch payload missing: {payload_panel}")

    if (
        "class ChunkSearchResult(BaseModel):" in models.read_text(encoding="utf-8", errors="ignore")
        and '@router.get("/search/chunks"' in api.read_text(encoding="utf-8", errors="ignore")
        and "export async function searchChunks(" in api_ts.read_text(encoding="utf-8", errors="ignore")
        and "<ChunkSearchPanel" in app.read_text(encoding="utf-8", errors="ignore")
        and "@click=\"emit('navigate', 'search')\"" in sidebar.read_text(encoding="utf-8", errors="ignore")
        and panel.exists()
    ):
        print("IMS chunk-search patch is already applied.")
        return 0

    for path in required:
        backup(path)
    if panel.exists():
        backup(panel)

    panel.parent.mkdir(parents=True, exist_ok=True)
    panel.write_text(
        "<!-- IMS_CHUNK_SEARCH_V1 -->\n" + payload_panel.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )

    models.write_text(patch_models(models.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    api.write_text(patch_api(api.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    api_ts.write_text(patch_api_ts(api_ts.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    app.write_text(patch_app(app.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    sidebar.write_text(patch_upload_panel(sidebar.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")

    py_compile.compile(str(models), doraise=True)
    py_compile.compile(str(api), doraise=True)

    print("Applied IMS separate chunk-search workspace.")
    print("Adds:")
    print(" - authenticated /api/search/chunks endpoint")
    print(" - Hybrid / Keyword / Semantic direct chunk search")
    print(" - optional document and chunk-type filters")
    print(" - separate 'Search chunks' sidebar workspace for all users")
    print(" - result filename, page, section, type, score, chunk index and full chunk text")
    print(" - Copy chunk and Download source PDF actions")
    print("Search does not create chats and does not call the OpenAI answer/query models.")
    print("No database migration, PDF reprocessing, or embedding rebuild is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
