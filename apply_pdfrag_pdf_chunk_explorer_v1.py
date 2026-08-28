from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_PDF_CHUNK_EXPLORER_V1"
BACKUP_SUFFIX = ".bak-before-pdf-chunk-explorer-v1"


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not target.exists():
        shutil.copy2(path, target)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_models(source: str) -> str:
    if "class DocumentChunkPageOut(BaseModel):" in source:
        return source

    anchor = "class HealthResponse(BaseModel):\n"
    addition = '''# IMS_PDF_CHUNK_EXPLORER_V1\nclass DocumentChunkOut(BaseModel):\n    id: uuid.UUID\n    chunk_index: int\n    page_number: int\n    page_end: int\n    content_type: str\n    parent_key: str = \"\"\n    section_path: list[str] = Field(default_factory=list)\n    heading: str = \"\"\n    table_id: uuid.UUID | None = None\n    table_row_index: int | None = None\n    extraction_confidence: float = 1.0\n    authority_status: str = \"unknown\"\n    metadata: dict[str, Any] = Field(default_factory=dict)\n    text: str\n    char_count: int\n\n\nclass DocumentChunkPageOut(BaseModel):\n    document_id: uuid.UUID\n    filename: str\n    run_id: uuid.UUID\n    processing_version: str\n    run_started_at: datetime | None = None\n    run_completed_at: datetime | None = None\n    run_metrics: dict[str, Any] = Field(default_factory=dict)\n    run_warnings: list[Any] = Field(default_factory=list)\n    total_chunks: int\n    filtered_chunks: int\n    offset: int\n    limit: int\n    content_types: list[str] = Field(default_factory=list)\n    authority_statuses: list[str] = Field(default_factory=list)\n    chunks: list[DocumentChunkOut] = Field(default_factory=list)\n\n\n'''
    return replace_once(source, anchor, addition + anchor, "DocumentChunk response models")


def patch_api(source: str) -> str:
    if "def inspect_document_v5_chunks(" in source:
        return source

    if "from sqlalchemy import select, update" in source:
        source = source.replace(
            "from sqlalchemy import select, update",
            "from sqlalchemy import select, text, update",
            1,
        )
    elif "from sqlalchemy import select, text, update" not in source:
        raise RuntimeError("Could not safely add sqlalchemy.text import to backend/app/api.py")

    if "    DocumentChunkOut,\n" not in source:
        source = replace_once(
            source,
            "    DocumentBatchResponse,\n",
            "    DocumentBatchResponse,\n    DocumentChunkOut,\n    DocumentChunkPageOut,\n",
            "DocumentChunk model imports",
        )

    anchor = '@router.get("/documents/{document_id}/download")\n'
    endpoint = r'''# IMS_PDF_CHUNK_EXPLORER_V1
@router.get("/documents/{document_id}/v5-chunks", response_model=DocumentChunkPageOut)
def inspect_document_v5_chunks(
    document_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    q: str = Query("", max_length=300),
    page: int | None = Query(None, ge=1),
    content_type: str = Query("", max_length=40),
    authority_status: str = Query("", max_length=40),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> DocumentChunkPageOut:
    """Inspect the active RAG v5 chunks for one PDF.

    This is a read-only authenticated diagnostics endpoint. It exposes the same
    source text already available through the document library, plus v5 structural
    metadata used for retrieval debugging. Embedding vectors are intentionally not
    returned.
    """
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")

    run = db.execute(
        text(
            """
            SELECT id, processing_version, metrics, warnings, started_at, completed_at
            FROM rag_v5_processing_runs
            WHERE document_id=:document_id
              AND is_active=true
              AND status='ready'
            ORDER BY started_at DESC
            LIMIT 1
            """
        ),
        {"document_id": document_id},
    ).mappings().first()
    if not run:
        raise HTTPException(404, "No active ready RAG v5 run exists for this document")

    filters = ["c.run_id=:run_id"]
    params: dict[str, object] = {"run_id": run["id"]}

    query_value = " ".join(q.split())
    if query_value:
        filters.append(
            "(c.text ILIKE :needle OR c.heading ILIKE :needle "
            "OR c.parent_key ILIKE :needle OR CAST(c.section_path AS text) ILIKE :needle)"
        )
        params["needle"] = f"%{query_value}%"
    if page is not None:
        filters.append("c.page_number <= :page AND c.page_end >= :page")
        params["page"] = page
    if content_type.strip():
        filters.append("c.content_type=:content_type")
        params["content_type"] = content_type.strip()
    if authority_status.strip():
        filters.append("c.authority_status=:authority_status")
        params["authority_status"] = authority_status.strip()

    where_sql = " AND ".join(filters)

    total_chunks = int(
        db.execute(
            text("SELECT COUNT(*) FROM rag_v5_chunks WHERE run_id=:run_id"),
            {"run_id": run["id"]},
        ).scalar_one()
        or 0
    )
    filtered_chunks = int(
        db.execute(
            text(f"SELECT COUNT(*) FROM rag_v5_chunks c WHERE {where_sql}"),
            params,
        ).scalar_one()
        or 0
    )

    content_types = [
        str(value)
        for value in db.scalars(
            text(
                """
                SELECT DISTINCT content_type
                FROM rag_v5_chunks
                WHERE run_id=:run_id
                ORDER BY content_type
                """
            ),
            {"run_id": run["id"]},
        ).all()
        if value
    ]
    authority_statuses = [
        str(value)
        for value in db.scalars(
            text(
                """
                SELECT DISTINCT authority_status
                FROM rag_v5_chunks
                WHERE run_id=:run_id
                ORDER BY authority_status
                """
            ),
            {"run_id": run["id"]},
        ).all()
        if value
    ]

    row_params = dict(params)
    row_params.update({"offset": offset, "limit": limit})
    rows = list(
        db.execute(
            text(
                f"""
                SELECT
                    c.id, c.chunk_index, c.page_number, c.page_end,
                    c.content_type, c.parent_key, c.section_path, c.heading,
                    c.table_id, c.table_row_index, c.extraction_confidence,
                    c.authority_status, c.metadata, c.text
                FROM rag_v5_chunks c
                WHERE {where_sql}
                ORDER BY c.chunk_index
                OFFSET :offset
                LIMIT :limit
                """
            ),
            row_params,
        ).mappings()
    )

    chunks = [
        DocumentChunkOut(
            id=row["id"],
            chunk_index=int(row["chunk_index"]),
            page_number=int(row["page_number"]),
            page_end=int(row["page_end"] or row["page_number"]),
            content_type=str(row["content_type"] or ""),
            parent_key=str(row["parent_key"] or ""),
            section_path=[str(item) for item in (row["section_path"] or [])],
            heading=str(row["heading"] or ""),
            table_id=row["table_id"],
            table_row_index=(
                int(row["table_row_index"])
                if row["table_row_index"] is not None
                else None
            ),
            extraction_confidence=float(row["extraction_confidence"] or 0.0),
            authority_status=str(row["authority_status"] or "unknown"),
            metadata=dict(row["metadata"] or {}),
            text=str(row["text"] or ""),
            char_count=len(str(row["text"] or "")),
        )
        for row in rows
    ]

    return DocumentChunkPageOut(
        document_id=document.id,
        filename=document.filename,
        run_id=run["id"],
        processing_version=str(run["processing_version"] or ""),
        run_started_at=run["started_at"],
        run_completed_at=run["completed_at"],
        run_metrics=dict(run["metrics"] or {}),
        run_warnings=list(run["warnings"] or []),
        total_chunks=total_chunks,
        filtered_chunks=filtered_chunks,
        offset=offset,
        limit=limit,
        content_types=content_types,
        authority_statuses=authority_statuses,
        chunks=chunks,
    )


'''
    return replace_once(source, anchor, endpoint + anchor, "document v5 chunk explorer endpoint")


def patch_api_ts(source: str) -> str:
    if "export interface DocumentChunkPage" not in source:
        anchor = "export interface DocumentBatchResponse {\n"
        addition = '''// IMS_PDF_CHUNK_EXPLORER_V1\nexport interface DocumentChunkRecord {\n  id: string\n  chunk_index: number\n  page_number: number\n  page_end: number\n  content_type: string\n  parent_key: string\n  section_path: string[]\n  heading: string\n  table_id: string | null\n  table_row_index: number | null\n  extraction_confidence: number\n  authority_status: string\n  metadata: Record<string, unknown>\n  text: string\n  char_count: number\n}\n\nexport interface DocumentChunkPage {\n  document_id: string\n  filename: string\n  run_id: string\n  processing_version: string\n  run_started_at?: string | null\n  run_completed_at?: string | null\n  run_metrics: Record<string, unknown>\n  run_warnings: unknown[]\n  total_chunks: number\n  filtered_chunks: number\n  offset: number\n  limit: number\n  content_types: string[]\n  authority_statuses: string[]\n  chunks: DocumentChunkRecord[]\n}\n\nexport interface DocumentChunkFilters {\n  offset?: number\n  limit?: number\n  q?: string\n  page?: number\n  content_type?: string\n  authority_status?: string\n}\n\n'''
        source = replace_once(source, anchor, addition + anchor, "frontend document chunk interfaces")

    if "export async function listDocumentChunks(" not in source:
        anchor = '''export async function downloadDocument(\n'''
        function = '''export async function listDocumentChunks(\n  documentId: string,\n  filters: DocumentChunkFilters = {},\n): Promise<DocumentChunkPage> {\n  const params = new URLSearchParams()\n  if (filters.offset !== undefined) params.set('offset', String(filters.offset))\n  if (filters.limit !== undefined) params.set('limit', String(filters.limit))\n  if (filters.q) params.set('q', filters.q)\n  if (filters.page !== undefined) params.set('page', String(filters.page))\n  if (filters.content_type) params.set('content_type', filters.content_type)\n  if (filters.authority_status) params.set('authority_status', filters.authority_status)\n  const queryString = params.toString()\n  const suffix = queryString ? `?${queryString}` : ''\n  return apiRequest<DocumentChunkPage>(\n    `/api/documents/${encodeURIComponent(documentId)}/v5-chunks${suffix}`,\n  )\n}\n\n'''
        source = replace_once(source, anchor, function + anchor, "frontend listDocumentChunks API")
    return source


def compatible_documents_panel(source: str) -> bool:
    if "DocumentChunkExplorer" in source:
        return True
    return (
        "documents-shell" in source
        and "listDocuments" in source
        and "document-grid" in source
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a PDF-wise active RAG v5 chunk explorer for retrieval diagnostics."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    here = Path(__file__).resolve().parent
    payload = here / "payload"

    paths = {
        "models": repo / "backend/app/models.py",
        "api": repo / "backend/app/api.py",
        "api_ts": repo / "frontend/src/services/api.ts",
        "documents": repo / "frontend/src/components/DocumentsPanel.vue",
        "explorer": repo / "frontend/src/components/DocumentChunkExplorer.vue",
        "test": repo / "backend/tests/test_pdf_chunk_explorer_contract.py",
    }
    payload_documents = payload / "frontend/src/components/DocumentsPanel.vue"
    payload_explorer = payload / "frontend/src/components/DocumentChunkExplorer.vue"
    payload_test = payload / "backend/tests/test_pdf_chunk_explorer_contract.py"

    required = [
        paths["models"],
        paths["api"],
        paths["api_ts"],
        paths["documents"],
        payload_documents,
        payload_explorer,
        payload_test,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Required file(s) not found:\n" + "\n".join(missing))

    current_documents = paths["documents"].read_text(encoding="utf-8-sig")
    if not compatible_documents_panel(current_documents):
        raise RuntimeError(
            "DocumentsPanel.vue is not a compatible IMS UI v2 document panel. "
            "No files were written. Inspect the local component before applying."
        )

    originals = {
        "models": paths["models"].read_text(encoding="utf-8-sig"),
        "api": paths["api"].read_text(encoding="utf-8-sig"),
        "api_ts": paths["api_ts"].read_text(encoding="utf-8-sig"),
    }
    transformed = {
        "models": patch_models(originals["models"]),
        "api": patch_api(originals["api"]),
        "api_ts": patch_api_ts(originals["api_ts"]),
    }

    # Preflight all Python transforms before changing the checkout.
    compile(transformed["models"], str(paths["models"]), "exec")
    compile(transformed["api"], str(paths["api"]), "exec")
    compile(payload_test.read_text(encoding="utf-8-sig"), str(paths["test"]), "exec")
    print("[preflight] backend Python transforms compile; no repository file written yet")

    for key in ("models", "api", "api_ts"):
        path = paths[key]
        if transformed[key] != originals[key]:
            backup(path)
            path.write_text(transformed[key], encoding="utf-8", newline="\n")
            print(f"[patched] {path}")
        else:
            print(f"[already patched] {path}")

    new_documents = payload_documents.read_text(encoding="utf-8-sig")
    if current_documents != new_documents:
        backup(paths["documents"])
        paths["documents"].write_text(new_documents, encoding="utf-8", newline="\n")
        print(f"[updated] {paths['documents']}")
    else:
        print(f"[already current] {paths['documents']}")

    paths["explorer"].parent.mkdir(parents=True, exist_ok=True)
    if paths["explorer"].exists():
        backup(paths["explorer"])
    shutil.copy2(payload_explorer, paths["explorer"])
    print(f"[copied] {paths['explorer']}")

    paths["test"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(payload_test, paths["test"])
    print(f"[copied] {paths['test']}")

    # Keep the v5.2 payload document panel aligned so reinstalling that package later
    # does not remove the explorer entry point.
    repo_payload_documents = repo / "payload/frontend/src/components/DocumentsPanel.vue"
    repo_payload_explorer = repo / "payload/frontend/src/components/DocumentChunkExplorer.vue"
    if repo_payload_documents.exists():
        backup(repo_payload_documents)
        repo_payload_documents.write_text(new_documents, encoding="utf-8", newline="\n")
        print(f"[updated] {repo_payload_documents}")
    repo_payload_explorer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(payload_explorer, repo_payload_explorer)
    print(f"[copied] {repo_payload_explorer}")

    for path in (paths["models"], paths["api"], paths["test"]):
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied PDF Chunk Explorer v1.")
    print(" - select any ready PDF from Documents -> Inspect chunks")
    print(" - browse every active RAG v5 chunk with pagination")
    print(" - filter by text, PDF page, content type and authority status")
    print(" - inspect chunk index, section path, heading, parent, confidence, table row and metadata")
    print(" - toggle clean vs raw [PDF STRUCTURE] text")
    print(" - copy raw chunk text/JSON and export the visible chunk page as JSON")
    print(" - read-only authenticated endpoint; embedding vectors are never returned")
    print()
    print("No DB migration, PDF reprocessing, OCR rerun, chunk rebuild or embedding rebuild is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
