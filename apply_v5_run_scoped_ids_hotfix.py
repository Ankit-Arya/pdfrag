from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = "def _run_scoped_id(run_id: uuid.UUID, kind: str, logical_id: str) -> str:"

def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)

def main() -> int:
    parser = argparse.ArgumentParser(description="Hotfix RAG v5 persisted IDs so retained/retried generations cannot collide.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    path = repo / "backend/app/rag/v5/ingestion.py"
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print("V5 run-scoped ID hotfix is already applied.")
        return 0

    backup = path.with_suffix(".py.bak-before-run-scoped-ids")
    if not backup.exists():
        shutil.copy2(path, backup)

    source = replace_once(source,
'''def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _body(value: str) -> str:
''',
'''def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _run_scoped_id(run_id: uuid.UUID, kind: str, logical_id: str) -> str:
    """Scope deterministic extraction IDs to one persisted processing generation."""
    return str(uuid.uuid5(run_id, f"{kind}:{logical_id}"))


def _persisted_chunk_metadata(chunk: V5Chunk, element_db_ids: dict[str, str]) -> dict:
    metadata = dict(chunk.metadata or {})
    element_ids = metadata.get("element_ids")
    if isinstance(element_ids, list):
        metadata["element_ids"] = [
            element_db_ids.get(str(element_id), str(element_id))
            for element_id in element_ids
        ]
    return metadata


def _body(value: str) -> str:
''',
"helper insertion")

    source = replace_once(source,
'''        if not chunks:
            raise ValueError("RAG v5 created no semantic chunks")
        directives = _authority_metadata(chunks)

        settings = get_settings()
''',
'''        if not chunks:
            raise ValueError("RAG v5 created no semantic chunks")
        directives = _authority_metadata(chunks)

        # Logical IDs are reproducible extraction identities. Database UUID primary keys must
        # be unique per retained processing run, otherwise retries/cross-document layouts collide.
        logical_element_ids = [element.element_id for element in layout.elements]
        logical_table_ids = [table.table_id for table in layout.tables]
        logical_chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(logical_element_ids)) != len(logical_element_ids):
            raise ValueError("RAG v5 generated duplicate logical element IDs inside one document")
        if len(set(logical_table_ids)) != len(logical_table_ids):
            raise ValueError("RAG v5 generated duplicate logical table IDs inside one document")
        if len(set(logical_chunk_ids)) != len(logical_chunk_ids):
            raise ValueError("RAG v5 generated duplicate logical chunk IDs inside one document")

        element_db_ids = {
            logical_id: _run_scoped_id(run_id, "element", logical_id)
            for logical_id in logical_element_ids
        }
        table_db_ids = {
            logical_id: _run_scoped_id(run_id, "table", logical_id)
            for logical_id in logical_table_ids
        }
        chunk_db_ids = {
            logical_id: _run_scoped_id(run_id, "chunk", logical_id)
            for logical_id in logical_chunk_ids
        }

        settings = get_settings()
''',
"map insertion")

    source = replace_once(source,
'''                    "id": element.element_id, "run_id": str(run_id), "document_id": str(document.id),
''',
'''                    "id": element_db_ids[element.element_id], "run_id": str(run_id), "document_id": str(document.id),
''',
"element id")

    source = replace_once(source,
'''                    "id": table.table_id, "run_id": str(run_id), "document_id": str(document.id),
''',
'''                    "id": table_db_ids[table.table_id], "run_id": str(run_id), "document_id": str(document.id),
''',
"table id")

    source = replace_once(source,
'''                        "table_id": table.table_id, "document_id": str(document.id),
''',
'''                        "table_id": table_db_ids[table.table_id], "document_id": str(document.id),
''',
"table row fk")

    source = replace_once(source,
'''                        "id": chunk.chunk_id, "run_id": str(run_id), "document_id": str(document.id),
''',
'''                        "id": chunk_db_ids[chunk.chunk_id], "run_id": str(run_id), "document_id": str(document.id),
''',
"chunk id")

    source = replace_once(source,
'''                        "heading": chunk.heading, "table_id": chunk.table_id or "",
''',
'''                        "heading": chunk.heading,
                        "table_id": table_db_ids[chunk.table_id] if chunk.table_id else "",
''',
"chunk table fk")

    source = replace_once(source,
'''                        "authority_status": chunk.authority_status, "metadata": _json(chunk.metadata),
''',
'''                        "authority_status": chunk.authority_status,
                        "metadata": _json(_persisted_chunk_metadata(chunk, element_db_ids)),
''',
"chunk metadata")

    source = replace_once(source,
'''                        "chunk_id": chunk.chunk_id, "page_number": chunk.page_number,
''',
'''                        "chunk_id": chunk_db_ids[chunk.chunk_id], "page_number": chunk.page_number,
''',
"terminology chunk id")

    path.write_text(source, encoding="utf-8", newline="\n")
    print("Applied RAG v5 run-scoped persistence-ID hotfix.")
    print(f"Backup: {backup}")
    print("RAG_V5_PROCESSING_VERSION is intentionally unchanged.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
