from __future__ import annotations

# ruff: noqa: E501

from sqlalchemy import text

from app.db import SessionLocal, initialize_database
from app.rag.smart_schema import ensure_smart_schema


def main() -> None:
    initialize_database()
    ensure_smart_schema()
    with SessionLocal() as db:
        counts = db.execute(
            text(
                """
                SELECT
                  (SELECT count(*) FROM documents WHERE status = 'ready') AS ready_documents,
                  (SELECT count(*) FROM document_chunks) AS chunks,
                  (SELECT count(*) FROM rag_terminology) AS terminology_rows,
                  (SELECT count(*) FROM rag_procedure_cards) AS procedure_cards,
                  (SELECT count(*) FROM rag_rules) AS rules
                """
            )
        ).mappings().one()
        print("Smart RAG counts")
        for key, value in counts.items():
            print(f"  {key}: {value}")

        rows = db.execute(
            text(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename IN ('document_chunks', 'rag_terminology', 'rag_procedure_cards', 'rag_rules')
                  AND (indexname LIKE '%hnsw%' OR indexname LIKE '%fts%' OR indexname LIKE 'ix_rag_%')
                ORDER BY indexname
                """
            )
        ).scalars()
        print("Smart RAG indexes")
        for name in rows:
            print(f"  {name}")


if __name__ == "__main__":
    main()
