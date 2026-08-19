from __future__ import annotations

# ruff: noqa: E501

import os

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
                  (SELECT count(*) FROM rag_rules) AS rules,
                  (SELECT count(*) FROM rag_authority_directives) AS authority_directives
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
                WHERE tablename IN ('document_chunks', 'rag_terminology', 'rag_procedure_cards', 'rag_rules', 'rag_authority_directives')
                  AND (indexname LIKE '%hnsw%' OR indexname LIKE '%fts%' OR indexname LIKE 'ix_rag_%')
                ORDER BY indexname
                """
            )
        ).scalars()
        print("Smart RAG indexes")
        for name in rows:
            print(f"  {name}")

        authority = db.execute(
            text(
                """
                SELECT directive_type, count(*) AS rows, count(DISTINCT document_id) AS documents
                FROM rag_authority_directives
                GROUP BY directive_type
                ORDER BY directive_type
                """
            )
        ).mappings()
        print("Authority directives")
        for row in authority:
            print(f"  {row['directive_type']}: {row['rows']} rows across {row['documents']} document(s)")

        conflicts = db.execute(
            text(
                """
                SELECT count(*)
                FROM (
                    SELECT alias_norm
                    FROM rag_terminology
                    GROUP BY alias_norm
                    HAVING count(DISTINCT canonical_norm) > 1
                ) x
                """
            )
        ).scalar_one()
        print(f"Terminology aliases with multiple corpus meanings: {conflicts}")

        print("AI-first understanding")
        for name, default in (
            ("SMART_RAG_AI_INTERPRETATION", "1"),
            ("SMART_RAG_AI_SEARCH_QUERIES", "4"),
            ("SMART_RAG_AI_EVIDENCE_REVIEW", "1"),
            ("SMART_RAG_AI_RETRY_QUERIES", "2"),
            ("SMART_RAG_AI_ANSWER_VERIFY", "1"),
            ("SMART_RAG_AI_VERIFY_SOURCES", "24"),
        ):
            print(f"  {name}: {os.getenv(name, default)}")


if __name__ == "__main__":
    main()
