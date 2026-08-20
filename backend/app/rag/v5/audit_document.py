from __future__ import annotations

import argparse
import json

from sqlalchemy import text

from app.db import SessionLocal
from app.rag.v5 import PROCESSING_VERSION
from app.rag.v5.schema import ensure_v5_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one active RAG v5 document generation.")
    parser.add_argument("--filename", required=True, help="Case-insensitive filename substring")
    parser.add_argument("--find", default="", help="Optional phrase to locate in v5 chunks/table rows")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    ensure_v5_schema()
    with SessionLocal() as db:
        docs = list(
            db.execute(
                text(
                    """
                    SELECT d.id, d.filename, r.id AS run_id, r.processing_version, r.metrics, r.warnings
                    FROM documents d
                    JOIN rag_v5_processing_runs r ON r.document_id=d.id
                    WHERE r.is_active=true AND r.status='ready'
                      AND lower(d.filename) LIKE lower(:pattern)
                    ORDER BY lower(d.filename)
                    LIMIT 10
                    """
                ),
                {"pattern": f"%{args.filename}%"},
            ).mappings()
        )
        if not docs:
            print("No active v5 generation matched that filename.")
            return 2
        for doc in docs:
            print(f"\nDOCUMENT: {doc['filename']}")
            print(f"run_id: {doc['run_id']}")
            print(f"processing_version: {doc['processing_version']} (expected {PROCESSING_VERSION})")
            print("metrics:")
            print(json.dumps(doc["metrics"], indent=2, ensure_ascii=False, default=str))
            if doc["warnings"]:
                print("warnings:")
                print(json.dumps(doc["warnings"], indent=2, ensure_ascii=False, default=str))
            if args.find:
                rows = list(
                    db.execute(
                        text(
                            """
                            SELECT c.page_number, c.page_end, c.content_type, c.authority_status,
                                   c.heading, c.section_path, c.text
                            FROM rag_v5_chunks c
                            WHERE c.run_id=CAST(:run_id AS uuid)
                              AND lower(c.text) LIKE lower(:needle)
                            ORDER BY c.page_number, c.chunk_index
                            LIMIT :limit
                            """
                        ),
                        {"run_id": str(doc["run_id"]), "needle": f"%{args.find}%", "limit": max(1, min(100, args.limit))},
                    ).mappings()
                )
                print(f"matches for {args.find!r}: {len(rows)}")
                for index, row in enumerate(rows, 1):
                    section = " > ".join(str(value) for value in (row["section_path"] or []))
                    print(
                        f"\n[{index}] page {row['page_number']}-{row['page_end']} "
                        f"type={row['content_type']} authority={row['authority_status']}"
                    )
                    if section:
                        print(f"section: {section}")
                    print(str(row["text"])[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
