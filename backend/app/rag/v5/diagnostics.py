from __future__ import annotations

import argparse
import json

from sqlalchemy import text

from app.db import SessionLocal
from app.rag.v5.schema import ensure_v5_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit RAG v5 corpus coverage and extraction quality.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any ready document lacks an active v5 generation")
    args = parser.parse_args()
    ensure_v5_schema()
    with SessionLocal() as db:
        row = db.execute(
            text(
                """
                SELECT
                    (SELECT count(*) FROM documents WHERE status='ready') AS ready_documents,
                    (SELECT count(DISTINCT document_id) FROM rag_v5_processing_runs WHERE is_active=true AND status='ready') AS v5_documents,
                    (SELECT count(*) FROM rag_v5_chunks c JOIN rag_v5_processing_runs r ON r.id=c.run_id WHERE r.is_active=true AND r.status='ready') AS v5_chunks,
                    (SELECT count(*) FROM rag_v5_tables t JOIN rag_v5_processing_runs r ON r.id=t.run_id WHERE r.is_active=true AND r.status='ready') AS tables,
                    (SELECT count(*) FROM rag_v5_table_rows tr JOIN rag_v5_tables t ON t.id=tr.table_id JOIN rag_v5_processing_runs r ON r.id=t.run_id WHERE r.is_active=true AND r.status='ready') AS table_rows,
                    (SELECT count(*) FROM rag_v5_pages p JOIN rag_v5_processing_runs r ON r.id=p.run_id WHERE r.is_active=true AND r.status='ready' AND p.ocr_used=true) AS ocr_pages,
                    (SELECT count(*) FROM rag_v5_pages p JOIN rag_v5_processing_runs r ON r.id=p.run_id WHERE r.is_active=true AND r.status='ready' AND p.quality_score < 0.55) AS low_quality_pages,
                    (SELECT count(*) FROM rag_v5_elements e JOIN rag_v5_processing_runs r ON r.id=e.run_id WHERE r.is_active=true AND r.status='ready' AND e.element_type='heading') AS headings,
                    (SELECT count(*) FROM rag_v5_elements e JOIN rag_v5_processing_runs r ON r.id=e.run_id WHERE r.is_active=true AND r.status='ready' AND e.element_type='figure') AS figures,
                    (SELECT count(*) FROM rag_v5_authority a JOIN rag_v5_processing_runs r ON r.id=a.run_id WHERE r.is_active=true AND r.status='ready') AS authority_directives,
                    (SELECT count(*) FROM rag_v5_terminology t JOIN rag_v5_processing_runs r ON r.id=t.run_id WHERE r.is_active=true AND r.status='ready') AS terminology,
                    (SELECT COALESCE(sum(jsonb_array_length(COALESCE(r.metrics->'table_rejected_pages', '[]'::jsonb))), 0) FROM rag_v5_processing_runs r WHERE r.is_active=true AND r.status='ready') AS table_rejected_pages
                """
            )
        ).mappings().one()
        missing = list(
            db.execute(
                text(
                    """
                    SELECT d.id, d.filename
                    FROM documents d
                    WHERE d.status='ready'
                      AND NOT EXISTS (
                        SELECT 1 FROM rag_v5_processing_runs r
                        WHERE r.document_id=d.id AND r.is_active=true AND r.status='ready'
                      )
                    ORDER BY lower(d.filename)
                    LIMIT 50
                    """
                )
            ).mappings()
        )
        worst = list(
            db.execute(
                text(
                    """
                    SELECT d.filename, p.page_number, round(p.quality_score::numeric, 3) AS quality,
                           p.ocr_used, p.warnings
                    FROM rag_v5_pages p
                    JOIN rag_v5_processing_runs r ON r.id=p.run_id AND r.is_active=true AND r.status='ready'
                    JOIN documents d ON d.id=p.document_id
                    WHERE p.quality_score < 0.65
                    ORDER BY p.quality_score, lower(d.filename), p.page_number
                    LIMIT 30
                    """
                )
            ).mappings()
        )
    payload = {key: int(value or 0) for key, value in row.items()}
    payload["coverage_percent"] = round(100.0 * payload["v5_documents"] / max(1, payload["ready_documents"]), 2)
    payload["missing_documents"] = [{"id": str(item["id"]), "filename": item["filename"]} for item in missing]
    payload["low_quality_samples"] = [dict(item) for item in worst]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print("RAG v5 diagnostics")
        print("------------------")
        for key in (
            "ready_documents", "v5_documents", "coverage_percent", "v5_chunks", "tables", "table_rows",
            "headings", "ocr_pages", "low_quality_pages", "figures", "terminology",
            "authority_directives", "table_rejected_pages",
        ):
            print(f"{key}: {payload[key]}")
        if missing:
            print("\nReady documents without active v5 generation (first 50):")
            for item in missing:
                print(f"- {item['filename']} ({item['id']})")
        if worst:
            print("\nLow-confidence pages (first 30):")
            for item in worst:
                print(f"- {item['filename']} p.{item['page_number']} quality={item['quality']} ocr={item['ocr_used']}")
    if args.strict and payload["v5_documents"] != payload["ready_documents"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
