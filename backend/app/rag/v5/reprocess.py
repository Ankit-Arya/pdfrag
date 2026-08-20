from __future__ import annotations

import argparse
import logging
import uuid

from sqlalchemy import select, text

from app.db import SessionLocal
from app.db_models import Document, DocumentStatus
from app.rag.v5 import PROCESSING_VERSION
from app.rag.v5.ingestion import process_document_v5
from app.rag.v5.schema import ensure_v5_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _already_current(db, document_id: uuid.UUID) -> bool:  # type: ignore[no-untyped-def]
    return bool(
        db.execute(
            text(
                """
                SELECT 1 FROM rag_v5_processing_runs
                WHERE document_id=CAST(:document_id AS uuid)
                  AND is_active=true AND status='ready' AND processing_version=:version
                LIMIT 1
                """
            ),
            {"document_id": str(document_id), "version": PROCESSING_VERSION},
        ).scalar()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structure-preserving RAG v5 generations without deleting v4 chunks.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the active v5 generation is current")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many documents (0 = all)")
    parser.add_argument("--document-id", action="append", default=[], help="Process only the given document UUID; may be repeated")
    args = parser.parse_args()

    ensure_v5_schema()
    requested_ids = {uuid.UUID(value) for value in args.document_id}
    with SessionLocal() as db:
        stmt = select(Document).where(Document.status == DocumentStatus.ready).order_by(Document.created_at, Document.id)
        documents = list(db.scalars(stmt))
        if requested_ids:
            documents = [document for document in documents if document.id in requested_ids]
        if args.limit > 0:
            documents = documents[: args.limit]

    attempted = succeeded = skipped = failed = 0
    for document_stub in documents:
        attempted += 1
        try:
            with SessionLocal() as db:
                document = db.get(Document, document_stub.id)
                if document is None:
                    continue
                if not args.force and _already_current(db, document.id):
                    skipped += 1
                    logger.info("SKIP current v5 generation: %s", document.filename)
                    continue
                logger.info("V5 PROCESS %s", document.filename)
                summary = process_document_v5(db, document, publish_document_state=False)
                succeeded += 1
                logger.info(
                    "V5 READY %s pages=%d chunks=%d tables=%d rows=%d ocr=%d low_quality=%d",
                    document.filename,
                    summary.pages,
                    summary.chunks,
                    summary.tables,
                    summary.table_rows,
                    summary.ocr_pages,
                    summary.low_quality_pages,
                )
        except Exception:
            failed += 1
            logger.exception("V5 FAILED %s", document_stub.filename)

    print(
        f"RAG v5 reprocess complete: attempted={attempted} succeeded={succeeded} "
        f"skipped_current={skipped} failed={failed} version={PROCESSING_VERSION}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
