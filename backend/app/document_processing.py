from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import update

from app.db import SessionLocal
from app.db_models import Document, DocumentStatus
from app.rag.service import rag_service

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessingSummary:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    missing: int = 0


def _mark_failed_if_still_processing(document_id: uuid.UUID) -> None:
    """Best-effort recovery for failures outside RagService's own handler."""
    try:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None or document.status != DocumentStatus.processing:
                return
            document.status = DocumentStatus.failed
            document.error = "Document processing failed. Check server logs for details."
            db.commit()
    except Exception:  # noqa: BLE001 - logging must not abort the remaining batch.
        logger.exception("Could not persist failure state for document %s", document_id)


def process_document_ids(document_ids: Iterable[uuid.UUID]) -> ProcessingSummary:
    """Process documents sequentially with one short-lived DB session per PDF.

    A failure in one document cannot abort the remaining batch. Closing the
    session after every item also releases its lazily loaded PDF payload and ORM
    identity map before the next document starts.
    """
    summary = ProcessingSummary()

    for document_id in document_ids:
        summary.attempted += 1
        filename = str(document_id)
        try:
            with SessionLocal() as db:
                document = db.get(Document, document_id)
                if document is None:
                    summary.missing += 1
                    logger.warning("Skipping missing document %s", document_id)
                    continue

                filename = document.filename
                rag_service.process_document(db, document)
        except Exception:  # noqa: BLE001 - isolate every document in the batch.
            summary.failed += 1
            logger.exception(
                "Background document processing failed for %s (%s)",
                filename,
                document_id,
            )
            _mark_failed_if_still_processing(document_id)
            continue

        summary.succeeded += 1

    logger.info(
        "Document batch finished: attempted=%s succeeded=%s failed=%s missing=%s",
        summary.attempted,
        summary.succeeded,
        summary.failed,
        summary.missing,
    )
    return summary


def recover_interrupted_processing() -> int:
    """Make documents retryable after a backend restart interrupted background work."""
    with SessionLocal() as db:
        result = db.execute(
            update(Document)
            .where(Document.status == DocumentStatus.processing)
            .values(
                status=DocumentStatus.uploaded,
                error=(
                    "Processing was interrupted by a backend restart. "
                    "Queue this document again."
                ),
            )
        )
        db.commit()
        recovered = int(result.rowcount or 0)

    if recovered:
        logger.warning("Recovered %s interrupted document(s) for retry", recovered)
    return recovered
