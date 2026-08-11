from __future__ import annotations

import logging

from sqlalchemy import select

from app.db import SessionLocal, initialize_database
from app.db_models import Document
from app.document_processing import process_document_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Rebuild stored chunks after changing chunking/retrieval logic.

    Run inside the backend container after applying this patch:

        python -m app.reprocess_documents

    Only document IDs are loaded up-front. Each PDF is then loaded and processed
    in a fresh DB session so a large collection does not keep every PDF binary in
    memory at the same time. One failed document does not abort the remaining
    batch.
    """
    initialize_database()
    with SessionLocal() as db:
        document_ids = list(
            db.scalars(select(Document.id).order_by(Document.created_at.asc()))
        )

    logger.info("Reprocessing %s document(s)", len(document_ids))
    summary = process_document_ids(document_ids)
    logger.info(
        "Finished reprocessing: succeeded=%s failed=%s missing=%s",
        summary.succeeded,
        summary.failed,
        summary.missing,
    )
    if summary.failed or summary.missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
