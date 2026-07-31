from __future__ import annotations

import logging

from sqlalchemy import select

from app.db import SessionLocal, initialize_database
from app.db_models import Document
from app.rag.service import rag_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Rebuild stored chunks after changing chunking/retrieval logic.

    Run inside the backend container after applying this patch:

        python -m app.reprocess_documents

    The PDF binary content is already stored in the documents table, so this does
    not require users to re-upload files.
    """
    initialize_database()
    with SessionLocal() as db:
        documents = list(db.scalars(select(Document).order_by(Document.created_at.asc())))
        logger.info("Reprocessing %s document(s)", len(documents))
        for document in documents:
            logger.info("Reprocessing %s (%s)", document.filename, document.id)
            rag_service.process_document(db, document)
        logger.info("Finished reprocessing documents")


if __name__ == "__main__":
    main()
