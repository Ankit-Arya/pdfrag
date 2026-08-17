from __future__ import annotations

import logging

from app.db import SessionLocal, initialize_database
from app.rag.embeddings import embedding_service
from app.rag.smart_index import backfill_all
from app.rag.smart_schema import ensure_smart_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    initialize_database()
    ensure_smart_schema()
    if not embedding_service.warmup():
        logger.warning(
            "Embedding backend is unavailable. Procedure-card embeddings may fail; "
            "terminology and deterministic rule extraction can still be created."
        )
    with SessionLocal() as db:
        totals = backfill_all(db)
    logger.info("Smart RAG backfill complete: %s", totals)
    print(totals)


if __name__ == "__main__":
    main()
