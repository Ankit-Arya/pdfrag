from __future__ import annotations

# ruff: noqa: E501

import logging
import os

from sqlalchemy import text

from app.config import get_settings
from app.db import engine

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def ensure_smart_schema() -> None:
    """Create additive smart-RAG tables and indexes without rewriting source chunks."""
    if not _env_bool("SMART_RAG_AUTO_SCHEMA", True):
        logger.info("SMART_RAG_AUTO_SCHEMA disabled; skipping smart RAG schema setup")
        return

    dimensions = int(get_settings().embedding_dimensions)
    if dimensions <= 0 or dimensions > 4096:
        raise RuntimeError(f"Unsafe embedding dimension for smart schema: {dimensions}")

    core_ddl = [
        """
        CREATE TABLE IF NOT EXISTS rag_terminology (
            id bigserial PRIMARY KEY,
            alias text NOT NULL,
            alias_norm text NOT NULL,
            canonical_name text NOT NULL,
            canonical_norm text NOT NULL,
            concept_type varchar(40) NOT NULL DEFAULT 'term',
            confidence double precision NOT NULL DEFAULT 0.80,
            verified boolean NOT NULL DEFAULT false,
            document_id uuid NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_id uuid NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
            page_number integer NULL,
            evidence text NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_rag_terminology_evidence
        ON rag_terminology (
            alias_norm,
            canonical_norm,
            COALESCE(chunk_id, '00000000-0000-0000-0000-000000000000'::uuid)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_terminology_alias ON rag_terminology(alias_norm)",
        "CREATE INDEX IF NOT EXISTS ix_rag_terminology_canonical ON rag_terminology(canonical_norm)",
        """
        CREATE TABLE IF NOT EXISTS rag_procedure_cards (
            id bigserial PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            section_key text NOT NULL,
            title text NOT NULL,
            search_text text NOT NULL,
            page_start integer NOT NULL DEFAULT 1,
            page_end integer NOT NULL DEFAULT 1,
            start_chunk_index integer NOT NULL DEFAULT 0,
            end_chunk_index integer NOT NULL DEFAULT 0,
            applicability jsonb NOT NULL DEFAULT '{}'::jsonb,
            embedding vector(%d) NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE(document_id, section_key)
        )
        """ % dimensions,
        "CREATE INDEX IF NOT EXISTS ix_rag_procedure_cards_document ON rag_procedure_cards(document_id)",
        """
        CREATE INDEX IF NOT EXISTS ix_rag_procedure_cards_fts_simple
        ON rag_procedure_cards USING gin (to_tsvector('simple', search_text))
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_rag_procedure_cards_fts_english
        ON rag_procedure_cards USING gin (to_tsvector('english', search_text))
        """,
        """
        CREATE TABLE IF NOT EXISTS rag_rules (
            id bigserial PRIMARY KEY,
            procedure_card_id bigint NULL REFERENCES rag_procedure_cards(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_id uuid NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
            page_number integer NOT NULL,
            field_tokens text[] NOT NULL DEFAULT ARRAY[]::text[],
            operator varchar(4) NOT NULL,
            threshold double precision NOT NULL,
            unit varchar(32) NOT NULL DEFAULT '',
            condition_text text NOT NULL,
            confidence double precision NOT NULL DEFAULT 0.95,
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE(chunk_id, operator, threshold, condition_text)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_rules_document ON rag_rules(document_id)",
        "CREATE INDEX IF NOT EXISTS ix_rag_rules_field_tokens ON rag_rules USING gin(field_tokens)",
    ]

    with engine.begin() as conn:
        for statement in core_ddl:
            conn.execute(text(statement))

    if _env_bool("SMART_RAG_BUILD_INDEXES", True):
        optional_indexes = [
            (
                "chunk HNSW",
                """
                CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """,
            ),
            (
                "chunk simple FTS",
                """
                CREATE INDEX IF NOT EXISTS ix_document_chunks_fts_simple
                ON document_chunks USING gin (to_tsvector('simple', text))
                """,
            ),
            (
                "chunk English FTS",
                """
                CREATE INDEX IF NOT EXISTS ix_document_chunks_fts_english
                ON document_chunks USING gin (to_tsvector('english', text))
                """,
            ),
            (
                "procedure-card HNSW",
                """
                CREATE INDEX IF NOT EXISTS ix_rag_procedure_cards_embedding_hnsw
                ON rag_procedure_cards USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """,
            ),
        ]
        for label, statement in optional_indexes:
            try:
                with engine.begin() as conn:
                    conn.execute(text(statement))
            except Exception:
                # Do not make the service undeployable if an older pgvector build or
                # constrained DB cannot build an ANN index. Retrieval still works,
                # but logs make the missing acceleration explicit.
                logger.exception("Could not create optional smart RAG index: %s", label)

    try:
        with engine.begin() as conn:
            for table in ("document_chunks", "rag_terminology", "rag_procedure_cards", "rag_rules"):
                conn.execute(text(f"ANALYZE {table}"))
    except Exception:
        logger.exception("Smart RAG ANALYZE failed; continuing")
    logger.info("Smart RAG schema ready")
