from __future__ import annotations

import logging

from sqlalchemy import text

from app.config import get_settings
from app.db import engine

logger = logging.getLogger(__name__)


def ensure_v5_schema() -> None:
    settings = get_settings()
    dimensions = int(settings.embedding_dimensions)
    if dimensions <= 0 or dimensions > 4096:
        raise RuntimeError(f"Unsafe embedding dimension for RAG v5: {dimensions}")

    ddl = [
        """
        CREATE TABLE IF NOT EXISTS rag_v5_processing_runs (
            id uuid PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            processing_version varchar(64) NOT NULL,
            status varchar(24) NOT NULL,
            is_active boolean NOT NULL DEFAULT false,
            metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
            warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
            error text NULL,
            started_at timestamptz NOT NULL DEFAULT now(),
            completed_at timestamptz NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_runs_document ON rag_v5_processing_runs(document_id, is_active)",
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_rag_v5_runs_one_active ON rag_v5_processing_runs(document_id) WHERE is_active",
        """
        CREATE TABLE IF NOT EXISTS rag_v5_pages (
            id bigserial PRIMARY KEY,
            run_id uuid NOT NULL REFERENCES rag_v5_processing_runs(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number integer NOT NULL,
            width double precision NOT NULL,
            height double precision NOT NULL,
            language varchar(64) NOT NULL DEFAULT '',
            native_chars integer NOT NULL DEFAULT 0,
            ocr_used boolean NOT NULL DEFAULT false,
            quality_score double precision NOT NULL DEFAULT 1.0,
            warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
            UNIQUE(run_id, page_number)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_pages_document_page ON rag_v5_pages(document_id, page_number)",
        """
        CREATE TABLE IF NOT EXISTS rag_v5_elements (
            id uuid PRIMARY KEY,
            run_id uuid NOT NULL REFERENCES rag_v5_processing_runs(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number integer NOT NULL,
            order_index integer NOT NULL,
            element_type varchar(32) NOT NULL,
            parent_key text NOT NULL DEFAULT '',
            text text NOT NULL,
            bbox jsonb NOT NULL DEFAULT '[]'::jsonb,
            heading_level integer NULL,
            confidence double precision NOT NULL DEFAULT 1.0,
            extraction_source varchar(64) NOT NULL DEFAULT 'native',
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_elements_document_page ON rag_v5_elements(document_id, page_number, order_index)",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_elements_type ON rag_v5_elements(element_type)",
        """
        CREATE TABLE IF NOT EXISTS rag_v5_tables (
            id uuid PRIMARY KEY,
            run_id uuid NOT NULL REFERENCES rag_v5_processing_runs(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            table_key text NOT NULL,
            title text NOT NULL DEFAULT '',
            page_start integer NOT NULL,
            page_end integer NOT NULL,
            columns jsonb NOT NULL DEFAULT '[]'::jsonb,
            bbox_by_page jsonb NOT NULL DEFAULT '{}'::jsonb,
            confidence double precision NOT NULL DEFAULT 1.0,
            extraction_source varchar(64) NOT NULL DEFAULT '',
            section_path jsonb NOT NULL DEFAULT '[]'::jsonb,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            UNIQUE(run_id, table_key)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_tables_document ON rag_v5_tables(document_id, page_start)",
        """
        CREATE TABLE IF NOT EXISTS rag_v5_table_rows (
            id bigserial PRIMARY KEY,
            table_id uuid NOT NULL REFERENCES rag_v5_tables(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            row_index integer NOT NULL,
            page_number integer NOT NULL,
            cells jsonb NOT NULL DEFAULT '[]'::jsonb,
            normalized_text text NOT NULL,
            bbox jsonb NOT NULL DEFAULT '[]'::jsonb,
            confidence double precision NOT NULL DEFAULT 1.0,
            extraction_source varchar(64) NOT NULL DEFAULT '',
            UNIQUE(table_id, row_index)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_rows_document_page ON rag_v5_table_rows(document_id, page_number)",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_rows_fts ON rag_v5_table_rows USING gin (to_tsvector('english', normalized_text))",
        """
        CREATE TABLE IF NOT EXISTS rag_v5_terminology (
            id bigserial PRIMARY KEY,
            run_id uuid NOT NULL REFERENCES rag_v5_processing_runs(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_id uuid NOT NULL,
            page_number integer NOT NULL,
            alias text NOT NULL,
            alias_norm text NOT NULL,
            canonical_name text NOT NULL,
            canonical_norm text NOT NULL,
            confidence double precision NOT NULL DEFAULT 0.95,
            evidence text NOT NULL DEFAULT '',
            UNIQUE(run_id, alias_norm, canonical_norm, chunk_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_terminology_alias ON rag_v5_terminology(alias_norm)",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_terminology_document ON rag_v5_terminology(document_id)",
        f"""
        CREATE TABLE IF NOT EXISTS rag_v5_chunks (
            id uuid PRIMARY KEY,
            run_id uuid NOT NULL REFERENCES rag_v5_processing_runs(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index integer NOT NULL,
            page_number integer NOT NULL,
            page_end integer NOT NULL,
            content_type varchar(32) NOT NULL,
            parent_key text NOT NULL,
            section_path jsonb NOT NULL DEFAULT '[]'::jsonb,
            heading text NOT NULL DEFAULT '',
            table_id uuid NULL REFERENCES rag_v5_tables(id) ON DELETE CASCADE,
            table_row_index integer NULL,
            extraction_confidence double precision NOT NULL DEFAULT 1.0,
            authority_status varchar(40) NOT NULL DEFAULT 'unknown',
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            text text NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            UNIQUE(run_id, chunk_index)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_chunks_document ON rag_v5_chunks(document_id, chunk_index)",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_chunks_parent ON rag_v5_chunks(document_id, parent_key, chunk_index)",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_chunks_content_type ON rag_v5_chunks(content_type)",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_chunks_fts_simple ON rag_v5_chunks USING gin (to_tsvector('simple', text))",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_chunks_fts_english ON rag_v5_chunks USING gin (to_tsvector('english', text))",
        """
        CREATE TABLE IF NOT EXISTS rag_v5_authority (
            id bigserial PRIMARY KEY,
            run_id uuid NOT NULL REFERENCES rag_v5_processing_runs(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            anchor_chunk_index integer NOT NULL,
            page_number integer NOT NULL,
            directive_type varchar(32) NOT NULL,
            target text NOT NULL DEFAULT '',
            target_norm text NOT NULL DEFAULT '',
            old_text text NOT NULL DEFAULT '',
            old_norm text NOT NULL DEFAULT '',
            new_text text NOT NULL DEFAULT '',
            new_norm text NOT NULL DEFAULT '',
            effective_year integer NULL,
            span_start_chunk integer NOT NULL,
            span_end_chunk integer NOT NULL,
            confidence double precision NOT NULL DEFAULT 0.99,
            evidence text NOT NULL DEFAULT ''
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_authority_document ON rag_v5_authority(document_id, effective_year)",
        "CREATE INDEX IF NOT EXISTS ix_rag_v5_authority_target ON rag_v5_authority(target_norm)",
    ]
    with engine.begin() as conn:
        for statement in ddl:
            conn.execute(text(statement))

    optional = [
        (
            "v5 chunk HNSW",
            """
            CREATE INDEX IF NOT EXISTS ix_rag_v5_chunks_embedding_hnsw
            ON rag_v5_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 96)
            """,
        ),
    ]
    for label, statement in optional:
        try:
            with engine.begin() as conn:
                conn.execute(text(statement))
        except Exception:
            logger.exception("Could not build optional %s index", label)

    try:
        with engine.begin() as conn:
            for table_name in (
                "rag_v5_processing_runs", "rag_v5_pages", "rag_v5_elements",
                "rag_v5_tables", "rag_v5_table_rows", "rag_v5_terminology", "rag_v5_chunks", "rag_v5_authority",
            ):
                conn.execute(text(f"ANALYZE {table_name}"))
    except Exception:
        logger.exception("RAG v5 ANALYZE failed; continuing")
    logger.info("RAG v5 schema ready")
