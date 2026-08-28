from __future__ import annotations

import uuid

from app.models import DocumentChunkOut, DocumentChunkPageOut


def test_document_chunk_page_contract_preserves_debug_metadata():
    document_id = uuid.uuid4()
    run_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    chunk = DocumentChunkOut(
        id=chunk_id,
        chunk_index=12,
        page_number=69,
        page_end=69,
        content_type="table_row",
        parent_key="table:test",
        section_path=["6.9.3 Important Cut-Out Cocks"],
        heading="6.9.3 Important Cut-Out Cocks",
        table_id=None,
        table_row_index=3,
        extraction_confidence=0.98,
        authority_status="current",
        metadata={"cells": ["Bic", "Brake Isolation Cock"]},
        text="[PDF STRUCTURE]\n...\n[/PDF STRUCTURE]\nBic (Brake Isolation Cock)",
        char_count=72,
    )

    page = DocumentChunkPageOut(
        document_id=document_id,
        filename="RS-3.pdf",
        run_id=run_id,
        processing_version="rag-v5.0.0",
        total_chunks=300,
        filtered_chunks=1,
        offset=0,
        limit=100,
        content_types=["table_row"],
        authority_statuses=["current"],
        chunks=[chunk],
    )

    assert page.chunks[0].metadata["cells"][1] == "Brake Isolation Cock"
    assert page.chunks[0].chunk_index == 12
    assert page.filtered_chunks == 1
