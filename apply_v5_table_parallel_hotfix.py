from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

LAYOUT_MARKER = "def _canonicalize_table_row_indexes(tables: list[V5Table]) -> None:"
INGESTION_MARKER = "ensure_schema: bool = True"
REPROCESS_MARKER = 'parser.add_argument("--workers"'

REPROCESS_SOURCE = r'''from __future__ import annotations

import argparse
import logging
import multiprocessing
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import SessionLocal, engine
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


def _advisory_lock_key(document_id: uuid.UUID) -> int:
    value = document_id.int & ((1 << 64) - 1)
    return value - (1 << 64) if value >= (1 << 63) else value


def _process_one(document_id_value: str, force: bool) -> dict[str, object]:
    document_id = uuid.UUID(document_id_value)
    lock_key = _advisory_lock_key(document_id)
    filename = document_id_value

    # A dedicated connection is kept for the full document so the session-level
    # advisory lock survives the commits performed by process_document_v5().
    with engine.connect() as connection:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": lock_key},
            ).scalar()
        )
        if not acquired:
            return {"status": "locked", "filename": filename, "document_id": document_id_value}

        try:
            with Session(bind=connection, expire_on_commit=False) as db:
                document = db.get(Document, document_id)
                if document is None:
                    return {"status": "missing", "filename": filename, "document_id": document_id_value}
                filename = document.filename
                if not force and _already_current(db, document.id):
                    return {"status": "skipped", "filename": filename, "document_id": document_id_value}

                logger.info("V5 PROCESS %s", filename)
                summary = process_document_v5(
                    db,
                    document,
                    publish_document_state=False,
                    ensure_schema=False,
                )
                return {
                    "status": "ready",
                    "filename": filename,
                    "document_id": document_id_value,
                    "pages": summary.pages,
                    "chunks": summary.chunks,
                    "tables": summary.tables,
                    "rows": summary.table_rows,
                    "ocr": summary.ocr_pages,
                    "low_quality": summary.low_quality_pages,
                }
        except Exception:
            return {
                "status": "failed",
                "filename": filename,
                "document_id": document_id_value,
                "error": traceback.format_exc(),
            }
        finally:
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key},
                )
                connection.commit()
            except Exception:
                connection.rollback()
                logger.exception("Could not release v5 advisory lock for %s", document_id_value)


def _record_result(result: dict[str, object], counts: dict[str, int]) -> None:
    status = str(result.get("status") or "failed")
    filename = str(result.get("filename") or result.get("document_id") or "unknown")
    if status == "ready":
        counts["succeeded"] += 1
        logger.info(
            "V5 READY %s pages=%s chunks=%s tables=%s rows=%s ocr=%s low_quality=%s",
            filename,
            result.get("pages"),
            result.get("chunks"),
            result.get("tables"),
            result.get("rows"),
            result.get("ocr"),
            result.get("low_quality"),
        )
    elif status == "skipped":
        counts["skipped"] += 1
        logger.info("SKIP current v5 generation: %s", filename)
    elif status == "locked":
        counts["locked"] += 1
        logger.info("SKIP locked by another v5 worker: %s", filename)
    elif status == "missing":
        counts["missing"] += 1
        logger.warning("SKIP document disappeared before processing: %s", filename)
    else:
        counts["failed"] += 1
        logger.error("V5 FAILED %s\n%s", filename, result.get("error") or "unknown error")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structure-preserving RAG v5 generations without deleting v4 chunks.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the active v5 generation is current")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many documents (0 = all)")
    parser.add_argument("--document-id", action="append", default=[], help="Process only the given document UUID; may be repeated")
    parser.add_argument("--workers", type=int, default=1, help="Process this many PDFs concurrently (default: 1; recommended start: 2)")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")

    # Schema creation/ANALYZE happens once in the parent. Child document workers
    # explicitly skip that repeated work.
    ensure_v5_schema()
    requested_ids = {uuid.UUID(value) for value in args.document_id}
    with SessionLocal() as db:
        stmt = select(Document.id, Document.filename).where(Document.status == DocumentStatus.ready).order_by(Document.created_at, Document.id)
        documents = [(document_id, filename) for document_id, filename in db.execute(stmt).all()]
        if requested_ids:
            documents = [(document_id, filename) for document_id, filename in documents if document_id in requested_ids]
        if args.limit > 0:
            documents = documents[: args.limit]

    counts = {"succeeded": 0, "skipped": 0, "locked": 0, "missing": 0, "failed": 0}
    attempted = len(documents)

    if args.workers == 1:
        for index, (document_id, _filename) in enumerate(documents, 1):
            result = _process_one(str(document_id), args.force)
            _record_result(result, counts)
            logger.info("V5 PROGRESS %d/%d", index, attempted)
    else:
        logger.info("Starting RAG v5 parallel reprocess with workers=%d documents=%d", args.workers, attempted)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {
                executor.submit(_process_one, str(document_id), args.force): (document_id, filename)
                for document_id, filename in documents
            }
            completed = 0
            for future in as_completed(futures):
                document_id, filename = futures[future]
                try:
                    result = future.result()
                except Exception:
                    result = {
                        "status": "failed",
                        "filename": filename,
                        "document_id": str(document_id),
                        "error": traceback.format_exc(),
                    }
                _record_result(result, counts)
                completed += 1
                logger.info(
                    "V5 PROGRESS %d/%d ready=%d failed=%d skipped=%d locked=%d",
                    completed,
                    attempted,
                    counts["succeeded"],
                    counts["failed"],
                    counts["skipped"],
                    counts["locked"],
                )

    print(
        f"RAG v5 reprocess complete: attempted={attempted} succeeded={counts['succeeded']} "
        f"skipped_current={counts['skipped']} skipped_locked={counts['locked']} "
        f"missing={counts['missing']} failed={counts['failed']} workers={args.workers} "
        f"version={PROCESSING_VERSION}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_layout(source: str) -> str:
    if LAYOUT_MARKER in source:
        return source

    old_offset = "                    offset = len(previous.rows)\n"
    new_offset = (
        "                    # Retained source rows can have gaps after sparse/noise rows are filtered.\n"
        "                    # Use the highest retained index rather than len(rows), then canonicalize below.\n"
        "                    offset = max((row.row_index for row in previous.rows), default=0)\n"
    )
    source = replace_once(source, old_offset, new_offset, "multipage table offset")

    function_anchor = "\n\ndef _column_name_score(value: str) -> float:\n"
    canonicalizer = '''\n\ndef _canonicalize_table_row_indexes(tables: list[V5Table]) -> None:\n    """Assign final contiguous row indexes after all table filtering and page merges."""\n    for table in tables:\n        for index, row in enumerate(table.rows, 1):\n            row.row_index = index\n\n\ndef _column_name_score(value: str) -> float:\n'''
    source = replace_once(source, function_anchor, canonicalizer, "table canonicalizer insertion")

    call_anchor = "    tables = _merge_multipage_tables(tables)\n    _assign_section_paths(elements, tables)\n"
    call_replacement = (
        "    tables = _merge_multipage_tables(tables)\n"
        "    _canonicalize_table_row_indexes(tables)\n"
        "    _assign_section_paths(elements, tables)\n"
    )
    source = replace_once(source, call_anchor, call_replacement, "table canonicalizer call")
    return source


def patch_ingestion(source: str) -> str:
    if INGESTION_MARKER not in source:
        signature_old = '''    publish_document_state: bool = True,\n) -> V5ProcessingSummary:\n'''
        signature_new = '''    publish_document_state: bool = True,\n    ensure_schema: bool = True,\n) -> V5ProcessingSummary:\n'''
        source = replace_once(source, signature_old, signature_new, "ingestion ensure_schema parameter")

        ensure_old = "    ensure_v5_schema()\n    run_id = uuid.uuid4()\n"
        ensure_new = "    if ensure_schema:\n        ensure_v5_schema()\n    run_id = uuid.uuid4()\n"
        source = replace_once(source, ensure_old, ensure_new, "conditional schema ensure")

    invariant_marker = "RAG v5 table row indexes are not canonical"
    if invariant_marker not in source:
        chunks_anchor = '''        chunks = build_v5_chunks(\n            layout,\n            target_chars=_target_chars(),\n            overlap_chars=_overlap_chars(),\n        )\n'''
        chunks_replacement = '''        # Database and retrieval code require one unambiguous row index per final table.\n        # Check the post-merge representation before chunk IDs or inserts are generated.\n        for table in layout.tables:\n            indexes = [row.row_index for row in table.rows]\n            expected = list(range(1, len(indexes) + 1))\n            if indexes != expected:\n                raise ValueError(\n                    f"RAG v5 table row indexes are not canonical for {table.table_key}: "\n                    f"expected={expected[:20]} actual={indexes[:20]}"\n                )\n\n        chunks = build_v5_chunks(\n            layout,\n            target_chars=_target_chars(),\n            overlap_chars=_overlap_chars(),\n        )\n'''
        source = replace_once(source, chunks_anchor, chunks_replacement, "table row invariant")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply RAG v5 table-row + parallel reprocess hotfix.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    layout_path = repo / "backend/app/rag/v5/layout.py"
    ingestion_path = repo / "backend/app/rag/v5/ingestion.py"
    reprocess_path = repo / "backend/app/rag/v5/reprocess.py"
    paths = [layout_path, ingestion_path, reprocess_path]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Not found: {path}")

    originals = {path: path.read_text(encoding="utf-8") for path in paths}
    updated = {
        layout_path: patch_layout(originals[layout_path]),
        ingestion_path: patch_ingestion(originals[ingestion_path]),
        reprocess_path: originals[reprocess_path] if REPROCESS_MARKER in originals[reprocess_path] else REPROCESS_SOURCE,
    }

    backups: dict[Path, Path] = {}
    for path in paths:
        backup = path.with_suffix(path.suffix + ".bak-v5-table-parallel")
        backups[path] = backup
        if not backup.exists():
            shutil.copy2(path, backup)

    try:
        for path, content in updated.items():
            path.write_text(content, encoding="utf-8", newline="\n")
        for path in paths:
            py_compile.compile(str(path), doraise=True)
    except Exception:
        for path, original in originals.items():
            path.write_text(original, encoding="utf-8", newline="\n")
        raise

    print("Applied RAG v5 table-row canonicalization and parallel reprocess hotfix.")
    print("Modified:")
    for path in paths:
        changed = updated[path] != originals[path]
        print(f"- {path} ({'changed' if changed else 'already patched'})")
    print("Backups:")
    for backup in backups.values():
        print(f"- {backup}")
    print("Syntax checks: PASS")
    print("Processing version intentionally unchanged; active successful v5 generations remain resumable/skippable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
