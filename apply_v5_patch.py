from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

TARGET_COMMIT = "665655ecb1011bad2f08497f879e07403c508f56"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one patch anchor, found {count}")
    return text.replace(old, new, 1)


def _head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return ""


def _backup(repo: Path, relative: str) -> None:
    source = repo / relative
    target = repo / "ROLLBACK_V5" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and not target.exists():
        shutil.copy2(source, target)


def _patch_config(repo: Path) -> None:
    path = repo / "backend/app/config.py"
    _backup(repo, "backend/app/config.py")
    text = path.read_text(encoding="utf-8")
    old = """    extract_tables: bool = True\n    table_min_rows: int = 2\n\n    max_file_size_mb: int = 25\n"""
    new = """    extract_tables: bool = True\n    table_min_rows: int = 2\n\n    # RAG v5 is an additive structured-ingestion generation. Keep query cutover off\n    # until all ready PDFs have an active v5 generation and diagnostics pass.\n    rag_v5_schema_enabled: bool = True\n    rag_v5_query_enabled: bool = False\n    rag_v5_processing_version: str = \"rag-v5.0.0\"\n    rag_v5_chunk_target_chars: int = Field(default=1000, ge=500, le=2400)\n    rag_v5_chunk_overlap_chars: int = Field(default=120, ge=0, le=400)\n    rag_v5_retrieval_per_arm: int = Field(default=48, ge=12, le=160)\n    rag_v5_final_evidence: int = Field(default=32, ge=12, le=80)\n    rag_v5_parent_window: int = Field(default=2, ge=0, le=5)\n    rag_v5_min_table_confidence: float = Field(default=0.62, ge=0.35, le=0.99)\n    rag_v5_ocr_image_coverage: float = Field(default=0.45, ge=0.05, le=0.95)\n    rag_v5_legacy_chunk_mirror: bool = True\n\n    max_file_size_mb: int = 25\n"""
    path.write_text(_replace_once(text, old, new, "config.py"), encoding="utf-8", newline="\n")


def _patch_main(repo: Path) -> None:
    path = repo / "backend/app/main.py"
    _backup(repo, "backend/app/main.py")
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "from app.rag.smart_runtime import install_smart_rag_patch\n",
        "from app.rag.smart_runtime import install_smart_rag_patch\nfrom app.rag.v5.schema import ensure_v5_schema\n",
        "main.py import",
    )
    old = """    # Additive smart-RAG upgrade: HNSW/GIN retrieval, organisation terminology,\n    # scenario-aware rule matching and bounded context. Existing broad RAG remains\n    # available as a low-confidence fallback.\n    install_smart_rag_patch()\n"""
    new = """    # RAG v5 schema is additive and can be built while v4 continues serving users.\n    if settings.rag_v5_schema_enabled:\n        ensure_v5_schema()\n\n    # Do not install the v4 monkeypatch chain after v5 query cutover. V5 is an explicit\n    # service implementation with its own interpretation -> evidence-needs -> retrieval\n    # -> coverage -> answer -> verification pipeline.\n    if not settings.rag_v5_query_enabled:\n        install_smart_rag_patch()\n    else:\n        logger.info(\"RAG v5 query pipeline enabled; Smart RAG v4 runtime patch skipped\")\n"""
    path.write_text(_replace_once(text, old, new, "main.py lifespan"), encoding="utf-8", newline="\n")


def _patch_service(repo: Path) -> None:
    path = repo / "backend/app/rag/service.py"
    _backup(repo, "backend/app/rag/service.py")
    text = path.read_text(encoding="utf-8")
    old = "rag_service = RagService()\n"
    new = """# Explicit generation cutover. During migration the default remains the proven v4\n# service while `python -m app.rag.v5.reprocess` builds v5 tables in parallel.\n# After `app.rag.v5.diagnostics --strict` passes, set RAG_V5_QUERY_ENABLED=1 and\n# restart; no runtime monkeypatch is required for the v5 query path.\nif get_settings().rag_v5_query_enabled:\n    from app.rag.v5.service import V5RagService\n\n    rag_service = V5RagService()\nelse:\n    rag_service = RagService()\n"""
    path.write_text(_replace_once(text, old, new, "service.py singleton"), encoding="utf-8", newline="\n")


def _patch_dockerfile(repo: Path) -> None:
    path = repo / "backend/Dockerfile"
    _backup(repo, "backend/Dockerfile")
    text = path.read_text(encoding="utf-8")
    old = """        tesseract-ocr \\
        tesseract-ocr-eng \\
"""
    new = """        tesseract-ocr \\
        tesseract-ocr-eng \\
        tesseract-ocr-hin \\
"""
    text = _replace_once(text, old, new, "Dockerfile OCR languages")
    text = _replace_once(
        text,
        "    && tesseract --list-langs | grep -qx eng \\\n",
        "    && tesseract --list-langs | grep -qx eng \\\n    && tesseract --list-langs | grep -qx hin \\\n",
        "Dockerfile OCR verification",
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the pdfrag RAG v5 structured-ingestion migration patch.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    parser.add_argument("--allow-drift", action="store_true", help="Allow applying when HEAD is not the validated target commit")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    required = [repo / "backend/app/config.py", repo / "backend/app/main.py", repo / "backend/app/rag/service.py"]
    if not all(path.exists() for path in required):
        raise SystemExit(f"Not a pdfrag repository root: {repo}")
    head = _head(repo)
    if head and head != TARGET_COMMIT and not args.allow_drift:
        raise SystemExit(
            f"Refusing to patch HEAD {head}. Validated target is {TARGET_COMMIT}. "
            "Update the patch or rerun with --allow-drift only after reviewing the diff."
        )

    _patch_config(repo)
    _patch_main(repo)
    _patch_service(repo)
    _patch_dockerfile(repo)
    print("RAG v5 patch anchors applied successfully.")
    print("Backups: ROLLBACK_V5/")
    print("Next: docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
