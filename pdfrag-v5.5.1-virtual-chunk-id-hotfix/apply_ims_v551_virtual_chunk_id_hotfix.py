from __future__ import annotations

import argparse
import json
import py_compile
import shutil
from pathlib import Path

MARKER = "IMS_RAG_V551_VIRTUAL_CHUNK_ID_GUARD"
BACKUP_SUFFIX = ".bak-before-ims-v551-virtual-chunk-id-guard"


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not target.exists():
        shutil.copy2(path, target)


def replace_function(source: str, name: str, replacement: str, next_name: str) -> str:
    start = source.find(f"def {name}(")
    if start < 0:
        raise RuntimeError(f"Function {name} not found")
    end = source.find(f"\ndef {next_name}(", start + 1)
    if end < 0:
        raise RuntimeError(f"Function boundary {next_name} after {name} not found")
    return source[:start] + replacement.rstrip() + "\n\n" + source[end + 1 :]


def patch_assistant_retrieval(source: str) -> str:
    if MARKER in source:
        return source
    if "def _section_keys_for_chunks(" not in source:
        raise RuntimeError(
            "backend/app/rag/v5/assistant_retrieval.py does not contain the expected section expansion helper"
        )

    if "from uuid import UUID\n" not in source:
        anchor = "from typing import Iterable, Sequence\n"
        if anchor not in source:
            raise RuntimeError("typing import anchor not found in assistant_retrieval.py")
        source = source.replace(anchor, anchor + "from uuid import UUID\n", 1)

    replacement = r'''def _physical_uuid_chunk_ids(chunk_ids: Sequence[str]) -> list[str]:
    """Return canonical database chunk UUIDs and ignore virtual/synthetic chunk ids.

    The retrieval pipeline can contain source-grounded virtual evidence objects whose
    chunk_id intentionally is not a rag_v5_chunks.id UUID (for example complete table
    aggregates and legacy recovery chunks). DB-backed section expansion must never cast
    those virtual ids to uuid[]. Filtering by actual UUID parse rather than by a known
    prefix keeps this safe for current and future virtual chunk types.
    """
    output: list[str] = []
    seen: set[str] = set()
    for value in chunk_ids:
        raw = str(value or "").strip()
        if not raw:
            continue
        try:
            canonical = str(UUID(raw))
        except (ValueError, TypeError, AttributeError):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        output.append(canonical)
    return output


def _section_keys_for_chunks(db: Session, chunk_ids: Sequence[str]) -> list[tuple[str, str]]:
    # Only physical rag_v5_chunks ids can participate in this DB lookup. Synthetic
    # evidence stays available to rerank/coverage/answer logic but is intentionally
    # skipped by DB-backed section expansion.
    physical_ids = _physical_uuid_chunk_ids(chunk_ids)
    if not physical_ids:
        return []
    rows = db.execute(
        text(
            """
            SELECT id, document_id, parent_key
            FROM rag_v5_chunks
            WHERE id = ANY(CAST(:chunk_ids AS uuid[]))
            """
        ),
        {"chunk_ids": physical_ids},
    ).mappings()
    by_chunk: dict[str, tuple[str, str]] = {}
    for row in rows:
        parent_key = str(row["parent_key"] or "")
        if parent_key:
            by_chunk[str(row["id"])] = (str(row["document_id"]), parent_key)

    # Preserve the original physical reranker order. SQL ANY() does not guarantee it.
    output: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk_id in physical_ids:
        key = by_chunk.get(chunk_id)
        if key is not None and key not in seen:
            seen.add(key)
            output.append(key)
    return output
'''
    source = replace_function(
        source,
        "_section_keys_for_chunks",
        replacement,
        "_section_rows",
    )
    return f"# {MARKER}\n" + source


def prepare(repo: Path, package: Path) -> tuple[dict[Path, str], list[tuple[Path, Path]]]:
    runtime = repo / "backend/app/rag/v5/assistant_retrieval.py"
    if not runtime.exists():
        raise SystemExit(f"Required runtime file missing: {runtime}")

    transforms: dict[Path, object] = {runtime: patch_assistant_retrieval}

    # Keep an existing payload mirror compatible when present. It is optional because
    # deployments differ; absence of a mirror must never block the live runtime fix.
    payload = repo / "payload/backend/app/rag/v5/assistant_retrieval.py"
    if payload.exists():
        transforms[payload] = patch_assistant_retrieval

    transformed: dict[Path, str] = {}
    for path, fn in transforms.items():
        transformed[path] = fn(path.read_text(encoding="utf-8-sig"))  # type: ignore[operator]

    copies = [
        (
            package / "backend/tests/test_v551_virtual_chunk_ids.py",
            repo / "backend/tests/test_v551_virtual_chunk_ids.py",
        )
    ]
    return transformed, copies


def validate(transformed: dict[Path, str], copies: list[tuple[Path, Path]]) -> None:
    for path, content in transformed.items():
        if path.suffix == ".py":
            compile(content, str(path), "exec")
    for src, dst in copies:
        if dst.suffix == ".py":
            compile(src.read_text(encoding="utf-8"), str(dst), "exec")


def write(
    transformed: dict[Path, str],
    copies: list[tuple[Path, Path]],
    repo: Path,
) -> None:
    for path, content in transformed.items():
        current = path.read_text(encoding="utf-8-sig")
        if current == content:
            print(f"[already compatible] {path.relative_to(repo)}")
            continue
        backup(path)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"[patched] {path.relative_to(repo)}")

    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            print(f"[already copied] {dst.relative_to(repo)}")
            continue
        if dst.exists():
            backup(dst)
        shutil.copy2(src, dst)
        print(f"[copied] {dst.relative_to(repo)}")

    snapshot = repo / "pdfrag-v5.5.1-replacement-files"
    for path in transformed:
        relative = path.relative_to(repo)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for _src, dst in copies:
        relative = dst.relative_to(repo)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, target)

    manifest = {
        "version": "rag-v5.5.1-virtual-chunk-id-guard",
        "replacement_files": [
            str(path.relative_to(repo)).replace("\\", "/") for path in transformed
        ] + [str(dst.relative_to(repo)).replace("\\", "/") for _src, dst in copies],
        "notes": [
            "Filters non-UUID virtual/synthetic chunk IDs before rag_v5_chunks UUID lookups.",
            "Keeps synthetic v5.4/v5.5 evidence in the retrieval pipeline; only DB-backed section expansion skips it.",
            "No database migration, embedding rebuild, or document reprocessing is required.",
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[snapshot] {snapshot.relative_to(repo)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply IMS RAG v5.5.1 virtual/synthetic chunk-id UUID guard."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    parser.add_argument(
        "--check",
        action="store_true",
        help="preflight transform/compile without writing",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    package = Path(__file__).resolve().parent
    transformed, copies = prepare(repo, package)
    validate(transformed, copies)
    print("[preflight] all transformed Python files compile in memory")
    if args.check:
        print("[check only] no repository files were changed")
        return 0

    write(transformed, copies, repo)
    for path in [
        repo / "backend/app/rag/v5/assistant_retrieval.py",
        repo / "backend/tests/test_v551_virtual_chunk_ids.py",
    ]:
        py_compile.compile(str(path), doraise=True)

    print()
    print("Applied IMS RAG v5.5.1 virtual chunk-id hotfix.")
    print(" - virtual/synthetic evidence ids are never cast to PostgreSQL uuid[]")
    print(" - physical UUID chunks still seed normal section expansion")
    print(" - synthetic procedure/table evidence remains available for rerank, coverage and answer synthesis")
    print(" - the guard is prefix-independent and protects future virtual chunk types too")
    print()
    print("No database migration, embedding rebuild, or PDF reprocessing is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
