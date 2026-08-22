from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

OLD_SQL = '''                SELECT DISTINCT d.id, d.filename
                FROM documents d
                JOIN rag_v5_processing_runs r
                  ON r.document_id=d.id AND r.is_active=true AND r.status='ready'
                WHERE d.status='ready'
                ORDER BY lower(d.filename), d.id
'''

NEW_SQL = '''                SELECT d.id, d.filename
                FROM documents d
                WHERE d.status='ready'
                  AND EXISTS (
                      SELECT 1
                      FROM rag_v5_processing_runs r
                      WHERE r.document_id=d.id
                        AND r.is_active=true
                        AND r.status='ready'
                  )
                ORDER BY lower(d.filename), d.id
'''

OLD_INTERPRET = '_int_env("SMART_RAG_AI_INTERPRET_MAX_TOKENS", 950, 500, 1600)'
NEW_INTERPRET = '_int_env("SMART_RAG_AI_INTERPRET_MAX_TOKENS", 1400, 500, 1600)'

OLD_COMPOSE = 'SMART_RAG_AI_INTERPRET_MAX_TOKENS: ${SMART_RAG_AI_INTERPRET_MAX_TOKENS:-950}'
NEW_COMPOSE = 'SMART_RAG_AI_INTERPRET_MAX_TOKENS: ${SMART_RAG_AI_INTERPRET_MAX_TOKENS:-1400}'

OLD_API_DEFAULT = '''    return {
        "detail": "The language model request failed",
        "code": "chat_failed",
    }
'''
NEW_API_DEFAULT = '''    return {
        "detail": "The request failed while searching the documents or preparing the answer. Please retry; if it persists, check backend logs.",
        "code": "chat_failed",
    }
'''


def replace_or_verify(path: Path, old: str, new: str, label: str) -> bool:
    if not path.exists():
        return False
    source = path.read_text(encoding="utf-8")
    if new in source:
        print(f"[already fixed] {label}: {path}")
        return False
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one old pattern in {path}, found {count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8", newline="\n")
    print(f"[patched] {label}: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix IMS Assistant v5.1 PostgreSQL routing query and harden query-planner output budget."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    actual_assistant = repo / "backend/app/rag/v5/assistant_retrieval.py"
    payload_assistant = repo / "payload/backend/app/rag/v5/assistant_retrieval.py"
    understanding = repo / "backend/app/rag/smart_understanding.py"
    compose = repo / "docker-compose.smart-rag.yml"
    api = repo / "backend/app/api.py"

    if not actual_assistant.exists():
        raise SystemExit(
            "backend/app/rag/v5/assistant_retrieval.py was not found. "
            "Apply IMS Assistant v5.1 first."
        )

    changed = False
    changed |= replace_or_verify(
        actual_assistant, OLD_SQL, NEW_SQL, "active-document PostgreSQL query"
    )

    if payload_assistant.exists():
        changed |= replace_or_verify(
            payload_assistant, OLD_SQL, NEW_SQL, "payload active-document PostgreSQL query"
        )

    if understanding.exists():
        changed |= replace_or_verify(
            understanding, OLD_INTERPRET, NEW_INTERPRET, "AI interpretation token default"
        )

    if compose.exists():
        changed |= replace_or_verify(
            compose, OLD_COMPOSE, NEW_COMPOSE, "Compose AI interpretation token default"
        )

    if api.exists():
        changed |= replace_or_verify(
            api, OLD_API_DEFAULT, NEW_API_DEFAULT, "generic chat-stream error message"
        )

    for path in [actual_assistant, understanding, api]:
        if path.exists():
            py_compile.compile(str(path), doraise=True)
    if payload_assistant.exists():
        py_compile.compile(str(payload_assistant), doraise=True)

    print()
    print("IMS v5.1 runtime hotfix complete.")
    print("Fixes:")
    print(" - PostgreSQL DISTINCT/ORDER BY failure in document routing")
    print(" - query-planner JSON truncation risk by raising default 950 -> 1400 tokens")
    print(" - misleading generic 'language model failed' fallback message")
    print("No database migration, PDF reprocessing, or embedding rebuild is required.")
    if not changed:
        print("No changes were necessary; all target fixes were already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
