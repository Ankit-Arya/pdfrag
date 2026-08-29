from __future__ import annotations

import argparse
from pathlib import Path

OLD = """const props = defineProps<{
  knowledge: KnowledgeStatus | null
}>()
"""

NEW = """defineProps<{
  knowledge: KnowledgeStatus | null
}>()
"""


def patch(path: Path) -> None:
    if not path.exists():
        print(f"[skip] missing: {path}")
        return
    source = path.read_text(encoding="utf-8-sig")
    if OLD in source:
        backup = path.with_suffix(path.suffix + ".bak-before-documents-props-fix")
        if not backup.exists():
            backup.write_text(source, encoding="utf-8", newline="\n")
        path.write_text(source.replace(OLD, NEW, 1), encoding="utf-8", newline="\n")
        print(f"[fixed] {path}")
        return
    if NEW in source and "const props = defineProps<{" not in source:
        print(f"[already fixed] {path}")
        return
    raise RuntimeError(f"Expected DocumentsPanel defineProps block was not found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix unused props declaration in IMS UI v2 DocumentsPanel.")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    for target in [
        repo / "frontend/src/components/DocumentsPanel.vue",
        repo / "payload/frontend/src/components/DocumentsPanel.vue",
    ]:
        patch(target)
    print("DocumentsPanel props hotfix applied.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
