from __future__ import annotations

import argparse
from pathlib import Path

OLD = """.chunk-search-shell {
  min-width: 0;
}
"""

NEW = """.chunk-search-shell {
  min-width: 0;
  height: 100vh;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior-y: contain;
  scrollbar-gutter: stable;
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Make the IMS Search chunks workspace vertically scrollable."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    target = Path(args.repo).resolve() / "frontend/src/components/ChunkSearchPanel.vue"
    if not target.exists():
        raise SystemExit(f"Missing file: {target}")

    source = target.read_text(encoding="utf-8")

    if NEW in source:
        print("Chunk Search scrolling hotfix is already applied.")
        return 0

    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one .chunk-search-shell style block, found {count}. "
            "Inspect ChunkSearchPanel.vue before applying."
        )

    backup = target.with_suffix(target.suffix + ".bak-before-scroll-hotfix")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8", newline="\n")

    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8", newline="\n")

    print("Applied Chunk Search scrolling hotfix.")
    print("Changed: frontend/src/components/ChunkSearchPanel.vue")
    print("No backend/database/PDF changes are required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
