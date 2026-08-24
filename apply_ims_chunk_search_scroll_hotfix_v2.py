from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "IMS_CHUNK_SEARCH_SCROLL_HOTFIX_V2"

CSS = f"""
/* {MARKER} */
.chunk-search-shell {{
  height: 100vh;
  min-height: 0;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  overscroll-behavior-y: contain;
  scrollbar-gutter: stable;
  -webkit-overflow-scrolling: touch;
}}
"""

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robustly enable vertical scrolling on the IMS Search chunks workspace."
    )
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()

    target = Path(args.repo).resolve() / "frontend/src/components/ChunkSearchPanel.vue"
    if not target.exists():
        raise SystemExit(f"Missing file: {target}")

    source = target.read_text(encoding="utf-8")

    if MARKER in source:
        print("Chunk Search scrolling hotfix v2 is already applied.")
        return 0

    style_close = source.rfind("</style>")
    if style_close < 0:
        raise RuntimeError(
            "Could not find </style> in ChunkSearchPanel.vue. "
            "Please inspect the component before applying."
        )

    backup = target.with_suffix(target.suffix + ".bak-before-scroll-hotfix-v2")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8", newline="\n")

    patched = source[:style_close] + CSS + "\n" + source[style_close:]
    target.write_text(patched, encoding="utf-8", newline="\n")

    print("Applied Chunk Search scrolling hotfix v2.")
    print("Changed: frontend/src/components/ChunkSearchPanel.vue")
    print("Method: appended an overriding scoped CSS rule before </style>.")
    print("No backend/database/PDF changes are required.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
