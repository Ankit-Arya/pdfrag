from __future__ import annotations

import argparse
from pathlib import Path

BAD = "<!-- IMS_CHUNK_SEARCH_V1 -->"
GOOD = "// IMS_CHUNK_SEARCH_V1"

def main() -> int:
    parser = argparse.ArgumentParser(description="Fix the IMS chunk-search marker in frontend/src/services/api.ts")
    parser.add_argument("--repo", default=".", help="pdfrag repository root")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    path = repo / "frontend/src/services/api.ts"
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    source = path.read_text(encoding="utf-8")
    if BAD in source:
        source = source.replace(BAD, GOOD, 1)
        path.write_text(source, encoding="utf-8", newline="\n")
        print("Fixed frontend/src/services/api.ts marker: HTML comment -> TypeScript comment")
    elif GOOD in source:
        print("frontend/src/services/api.ts marker is already fixed")
    else:
        print("No IMS_CHUNK_SEARCH_V1 marker found; no change made")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
