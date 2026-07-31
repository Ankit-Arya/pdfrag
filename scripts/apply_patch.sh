#!/usr/bin/env bash
set -euo pipefail
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$(pwd)}"
if [[ ! -d "$TARGET/backend/app" ]]; then
  echo "Target does not look like a pdfrag repository root: $TARGET" >&2
  exit 1
fi
cp -R "$PATCH_DIR/backend/app/" "$TARGET/backend/app/"
echo "Patch files copied. Rebuild and run: docker compose exec backend python -m app.reprocess_documents"
