#!/usr/bin/env sh
set -eu
BASE_URL="${BASE_URL:-http://localhost:8080}"
echo "Checking ${BASE_URL}/api/health"
curl --fail --silent --show-error "${BASE_URL}/api/health"
echo
echo "Smoke test passed. Upload/chat requires a PDF and configured LLM credentials."
