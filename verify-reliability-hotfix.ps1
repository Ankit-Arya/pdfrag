$ErrorActionPreference = 'Stop'

$required = @(
  'docker-compose.yml',
  'docker-compose.smart-rag.yml',
  'backend/app/rag/service.py',
  'backend/app/rag/authority.py',
  'backend/app/rag/smart_runtime.py',
  'backend/app/rag/smart_retrieval.py',
  'backend/app/rag/terminology.py',
  'frontend/src/App.vue',
  'nginx/nginx.conf'
)

$missing = @($required | Where-Object { -not (Test-Path $_) })
if ($missing.Count -gt 0) {
  Write-Error ("Refusing to continue. Existing pdfrag checkout is incomplete. Missing: " + ($missing -join ', '))
}

Write-Host 'Required base/UI/Smart-RAG files are present.'
Write-Host 'Effective Smart RAG / embedding / healthcheck settings:'
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml config |
  Select-String 'SMART_RAG_|PRELOAD_EMBEDDING|REQUIRE_EMBEDDING|EMBEDDING_DOWNLOAD|EMBEDDING_LOCAL|start_period|timeout:|retries:'

if ($LASTEXITCODE -ne 0) {
  throw 'docker compose config failed.'
}

Write-Host ''
Write-Host 'Verification complete. Review the settings above before building.'
