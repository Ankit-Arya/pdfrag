$ErrorActionPreference = "Stop"

$required = @(
  "backend/app/rag/smart_understanding.py",
  "backend/app/rag/smart_runtime.py",
  "backend/app/rag/smart_retrieval.py",
  "backend/app/rag/scenario_reasoning.py",
  "backend/app/rag/authority.py",
  "frontend/src/App.vue",
  "docker-compose.smart-rag.yml"
)

foreach ($path in $required) {
  if (-not (Test-Path $path)) {
    throw "Missing required repository file: $path. Extract v3 over the complete pdfrag + Smart RAG v2 repository, not into an empty directory."
  }
}

$runtime = Get-Content "backend/app/rag/smart_runtime.py" -Raw
$understanding = Get-Content "backend/app/rag/smart_understanding.py" -Raw
$compose = Get-Content "docker-compose.smart-rag.yml" -Raw

if ($runtime -notmatch "AI-first understanding v3") { throw "v3 runtime marker not found." }
if ($runtime -notmatch "review_retrieved_evidence") { throw "AI evidence-review hook not found in runtime." }
if ($understanding -notmatch "evidence_needs") { throw "AI interpretation contract not found." }
if ($understanding -notmatch "CLOSED-BOOK RAG") { throw "Closed-book grounding guardrail not found." }
if ($compose -notmatch "SMART_RAG_AI_INTERPRETATION") { throw "v3 Compose environment controls not found." }
if ($compose -notmatch "start_period:\s*300s") { throw "Smart RAG backend healthcheck grace period is missing." }

Write-Host "AI-first Smart RAG v3 overlay verification: PASS"
Write-Host "Frontend files remain present. No Reprocess All is required for this overlay."
