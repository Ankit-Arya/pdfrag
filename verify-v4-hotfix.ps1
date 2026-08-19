$ErrorActionPreference = "Stop"

$required = @(
  "backend/app/rag/smart_understanding.py",
  "backend/app/rag/smart_runtime.py",
  "backend/tests/test_smart_ai_understanding.py",
  "HOTFIX_V4_README.md",
  "HOTFIX_V4.patch",
  "CHECKSUMS.sha256"
)

foreach ($path in $required) {
  if (-not (Test-Path $path)) {
    throw "Missing v4 hotfix file: $path"
  }
}

$runtime = Get-Content "backend/app/rag/smart_runtime.py" -Raw
$understanding = Get-Content "backend/app/rag/smart_understanding.py" -Raw

if ($runtime -notmatch "ai_postmerge_evidence_review") {
  throw "v4 post-merge evidence review marker not found"
}
if ($runtime -notmatch "AI-first understanding v4") {
  throw "v4 runtime startup marker not found"
}
if ($understanding -notmatch "v4 context isolation") {
  throw "v4 standalone-first context isolation marker not found"
}

Get-Content CHECKSUMS.sha256 | ForEach-Object {
  if ($_ -match '^([0-9a-fA-F]{64})\s+(.+)$') {
    $expected = $matches[1].ToLower()
    $file = $matches[2]
    if (-not (Test-Path $file)) { throw "Checksum target missing: $file" }
    $actual = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
    if ($actual -ne $expected) { throw "Checksum mismatch: $file" }
  }
}

Write-Host "Smart RAG v4 overlay verification: PASS"
