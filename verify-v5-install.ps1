param(
    [string]$Repo = "."
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path $Repo).Path
Push-Location $Repo
try {
    $target = "665655ecb1011bad2f08497f879e07403c508f56"
    $head = (git rev-parse HEAD).Trim()
    if ($head -ne $target) {
        Write-Warning "HEAD is $head; this package was validated against $target. Review the diff before continuing."
    }

    $required = @(
        "backend/app/rag/v5/layout.py",
        "backend/app/rag/v5/chunking.py",
        "backend/app/rag/v5/schema.py",
        "backend/app/rag/v5/ingestion.py",
        "backend/app/rag/v5/retrieval.py",
        "backend/app/rag/v5/service.py",
        "backend/app/rag/v5/terminology.py",
        "backend/app/rag/v5/reprocess.py",
        "backend/app/rag/v5/diagnostics.py",
        "docker-compose.v5.yml"
    )
    foreach ($file in $required) {
        if (-not (Test-Path $file)) { throw "Missing v5 file: $file" }
    }

    if (-not (Select-String -Path "backend/app/config.py" -Pattern "rag_v5_query_enabled" -Quiet)) {
        throw "config.py has not been patched. Run apply_v5_patch.py first."
    }
    if (-not (Select-String -Path "backend/app/main.py" -Pattern "ensure_v5_schema" -Quiet)) {
        throw "main.py has not been patched."
    }
    if (-not (Select-String -Path "backend/app/rag/service.py" -Pattern "V5RagService" -Quiet)) {
        throw "service.py has not been patched."
    }
    if (-not (Select-String -Path "backend/Dockerfile" -Pattern "tesseract-ocr-hin" -Quiet)) {
        throw "Dockerfile Hindi OCR package is missing."
    }

    docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml config | Out-Null
    Write-Host "V5 installation verification passed."
} finally {
    Pop-Location
}
