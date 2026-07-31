param(
    [string]$Target = (Get-Location).Path
)

$PatchRoot = Split-Path -Parent $PSScriptRoot
$BackendApp = Join-Path $Target "backend\app"
if (!(Test-Path $BackendApp)) {
    Write-Error "Target does not look like a pdfrag repository root: $Target"
    exit 1
}
Copy-Item -Path (Join-Path $PatchRoot "backend\app\*") -Destination $BackendApp -Recurse -Force
Write-Host "Patch files copied. Rebuild and run: docker compose exec backend python -m app.reprocess_documents"
