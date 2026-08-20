param(
    [switch]$EnableQuery,
    [switch]$DisableQuery,
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"
if ($EnableQuery -and $DisableQuery) {
    throw "Use either -EnableQuery or -DisableQuery, not both."
}
if (-not (Test-Path $EnvFile)) {
    throw "Environment file not found: $EnvFile"
}

$values = [ordered]@{
    "RAG_V5_SCHEMA_ENABLED" = "1"
    "RAG_V5_QUERY_ENABLED" = if ($EnableQuery) { "1" } else { "0" }
    "RAG_V5_PROCESSING_VERSION" = "rag-v5.0.0"
    "RAG_V5_CHUNK_TARGET_CHARS" = "1000"
    "RAG_V5_CHUNK_OVERLAP_CHARS" = "120"
    "RAG_V5_RETRIEVAL_PER_ARM" = "48"
    "RAG_V5_FINAL_EVIDENCE" = "32"
    "RAG_V5_PARENT_WINDOW" = "2"
    "RAG_V5_MIN_TABLE_CONFIDENCE" = "0.62"
    "RAG_V5_OCR_IMAGE_COVERAGE" = "0.45"
    "RAG_V5_LEGACY_CHUNK_MIRROR" = "1"
    "OCR_LANGUAGES" = "eng+hin"
}
if ($DisableQuery) {
    $values["RAG_V5_QUERY_ENABLED"] = "0"
}

$lines = [System.Collections.Generic.List[string]]::new()
Get-Content -LiteralPath $EnvFile | ForEach-Object { [void]$lines.Add($_) }

foreach ($key in $values.Keys) {
    $replacement = "$key=$($values[$key])"
    $matches = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($key))\s*=") {
            $matches += $i
        }
    }
    if ($matches.Count -gt 0) {
        $lines[$matches[0]] = $replacement
        for ($j = $matches.Count - 1; $j -ge 1; $j--) {
            $lines.RemoveAt($matches[$j])
        }
    } else {
        [void]$lines.Add($replacement)
    }
}

Set-Content -LiteralPath $EnvFile -Value $lines -Encoding utf8
Write-Host "Updated only RAG v5/OCR keys in $EnvFile. Existing secrets and other settings were preserved."
Write-Host "RAG_V5_QUERY_ENABLED=$($values['RAG_V5_QUERY_ENABLED'])"
