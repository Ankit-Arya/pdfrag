param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryPath
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $ScriptDir "apply_patch.py") $RepositoryPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
