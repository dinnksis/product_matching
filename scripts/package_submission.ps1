$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $RepoRoot "dist"
$Archive = Join-Path $DistDir "submission.zip"
$StageDir = Join-Path $DistDir "submission-stage"

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
if (Test-Path -LiteralPath $Archive) {
    Remove-Item -LiteralPath $Archive
}
if (Test-Path -LiteralPath $StageDir) {
    Remove-Item -LiteralPath $StageDir -Recurse
}

New-Item -ItemType Directory -Path (Join-Path $StageDir "src") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageDir "model") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "run.py") -Destination $StageDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "metadata.json") -Destination $StageDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "Dockerfile") -Destination $StageDir
Copy-Item -Path (Join-Path $RepoRoot "src\*.py") -Destination (Join-Path $StageDir "src")
Copy-Item -Path (Join-Path $RepoRoot "model\*.json") -Destination (Join-Path $StageDir "model")

Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $Archive -CompressionLevel Optimal
Remove-Item -LiteralPath $StageDir -Recurse
Write-Host "Created $Archive"
