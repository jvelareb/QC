# auto_push.ps1 — Watch + commit + force-push automation
# Rewrites the remote history on every cycle and uploads the current state.
# VS Code launches this automatically on folder open via .vscode/tasks.json

param(
    [int]$IntervalSec = 30   # seconds between checks
)

$repo = $PSScriptRoot

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  AUTO-PUSH active  |  repo: $repo" -ForegroundColor Cyan
Write-Host "  Interval: $IntervalSec s  |  Ctrl+C to stop" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $repo

while ($true) {
    $status = & git status --porcelain 2>&1
    $hasChanges = ($status | Where-Object { $_ -ne "" }).Count -gt 0

    if ($hasChanges) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$ts] Changes detected — pushing..." -ForegroundColor Yellow

        # Stage all files (respects .gitignore)
        & git add -A 2>&1 | Out-Null

        # Commit with timestamp
        & git commit -m "auto: $ts" 2>&1 | Out-Null

        # Force-push: rewrites remote with current state
        $push = & git push --force origin main 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$ts] OK — pushed to GitHub" -ForegroundColor Green
        } else {
            Write-Host "[$ts] ERROR on push: $push" -ForegroundColor Red
        }
    } else {
        $ts = Get-Date -Format "HH:mm:ss"
        Write-Host "[$ts] No changes" -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds $IntervalSec
}
