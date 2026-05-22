$ErrorActionPreference = "Stop"

$root = "C:\work\ai-log-agent"

Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", "$root\scripts\run_backend.ps1") `
  -WorkingDirectory $root

Start-Sleep -Seconds 1

Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", "$root\scripts\run_frontend.ps1") `
  -WorkingDirectory "$root\frontend"

Write-Host "Started backend and frontend in separate PowerShell windows."
