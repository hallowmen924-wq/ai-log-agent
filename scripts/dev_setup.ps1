$ErrorActionPreference = "Stop"

$root = "C:\work\ai-log-agent"
Set-Location $root

if (-not (Test-Path ".venv")) {
  Write-Host "[setup] Create virtualenv (.venv)"
  py -3.13 -m venv .venv
}

Write-Host "[setup] Upgrade pip"
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "[setup] Install backend requirements"
& ".\.venv\Scripts\pip.exe" install -r ".\requirements.txt"

Write-Host "[setup] Install frontend packages"
Set-Location "$root\frontend"

try {
  npm.cmd ci
} catch {
  Write-Host "[setup] npm ci failed. Trying auto-recovery for locked esbuild..."
  try { Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue } catch {}
  Start-Sleep -Seconds 1
  try { Remove-Item -LiteralPath ".\node_modules\@esbuild\win32-x64\esbuild.exe" -Force -ErrorAction SilentlyContinue } catch {}
  try { Remove-Item -LiteralPath ".\node_modules" -Recurse -Force -ErrorAction SilentlyContinue } catch {}
  npm.cmd install
}

Set-Location $root
Write-Host "[setup] Done"
