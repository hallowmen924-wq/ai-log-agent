$ErrorActionPreference = "Stop"

$root = "C:\work\ai-log-agent"
Set-Location $root

$py313 = "C:\Users\KBCARD\AppData\Local\Programs\Python\Python313\python.exe"

if (-not (Test-Path $py313)) {
  throw "Python 3.13 not found at: $py313"
}

if (-not (Test-Path ".venv313")) {
  Write-Host "[setup-313] Create virtualenv (.venv313)"
  & $py313 -m venv .venv313
}

Write-Host "[setup-313] Upgrade pip"
& ".\.venv313\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "[setup-313] Install backend requirements"
& ".\.venv313\Scripts\pip.exe" install -r ".\requirements.txt"

Write-Host "[setup-313] Install OCR packages (PaddleOCR + PaddlePaddle + pytesseract)"
& ".\.venv313\Scripts\pip.exe" install paddlepaddle paddleocr pytesseract

Write-Host "[setup-313] Install frontend packages"
Set-Location "$root\frontend"

try {
  npm.cmd ci
} catch {
  Write-Host "[setup-313] npm ci failed. Trying auto-recovery for locked esbuild..."
  try { Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue } catch {}
  Start-Sleep -Seconds 1
  try { Remove-Item -LiteralPath ".\node_modules\@esbuild\win32-x64\esbuild.exe" -Force -ErrorAction SilentlyContinue } catch {}
  try { Remove-Item -LiteralPath ".\node_modules" -Recurse -Force -ErrorAction SilentlyContinue } catch {}
  npm.cmd install
}

Set-Location $root
Write-Host "[setup-313] Done"
