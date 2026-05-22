$ErrorActionPreference = "Stop"

$root = "C:\work\ai-log-agent"
Set-Location $root

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  throw ".venv not found. Run .\scripts\dev_setup.ps1 first."
}

# Product debate defaults (stable local profile)
$env:PRODUCT_DEBATE_MAX_AGENTS = "3"
$env:PRODUCT_DEBATE_MAX_TURNS = "2"
$env:PRODUCT_DEBATE_TEMPERATURE = "0.3"
$env:PRODUCT_DEBATE_MEMORY_ENABLED = "0"
$env:PRODUCT_DEBATE_USE_AUTOGEN = "0"
$env:PRODUCT_DEBATE_AGENT_TIMEOUT_SEC = "20"
$env:PRODUCT_DEBATE_AUTOGEN_TIMEOUT_SEC = "120"
$env:PRODUCT_DEBATE_TOTAL_BUDGET_SEC = "45"

& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 18000
