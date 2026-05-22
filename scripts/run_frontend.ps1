$ErrorActionPreference = "Stop"

$root = "C:\work\ai-log-agent\frontend"
Set-Location $root

npm.cmd run dev -- --host 127.0.0.1 --port 3001
