import json
import pathlib
import requests

BASE = "http://127.0.0.1:18000"
PATH = pathlib.Path("c:/work/ai-log-agent/data/regulation_test_policy.txt")
OUT = pathlib.Path("c:/work/ai-log-agent/data/tmp_upload_check_result.json")

result = {
    "upload_status": None,
    "upload_body": "",
    "health_status": None,
    "reg_status": None,
    "reg_summary_len": 0,
}

try:
    with PATH.open("rb") as stream:
        files = [("files", (PATH.name, stream.read(), "text/plain"))]
        response = requests.post(f"{BASE}/regulation/upload", files=files, timeout=120)

    result["upload_status"] = response.status_code
    result["upload_body"] = response.text[:4000]
except Exception as error:
    result["upload_error"] = str(error)

try:
    health = requests.get(f"{BASE}/health", timeout=30)
    result["health_status"] = health.status_code
    payload = health.json()
    result["reg_status"] = (payload.get("agent_statuses") or {}).get("regulation_agent")
    result["reg_summary_len"] = len(payload.get("latest_regulation_analysis") or "")
    result["vector_count"] = payload.get("vector_count")
except Exception as error:
    result["health_error"] = str(error)

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
