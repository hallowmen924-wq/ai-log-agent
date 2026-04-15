from __future__ import annotations

from typing import Any

import requests

# Streamlit이 FastAPI를 쉽게 부를 수 있도록 만든 작은 API 클라이언트입니다.
# 화면 코드는 URL 문자열을 직접 조립하지 않고 이 클래스 메서드만 사용하면 됩니다.


class BackendClient:
    def __init__(self, base_url: str = "http://127.0.0.1:18000") -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        return requests.get(f"{self.base_url}/health", timeout=10).json()

    def start_worker(self, interval_seconds: int = 10) -> dict[str, Any]:
        return requests.post(
            f"{self.base_url}/worker/start",
            json={"interval_seconds": interval_seconds},
            timeout=10,
        ).json()

    def run_full_analysis(self, log_dir: str = "data/logs") -> dict[str, Any]:
        return requests.post(
            f"{self.base_url}/analysis/run",
            params={"log_dir": log_dir},
            timeout=180,
        ).json()

    def get_status(self) -> dict[str, Any]:
        return requests.get(f"{self.base_url}/analysis/status", timeout=10).json()

    def get_charts(self) -> dict[str, Any]:
        return requests.get(f"{self.base_url}/charts", timeout=30).json()

    def get_chart(self, chart_name: str) -> dict[str, Any]:
        return requests.get(f"{self.base_url}/charts/{chart_name}", timeout=30).json()

    def collect_news(self) -> dict[str, Any]:
        return requests.post(f"{self.base_url}/news/collect", timeout=30).json()

    def analyze_logs(
        self, raw_logs: str | None = None, log_dir: str = "data/logs"
    ) -> dict[str, Any]:
        return requests.post(
            f"{self.base_url}/logs/analyze",
            json={"raw_logs": raw_logs, "log_dir": log_dir},
            timeout=180,
        ).json()

    def build_faiss(self) -> dict[str, Any]:
        return requests.post(
            f"{self.base_url}/faiss/build", json={}, timeout=180
        ).json()

    def search_faiss(self, query: str, k: int = 5) -> dict[str, Any]:
        return requests.post(
            f"{self.base_url}/faiss/search",
            json={"query": query, "k": k},
            timeout=30,
        ).json()

    def get_faiss_entries(self, limit: int = 200) -> dict[str, Any]:
        return requests.get(f"{self.base_url}/faiss/entries", params={"limit": limit}, timeout=30).json()

    def export_faiss(self, format: str = "json", limit: int = 200) -> requests.Response:
        return requests.get(f"{self.base_url}/faiss/export", params={"format": format, "limit": limit}, timeout=60)

    def strategy_chat(self, question: str) -> dict[str, Any]:
        return requests.post(
            f"{self.base_url}/chat/strategy",
            json={"question": question},
            timeout=180,
        ).json()
