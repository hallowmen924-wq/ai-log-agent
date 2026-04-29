from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import threading
import time
from typing import Any

from agent.log_generator import append_synthetic_log
from agent.news_agent import analyze_news, collect_news
from agent.strategy_chat import (
    OLLAMA_EXECUTION_LOCK,
    lightweight_ollama_generate,
    run_periodic_log_agent,
    run_periodic_news_agent,
    strategy_chat,
)
from analyzer.log_analyzer import analyze_logs
from analyzer.risk_analyzer import calculate_risk
from rag.vector_db import (
    FAISS_STORE_CUSTOMER,
    FAISS_STORE_DOCUMENT,
    FAISS_STORE_LOGS,
    FAISS_STORE_NEWS,
    append_news_documents,
    append_structured_log_documents,
    build_vector_db,
    get_store_path,
    get_store_news_keys,
    get_store_document_count,
    get_vector_count,
    list_vectors,
    prepare_log_ingest_preview,
    search_context,
    search_customer_context,
    search_news_context,
    warmup_embeddings,
)
from rag.product_pattern_summary import DEFAULT_SUMMARY_PATH, write_product_pattern_summary

# services.py는 "실제 일"을 하는 계층입니다.
# API 파일은 요청/응답만 담당하고, 여기서 로그 분석, 뉴스 수집, 벡터 생성, 차트 데이터 계산을 처리합니다.


class BackendState:
    def __init__(self) -> None:
        # 메모리에 현재 분석 결과와 화면 상태를 캐시해 두는 저장소입니다.
        self.lock = threading.Lock()
        self.running = False
        self.results: list[dict[str, Any]] = []
        self.news: list[dict[str, Any]] = []
        self.issues: list[str] = []
        self.file_count = 0
        self.total_time = 0.0
        self.last_news_time: datetime.datetime | None = None
        self.last_new_item_time: datetime.datetime | None = None
        self.last_run_time: datetime.datetime | None = None
        self.latest_strategy_question: str | None = None
        self.last_strategy_time: datetime.datetime | None = None
        self.last_log_ingest_time: datetime.datetime | None = None
        self.last_log_vectorized_count = 0
        self.last_log_vectorized_signature: tuple[tuple[str, int, int], ...] = ()
        self.latest_log_briefing: str | None = None
        self.last_log_briefing_time: datetime.datetime | None = None
        self.latest_log_prompt_input: dict[str, Any] | None = None
        self.last_log_prompt_input_time: datetime.datetime | None = None
        self.log_prompt_template_override: str | None = None
        self.latest_news_briefing: str | None = None
        self.last_news_briefing_time: datetime.datetime | None = None
        self.latest_news_prompt_input: dict[str, Any] | None = None
        self.last_news_prompt_input_time: datetime.datetime | None = None
        self.news_prompt_template_override: str | None = None
        self.agent_statuses: dict[str, dict[str, Any]] = {}
        self.agent_activity_log: list[dict[str, Any]] = []
        self.vector_events: list[dict[str, Any]] = []
        self.chart_payloads: dict[str, Any] = {}
        self.last_chart_time: datetime.datetime | None = None
        self.last_faiss_time: datetime.datetime | str | None = None
        self.worker_runtime_stats: dict[str, Any] = {}
        self.full_faiss_items: list[dict[str, Any]] = []
        self.news_crawl_running = False
        self.news_crawl_target_count = 0
        self.news_crawl_success_count = 0
        self.news_crawl_failure_count = 0
        self.last_news_crawl_time: datetime.datetime | None = None
        self.last_news_crawl_error: str | None = None
        self.static_log_results: list[dict[str, Any]] = []
        self.static_log_signature: tuple[tuple[str, int, int], ...] = ()
        self.static_log_file_count = 0
        self.static_log_results_by_file: dict[str, list[dict[str, Any]]] = {}
        self.generated_log_results: list[dict[str, Any]] = []
        self.generated_log_offset = 0
        self.generated_log_pending_text = ""
        self.ollama_runtime: dict[str, Any] = {
            "agent": None,
            "status": "idle",
            "model": None,
            "prompt": "",
            "response_text": "",
            "error": None,
            "started_at": None,
            "updated_at": None,
            "completed_at": None,
        }
        self.cardloan_debate: dict[str, Any] = {
            "status": "idle",
            "question": None,
            "summary": "",
            "current_stage": None,
            "started_at": None,
            "updated_at": None,
            "completed_at": None,
            "error": None,
            "round_results": [],
            "reviewer_prompts": {},
        }

    def snapshot(self, include_faiss_items: bool = True) -> dict[str, Any]:
        try:
            vector_count = get_vector_count()
        except Exception:
            vector_count = 0
        cached_faiss_items: list[dict[str, Any]] = []
        if include_faiss_items:
            with self.lock:
                cached_faiss_items = safe_serialize(self.full_faiss_items)
        if include_faiss_items and vector_count > 0 and not cached_faiss_items:
            try:
                from rag.vector_db import list_vectors

                cached_faiss_items = safe_serialize(list_vectors(limit=1000))
                with self.lock:
                    self.full_faiss_items = cached_faiss_items
            except Exception:
                cached_faiss_items = []
        with self.lock:
            return {
                "running": self.running,
                "results": self.results,
                "news": self.news,
                "issues": self.issues,
                "file_count": self.file_count,
                "vector_count": vector_count,
                "total_time": self.total_time,
                "last_news_time": (
                    self.last_news_time.isoformat() if self.last_news_time else None
                ),
                "last_new_item_time": (
                    self.last_new_item_time.isoformat()
                    if self.last_new_item_time
                    else None
                ),
                "last_run_time": (
                    self.last_run_time.isoformat() if self.last_run_time else None
                ),
                "latest_strategy_question": self.latest_strategy_question,
                "last_strategy_time": (
                    self.last_strategy_time.isoformat()
                    if self.last_strategy_time
                    else None
                ),
                "last_log_ingest_time": (
                    self.last_log_ingest_time.isoformat()
                    if self.last_log_ingest_time
                    else None
                ),
                "last_log_vectorized_count": int(self.last_log_vectorized_count or 0),
                "last_log_vectorized_signature_size": len(
                    self.last_log_vectorized_signature or ()
                ),
                "latest_log_briefing": self.latest_log_briefing,
                "last_log_briefing_time": (
                    self.last_log_briefing_time.isoformat()
                    if self.last_log_briefing_time
                    else None
                ),
                "latest_log_prompt_input": safe_serialize(
                    self.latest_log_prompt_input or {}
                ),
                "last_log_prompt_input_time": (
                    self.last_log_prompt_input_time.isoformat()
                    if self.last_log_prompt_input_time
                    else None
                ),
                "log_prompt_template_override": self.log_prompt_template_override,
                "latest_news_briefing": self.latest_news_briefing,
                "last_news_briefing_time": (
                    self.last_news_briefing_time.isoformat()
                    if self.last_news_briefing_time
                    else None
                ),
                "latest_news_prompt_input": safe_serialize(
                    self.latest_news_prompt_input or {}
                ),
                "last_news_prompt_input_time": (
                    self.last_news_prompt_input_time.isoformat()
                    if self.last_news_prompt_input_time
                    else None
                ),
                "news_prompt_template_override": self.news_prompt_template_override,
                "agent_statuses": safe_serialize(self.agent_statuses),
                "agent_activity_log": safe_serialize(self.agent_activity_log),
                "vector_events": safe_serialize(self.vector_events),
                "last_chart_time": (
                    self.last_chart_time.isoformat() if self.last_chart_time else None
                ),
                "last_faiss_time": (
                    self.last_faiss_time.isoformat()
                    if isinstance(self.last_faiss_time, datetime.datetime)
                    else self.last_faiss_time
                ),
                "worker_runtime_stats": safe_serialize(self.worker_runtime_stats),
                "ollama_runtime": safe_serialize(self.ollama_runtime),
                "cardloan_debate": safe_serialize(self.cardloan_debate),
                "chart_payloads": self.chart_payloads,
                "full_faiss_items": cached_faiss_items,
                "news_crawl_running": self.news_crawl_running,
                "news_crawl_target_count": self.news_crawl_target_count,
                "news_crawl_success_count": self.news_crawl_success_count,
                "news_crawl_failure_count": self.news_crawl_failure_count,
                "last_news_crawl_time": (
                    self.last_news_crawl_time.isoformat()
                    if self.last_news_crawl_time
                    else None
                ),
                "last_news_crawl_error": self.last_news_crawl_error,
            }


state = BackendState()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_LOG_PRODUCT_SEQUENCE = ("C6", "C9", "C11", "C12")
TEST_LOG_BURST_COUNT = 4
_test_log_product_index = 0


def hydrate_state_from_existing_artifacts() -> bool:
    with state.lock:
        already_hydrated = bool(state.last_run_time and state.last_faiss_time)
    if already_hydrated:
        return False

    summary_path = pathlib.Path(DEFAULT_SUMMARY_PATH)
    faiss_timestamps: list[datetime.datetime] = []
    for store_name in (FAISS_STORE_LOGS, FAISS_STORE_NEWS):
        store_path = pathlib.Path(get_store_path(store_name))
        index_path = store_path / "index.faiss"
        metadata_path = store_path / "index.pkl"
        if index_path.exists() and metadata_path.exists():
            latest_mtime = max(index_path.stat().st_mtime, metadata_path.stat().st_mtime)
            faiss_timestamps.append(datetime.datetime.fromtimestamp(latest_mtime))

    if not faiss_timestamps:
        return False

    last_faiss_time = max(faiss_timestamps)
    last_run_time = last_faiss_time
    if summary_path.exists():
        summary_time = datetime.datetime.fromtimestamp(summary_path.stat().st_mtime)
        last_run_time = max(last_run_time, summary_time)

    with state.lock:
        if state.last_faiss_time is None:
            state.last_faiss_time = last_faiss_time
        if state.last_run_time is None:
            state.last_run_time = last_run_time
        if state.last_news_time is None and any(store_time for store_time in faiss_timestamps):
            state.last_news_time = last_faiss_time

    record_activity_event(
        "startup_sequence",
        "completed",
        "기존 FAISS/요약 아티팩트를 복구해 startup full_analysis를 건너뛰었습니다.",
        update_status=True,
    )
    return True


def _push_front(
    items: list[dict[str, Any]], item: dict[str, Any], limit: int = 30
) -> list[dict[str, Any]]:
    return ([item] + items)[:limit]


def _parse_event_time(value: Any) -> datetime.datetime | None:
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(text)
    except Exception:
        return None


def build_backend_diagnostics(
    worker_running: bool | None = None,
    worker_interval_seconds: int | None = None,
) -> dict[str, Any]:
    now = datetime.datetime.now()
    with state.lock:
        activity_log = list(state.agent_activity_log)
        vector_events = list(state.vector_events)
        news_crawl_running = bool(state.news_crawl_running)
        news_crawl_target_count = int(state.news_crawl_target_count or 0)
        news_crawl_success_count = int(state.news_crawl_success_count or 0)
        news_crawl_failure_count = int(state.news_crawl_failure_count or 0)
        last_news_crawl_time = state.last_news_crawl_time
        last_faiss_time = state.last_faiss_time
        worker_runtime_stats = dict(state.worker_runtime_stats)

    activity_events_last_60s = 0
    vector_events_last_60s = 0
    activity_by_source: dict[str, int] = {}
    vector_by_source: dict[str, int] = {}

    for event in activity_log:
        source = str(event.get("source") or "unknown")
        activity_by_source[source] = activity_by_source.get(source, 0) + 1
        ts = _parse_event_time(event.get("timestamp"))
        if ts is not None and (now - ts).total_seconds() <= 60:
            activity_events_last_60s += 1

    for event in vector_events:
        source = str(event.get("source") or "unknown")
        vector_by_source[source] = vector_by_source.get(source, 0) + 1
        ts = _parse_event_time(event.get("timestamp"))
        if ts is not None and (now - ts).total_seconds() <= 60:
            vector_events_last_60s += 1

    latest_activity = activity_log[0] if activity_log else {}
    latest_vector = vector_events[0] if vector_events else {}
    crawl_backlog = max(news_crawl_target_count - news_crawl_success_count - news_crawl_failure_count, 0)

    hotspots: list[str] = []
    if worker_running:
        hotspots.append(f"worker loop active every {worker_interval_seconds or '?'}s")
    if activity_events_last_60s >= 6:
        hotspots.append(f"activity events last 60s: {activity_events_last_60s}")
    if vector_events_last_60s >= 2:
        hotspots.append(f"vector rebuild events last 60s: {vector_events_last_60s}")
    if news_crawl_running:
        hotspots.append(f"news crawl backlog: {crawl_backlog}")

    return {
        "worker_running": bool(worker_running),
        "worker_interval_seconds": worker_interval_seconds,
        "activity_events_last_60s": activity_events_last_60s,
        "vector_events_last_60s": vector_events_last_60s,
        "last_activity_time": latest_activity.get("timestamp"),
        "last_activity_source": latest_activity.get("source"),
        "last_vector_event_time": latest_vector.get("timestamp"),
        "last_vector_event_source": latest_vector.get("source"),
        "last_faiss_time": (
            last_faiss_time.isoformat()
            if isinstance(last_faiss_time, datetime.datetime)
            else last_faiss_time
        ),
        "news_crawl_running": news_crawl_running,
        "news_crawl_target_count": news_crawl_target_count,
        "news_crawl_success_count": news_crawl_success_count,
        "news_crawl_failure_count": news_crawl_failure_count,
        "news_crawl_backlog": crawl_backlog,
        "last_news_crawl_time": (
            last_news_crawl_time.isoformat() if last_news_crawl_time else None
        ),
        "top_activity_sources": sorted(
            activity_by_source.items(), key=lambda item: item[1], reverse=True
        )[:5],
        "top_vector_sources": sorted(
            vector_by_source.items(), key=lambda item: item[1], reverse=True
        )[:5],
        "hotspots": hotspots,
        "worker_runtime": worker_runtime_stats,
    }


def update_worker_runtime_stats(stats: dict[str, Any]) -> None:
    with state.lock:
        state.worker_runtime_stats = safe_serialize(stats)


def _trim_runtime_text(text: str, limit: int = 6000) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[-limit:]


def update_ollama_runtime(
    agent: str,
    status: str,
    *,
    model: str | None = None,
    prompt: str | None = None,
    response_text: str | None = None,
    error: str | None = None,
) -> None:
    timestamp = datetime.datetime.now().isoformat()
    with state.lock:
        runtime = dict(state.ollama_runtime or {})
        should_initialize_running = (
            status == "running"
            and (
                prompt is not None
                or runtime.get("agent") != agent
                or runtime.get("status") != "running"
            )
        )
        if should_initialize_running:
            runtime = {
                "agent": agent,
                "status": status,
                "model": model,
                "prompt": _trim_runtime_text(prompt or "", limit=12000),
                "response_text": "",
                "error": None,
                "started_at": timestamp,
                "updated_at": timestamp,
                "completed_at": None,
            }
        else:
            runtime["agent"] = agent
            runtime["status"] = status
            if model is not None:
                runtime["model"] = model
            if prompt is not None:
                runtime["prompt"] = _trim_runtime_text(prompt, limit=12000)
            if response_text is not None:
                runtime["response_text"] = _trim_runtime_text(response_text)
            if error is not None:
                runtime["error"] = error
            runtime["updated_at"] = timestamp
            if status in {"completed", "failed"}:
                runtime["completed_at"] = timestamp
        state.ollama_runtime = runtime


def record_prompt_input(agent: str, prompt_input: dict[str, Any]) -> None:
    timestamp = datetime.datetime.now().isoformat()
    with state.lock:
        if agent == "log_agent":
            state.latest_log_prompt_input = safe_serialize(prompt_input)
            state.last_log_prompt_input_time = datetime.datetime.fromisoformat(timestamp)
        elif agent == "news_agent":
            state.latest_news_prompt_input = safe_serialize(prompt_input)
            state.last_news_prompt_input_time = datetime.datetime.fromisoformat(timestamp)


def record_ollama_progress(agent: str, event: str, payload: dict[str, Any]) -> None:
    model = str(payload.get("model") or "").strip() or None
    if event == "start":
        update_ollama_runtime(
            agent,
            "running",
            model=model,
            prompt=str(payload.get("prompt") or ""),
        )
        return
    if event == "chunk":
        update_ollama_runtime(
            agent,
            "running",
            model=model,
            response_text=str(payload.get("text") or ""),
        )
        return
    if event == "completed":
        update_ollama_runtime(
            agent,
            "completed",
            model=model,
            response_text=str(payload.get("text") or ""),
        )
        return
    if event == "failed":
        update_ollama_runtime(
            agent,
            "failed",
            model=model,
            error=str(payload.get("error") or "Ollama 실행 실패"),
            response_text=str(payload.get("text") or ""),
        )
        return


def record_activity_event(
    source: str, status: str, detail: str, update_status: bool = False
) -> None:
    timestamp = datetime.datetime.now().isoformat()
    event = {
        "source": source,
        "status": status,
        "detail": detail,
        "timestamp": timestamp,
    }
    with state.lock:
        state.agent_activity_log = _push_front(state.agent_activity_log, event)
        if update_status:
            state.agent_statuses[source] = {
                "status": status,
                "detail": detail,
                "updated_at": timestamp,
            }


def record_vector_event(
    source: str, action: str, before_count: int, after_count: int, detail: str
) -> None:
    timestamp = datetime.datetime.now().isoformat()
    event = {
        "source": source,
        "action": action,
        "before_count": before_count,
        "after_count": after_count,
        "added_count": after_count - before_count,
        "detail": detail,
        "timestamp": timestamp,
    }
    with state.lock:
        state.vector_events = _push_front(state.vector_events, event)
    # update a full FAISS snapshot for UI consumers
    try:
        from rag.vector_db import list_vectors

        try:
            items = list_vectors(limit=1000)
        except Exception:
            items = []
        with state.lock:
            state.full_faiss_items = items
            state.last_faiss_time = timestamp
    except Exception:
        # non-fatal: just skip snapshot update
        pass


def reset_strategy_runtime(question: str) -> None:
    timestamp = datetime.datetime.now()
    default_statuses = {
        "orchestrator": {
            "status": "running",
            "detail": "질문을 접수하고 실행 순서를 준비 중입니다.",
            "updated_at": timestamp.isoformat(),
        },
        "log_agent": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
        "news_agent": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
        "regulation_agent": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
        "decision_agent": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
        "vector_store": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
    }
    with state.lock:
        state.latest_strategy_question = question
        state.last_strategy_time = timestamp
        state.agent_statuses = default_statuses
    record_activity_event(
        "orchestrator", "running", f"질문 접수: {question}", update_status=True
    )


def _extract_first_json_block(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    try:
        json.loads(value)
        return value
    except Exception:
        pass
    match = re.search(r"\{.*\}", value, flags=re.S)
    return match.group(0).strip() if match else value


def _parse_json_payload(text: str) -> dict[str, Any]:
    candidate = _extract_first_json_block(text)
    if not candidate:
        return {}
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else {"raw_text": candidate}
    except Exception:
        return {"raw_text": str(text or "").strip()}


def _safe_number(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except Exception:
        return 0.0


def _keyword_bonus(text: str, keywords: tuple[str, ...]) -> int:
    haystack = str(text or "").lower()
    return sum(1 for keyword in keywords if keyword.lower() in haystack)


def _build_cardloan_debate_state(question: str, reviewer_prompts: dict[str, str]) -> dict[str, Any]:
    timestamp = datetime.datetime.now().isoformat()
    return {
        "status": "running",
        "question": question,
        "summary": "카드론 토론실 실행 준비 중",
        "current_stage": "신용기획부",
        "started_at": timestamp,
        "updated_at": timestamp,
        "completed_at": None,
        "error": None,
        "round_results": [],
        "reviewer_prompts": safe_serialize(reviewer_prompts),
    }


def _update_cardloan_debate_state(**updates: Any) -> None:
    timestamp = datetime.datetime.now().isoformat()
    with state.lock:
        current = dict(state.cardloan_debate or {})
        current.update(safe_serialize(updates))
        current["updated_at"] = timestamp
        state.cardloan_debate = current


def reset_cardloan_debate_runtime(question: str, reviewer_prompts: dict[str, str]) -> None:
    timestamp = datetime.datetime.now()
    default_statuses = {
        "orchestrator": {
            "status": "running",
            "detail": "카드론 토론실 실행 순서를 준비 중입니다.",
            "updated_at": timestamp.isoformat(),
        },
        "credit_planning_agent": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
        "sales_strategy_agent": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
        "solution_planning_agent": {
            "status": "pending",
            "detail": "대기 중",
            "updated_at": timestamp.isoformat(),
        },
    }
    with state.lock:
        statuses = dict(state.agent_statuses or {})
        statuses.update(default_statuses)
        state.agent_statuses = statuses
        state.latest_strategy_question = question
        state.last_strategy_time = timestamp
        state.cardloan_debate = _build_cardloan_debate_state(question, reviewer_prompts)
    record_activity_event(
        "orchestrator",
        "running",
        f"카드론 토론실 시작: {question}",
        update_status=True,
    )


def _build_news_signal_candidates(limit: int = 60) -> list[dict[str, Any]]:
    news_items = list_vectors(limit=limit, store_name=FAISS_STORE_NEWS)
    candidates: list[dict[str, Any]] = []
    for item in news_items:
        item_type = str(item.get("type") or "").strip().lower()
        if item_type not in {"signal_news", "generated_news", "news"}:
            continue
        features = item.get("features") or {}
        snippet = str(item.get("snippet") or "").strip()
        risk_signal = [str(value).strip() for value in features.get("risk_signal") or [] if str(value).strip()]
        opportunity_signal = [str(value).strip() for value in features.get("opportunity_signal") or [] if str(value).strip()]
        linked_decision = [str(value).strip() for value in features.get("linked_decision") or [] if str(value).strip()]
        tags = [str(value).strip() for value in features.get("tags") or [] if str(value).strip()]
        signal_summary = str(features.get("signal_summary") or snippet or "시장 신호 요약 없음").strip()
        impact_score = min(5.0, 1.0 + len(linked_decision) + (_keyword_bonus(signal_summary + " " + snippet, ("금리", "규제", "dsr", "연체", "부실")) * 0.5))
        urgency_score = min(5.0, 1.0 + len(risk_signal) + (_keyword_bonus(signal_summary + " " + snippet, ("긴급", "즉시", "변경", "강화", "충격")) * 0.5))
        risk_score = min(5.0, 1.0 + len(risk_signal) + (_keyword_bonus(signal_summary + " " + snippet, ("리스크", "연체", "부실", "규제", "심사 강화")) * 0.5))
        composite_score = impact_score * 0.5 + urgency_score * 0.3 + risk_score * 0.2
        candidates.append(
            {
                "title": str(item.get("name") or item.get("id") or "시장 신호").strip(),
                "summary": signal_summary,
                "tags": tags,
                "risk_signal": risk_signal,
                "opportunity_signal": opportunity_signal,
                "linked_decision": linked_decision,
                "impact_score": round(impact_score, 2),
                "urgency_score": round(urgency_score, 2),
                "risk_score": round(risk_score, 2),
                "composite_score": round(composite_score, 2),
            }
        )
    candidates.sort(key=lambda row: row.get("composite_score", 0), reverse=True)
    return candidates[:5]


def _resolve_case_decision(item: dict[str, Any]) -> str:
    out_fields = item.get("out_fields") or {}
    features = item.get("features") or {}
    reject_reasons = item.get("reject_reason_details") or item.get("reject_reason_codes") or []
    decision = str(
        out_fields.get("승인 여부")
        or out_fields.get("심사결과")
        or features.get("decision")
        or features.get("심사결과")
        or ""
    ).strip()
    if "거절" in decision or reject_reasons:
        return "거절"
    if "승인" in decision:
        return "승인"
    return decision or "미상"


def _build_case_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    features = item.get("features") or {}
    reasons = item.get("reject_reason_details") or item.get("reject_reason_codes") or []
    reason_texts: list[str] = []
    for reason in reasons[:3]:
        if isinstance(reason, dict):
            rendered = str(reason.get("description") or reason.get("code") or "").strip()
        else:
            rendered = str(reason).strip()
        if rendered:
            reason_texts.append(rendered)

    return {
        "product": str(item.get("product") or "UNKNOWN").strip(),
        "decision": _resolve_case_decision(item),
        "available_amount": int(_safe_number(features.get("available_amount") or features.get("최종대출가능금액") or 0)),
        "applied_rate": round(_safe_number(features.get("applied_rate") or features.get("금리") or 0), 2),
        "recognized_income": int(_safe_number(features.get("recognized_income") or features.get("annual_income") or 0)),
        "dsr_ratio": round(_safe_number(features.get("dsr_ratio") or features.get("dsr비율") or 0), 2),
        "dti": round(_safe_number(features.get("dti") or 0), 2),
        "reason": reason_texts,
        "snippet": str(item.get("snippet") or "").strip()[:220],
    }


def _select_sales_case_sets(limit: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_items = list_vectors(limit=80, store_name=FAISS_STORE_CUSTOMER) + list_vectors(limit=80, store_name=FAISS_STORE_LOGS)
    case_snapshots = [_build_case_snapshot(item) for item in candidate_items if str(item.get("type") or "").strip().lower() in {"customer_pattern", "log", "generated_log"}]
    approved_cases = [item for item in case_snapshots if item.get("decision") == "승인"]
    rejected_cases = [item for item in case_snapshots if item.get("decision") == "거절"]
    approved_cases.sort(key=lambda row: (row.get("available_amount", 0), row.get("applied_rate", 0)), reverse=True)
    rejected_cases.sort(key=lambda row: (len(row.get("reason") or []), row.get("dsr_ratio", 0), row.get("dti", 0)), reverse=True)
    current_customer = (rejected_cases or approved_cases or [{"product": "카드론", "decision": "미상", "reason": [], "snippet": "현재 고객 요약 없음"}])[0]
    return current_customer, approved_cases[:limit], rejected_cases[:limit]


def _compact_market_signal_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "제목": str(item.get("title") or "시장 신호").strip()[:40],
        "요약": str(item.get("summary") or "").strip()[:90],
        "위험": [str(value).strip()[:40] for value in (item.get("risk_signal") or [])[:2] if str(value).strip()],
        "기회": [str(value).strip()[:40] for value in (item.get("opportunity_signal") or [])[:1] if str(value).strip()],
        "심사연결": [str(value).strip()[:42] for value in (item.get("linked_decision") or [])[:2] if str(value).strip()],
    }


def _compact_case_prompt_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "상품": str(item.get("product") or "-").strip(),
        "판단": str(item.get("decision") or "미상").strip(),
        "가능한도": int(item.get("available_amount") or 0),
        "금리": round(_safe_number(item.get("applied_rate") or 0), 2),
        "인정소득": int(item.get("recognized_income") or 0),
        "DSR": round(_safe_number(item.get("dsr_ratio") or 0), 2),
        "DTI": round(_safe_number(item.get("dti") or 0), 2),
        "거절사유": [str(value).strip()[:40] for value in (item.get("reason") or [])[:2] if str(value).strip()],
        "요약": str(item.get("snippet") or "").strip()[:90],
    }


def _cardloan_stage_output_rules(schema_text: str) -> str:
    return (
        "출력 규칙:\n"
        "- 반드시 JSON만 출력하라\n"
        "- JSON 내부 값은 모두 한국어로만 작성하라\n"
        "- 영어 문장과 영어 불릿을 쓰지 마라\n"
        "- 각 배열은 최대 2개만 작성하라\n"
        "- 각 항목은 짧고 바로 읽히는 문장으로 작성하라\n"
        "- 숫자는 필요한 경우만 간단히 포함하라\n"
        f"{schema_text}"
    )


def _build_stage_preview(stage_key: str, payload: dict[str, Any]) -> str:
    if stage_key == "credit_planning_agent":
        risk_text = str(((payload.get("risk_forecast") or [""])[0]) or "").strip()
        weakness_text = str(((payload.get("policy_weakness") or [""])[0]) or "").strip()
        rule_text = str(((payload.get("new_rules") or [""])[0]) or "").strip()
        parts = []
        if risk_text:
            parts.append(f"지금 보면 {risk_text}")
        if weakness_text:
            parts.append(f"현재 심사에서는 {weakness_text}")
        if rule_text:
            parts.append(f"그래서 {rule_text} 기준을 먼저 보완하겠습니다")
        return ". ".join(parts) + ("." if parts else "") or "미래 리스크를 아직 정리하지 못했습니다."
    if stage_key == "sales_strategy_agent":
        status_text = str(payload.get("current_status") or "현재 고객 상태 판단이 아직 없습니다").strip()
        reason_text = str(((payload.get("rejection_reason") or [""])[0]) or "").strip()
        condition_text = str(((payload.get("conversion_conditions") or [""])[0]) or "").strip()
        action_text = str(((payload.get("action_plan") or [""])[0]) or "").strip()
        parts = [f"제 판단으로 현재 고객은 {status_text}"] if status_text else []
        if reason_text:
            parts.append(f"핵심 걸림돌은 {reason_text}")
        if condition_text:
            parts.append(f"전환은 {condition_text} 조건으로 시도하겠습니다")
        if action_text:
            parts.append(f"실행은 {action_text}부터 진행하겠습니다")
        return ". ".join(parts) + ("." if parts else "") or "승인 전환 전략을 아직 정리하지 못했습니다."
    new_product = payload.get("new_product") or {}
    product_name = str(new_product.get("name") or "").strip()
    target_text = str(new_product.get("target") or "").strip()
    structure_text = str(new_product.get("structure") or "").strip()
    conflict_text = str(payload.get("conflict_analysis") or "").strip()
    parts = []
    if conflict_text:
        parts.append(f"두 부서 의견을 비교해 보니 {conflict_text}")
    if product_name:
        parts.append(f"그래서 {product_name} 상품을 제안합니다")
    if target_text:
        parts.append(f"대상은 {target_text}입니다")
    if structure_text:
        parts.append(f"구조는 {structure_text}로 설계하겠습니다")
    return ". ".join(parts) + ("." if parts else "") or "상품 구조 제안을 아직 정리하지 못했습니다."


def _build_stage_evidence(stage_key: str, payload: dict[str, Any]) -> str:
    if stage_key == "credit_planning_agent":
        effect_text = str(((payload.get("expected_effect") or [""])[0]) or "").strip()
        rule_text = str(((payload.get("new_rules") or [""])[1]) or ((payload.get("new_rules") or [""])[0] if (payload.get("new_rules") or []) else "")).strip()
        parts = []
        if rule_text:
            parts.append(f"추가로 {rule_text}")
        if effect_text:
            parts.append(f"이렇게 바꾸면 {effect_text}")
        return ". ".join(parts) + ("." if parts else "") or "정책 보완 근거를 아직 정리하지 못했습니다."
    if stage_key == "sales_strategy_agent":
        action_text = str(((payload.get("action_plan") or [""])[1]) or ((payload.get("action_plan") or [""])[0] if (payload.get("action_plan") or []) else "")).strip()
        reason_text = str(((payload.get("rejection_reason") or [""])[1]) or "").strip()
        parts = []
        if reason_text:
            parts.append(f"보조 원인은 {reason_text}")
        if action_text:
            parts.append(f"현장 실행은 {action_text}까지 같이 보겠습니다")
        return ". ".join(parts) + ("." if parts else "") or "영업 전환 근거를 아직 정리하지 못했습니다."
    improvement_text = str(((payload.get("improvement") or [""])[0]) or "").strip()
    risk_control = ", ".join(str(item).strip() for item in ((payload.get("new_product") or {}).get("risk_control") or [])[:2] if str(item).strip())
    profit_text = str((payload.get("new_product") or {}).get("profit_model") or "").strip()
    parts = []
    if improvement_text:
        parts.append(f"기존 상품은 {improvement_text}")
    if risk_control:
        parts.append(f"리스크 통제는 {risk_control} 중심으로 가져가겠습니다")
    if profit_text:
        parts.append(f"수익 구조는 {profit_text}로 맞추겠습니다")
    return ". ".join(parts) + ("." if parts else "") or "상품 설계 근거를 아직 정리하지 못했습니다."


def _build_cardloan_debate_summary(round_results: list[dict[str, Any]]) -> str:
    if not round_results:
        return "카드론 토론실 결과가 아직 없습니다."
    lines = [f"{item.get('name')}: {item.get('preview')}" for item in round_results]
    return "카드론 토론실 완료\n\n" + "\n".join(lines)


def _run_cardloan_stage(
    *,
    agent_key: str,
    name: str,
    display: str,
    tone: str,
    stage_title: str,
    verdict_label: str,
    prompt: str,
) -> dict[str, Any]:
    _update_cardloan_debate_state(current_stage=name, summary=f"{name} 단계 실행 중")
    record_activity_event(agent_key, "running", f"{name} 관점의 Ollama 분석을 시작합니다.", update_status=True)
    raw_text = lightweight_ollama_generate(
        prompt,
        progress_callback=lambda event, payload: record_ollama_progress(agent_key, event, payload),
    )
    payload = _parse_json_payload(raw_text)
    stage_result = {
        "persona_id": agent_key,
        "name": name,
        "display": display,
        "tone": tone,
        "stage_title": stage_title,
        "verdict": verdict_label,
        "preview": _build_stage_preview(agent_key, payload),
        "evidence": _build_stage_evidence(agent_key, payload),
        "response": {
            "answer": raw_text,
            "parsed": payload,
            "raw_text": raw_text,
        },
        "generated_at": datetime.datetime.now().isoformat(),
    }
    with state.lock:
        debate_state = dict(state.cardloan_debate or {})
        round_results = list(debate_state.get("round_results") or [])
        round_results.append(safe_serialize(stage_result))
        debate_state["round_results"] = round_results
        debate_state["current_stage"] = name
        debate_state["summary"] = f"{name} 단계 완료"
        debate_state["updated_at"] = datetime.datetime.now().isoformat()
        state.cardloan_debate = debate_state
    record_activity_event(agent_key, "completed", f"{name} 단계 결과를 생성했습니다.", update_status=True)
    return stage_result


def ask_cardloan_debate(question: str, reviewer_prompts: dict[str, str] | None = None) -> dict[str, Any]:
    effective_prompts = {key: str(value).strip() for key, value in (reviewer_prompts or {}).items() if str(value).strip()}
    reset_cardloan_debate_runtime(question, effective_prompts)

    try:
        with OLLAMA_EXECUTION_LOCK:
            market_signals = [_compact_market_signal_entry(item) for item in _build_news_signal_candidates(limit=60)[:4]]
            current_customer, approved_cases, rejected_cases = _select_sales_case_sets(limit=4)
            current_customer_prompt = _compact_case_prompt_entry(current_customer)
            approved_cases_prompt = [_compact_case_prompt_entry(item) for item in approved_cases[:3]]
            rejected_cases_prompt = [_compact_case_prompt_entry(item) for item in rejected_cases[:3]]

            credit_prompt = (
                f"{effective_prompts.get('credit_planning_agent') or '너는 신용기획부 리스크 정책 담당자다. 미래 리스크를 선제적으로 차단하고 심사 기준을 개선하라.'}\n\n"
                f"[토론 주제]\n{question}\n\n"
                f"[시장 신호 TOP5]\n{json.dumps(market_signals, ensure_ascii=False, indent=2)}\n\n"
                "지시:\n"
                "1. 향후 발생할 주요 리스크를 예측하라\n"
                "2. 현재 심사 정책의 취약점을 도출하라\n"
                "3. 보완해야 할 심사 기준을 제안하라\n"
                "4. 구체적인 룰(조건)을 작성하라\n\n"
                f"{_cardloan_stage_output_rules('{\"risk_forecast\": [], \"policy_weakness\": [], \"new_rules\": [], \"expected_effect\": []}') }"
            )
            credit_result = _run_cardloan_stage(
                agent_key="credit_planning_agent",
                name="신용기획부",
                display="신용기획부",
                tone="리스크 정책",
                stage_title="미래 리스크 예측 및 심사 룰 개편",
                verdict_label="정책 선제 수정",
                prompt=credit_prompt,
            )

            sales_prompt = (
                f"{effective_prompts.get('sales_strategy_agent') or '너는 금융영업부 전략 담당자다. 거절된 고객을 승인 가능한 고객으로 전환하고 승인율과 수익, 채널 전략을 함께 설계하라.'}\n\n"
                f"[토론 주제]\n{question}\n\n"
                f"[현재 고객]\n{json.dumps({'question': question, 'current_customer': current_customer_prompt}, ensure_ascii=False, indent=2)}\n\n"
                f"[고금액 승인 사례]\n{json.dumps(approved_cases_prompt, ensure_ascii=False, indent=2)}\n\n"
                f"[유사 거절 사례]\n{json.dumps(rejected_cases_prompt, ensure_ascii=False, indent=2)}\n\n"
                "지시:\n"
                "1. 승인 사례와 거절 사례의 차이를 분석하라\n"
                "2. 현재 고객이 거절된 핵심 원인을 찾아라\n"
                "3. 승인으로 전환하기 위한 조건을 제시하라\n"
                "4. 실행 가능한 전략을 구체적으로 작성하라\n\n"
                f"{_cardloan_stage_output_rules('{\"current_status\": \"\", \"rejection_reason\": [], \"conversion_conditions\": [], \"action_plan\": []}') }"
            )
            sales_result = _run_cardloan_stage(
                agent_key="sales_strategy_agent",
                name="금융영업부",
                display="금융영업부",
                tone="전환 영업",
                stage_title="승인 전환 전략 수립",
                verdict_label="영업 전환 전략",
                prompt=sales_prompt,
            )

            solution_prompt = (
                f"{effective_prompts.get('solution_planning_agent') or '너는 금융솔루션부 상품 기획자다. 리스크를 통제하면서도 매출을 확대하는 카드론 상품 구조를 설계하라.'}\n\n"
                f"[토론 주제]\n{question}\n\n"
                f"[리스크 정책]\n{json.dumps(credit_result.get('response', {}).get('parsed', {}), ensure_ascii=False, indent=2)}\n\n"
                f"[영업 전략]\n{json.dumps(sales_result.get('response', {}).get('parsed', {}), ensure_ascii=False, indent=2)}\n\n"
                "지시:\n"
                "1. 두 전략의 충돌 지점을 분석하라\n"
                "2. 이를 해결할 수 있는 상품 구조를 설계하라\n"
                "3. 신상품 1개를 제안하라\n"
                "4. 기존 상품 개선안도 제시하라\n\n"
                f"{_cardloan_stage_output_rules('{\"conflict_analysis\": \"\", \"new_product\": {\"name\": \"\", \"target\": \"\", \"structure\": \"\", \"risk_control\": [], \"profit_model\": \"\"}, \"improvement\": []}') }"
            )
            solution_result = _run_cardloan_stage(
                agent_key="solution_planning_agent",
                name="금융솔루션부",
                display="금융솔루션부",
                tone="상품 기획",
                stage_title="상품 구조 설계 및 신상품 제안",
                verdict_label="상품 설계",
                prompt=solution_prompt,
            )

        round_results = [credit_result, sales_result, solution_result]
        summary = _build_cardloan_debate_summary(round_results)
        completed_at = datetime.datetime.now().isoformat()
        with state.lock:
            debate_state = dict(state.cardloan_debate or {})
            debate_state.update(
                {
                    "status": "completed",
                    "summary": summary,
                    "current_stage": "완료",
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                    "round_results": safe_serialize(round_results),
                }
            )
            state.cardloan_debate = debate_state
            state.last_strategy_time = datetime.datetime.now()
        record_activity_event("orchestrator", "completed", "카드론 토론실 3단계 실행이 완료되었습니다.", update_status=True)
        return {
            "status": "completed",
            "question": question,
            "summary": summary,
            "round_results": safe_serialize(round_results),
            "current_stage": "완료",
            "started_at": state.cardloan_debate.get("started_at"),
            "completed_at": completed_at,
        }
    except Exception as error:
        failed_at = datetime.datetime.now().isoformat()
        with state.lock:
            debate_state = dict(state.cardloan_debate or {})
            debate_state.update(
                {
                    "status": "failed",
                    "error": str(error),
                    "summary": f"카드론 토론실 실패: {error}",
                    "completed_at": failed_at,
                    "updated_at": failed_at,
                }
            )
            state.cardloan_debate = debate_state
        record_activity_event("orchestrator", "failed", f"카드론 토론실 실행 실패: {error}", update_status=True)
        raise


def resolve_project_path(path_str: str) -> str:
    # backend 폴더에서 서버를 띄워도 data/logs 같은 상대경로를 프로젝트 루트 기준으로 맞춰줍니다.
    candidate = pathlib.Path(path_str)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _iter_log_dir_candidates(log_dir: str) -> list[str]:
    resolved = resolve_project_path(log_dir)
    candidates = [resolved]

    normalized = pathlib.Path(log_dir).as_posix().strip().lower()
    if normalized == "data/logs":
        fallback_dir = str((PROJECT_ROOT / "logs").resolve())
        if fallback_dir not in candidates:
            candidates.append(fallback_dir)

    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _read_log_file_text(file_path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(file_path, encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as file:
            return file.read()
    except OSError:
        return ""


def _iter_log_analysis_batches(
    file_path: str,
    batch_pair_limit: int = 2000,
):
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(file_path, encoding=encoding) as file:
                current_in = None
                batch_lines: list[str] = []
                pair_count = 0

                for line in file:
                    if "in_data =" in line:
                        current_in = line.rstrip("\r\n")
                        continue

                    if "out_data =" in line and current_in:
                        batch_lines.append(current_in)
                        batch_lines.append(line.rstrip("\r\n"))
                        current_in = None
                        pair_count += 1

                        if pair_count >= batch_pair_limit:
                            yield "\n".join(batch_lines)
                            batch_lines = []
                            pair_count = 0

                if batch_lines:
                    yield "\n".join(batch_lines)
            return
        except UnicodeDecodeError:
            continue
        except OSError:
            return

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as file:
            current_in = None
            batch_lines: list[str] = []
            pair_count = 0

            for line in file:
                if "in_data =" in line:
                    current_in = line.rstrip("\r\n")
                    continue

                if "out_data =" in line and current_in:
                    batch_lines.append(current_in)
                    batch_lines.append(line.rstrip("\r\n"))
                    current_in = None
                    pair_count += 1

                    if pair_count >= batch_pair_limit:
                        yield "\n".join(batch_lines)
                        batch_lines = []
                        pair_count = 0

            if batch_lines:
                yield "\n".join(batch_lines)
    except OSError:
        return


def load_all_logs(log_dir: str = "data/logs") -> tuple[str, int]:
    logs = ""
    count = 0
    loaded_files: set[str] = set()
    for candidate_dir in _iter_log_dir_candidates(log_dir):
        if not os.path.exists(candidate_dir):
            continue
        for name in os.listdir(candidate_dir):
            lowered = name.lower()
            if lowered.endswith(".txt") or lowered.endswith(".log"):
                file_path = os.path.abspath(os.path.join(candidate_dir, name))
                if file_path in loaded_files:
                    continue
                file_text = _read_log_file_text(file_path)
                if not file_text.strip():
                    continue
                logs += file_text
                count += 1
                loaded_files.add(file_path)
    return logs, count


def _collect_log_file_paths(log_dir: str = "data/logs") -> list[str]:
    file_paths: list[str] = []
    seen: set[str] = set()
    for candidate_dir in _iter_log_dir_candidates(log_dir):
        if not os.path.exists(candidate_dir):
            continue
        for name in sorted(os.listdir(candidate_dir)):
            lowered = name.lower()
            if not (lowered.endswith(".txt") or lowered.endswith(".log")):
                continue
            file_path = os.path.abspath(os.path.join(candidate_dir, name))
            if file_path in seen:
                continue
            seen.add(file_path)
            file_paths.append(file_path)
    return file_paths


def _build_log_file_signature(file_paths: list[str]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for file_path in file_paths:
        try:
            stat = os.stat(file_path)
            signature.append((file_path, int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            continue
    return tuple(signature)


def _signature_index_by_path(
    signature: tuple[tuple[str, int, int], ...],
) -> dict[str, tuple[int, int]]:
    return {file_path: (mtime_ns, size) for file_path, mtime_ns, size in signature}


def _load_static_log_results(
    log_paths: list[str],
    cached_signature_by_path: dict[str, tuple[int, int]] | None = None,
    cached_results_by_file: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, list[dict[str, Any]]]]:
    results: list[dict[str, Any]] = []
    file_count = 0
    results_by_file: dict[str, list[dict[str, Any]]] = {}
    cached_signature_by_path = cached_signature_by_path or {}
    cached_results_by_file = cached_results_by_file or {}

    for file_path in log_paths:
        try:
            stat = os.stat(file_path)
            current_signature = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            continue

        cached_signature = cached_signature_by_path.get(file_path)
        if cached_signature == current_signature:
            cached_results = list(cached_results_by_file.get(file_path) or [])
            if cached_results:
                results.extend(cached_results)
                results_by_file[file_path] = cached_results
            file_count += 1
            continue

        file_results: list[dict[str, Any]] = []
        file_had_content = False
        for batch_text in _iter_log_analysis_batches(file_path):
            if not batch_text.strip():
                continue
            file_had_content = True
            try:
                batch_results = analyze_logs(batch_text)
            except MemoryError:
                continue
            file_results.extend(batch_results)
            results.extend(batch_results)
        if file_had_content:
            results_by_file[file_path] = file_results
            file_count += 1
    if not results and file_count == 0:
        return [], 0, {}
    return results, file_count, results_by_file


def _read_generated_log_delta(
    file_path: str,
    previous_offset: int,
    previous_pending_text: str,
) -> tuple[list[dict[str, Any]], int, str, bool, int]:
    try:
        current_size = int(os.path.getsize(file_path) or 0)
    except OSError:
        return [], 0, "", False, 0

    reset_required = current_size < previous_offset
    offset = 0 if reset_required else previous_offset
    pending_text = "" if reset_required else previous_pending_text

    try:
        with open(file_path, "rb") as file:
            file.seek(offset)
            chunk = file.read()
    except OSError:
        return [], offset, pending_text, reset_required, 0

    if not chunk and not reset_required:
        return [], offset, pending_text, False, 1 if current_size > 0 else 0

    delta_text = pending_text + chunk.decode("utf-8", errors="ignore")
    if not delta_text:
        return [], offset + len(chunk), "", reset_required, 1 if current_size > 0 else 0

    if delta_text.endswith(("\n", "\r")):
        complete_text = delta_text
        next_pending_text = ""
    else:
        last_newline = max(delta_text.rfind("\n"), delta_text.rfind("\r"))
        if last_newline == -1:
            complete_text = ""
            next_pending_text = delta_text
        else:
            complete_text = delta_text[: last_newline + 1]
            next_pending_text = delta_text[last_newline + 1 :]

    analyzed_results = analyze_logs(complete_text) if complete_text.strip() else []
    next_offset = offset + len(chunk)
    return analyzed_results, next_offset, next_pending_text, reset_required, 1 if current_size > 0 else 0


def analyze_logs_incremental(log_dir: str = "data/logs") -> tuple[list[dict[str, Any]], int]:
    all_log_paths = _collect_log_file_paths(log_dir)
    generated_path = str((PROJECT_ROOT / "logs" / "generated_live.log").resolve())
    static_paths = [path for path in all_log_paths if os.path.abspath(path) != generated_path]
    static_signature = _build_log_file_signature(static_paths)

    with state.lock:
        cached_static_signature = state.static_log_signature
        cached_static_results = list(state.static_log_results)
        cached_static_file_count = int(state.static_log_file_count or 0)
        cached_static_results_by_file = {
            file_path: list(items)
            for file_path, items in (state.static_log_results_by_file or {}).items()
        }
        cached_generated_results = list(state.generated_log_results)
        cached_generated_offset = int(state.generated_log_offset or 0)
        cached_generated_pending = str(state.generated_log_pending_text or "")

    if static_signature != cached_static_signature:
        static_results, static_file_count, static_results_by_file = _load_static_log_results(
            static_paths,
            cached_signature_by_path=_signature_index_by_path(cached_static_signature),
            cached_results_by_file=cached_static_results_by_file,
        )
        cached_generated_results = []
        cached_generated_offset = 0
        cached_generated_pending = ""
    else:
        static_results = cached_static_results
        static_file_count = cached_static_file_count
        static_results_by_file = cached_static_results_by_file

    generated_results = list(cached_generated_results)
    generated_file_count = 0
    if os.path.exists(generated_path):
        delta_results, next_offset, next_pending, reset_required, generated_file_count = _read_generated_log_delta(
            generated_path,
            cached_generated_offset,
            cached_generated_pending,
        )
        if reset_required:
            generated_results = delta_results
        elif delta_results:
            generated_results.extend(delta_results)
        cached_generated_offset = next_offset
        cached_generated_pending = next_pending

    with state.lock:
        state.static_log_signature = static_signature
        state.static_log_results = list(static_results)
        state.static_log_file_count = static_file_count
        state.static_log_results_by_file = {
            file_path: list(items) for file_path, items in static_results_by_file.items()
        }
        state.generated_log_results = list(generated_results)
        state.generated_log_offset = cached_generated_offset
        state.generated_log_pending_text = cached_generated_pending

    return list(static_results) + list(generated_results), static_file_count + generated_file_count


def safe_serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {key: safe_serialize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(value) for value in obj]
    return str(obj)


def merge_news_items(
    existing_news: list[dict[str, Any]],
    new_news: list[dict[str, Any]],
    max_items: int = 400,
    max_new_items_to_add: int | None = None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    # 같은 제목/링크 조합은 중복으로 보고 하나만 유지합니다.
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    existing_keys = {
        (str(item.get("title", "")).strip(), str(item.get("link", "")).strip())
        for item in existing_news
    }
    new_unique_count = 0
    added_items: list[dict[str, Any]] = []
    max_new_items = (
        max_new_items_to_add
        if max_new_items_to_add is None
        else max(0, int(max_new_items_to_add))
    )

    for item in list(new_news) + list(existing_news):
        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()
        key = (title, link)
        if key in seen:
            continue
        if key not in existing_keys:
            if max_new_items is not None and new_unique_count >= max_new_items:
                continue
            new_unique_count += 1
            added_items.append(item)
        seen.add(key)
        merged.append(item)
        if len(merged) >= max_items:
            break

    return merged, new_unique_count, added_items


def _news_item_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        str(item.get("title", "")).strip(),
        str(item.get("link", "")).strip(),
    )


def enrich_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 원본 로그 분석 결과에 risk 계산 결과를 붙여서 화면이 바로 쓸 수 있는 형태로 바꿉니다.
    enriched: list[dict[str, Any]] = []
    for row in results:
        try:
            product = row.get("product")
            in_fields = row.get("in_fields", {})
            out_fields = row.get("out_fields", {})
            in_mapping = row.get("in_mapping", {})
            out_mapping = row.get("out_mapping", {})
            risk = calculate_risk(
                in_fields, out_fields, in_mapping, out_mapping, product=product
            )
            enriched.append(
                {
                    "product": product,
                    "in_fields": safe_serialize(in_fields),
                    "out_fields": safe_serialize(out_fields),
                    "in_mapping": safe_serialize(in_mapping),
                    "out_mapping": safe_serialize(out_mapping),
                    "reject_reason_codes": safe_serialize(
                        row.get("reject_reason_codes", [])
                    ),
                    "reject_reason_details": safe_serialize(
                        row.get("reject_reason_details", [])
                    ),
                    "risk": safe_serialize(risk),
                }
            )
        except Exception as error:
            enriched.append({"error": str(error)})
    return enriched


def build_chart_payloads(results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # 메인 화면 4개 차트가 공통으로 쓰는 데이터를 한 번에 계산합니다.
    # Streamlit은 이 결과만 받아서 그리므로 화면이 가벼워집니다.
    source_results = results if results is not None else state.results
    enriched = enrich_results(source_results)

    score_labels: list[str] = []
    score_values: list[float] = []
    component_series = {
        "financial": [],
        "credit": [],
        "behavior": [],
        "regulation": [],
    }
    grade_counts: dict[str, int] = {}
    product_grade_counts: dict[str, dict[str, int]] = {}

    for index, row in enumerate(enriched):
        risk = row.get("risk", {}) if isinstance(row, dict) else {}
        details = risk.get("details", {}) if isinstance(risk, dict) else {}
        product = row.get("product", "N/A") if isinstance(row, dict) else "N/A"
        grade = risk.get("grade", "N/A") if isinstance(risk, dict) else "N/A"

        score_labels.append(f"{product}-{index + 1}")
        score_values.append(float(risk.get("score", 0)))
        component_series["financial"].append(float(details.get("financial", 0)))
        component_series["credit"].append(float(details.get("credit", 0)))
        component_series["behavior"].append(float(details.get("behavior", 0)))
        component_series["regulation"].append(float(details.get("regulation", 0)))

        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        if product not in product_grade_counts:
            product_grade_counts[product] = {}
        product_grade_counts[product][grade] = (
            product_grade_counts[product].get(grade, 0) + 1
        )

    try:
        vector_count = get_vector_count()
    except Exception:
        vector_count = 0

    news_count = len(state.news)
    issues_count = len(state.issues)

    payloads = {
        "score_trend": {
            "labels": score_labels,
            "scores": score_values,
        },
        "risk_components": {
            "labels": score_labels,
            "series": component_series,
        },
        "grade_distribution": {
            "grades": grade_counts,
            "by_product": product_grade_counts,
        },
        "vector_status": {
            "vector_count": vector_count,
            "news_count": news_count,
            "issues_count": issues_count,
        },
    }

    with state.lock:
        state.chart_payloads = payloads
        state.last_chart_time = datetime.datetime.now()

    return payloads


def collect_news_bundle(
    accumulate: bool = True,
    max_new_items_to_add: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    record_activity_event("news_collector", "running", "뉴스 RSS를 수집하고 있습니다.")
    news = collect_news()
    with state.lock:
        existing_news = list(state.news)

    new_unique_count = 0
    added_news_items: list[dict[str, Any]] = []
    if accumulate:
        effective_news, new_unique_count, added_news_items = merge_news_items(
            existing_news,
            news,
            max_new_items_to_add=max_new_items_to_add,
        )
    else:
        effective_news = news
        new_unique_count = len(news)
        added_news_items = list(news)

    issues = analyze_news(effective_news)
    collected_at = datetime.datetime.now()

    with state.lock:
        state.news = effective_news
        state.issues = issues
        state.last_news_time = collected_at
        if new_unique_count > 0:
            state.last_new_item_time = collected_at
    record_activity_event(
        "news_collector",
        "completed",
        f"뉴스 {len(effective_news)}건 유지, 신규 기사 {new_unique_count}건 반영",
    )
    build_chart_payloads()

    # If there are new items, fetch article contents in background and then build FAISS
    if new_unique_count > 0:
        def _bg_fetch_and_index(
            news_snapshot: list[dict[str, Any]],
            crawl_candidates: list[dict[str, Any]],
        ):
            crawl_targets = [item for item in news_snapshot if not item.get("content")]
            if crawl_candidates:
                crawl_target_keys = {
                    (str(item.get("title", "")).strip(), str(item.get("link", "")).strip())
                    for item in crawl_candidates
                }
                crawl_targets = [
                    item
                    for item in crawl_targets
                    if (str(item.get("title", "")).strip(), str(item.get("link", "")).strip())
                    in crawl_target_keys
                ]
            record_activity_event(
                "news_agent",
                "pending",
                f"뉴스 본문 크롤링 완료를 기다리는 중입니다. 대상 {len(crawl_targets)}건",
                update_status=True,
            )
            with state.lock:
                state.news_crawl_running = True
                state.news_crawl_target_count = len(crawl_targets)
                state.news_crawl_success_count = 0
                state.news_crawl_failure_count = 0
                state.last_news_crawl_time = datetime.datetime.now()
                state.last_news_crawl_error = None
            record_activity_event(
                "news_crawler",
                "running",
                f"뉴스 본문을 백그라운드로 수집하고 FAISS에 적재합니다. 대상 {len(crawl_targets)}건",
                update_status=True,
            )
            crawled_new_item_count = 0
            crawl_target_links = {
                str(item.get("link", "")).strip()
                for item in crawl_targets
                if str(item.get("link", "")).strip()
            }
            try:
                from agent.news_agent import fetch_article_text

                for i, item in enumerate(news_snapshot):
                    # if content already present, skip
                    if item.get("content"):
                        continue
                    try:
                        txt = fetch_article_text(item.get("link", ""))
                        if txt:
                            # update global state.news matching by link
                            item_link = str(item.get("link", "")).strip()
                            with state.lock:
                                for s in state.news:
                                    if s.get("link") == item.get("link") and not s.get("content"):
                                        s["content"] = txt
                                        state.news_crawl_success_count += 1
                                        state.last_news_crawl_time = datetime.datetime.now()
                                        if item_link and item_link in crawl_target_links:
                                            crawled_new_item_count += 1
                                        break
                        else:
                            with state.lock:
                                state.news_crawl_failure_count += 1
                                state.last_news_crawl_time = datetime.datetime.now()
                                state.last_news_crawl_error = "empty_content"
                    except Exception:
                        with state.lock:
                            state.news_crawl_failure_count += 1
                            state.last_news_crawl_time = datetime.datetime.now()
                            state.last_news_crawl_error = "fetch_failed"
                    # small sleep to be polite
                    time.sleep(0.15)

                # refresh the news agent prompt/briefing only after crawled content exists
                try:
                    if crawled_new_item_count > 0:
                        run_background_news_agent_cycle(should_persist=True)
                    else:
                        record_activity_event(
                            "news_agent",
                            "pending",
                            "신규로 파싱된 뉴스 본문이 없어 뉴스 브리핑 실행을 건너뛰었습니다.",
                            update_status=True,
                        )
                except Exception as e:
                    with state.lock:
                        state.last_news_crawl_error = f"news_agent_failed: {e}"
                    record_activity_event(
                        "news_agent",
                        "failed",
                        f"뉴스 브리핑 생성 실패: {e}",
                        update_status=True,
                    )

                # after fetching all contents, append only newly collected articles into FAISS
                try:
                    appended_count = append_news_vectors_bundle(source="news_crawler")
                    record_activity_event(
                        "news_crawler",
                        "completed",
                        f"뉴스 크롤링 후 신규 기사 {appended_count}건을 FAISS에 증분 적재했습니다.",
                        update_status=True,
                    )
                except Exception as e:
                    record_activity_event(
                        "news_crawler",
                        "failed",
                        f"크롤링 후 FAISS 적재 실패: {e}",
                        update_status=True,
                    )
                build_chart_payloads()
                with state.lock:
                    state.news_crawl_running = False
                    state.last_news_crawl_time = datetime.datetime.now()
            except Exception as e:
                with state.lock:
                    state.news_crawl_running = False
                    state.last_news_crawl_time = datetime.datetime.now()
                    state.last_news_crawl_error = str(e)
                record_activity_event("news_crawler", "failed", f"백그라운드 크롤러 실패: {e}", update_status=True)

        # snapshot to pass into thread
        news_snapshot = list(effective_news)
        crawl_candidates = list(added_news_items)
        t = threading.Thread(
            target=_bg_fetch_and_index,
            args=(news_snapshot, crawl_candidates),
            daemon=True,
        )
        t.start()

    return effective_news, issues


def analyze_logs_bundle(
    raw_logs: str | None = None, log_dir: str = "data/logs"
) -> tuple[list[dict[str, Any]], int]:
    record_activity_event("log_analyzer", "running", "로그 분석을 시작했습니다.")
    if raw_logs is not None:
        raw_text, file_count = raw_logs, 0
        results = analyze_logs(raw_text or "")
    else:
        results, file_count = analyze_logs_incremental(log_dir)
    with state.lock:
        state.results = results
        state.file_count = file_count
    try:
        summary_path = write_product_pattern_summary(results)
        record_activity_event(
            "log_summary",
            "completed",
            f"상품 패턴 요약 JSON 갱신 완료: {summary_path.name}",
        )
    except Exception as error:
        record_activity_event(
            "log_summary",
            "failed",
            f"상품 패턴 요약 JSON 갱신 실패: {error}",
        )
    record_activity_event(
        "log_analyzer",
        "completed",
        f"로그 파일 {file_count}개, 분석 결과 {len(results)}건",
    )
    build_chart_payloads(results)
    return results, file_count


def build_faiss_bundle(
    logs: list[dict[str, Any]] | None = None,
    news: list[dict[str, Any]] | None = None,
    source: str = "faiss_builder",
    stores: set[str] | None = None,
    force_log_rebuild: bool = True,
) -> int:
    with state.lock:
        effective_logs = list(logs) if logs is not None else list(state.results)
        effective_news = list(news) if news is not None else list(state.news)

    requested_stores = set(stores or {
        FAISS_STORE_LOGS,
        FAISS_STORE_NEWS,
        FAISS_STORE_CUSTOMER,
        FAISS_STORE_DOCUMENT,
    })

    if logs is None and not effective_logs:
        raw_text, file_count = load_all_logs()
        if raw_text.strip():
            effective_logs = analyze_logs(raw_text)
            with state.lock:
                state.results = effective_logs
                state.file_count = file_count

    try:
        before_count = get_vector_count()
    except Exception:
        before_count = 0
    record_activity_event(
        source,
        "running",
        f"벡터 DB를 갱신 중입니다. 스토어 {', '.join(sorted(requested_stores))} | 로그 {len(effective_logs)}건, 뉴스 {len(effective_news)}건",
    )
    appended_only = True

    if FAISS_STORE_LOGS in requested_stores and not force_log_rebuild:
        with state.lock:
            last_vectorized_signature = tuple(state.last_log_vectorized_signature or ())
            current_static_signature = tuple(state.static_log_signature or ())
        existing_structured_logs = get_store_document_count(FAISS_STORE_LOGS, {"log"})
        current_structured_logs = len(effective_logs)
        signature_changed = (
            bool(last_vectorized_signature)
            and current_static_signature
            and current_static_signature != last_vectorized_signature
        )
        if current_structured_logs >= existing_structured_logs and not (
            signature_changed and current_structured_logs <= existing_structured_logs
        ):
            new_logs = effective_logs[existing_structured_logs:]
            if new_logs:
                log_before_count = get_vector_count(FAISS_STORE_LOGS)
                appended_count, log_after_count = append_structured_log_documents(new_logs)
                record_vector_event(
                    source,
                    "append",
                    log_before_count,
                    log_after_count,
                    f"심사 로그 {appended_count}건 증분 적재",
                )
            with state.lock:
                state.last_log_vectorized_count = current_structured_logs
                state.last_log_vectorized_signature = current_static_signature
            requested_stores.discard(FAISS_STORE_LOGS)
        else:
            requested_stores.discard(FAISS_STORE_LOGS)
            record_activity_event(
                source,
                "running",
                "로그 벡터 스토어는 add-only 모드라 감소/재정렬 또는 동일 건수 수정 이력은 증분 적재에서 제외했습니다.",
            )

    if FAISS_STORE_NEWS in requested_stores:
        existing_news_keys = get_store_news_keys(FAISS_STORE_NEWS)
        new_news = [
            item for item in effective_news if _news_item_key(item) not in existing_news_keys
        ]
        if new_news:
            news_before_count = get_vector_count(FAISS_STORE_NEWS)
            appended_count, news_after_count = append_news_documents(new_news)
            record_vector_event(
                source,
                "append",
                news_before_count,
                news_after_count,
                f"뉴스 문서 {appended_count}건 증분 적재",
            )
        requested_stores.discard(FAISS_STORE_NEWS)

    ingest_preview = prepare_log_ingest_preview(effective_logs)
    with state.lock:
        state.latest_log_prompt_input = ingest_preview
        state.last_log_prompt_input_time = datetime.datetime.now()
        state.last_log_vectorized_count = len(effective_logs)
        state.last_log_vectorized_signature = tuple(state.static_log_signature or ())

    if requested_stores:
        appended_only = False
        build_vector_db(effective_logs, effective_news, rebuild_stores=requested_stores)
    count = get_vector_count()
    record_vector_event(
        source,
        "append" if appended_only else "rebuild",
        before_count,
        count,
        (
            f"스토어 {', '.join(sorted(stores or requested_stores or []))} 기준으로 증분 동기화"
            if appended_only
            else f"스토어 {', '.join(sorted(stores or requested_stores or []))} 기준으로 갱신"
        ),
    )
    record_activity_event(source, "completed", f"벡터 DB 갱신 완료. 총 벡터 {count}건")
    build_chart_payloads(effective_logs)
    return count


def append_log_vectors_bundle(
    logs: list[dict[str, Any]] | None = None,
    source: str = "log_vector_append",
) -> int:
    with state.lock:
        effective_logs = list(logs) if logs is not None else list(state.results)
        previous_count = int(state.last_log_vectorized_count or 0)
        previous_signature = tuple(state.last_log_vectorized_signature or ())
        current_signature = tuple(state.static_log_signature or ())

    current_count = len(effective_logs)
    if current_count <= 0:
        return 0

    try:
        existing_structured_count = get_store_document_count(FAISS_STORE_LOGS, {"log"})
    except Exception:
        existing_structured_count = 0

    previous_count = max(previous_count, existing_structured_count)

    if current_count < previous_count or (
        previous_signature and current_signature and previous_signature != current_signature and current_count <= previous_count
    ):
        record_activity_event(
            source,
            "running",
            "로그 파일 구조가 append-only 조건을 벗어나 증분 적재를 건너뜁니다.",
            update_status=True,
        )
        return 0

    new_logs = effective_logs[previous_count:]
    if not new_logs:
        return 0

    before_count = get_vector_count(FAISS_STORE_LOGS)
    appended_count, after_count = append_structured_log_documents(new_logs)
    generated_at = datetime.datetime.now()
    ingest_preview = prepare_log_ingest_preview(new_logs)
    with state.lock:
        state.last_log_vectorized_count = current_count
        state.last_log_vectorized_signature = current_signature
        state.latest_log_prompt_input = ingest_preview
        state.last_log_prompt_input_time = generated_at

    record_vector_event(
        source,
        "append",
        before_count,
        after_count,
        f"신규 구조화 심사로그 {appended_count}건 증분 적재",
    )
    record_activity_event(
        source,
        "completed",
        f"신규 구조화 심사로그 {appended_count}건 증분 적재 완료",
        update_status=True,
    )
    build_chart_payloads(effective_logs)
    return appended_count


def append_news_vectors_bundle(
    news_items: list[dict[str, Any]] | None = None,
    source: str = "news_vector_append",
) -> int:
    with state.lock:
        effective_news = list(news_items) if news_items is not None else list(state.news)

    if not effective_news:
        return 0

    existing_news_keys = get_store_news_keys(FAISS_STORE_NEWS)
    new_news = [
        item for item in effective_news if _news_item_key(item) not in existing_news_keys
    ]
    if not new_news:
        return 0

    before_count = get_vector_count(FAISS_STORE_NEWS)
    appended_count, after_count = append_news_documents(new_news)

    record_vector_event(
        source,
        "append",
        before_count,
        after_count,
        f"신규 뉴스 {appended_count}건 증분 적재",
    )
    record_activity_event(
        source,
        "completed",
        f"신규 뉴스 {appended_count}건 증분 적재 완료",
        update_status=True,
    )
    build_chart_payloads()
    return appended_count


def run_full_analysis(
    log_dir: str = "data/logs",
    collect_news: bool = True,
) -> dict[str, Any]:
    # 사용자가 "전체 분석 실행"을 눌렀을 때 호출되는 핵심 파이프라인입니다.
    # 로그 분석과 뉴스 수집은 서로 독립적으로 시작하고,
    # 두 결과가 모두 준비되면 FAISS/상태 스냅샷을 갱신합니다.
    with state.lock:
        state.running = True
    record_activity_event("system", "running", "전체 분석 파이프라인을 시작했습니다.")
    record_activity_event(
        "startup_sequence",
        "running",
        "초기 전체 분석 시작: 로그 분석과 뉴스 수집을 병렬로 시작합니다.",
        update_status=True,
    )
    start = time.time()
    warmup_result: dict[str, int] = {}
    log_results: tuple[list[dict[str, Any]], int] = ([], 0)
    with state.lock:
        existing_news = list(state.news)
        existing_issues = list(state.issues)
    news_results: tuple[list[dict[str, Any]], list[str]] = (existing_news, existing_issues)
    pipeline_errors: list[BaseException] = []

    def _warm_embeddings_in_background() -> None:
        nonlocal warmup_result
        try:
            warmup_result = warmup_embeddings()
        except Exception:
            warmup_result = {}

    def _run_log_analysis() -> None:
        nonlocal log_results
        try:
            log_results = analyze_logs_bundle(log_dir=log_dir)
        except BaseException as error:
            pipeline_errors.append(error)

    def _run_news_collection() -> None:
        nonlocal news_results
        try:
            news_results = collect_news_bundle(accumulate=True)
        except BaseException as error:
            pipeline_errors.append(error)

    warmup_thread = threading.Thread(target=_warm_embeddings_in_background, daemon=True)
    log_thread = threading.Thread(target=_run_log_analysis, daemon=True)
    news_thread = threading.Thread(target=_run_news_collection, daemon=True) if collect_news else None
    warmup_thread.start()
    log_thread.start()
    if news_thread is not None:
        news_thread.start()
    else:
        record_activity_event(
            "startup_sequence",
            "running",
            "재기동 fast bootstrap에서는 뉴스 수집을 생략하고 worker 주기로만 재개합니다.",
            update_status=True,
        )
    try:
        log_thread.join()
        if news_thread is not None:
            news_thread.join()
        if pipeline_errors:
            raise pipeline_errors[0]

        results, file_count = log_results
        news, issues = news_results
        warmup_thread.join(timeout=1)
        record_activity_event(
            "startup_sequence",
            "completed",
            "초기 전체 분석의 핵심 구간이 종료되었습니다.",
            update_status=True,
        )
        try:
            build_faiss_bundle(
                results,
                news,
                source="full_analysis",
                stores={FAISS_STORE_LOGS, FAISS_STORE_NEWS},
                force_log_rebuild=False,
            )
        except Exception:
            pass
        with state.lock:
            state.file_count = file_count
            state.total_time = time.time() - start
            state.last_run_time = datetime.datetime.now()
            state.running = False
        record_activity_event(
            "system", "completed", f"전체 분석 완료. 소요 {state.total_time:.1f}초 (임베딩 워밍업 {warmup_result.get('total_ms', 0)}ms)"
        )
        snapshot = state.snapshot()
        snapshot["results"] = enrich_results(results)
        snapshot["issues"] = issues
        snapshot["news"] = safe_serialize(news)
        snapshot["chart_payloads"] = build_chart_payloads(results)
        return snapshot
    except Exception:
        with state.lock:
            state.running = False
        record_activity_event(
            "system", "failed", "전체 분석 파이프라인이 실패했습니다."
        )
        raise


def ask_strategy(
    question: str,
    news_prompt_template: str | None = None,
    log_prompt_template: str | None = None,
) -> dict[str, Any]:
    reset_strategy_runtime(question)

    def on_agent_event(agent: str, status: str, detail: str) -> None:
        record_activity_event(agent, status, detail, update_status=True)

    def on_vector_event(
        source: str, action: str, before_count: int, after_count: int, detail: str
    ) -> None:
        record_vector_event(source, action, before_count, after_count, detail)

    def on_prompt_input(agent: str, prompt_input: dict[str, Any]) -> None:
        record_prompt_input(agent, prompt_input)

    def on_ollama_progress(agent: str, event: str, payload: dict[str, Any]) -> None:
        record_ollama_progress(agent, event, payload)

    try:
        with state.lock:
            effective_news_prompt_template = (
                news_prompt_template or state.news_prompt_template_override
            )
            effective_log_prompt_template = (
                log_prompt_template or state.log_prompt_template_override
            )
        result = strategy_chat(
            question,
            event_callback=on_agent_event,
            vector_callback=on_vector_event,
            prompt_input_callback=on_prompt_input,
            ollama_progress_callback=on_ollama_progress,
            news_prompt_template=effective_news_prompt_template,
            log_prompt_template=effective_log_prompt_template,
        )
        record_activity_event(
            "orchestrator",
            "completed",
            "멀티 에이전트 보고서 생성이 완료되었습니다.",
            update_status=True,
        )
        with state.lock:
            state.last_strategy_time = datetime.datetime.now()
            state.latest_log_prompt_input = result.get("prompt_inputs", {}).get(
                "log_agent"
            )
            state.last_log_prompt_input_time = datetime.datetime.now()
            state.latest_news_prompt_input = result.get("prompt_inputs", {}).get(
                "news_agent"
            )
            state.last_news_prompt_input_time = datetime.datetime.now()
        return result
    except Exception:
        record_activity_event(
            "orchestrator",
            "failed",
            "멀티 에이전트 실행 중 오류가 발생했습니다.",
            update_status=True,
        )
        raise


def run_background_news_agent_cycle(should_persist: bool = True) -> dict[str, Any]:
    with state.lock:
        effective_news = list(state.news)
        last_news_crawl_error = state.last_news_crawl_error
        news_prompt_template_override = state.news_prompt_template_override
    if not any(str(item.get("content", "")).strip() for item in effective_news):
        detail = "크롤링된 뉴스 본문이 없어 뉴스 브리핑을 생성하지 못했습니다."
        if last_news_crawl_error:
            detail = f"뉴스 브리핑 대기 중: 본문 확보 실패 ({last_news_crawl_error})"
        record_activity_event("news_agent", "failed", detail, update_status=True)
        return {
            "analysis": state.latest_news_briefing,
            "prompt_input": state.latest_news_prompt_input,
            "skipped": True,
            "reason": "no_crawled_news_content",
        }

    def on_agent_event(agent: str, status: str, detail: str) -> None:
        record_activity_event(agent, status, detail, update_status=True)

    def on_vector_event(
        source: str, action: str, before_count: int, after_count: int, detail: str
    ) -> None:
        record_vector_event(source, action, before_count, after_count, detail)

    def on_prompt_input(agent: str, prompt_input: dict[str, Any]) -> None:
        record_prompt_input(agent, prompt_input)

    def on_ollama_progress(agent: str, event: str, payload: dict[str, Any]) -> None:
        record_ollama_progress(agent, event, payload)

    result = run_periodic_news_agent(
        effective_news,
        should_persist=should_persist,
        event_callback=on_agent_event,
        vector_callback=on_vector_event,
        prompt_input_callback=on_prompt_input,
        ollama_progress_callback=on_ollama_progress,
        news_prompt_template=news_prompt_template_override,
    )
    with state.lock:
        state.latest_news_briefing = result.get("analysis")
        state.last_news_briefing_time = datetime.datetime.now()
        state.latest_news_prompt_input = result.get("prompt_input")
        state.last_news_prompt_input_time = datetime.datetime.now()
    build_chart_payloads()
    return result


def generate_test_log_cycle(record_event: bool = True) -> dict[str, Any]:
    global _test_log_product_index

    product = TEST_LOG_PRODUCT_SEQUENCE[
        _test_log_product_index % len(TEST_LOG_PRODUCT_SEQUENCE)
    ]
    _test_log_product_index += 1

    forced_decision = "승인" if product in {"C11", "C12"} else None
    generated = append_synthetic_log(product, force_decision=forced_decision)
    generated_at = datetime.datetime.now()
    with state.lock:
        state.last_log_ingest_time = generated_at
    if record_event:
        record_activity_event(
            "log_ingestor",
            "completed",
            f"테스트 로그 생성 완료. 상품 {generated.get('product', product)} | 파일 {generated.get('file_path', '')}",
            update_status=True,
        )
    return generated


def generate_test_log_burst(count: int = TEST_LOG_BURST_COUNT) -> dict[str, Any]:
    burst_count = max(1, int(count or 1))
    generated_items = [generate_test_log_cycle(record_event=False) for _ in range(burst_count)]

    product_counts: dict[str, int] = {}
    for item in generated_items:
        product = str(item.get("product") or "UNKNOWN").strip().upper() or "UNKNOWN"
        product_counts[product] = product_counts.get(product, 0) + 1

    product_summary = ", ".join(
        f"{product} {count}건" for product, count in sorted(product_counts.items())
    )
    generated_at = datetime.datetime.now()
    with state.lock:
        state.last_log_ingest_time = generated_at

    record_activity_event(
        "log_ingestor",
        "completed",
        f"테스트 로그 {burst_count}건 생성 완료. 상품 분포 {product_summary} | 파일 {generated_items[-1].get('file_path', '')}",
        update_status=True,
    )
    return {
        "count": burst_count,
        "products": product_counts,
        "product_summary": product_summary,
        "generated_at": generated_at.isoformat(),
        "file_path": generated_items[-1].get("file_path", ""),
        "items": generated_items,
    }


def run_background_log_agent_cycle(should_persist: bool = True) -> dict[str, Any]:
    with state.lock:
        effective_results = list(state.results)
        log_prompt_template_override = state.log_prompt_template_override

    def on_agent_event(agent: str, status: str, detail: str) -> None:
        record_activity_event(agent, status, detail, update_status=True)

    def on_vector_event(
        source: str, action: str, before_count: int, after_count: int, detail: str
    ) -> None:
        record_vector_event(source, action, before_count, after_count, detail)

    def on_prompt_input(agent: str, prompt_input: dict[str, Any]) -> None:
        record_prompt_input(agent, prompt_input)

    def on_ollama_progress(agent: str, event: str, payload: dict[str, Any]) -> None:
        record_ollama_progress(agent, event, payload)

    result = run_periodic_log_agent(
        effective_results,
        should_persist=should_persist,
        event_callback=on_agent_event,
        vector_callback=on_vector_event,
        prompt_input_callback=on_prompt_input,
        ollama_progress_callback=on_ollama_progress,
        log_prompt_template=log_prompt_template_override,
    )
    with state.lock:
        state.latest_log_briefing = result.get("analysis")
        state.last_log_briefing_time = datetime.datetime.now()
        state.latest_log_prompt_input = result.get("prompt_input")
        state.last_log_prompt_input_time = datetime.datetime.now()
    build_chart_payloads()
    return result


def search_faiss(query: str, k: int = 5, store_name: str | None = None) -> dict[str, list[str]]:
    normalized_store = str(store_name or "").strip().lower() or None
    if normalized_store == FAISS_STORE_LOGS:
        logs, _, _ = search_context(query, k=k)
        return {"logs": logs, "news": [], "rules": [], "customer": []}
    if normalized_store == FAISS_STORE_NEWS:
        news, rules = search_news_context(query, k=k)
        return {"logs": [], "news": news, "rules": rules, "customer": []}
    if normalized_store == FAISS_STORE_DOCUMENT:
        _, rules = search_news_context(query, k=k)
        return {"logs": [], "news": [], "rules": rules, "customer": []}
    if normalized_store == FAISS_STORE_CUSTOMER:
        customer = search_customer_context(query, k=k)
        return {"logs": [], "news": [], "rules": [], "customer": customer}

    logs, news, rules = search_context(query, k=k)
    return {"logs": logs, "news": news, "rules": rules, "customer": []}


def get_chart_payloads() -> dict[str, Any]:
    with state.lock:
        has_payloads = bool(state.chart_payloads)
    if not has_payloads:
        return build_chart_payloads()
    with state.lock:
        return state.chart_payloads
