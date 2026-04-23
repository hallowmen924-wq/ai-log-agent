from __future__ import annotations

import datetime
import os
import pathlib
import threading
import time
from typing import Any

from agent.log_generator import append_synthetic_log
from agent.news_agent import analyze_news, collect_news
from agent.strategy_chat import (
    run_periodic_log_agent,
    run_periodic_news_agent,
    strategy_chat,
)
from analyzer.log_analyzer import analyze_logs
from analyzer.risk_analyzer import calculate_risk
from rag.vector_db import build_vector_db, get_vector_count, search_context

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
        self.latest_log_briefing: str | None = None
        self.last_log_briefing_time: datetime.datetime | None = None
        self.latest_log_prompt_input: dict[str, Any] | None = None
        self.last_log_prompt_input_time: datetime.datetime | None = None
        self.latest_news_briefing: str | None = None
        self.last_news_briefing_time: datetime.datetime | None = None
        self.latest_news_prompt_input: dict[str, Any] | None = None
        self.last_news_prompt_input_time: datetime.datetime | None = None
        self.agent_statuses: dict[str, dict[str, Any]] = {}
        self.agent_activity_log: list[dict[str, Any]] = []
        self.vector_events: list[dict[str, Any]] = []
        self.chart_payloads: dict[str, Any] = {}
        self.last_chart_time: datetime.datetime | None = None
        self.full_faiss_items: list[dict[str, Any]] = []
        self.news_crawl_running = False
        self.news_crawl_target_count = 0
        self.news_crawl_success_count = 0
        self.news_crawl_failure_count = 0
        self.last_news_crawl_time: datetime.datetime | None = None
        self.last_news_crawl_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        try:
            vector_count = get_vector_count()
        except Exception:
            vector_count = 0
        cached_faiss_items: list[dict[str, Any]] = []
        with self.lock:
            cached_faiss_items = safe_serialize(self.full_faiss_items)
        if vector_count > 0 and not cached_faiss_items:
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
                "agent_statuses": safe_serialize(self.agent_statuses),
                "agent_activity_log": safe_serialize(self.agent_activity_log),
                "vector_events": safe_serialize(self.vector_events),
                "last_chart_time": (
                    self.last_chart_time.isoformat() if self.last_chart_time else None
                ),
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


def _push_front(
    items: list[dict[str, Any]], item: dict[str, Any], limit: int = 30
) -> list[dict[str, Any]]:
    return ([item] + items)[:limit]


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


def resolve_project_path(path_str: str) -> str:
    # backend 폴더에서 서버를 띄워도 data/logs 같은 상대경로를 프로젝트 루트 기준으로 맞춰줍니다.
    candidate = pathlib.Path(path_str)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def load_all_logs(log_dir: str = "data/logs") -> tuple[str, int]:
    log_dir = resolve_project_path(log_dir)
    logs = ""
    count = 0
    if not os.path.exists(log_dir):
        return logs, count
    for name in os.listdir(log_dir):
        if name.endswith(".txt") or name.endswith(".log"):
            file_path = os.path.join(log_dir, name)
            try:
                with open(file_path, encoding="utf-8") as file:
                    logs += file.read()
                    count += 1
            except Exception:
                continue
    return logs, count


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
) -> tuple[list[dict[str, Any]], int]:
    # 같은 제목/링크 조합은 중복으로 보고 하나만 유지합니다.
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    existing_keys = {
        (str(item.get("title", "")).strip(), str(item.get("link", "")).strip())
        for item in existing_news
    }
    new_unique_count = 0

    for item in list(new_news) + list(existing_news):
        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()
        key = (title, link)
        if key in seen:
            continue
        seen.add(key)
        if key not in existing_keys:
            new_unique_count += 1
        merged.append(item)
        if len(merged) >= max_items:
            break

    return merged, new_unique_count


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
) -> tuple[list[dict[str, Any]], list[str]]:
    record_activity_event("news_collector", "running", "뉴스 RSS를 수집하고 있습니다.")
    news = collect_news()
    with state.lock:
        existing_news = list(state.news)

    new_unique_count = 0
    if accumulate:
        effective_news, new_unique_count = merge_news_items(existing_news, news)
    else:
        effective_news = news
        new_unique_count = len(news)

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
        def _bg_fetch_and_index(news_snapshot: list[dict[str, Any]]):
            crawl_targets = [item for item in news_snapshot if not item.get("content")]
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
                            with state.lock:
                                for s in state.news:
                                    if s.get("link") == item.get("link") and not s.get("content"):
                                        s["content"] = txt
                                        state.news_crawl_success_count += 1
                                        state.last_news_crawl_time = datetime.datetime.now()
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
                    with state.lock:
                        has_crawled_news = any(
                            str(news_item.get("content", "")).strip()
                            for news_item in state.news
                        )
                    if has_crawled_news:
                        run_background_news_agent_cycle(should_persist=True)
                    else:
                        record_activity_event(
                            "news_agent",
                            "failed",
                            "크롤링된 뉴스 본문이 없어 뉴스 브리핑을 생성하지 못했습니다.",
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

                # after fetching all contents, rebuild FAISS using current state.news
                try:
                    count = build_faiss_bundle(logs=None, news=None, source="news_crawler")
                    record_activity_event(
                        "news_crawler",
                        "completed",
                        f"뉴스 크롤링 및 FAISS 적재 완료. 총 벡터 {count}건",
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
        t = threading.Thread(target=_bg_fetch_and_index, args=(news_snapshot,), daemon=True)
        t.start()

    return effective_news, issues


def analyze_logs_bundle(
    raw_logs: str | None = None, log_dir: str = "data/logs"
) -> tuple[list[dict[str, Any]], int]:
    record_activity_event("log_analyzer", "running", "로그 분석을 시작했습니다.")
    raw_text, file_count = (
        (raw_logs, 0) if raw_logs is not None else load_all_logs(log_dir)
    )
    results = analyze_logs(raw_text or "")
    with state.lock:
        state.results = results
        state.file_count = file_count
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
) -> int:
    with state.lock:
        effective_logs = list(logs) if logs is not None else list(state.results)
        effective_news = list(news) if news is not None else list(state.news)

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
        f"벡터 DB를 갱신 중입니다. 로그 {len(effective_logs)}건, 뉴스 {len(effective_news)}건",
    )
    build_vector_db(effective_logs, effective_news)
    count = get_vector_count()
    record_vector_event(
        source,
        "rebuild",
        before_count,
        count,
        f"로그 {len(effective_logs)}건과 뉴스 {len(effective_news)}건으로 재구성",
    )
    record_activity_event(source, "completed", f"벡터 DB 갱신 완료. 총 벡터 {count}건")
    build_chart_payloads(effective_logs)
    return count


def run_full_analysis(log_dir: str = "data/logs") -> dict[str, Any]:
    # 사용자가 "전체 분석 실행"을 눌렀을 때 호출되는 핵심 파이프라인입니다.
    # 순서: 로그 분석 -> 뉴스 수집 -> FAISS 생성 -> 상태/차트 스냅샷 갱신
    with state.lock:
        state.running = True
    record_activity_event("system", "running", "전체 분석 파이프라인을 시작했습니다.")
    start = time.time()
    try:
        results, file_count = analyze_logs_bundle(log_dir=log_dir)
        news, issues = collect_news_bundle(accumulate=True)
        try:
            build_faiss_bundle(results, news, source="full_analysis")
        except Exception:
            pass
        with state.lock:
            state.file_count = file_count
            state.total_time = time.time() - start
            state.last_run_time = datetime.datetime.now()
            state.running = False
        record_activity_event(
            "system", "completed", f"전체 분석 완료. 소요 {state.total_time:.1f}초"
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


def ask_strategy(question: str) -> dict[str, Any]:
    reset_strategy_runtime(question)

    def on_agent_event(agent: str, status: str, detail: str) -> None:
        record_activity_event(agent, status, detail, update_status=True)

    def on_vector_event(
        source: str, action: str, before_count: int, after_count: int, detail: str
    ) -> None:
        record_vector_event(source, action, before_count, after_count, detail)

    try:
        result = strategy_chat(
            question, event_callback=on_agent_event, vector_callback=on_vector_event
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

    result = run_periodic_news_agent(
        effective_news,
        should_persist=should_persist,
        event_callback=on_agent_event,
        vector_callback=on_vector_event,
    )
    with state.lock:
        state.latest_news_briefing = result.get("analysis")
        state.last_news_briefing_time = datetime.datetime.now()
        state.latest_news_prompt_input = result.get("prompt_input")
        state.last_news_prompt_input_time = datetime.datetime.now()
    build_chart_payloads()
    return result


def generate_test_log_cycle() -> dict[str, Any]:
    generated = append_synthetic_log()
    generated_at = datetime.datetime.now()
    with state.lock:
        state.last_log_ingest_time = generated_at
    record_activity_event(
        "log_ingestor",
        "completed",
        f"테스트 로그 생성 완료. 상품 {generated.get('product', 'N/A')} | 파일 {generated.get('file_path', '')}",
        update_status=True,
    )
    return generated


def run_background_log_agent_cycle(should_persist: bool = True) -> dict[str, Any]:
    with state.lock:
        effective_results = list(state.results)

    def on_agent_event(agent: str, status: str, detail: str) -> None:
        record_activity_event(agent, status, detail, update_status=True)

    def on_vector_event(
        source: str, action: str, before_count: int, after_count: int, detail: str
    ) -> None:
        record_vector_event(source, action, before_count, after_count, detail)

    result = run_periodic_log_agent(
        effective_results,
        should_persist=should_persist,
        event_callback=on_agent_event,
        vector_callback=on_vector_event,
    )
    with state.lock:
        state.latest_log_briefing = result.get("analysis")
        state.last_log_briefing_time = datetime.datetime.now()
        state.latest_log_prompt_input = result.get("prompt_input")
        state.last_log_prompt_input_time = datetime.datetime.now()
    build_chart_payloads()
    return result


def search_faiss(query: str, k: int = 5) -> dict[str, list[str]]:
    logs, news, rules = search_context(query, k=k)
    return {"logs": logs, "news": news, "rules": rules}


def get_chart_payloads() -> dict[str, Any]:
    with state.lock:
        has_payloads = bool(state.chart_payloads)
    if not has_payloads:
        return build_chart_payloads()
    with state.lock:
        return state.chart_payloads
