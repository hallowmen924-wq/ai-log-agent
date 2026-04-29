from __future__ import annotations

import time
import threading

from backend.services import (
    FAISS_STORE_NEWS,
    analyze_logs_bundle,
    append_log_vectors_bundle,
    append_news_vectors_bundle,
    build_faiss_bundle,
    collect_news_bundle,
    generate_test_log_burst,
    record_activity_event,
    run_background_log_agent_cycle,
    state,
    update_worker_runtime_stats,
)

# 이 워커는 백그라운드에서 주기적으로 뉴스를 새로 받고,
# 이미 분석된 로그가 있으면 FAISS 벡터 DB도 다시 빌드합니다.
# 즉, 메인 화면을 다시 열지 않아도 데이터가 조금씩 최신 상태로 갱신됩니다.


class NewsVectorWorker:
    def __init__(self, interval_seconds: int = 10) -> None:
        self.interval_seconds = max(1, interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_trace_pending = True
        self._last_log_cycle_at = 0.0
        self._last_log_analysis_at = 0.0
        self._last_log_agent_cycle_at = 0.0
        self._last_news_cycle_at = 0.0
        self._last_faiss_cycle_at = 0.0
        self._recalculate_task_intervals()

    def _recalculate_task_intervals(self) -> None:
        base = max(1, self.interval_seconds)
        self.log_cycle_seconds = 1
        self.log_burst_count = 4
        self.log_analysis_seconds = 1
        self.log_agent_seconds = max(3, base * 3)
        self.news_cycle_seconds = 10
        self.faiss_cycle_seconds = max(180, base * 60)
        self.faiss_log_rebuild_threshold = max(24, self.log_burst_count * 8)

    def _should_run(self, last_run_at: float, cadence_seconds: int, now: float) -> bool:
        return last_run_at <= 0 or (now - last_run_at) >= cadence_seconds

    def _run_log_cycle(self) -> dict[str, object]:
        try:
            return generate_test_log_burst(self.log_burst_count)
        except Exception:
            return {}

    def _run_log_analysis_cycle(self) -> bool:
        try:
            analyze_logs_bundle(log_dir="data/logs")
            return True
        except Exception:
            return False

    def _run_log_vector_append_cycle(self) -> int:
        try:
            return append_log_vectors_bundle(source="worker_log_append")
        except Exception:
            return 0

    def _run_log_agent_cycle(self) -> bool:
        try:
            run_background_log_agent_cycle(should_persist=True)
            return True
        except Exception:
            return False

    def _run_news_cycle(self) -> tuple[bool, bool]:
        try:
            news, _ = collect_news_bundle(accumulate=True, max_new_items_to_add=1)
        except Exception:
            return False, False

        with state.lock:
            has_new_items = bool(
                state.last_new_item_time
                and state.last_news_time
                and state.last_new_item_time == state.last_news_time
            )
            has_crawled_news = any(
                str(item.get("content", "")).strip() for item in state.news
            )
        return has_new_items, bool(news and has_crawled_news)

    def _run_faiss_cycle(self) -> bool:
        try:
            append_news_vectors_bundle(source="worker")
            return True
        except Exception:
            return False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        # 이미 실행 중이면 중복 스레드를 만들지 않습니다.
        if self.running:
            return False
        self._stop_event.clear()
        self._startup_trace_pending = True
        record_activity_event(
            "startup_sequence",
            "running",
            f"worker 시작 요청 수신. base interval={self.interval_seconds}s",
            update_status=True,
        )
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> bool:
        if not self.running:
            return False
        self._stop_event.set()
        return True

    def update_interval(self, interval_seconds: int) -> None:
        self.interval_seconds = max(1, interval_seconds)
        self._recalculate_task_intervals()

    def _run_loop(self) -> None:
        # stop 요청이 들어오기 전까지 interval_seconds 간격으로 반복 실행됩니다.
        while not self._stop_event.is_set():
            loop_started_at = time.monotonic()
            startup_trace_active = self._startup_trace_pending
            runtime_stats = {
                "last_loop_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_interval_seconds": self.interval_seconds,
                "log_cycle_seconds": self.log_cycle_seconds,
                "log_burst_count": self.log_burst_count,
                "log_analysis_seconds": self.log_analysis_seconds,
                "log_agent_seconds": self.log_agent_seconds,
                "news_cycle_seconds": self.news_cycle_seconds,
                "faiss_cycle_seconds": self.faiss_cycle_seconds,
                "log_cycle_ran": False,
                "log_generated_count": 0,
                "log_generated_products": "-",
                "log_analysis_ran": False,
                "log_agent_ran": False,
                "log_vector_append_ran": False,
                "log_vector_appended_count": 0,
                "news_cycle_ran": False,
                "faiss_cycle_ran": False,
                "log_cycle_elapsed_ms": 0,
                "log_analysis_elapsed_ms": 0,
                "log_vector_append_elapsed_ms": 0,
                "log_agent_elapsed_ms": 0,
                "news_cycle_elapsed_ms": 0,
                "faiss_cycle_elapsed_ms": 0,
                "last_loop_elapsed_ms": 0,
                "faiss_log_rebuild_threshold": self.faiss_log_rebuild_threshold,
                "faiss_rebuild_reason": "뉴스/로그 신규 유입분은 FAISS add_documents로 증분 반영",
            }
            try:
                now = time.monotonic()
                log_cycle_ran = False
                log_analysis_ran = False
                has_new_items = False
                has_crawled_news = False

                if startup_trace_active:
                    record_activity_event(
                        "startup_sequence",
                        "running",
                        "첫 worker 루프 시작. 뉴스 주기 점검을 먼저 실행합니다.",
                        update_status=True,
                    )

                if self._should_run(self._last_news_cycle_at, self.news_cycle_seconds, now):
                    if startup_trace_active:
                        record_activity_event(
                            "startup_sequence",
                            "running",
                            "초기 뉴스 수집/브리핑 경로를 먼저 평가합니다.",
                            update_status=True,
                        )
                    phase_started_at = time.monotonic()
                    has_new_items, has_crawled_news = self._run_news_cycle()
                    runtime_stats["news_cycle_ran"] = True
                    runtime_stats["news_cycle_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )
                    runtime_stats["news_cycle_has_new_items"] = has_new_items
                    runtime_stats["news_cycle_has_crawled_news"] = has_crawled_news
                    self._last_news_cycle_at = now
                else:
                    with state.lock:
                        has_crawled_news = any(
                            str(item.get("content", "")).strip()
                            for item in state.news
                        )

                if self._should_run(self._last_log_cycle_at, self.log_cycle_seconds, now):
                    if startup_trace_active:
                        record_activity_event(
                            "startup_sequence",
                            "running",
                            "이어서 로그 생성/분석 경로를 평가합니다.",
                            update_status=True,
                        )
                    phase_started_at = time.monotonic()
                    log_cycle_result = self._run_log_cycle()
                    log_cycle_ran = bool(log_cycle_result)
                    runtime_stats["log_cycle_ran"] = log_cycle_ran
                    runtime_stats["log_generated_count"] = int(log_cycle_result.get("count", 0) or 0)
                    runtime_stats["log_generated_products"] = str(log_cycle_result.get("product_summary") or "-")
                    runtime_stats["log_cycle_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )
                    if log_cycle_ran:
                        self._last_log_cycle_at = now

                if log_cycle_ran and self._should_run(self._last_log_analysis_at, self.log_analysis_seconds, now):
                    phase_started_at = time.monotonic()
                    log_analysis_ran = self._run_log_analysis_cycle()
                    runtime_stats["log_analysis_ran"] = log_analysis_ran
                    runtime_stats["log_analysis_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )
                    if log_analysis_ran:
                        self._last_log_analysis_at = now

                if log_analysis_ran:
                    phase_started_at = time.monotonic()
                    appended_count = self._run_log_vector_append_cycle()
                    runtime_stats["log_vector_append_ran"] = appended_count > 0
                    runtime_stats["log_vector_appended_count"] = int(appended_count or 0)
                    runtime_stats["log_vector_append_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )

                if log_analysis_ran and self._should_run(self._last_log_agent_cycle_at, self.log_agent_seconds, now):
                    phase_started_at = time.monotonic()
                    runtime_stats["log_agent_ran"] = self._run_log_agent_cycle()
                    runtime_stats["log_agent_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )
                    if runtime_stats["log_agent_ran"]:
                        self._last_log_agent_cycle_at = now

                with state.lock:
                    has_results = bool(state.results)

                appended_count = int(runtime_stats.get("log_vector_appended_count", 0) or 0)
                enough_new_logs_for_rebuild = appended_count >= self.faiss_log_rebuild_threshold
                faiss_input_changed = has_new_items or enough_new_logs_for_rebuild
                runtime_stats["faiss_input_changed"] = faiss_input_changed
                runtime_stats["faiss_rebuild_due_to_news"] = has_new_items
                runtime_stats["faiss_rebuild_due_to_logs"] = enough_new_logs_for_rebuild
                should_rebuild_faiss = (
                    has_results
                    and faiss_input_changed
                    and (has_new_items or has_crawled_news)
                )
                runtime_stats["faiss_rebuild_ready"] = should_rebuild_faiss
                if (
                    should_rebuild_faiss
                    and self._should_run(self._last_faiss_cycle_at, self.faiss_cycle_seconds, now)
                ):
                    phase_started_at = time.monotonic()
                    if self._run_faiss_cycle():
                        runtime_stats["faiss_cycle_ran"] = True
                        self._last_faiss_cycle_at = now
                    runtime_stats["faiss_cycle_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )
                if startup_trace_active:
                    record_activity_event(
                        "startup_sequence",
                        "completed",
                        "초기 worker 시퀀스 기록 완료. 이후부터는 주기 실행만 반복합니다.",
                        update_status=True,
                    )
                    self._startup_trace_pending = False
            except Exception:
                pass
            runtime_stats["last_loop_elapsed_ms"] = int(
                (time.monotonic() - loop_started_at) * 1000
            )
            update_worker_runtime_stats(runtime_stats)
            self._stop_event.wait(self.interval_seconds)


worker = NewsVectorWorker(interval_seconds=1)
