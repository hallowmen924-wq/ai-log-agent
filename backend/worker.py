from __future__ import annotations

import time
import threading

from backend.services import (
    analyze_logs_bundle,
    build_faiss_bundle,
    collect_news_bundle,
    generate_test_log_cycle,
    run_background_log_agent_cycle,
    run_background_news_agent_cycle,
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
        self._last_log_cycle_at = 0.0
        self._last_news_cycle_at = 0.0
        self._last_faiss_cycle_at = 0.0
        self._recalculate_task_intervals()

    def _recalculate_task_intervals(self) -> None:
        base = max(1, self.interval_seconds)
        self.log_cycle_seconds = max(30, base * 3)
        self.news_cycle_seconds = max(20, base * 2)
        self.faiss_cycle_seconds = max(60, base * 6)

    def _should_run(self, last_run_at: float, cadence_seconds: int, now: float) -> bool:
        return last_run_at <= 0 or (now - last_run_at) >= cadence_seconds

    def _run_log_cycle(self) -> bool:
        try:
            generate_test_log_cycle()
            analyze_logs_bundle(log_dir="data/logs")
            run_background_log_agent_cycle(should_persist=True)
            return True
        except Exception:
            return False

    def _run_news_cycle(self) -> tuple[bool, bool]:
        try:
            news, _ = collect_news_bundle(accumulate=True)
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

        if news and has_new_items and has_crawled_news:
            try:
                run_background_news_agent_cycle(should_persist=True)
            except Exception:
                pass
        return has_new_items, bool(news and has_crawled_news)

    def _run_faiss_cycle(self) -> bool:
        try:
            build_faiss_bundle(news=None, source="worker")
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
            runtime_stats = {
                "last_loop_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_interval_seconds": self.interval_seconds,
                "log_cycle_seconds": self.log_cycle_seconds,
                "news_cycle_seconds": self.news_cycle_seconds,
                "faiss_cycle_seconds": self.faiss_cycle_seconds,
                "log_cycle_ran": False,
                "news_cycle_ran": False,
                "faiss_cycle_ran": False,
                "log_cycle_elapsed_ms": 0,
                "news_cycle_elapsed_ms": 0,
                "faiss_cycle_elapsed_ms": 0,
                "last_loop_elapsed_ms": 0,
                "faiss_rebuild_reason": "입력 변경 시 검색 인덱스를 최신 로그/뉴스 상태와 맞추기 위해 재구성",
            }
            try:
                now = time.monotonic()
                log_cycle_ran = False
                has_new_items = False
                has_crawled_news = False

                if self._should_run(self._last_log_cycle_at, self.log_cycle_seconds, now):
                    phase_started_at = time.monotonic()
                    log_cycle_ran = self._run_log_cycle()
                    runtime_stats["log_cycle_ran"] = log_cycle_ran
                    runtime_stats["log_cycle_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )
                    if log_cycle_ran:
                        self._last_log_cycle_at = now

                if self._should_run(self._last_news_cycle_at, self.news_cycle_seconds, now):
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

                with state.lock:
                    has_results = bool(state.results)

                faiss_input_changed = log_cycle_ran or has_new_items
                runtime_stats["faiss_input_changed"] = faiss_input_changed
                if (
                    has_results
                    and has_crawled_news
                    and faiss_input_changed
                    and self._should_run(self._last_faiss_cycle_at, self.faiss_cycle_seconds, now)
                ):
                    phase_started_at = time.monotonic()
                    if self._run_faiss_cycle():
                        runtime_stats["faiss_cycle_ran"] = True
                        self._last_faiss_cycle_at = now
                    runtime_stats["faiss_cycle_elapsed_ms"] = int(
                        (time.monotonic() - phase_started_at) * 1000
                    )
            except Exception:
                pass
            runtime_stats["last_loop_elapsed_ms"] = int(
                (time.monotonic() - loop_started_at) * 1000
            )
            update_worker_runtime_stats(runtime_stats)
            self._stop_event.wait(self.interval_seconds)


worker = NewsVectorWorker(interval_seconds=10)
