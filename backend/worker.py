from __future__ import annotations

import threading

from backend.services import (
    analyze_logs_bundle,
    build_faiss_bundle,
    collect_news_bundle,
    generate_test_log_cycle,
    run_background_log_agent_cycle,
    run_background_news_agent_cycle,
    state,
)

# 이 워커는 백그라운드에서 주기적으로 뉴스를 새로 받고,
# 이미 분석된 로그가 있으면 FAISS 벡터 DB도 다시 빌드합니다.
# 즉, 메인 화면을 다시 열지 않아도 데이터가 조금씩 최신 상태로 갱신됩니다.


class NewsVectorWorker:
    def __init__(self, interval_seconds: int = 10) -> None:
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

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

    def _run_loop(self) -> None:
        # stop 요청이 들어오기 전까지 interval_seconds 간격으로 반복 실행됩니다.
        while not self._stop_event.is_set():
            try:
                try:
                    generate_test_log_cycle()
                    analyze_logs_bundle(log_dir="data/logs")
                    run_background_log_agent_cycle(should_persist=True)
                except Exception:
                    pass

                # 워커는 뉴스를 누적 모드로 모아서 새 기사만 계속 state.news에 쌓습니다.
                news, _ = collect_news_bundle(accumulate=True)
                with state.lock:
                    has_new_items = bool(
                        state.last_new_item_time
                        and state.last_news_time
                        and state.last_new_item_time == state.last_news_time
                    )
                with state.lock:
                    has_results = bool(state.results)
                if news:
                    try:
                        run_background_news_agent_cycle(should_persist=has_new_items)
                    except Exception:
                        pass
                if has_results and news:
                    try:
                        # 로그 결과는 그대로 두고, 최신 뉴스 기준으로 벡터 DB를 다시 구성합니다.
                        build_faiss_bundle(news=news, source="worker")
                    except Exception:
                        pass
            except Exception:
                pass
            self._stop_event.wait(self.interval_seconds)


worker = NewsVectorWorker(interval_seconds=10)
