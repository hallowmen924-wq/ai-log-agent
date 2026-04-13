from __future__ import annotations

import pathlib
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 이 파일은 FastAPI의 실제 진입점입니다.
# Streamlit은 이 서버에 HTTP 요청을 보내고, 서버는 분석/벡터/챗 작업을 수행합니다.

from backend.schemas import (
    FaissBuildRequest,
    FullAnalysisResponse,
    GenericMessage,
    LogAnalyzeRequest,
    LogAnalyzeResponse,
    NewsCollectResponse,
    SearchRequest,
    StrategyChatRequest,
    StrategyChatResponse,
    WorkerConfigRequest,
)
from backend.services import (
    analyze_logs_bundle,
    ask_strategy,
    build_faiss_bundle,
    collect_news_bundle,
    enrich_results,
    get_chart_payloads,
    search_faiss,
    state,
    run_full_analysis,
)
from backend.worker import worker

app = FastAPI(title="AI Log Agent API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    # 화면에서 가장 먼저 확인하는 상태 API입니다.
    # 서버가 살아있는지, 워커가 도는지, 최근 분석 상태가 어떤지 전달합니다.
    return {"status": "ok", "worker_running": worker.running, **state.snapshot()}


@app.post("/news/collect", response_model=NewsCollectResponse)
def news_collect() -> NewsCollectResponse:
    news, issues = collect_news_bundle(accumulate=True)
    snapshot = state.snapshot()
    return NewsCollectResponse(
        news=news,
        issues=issues,
        count=len(news),
        last_new_item_time=snapshot.get("last_new_item_time"),
    )


@app.post("/logs/analyze", response_model=LogAnalyzeResponse)
def logs_analyze(payload: LogAnalyzeRequest) -> LogAnalyzeResponse:
    results, file_count = analyze_logs_bundle(raw_logs=payload.raw_logs, log_dir=payload.log_dir)
    return LogAnalyzeResponse(file_count=file_count, results=enrich_results(results))


@app.post("/faiss/build")
def faiss_build(payload: FaissBuildRequest) -> dict:
    try:
        count = build_faiss_bundle(logs=payload.logs, news=payload.news)
        return {"status": "ok", "vector_count": count}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/faiss/search")
def faiss_search(payload: SearchRequest) -> dict:
    try:
        return search_faiss(payload.query, payload.k)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/chat/strategy", response_model=StrategyChatResponse)
def chat_strategy(payload: StrategyChatRequest) -> StrategyChatResponse:
    try:
        return StrategyChatResponse(**ask_strategy(payload.question))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/analysis/run", response_model=FullAnalysisResponse)
def analysis_run(log_dir: str = "data/logs") -> FullAnalysisResponse:
    try:
        snapshot = run_full_analysis(log_dir=log_dir)
        return FullAnalysisResponse(**snapshot)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/analysis/status", response_model=FullAnalysisResponse)
def analysis_status() -> FullAnalysisResponse:
    snapshot = state.snapshot()
    snapshot["results"] = enrich_results(snapshot["results"])
    return FullAnalysisResponse(**snapshot)


@app.get("/charts")
def charts_all() -> dict:
    # 메인 대시보드의 4개 차트가 한 번에 가져가는 스냅샷 API입니다.
    snapshot = state.snapshot()
    return {
        "status": "ok",
        "last_chart_time": snapshot.get("last_chart_time"),
        "charts": get_chart_payloads(),
    }


@app.get("/charts/{chart_name}")
def charts_one(chart_name: str) -> dict:
    payloads = get_chart_payloads()
    if chart_name not in payloads:
        raise HTTPException(status_code=404, detail=f"unknown chart: {chart_name}")
    snapshot = state.snapshot()
    return {
        "status": "ok",
        "last_chart_time": snapshot.get("last_chart_time"),
        "chart_name": chart_name,
        "data": payloads[chart_name],
    }


@app.post("/worker/start", response_model=GenericMessage)
def worker_start(payload: WorkerConfigRequest) -> GenericMessage:
    worker.update_interval(payload.interval_seconds)
    started = worker.start()
    detail = f"worker interval={worker.interval_seconds}s"
    return GenericMessage(status="started" if started else "already_running", detail=detail)


@app.post("/worker/stop", response_model=GenericMessage)
def worker_stop() -> GenericMessage:
    stopped = worker.stop()
    return GenericMessage(status="stopped" if stopped else "already_stopped")
