from __future__ import annotations

import pathlib
import sys
import time

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
    run_full_analysis,
    search_faiss,
    state,
)
from backend.worker import worker
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
import asyncio

app = FastAPI(title="AI Log Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_faiss_stats_cache: dict[str, object] = {
    "version": None,
    "data": None,
    "cached_at": 0.0,
}


def build_ws_snapshot(snapshot: dict) -> dict:
    return {
        "vector_count": snapshot.get("vector_count"),
        "vector_events": snapshot.get("vector_events", []),
        "agent_activity_log": snapshot.get("agent_activity_log", []),
        "agent_statuses": snapshot.get("agent_statuses", {}),
        "latest_news_prompt_input": snapshot.get("latest_news_prompt_input", {}),
        "last_news_prompt_input_time": snapshot.get("last_news_prompt_input_time"),
        "latest_log_prompt_input": snapshot.get("latest_log_prompt_input", {}),
        "last_log_prompt_input_time": snapshot.get("last_log_prompt_input_time"),
        "latest_news_briefing": snapshot.get("latest_news_briefing"),
        "last_news_briefing_time": snapshot.get("last_news_briefing_time"),
        "latest_log_briefing": snapshot.get("latest_log_briefing"),
        "last_log_briefing_time": snapshot.get("last_log_briefing_time"),
        "last_news_time": snapshot.get("last_news_time"),
        "last_new_item_time": snapshot.get("last_new_item_time"),
        "news_crawl_running": snapshot.get("news_crawl_running"),
        "news_crawl_target_count": snapshot.get("news_crawl_target_count"),
        "news_crawl_success_count": snapshot.get("news_crawl_success_count"),
        "news_crawl_failure_count": snapshot.get("news_crawl_failure_count"),
        "last_news_crawl_time": snapshot.get("last_news_crawl_time"),
        "last_news_crawl_error": snapshot.get("last_news_crawl_error"),
    }


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
    results, file_count = analyze_logs_bundle(
        raw_logs=payload.raw_logs, log_dir=payload.log_dir
    )
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


@app.get("/faiss/entries")
def faiss_entries(limit: int = 200) -> dict:
    try:
        from rag.vector_db import list_vectors

        items = list_vectors(limit=limit)
        return {"status": "ok", "count": len(items), "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/export")
def faiss_export(format: str = "json", limit: int = 200):
    try:
        from rag.vector_db import list_vectors
        import json, csv, io

        items = list_vectors(limit=limit)
        if format == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["id", "type", "product", "agent", "source", "name", "snippet"])
            writer.writeheader()
            for row in items:
                writer.writerow(row)
            return Response(content=output.getvalue(), media_type="text/csv")
        else:
            return {"status": "ok", "count": len(items), "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/entry/{doc_id}")
def faiss_entry(doc_id: str):
    try:
        from rag.vector_db import get_vector_by_id

        item = get_vector_by_id(doc_id)
        if item is None:
            raise HTTPException(status_code=404, detail="entry not found")
        return {"status": "ok", "item": item}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/stats")
def faiss_stats() -> dict:
    """Compute per-product aggregates from FAISS stored vectors.

    Returns average applied rate, average available amount, approval rate,
    common rejection reasons, and average credit grades per product code.
    """
    try:
        from rag.vector_db import list_vectors

        snapshot = state.snapshot()
        latest_event = (snapshot.get("vector_events") or [{}])[0]
        cache_version = (
            latest_event.get("timestamp"),
            snapshot.get("vector_count"),
        )
        now = time.time()

        if (
            _faiss_stats_cache.get("data") is not None
            and _faiss_stats_cache.get("version") == cache_version
            and (now - float(_faiss_stats_cache.get("cached_at") or 0.0)) < 30
        ):
            return {
                "status": "ok",
                "products": _faiss_stats_cache.get("data"),
                "cached": True,
            }

        items = list_vectors(limit=10000)
        per_prod: dict[str, list[dict]] = {}
        for it in items:
            prod = it.get("product") or "UNKNOWN"
            per_prod.setdefault(prod, []).append(it)

        stats: dict = {}
        for prod, product_items in per_prod.items():
            cnt = 0
            sum_rate = 0.0
            rate_count = 0
            sum_limit = 0.0
            limit_count = 0
            approvals = 0
            approval_count = 0
            reject_reasons: dict[str, int] = {}
            kcb_scores = []
            nice_scores = []

            for item in product_items:
                features = item.get("features", {}) or {}
                meta = {
                    "in_fields": item.get("in_fields") or {},
                    "out_fields": item.get("out_fields") or {},
                }

                # applied_rate
                ar = None
                try:
                    ar = features.get("applied_rate") if isinstance(features, dict) else None
                except Exception:
                    ar = None
                if ar is not None:
                    try:
                        sum_rate += float(ar)
                        rate_count += 1
                    except Exception:
                        pass

                # available_amount
                aa = None
                try:
                    aa = features.get("available_amount") if isinstance(features, dict) else None
                except Exception:
                    aa = None
                if aa is not None:
                    try:
                        sum_limit += float(aa)
                        limit_count += 1
                    except Exception:
                        pass

                # approval detection: look into metadata in_fields/out_fields
                approved = None
                for field_container in (meta.get("out_fields") or {}, meta.get("in_fields") or {}, features or {}):
                    if not isinstance(field_container, dict):
                        continue
                    for k, v in field_container.items():
                        sval = str(v).lower() if v is not None else ""
                        if any(tok in sval for tok in ("승인", "승", "approve", "approved", "ok", " 승인 ")):
                            approved = True
                            break
                        if any(tok in sval for tok in ("거절", "불가", "reject", "denied", "불허")):
                            approved = False
                            # capture reason nearby
                            if len(sval) > 2:
                                reject_reasons[sval[:200]] = reject_reasons.get(sval[:200], 0) + 1
                            break
                    if approved is not None:
                        break

                if approved is True:
                    approvals += 1
                if approved is not None:
                    approval_count += 1

                # credit scores
                try:
                    cs = features.get("credit_score")
                    if cs is not None:
                        nice_scores.append(float(cs))
                except Exception:
                    pass
                try:
                    cg = features.get("credit_grade")
                    if cg is not None:
                        # map grades A,B,C -> numeric fallback
                        if isinstance(cg, str) and cg.isalpha():
                            # A=4,B=3,C=2,D=1,S=5
                            mapping = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                            val = mapping.get(cg.upper())
                            if val is not None:
                                kcb_scores.append(val)
                        else:
                            try:
                                kcb_scores.append(float(cg))
                            except Exception:
                                pass
                except Exception:
                    pass

                cnt += 1

            stats[prod] = {
                "count": cnt,
                "avg_applied_rate": (sum_rate / rate_count) if rate_count else None,
                "avg_available_amount": (sum_limit / limit_count) if limit_count else None,
                "approval_rate": (approvals / approval_count) if approval_count else None,
                "top_reject_reasons": sorted(reject_reasons.items(), key=lambda x: -x[1])[:5],
                "avg_kcb_grade": (sum(kcb_scores) / len(kcb_scores)) if kcb_scores else None,
                "avg_credit_score": (sum(nice_scores) / len(nice_scores)) if nice_scores else None,
            }

        _faiss_stats_cache["version"] = cache_version
        _faiss_stats_cache["data"] = stats
        _faiss_stats_cache["cached_at"] = now

        return {"status": "ok", "products": stats, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/search_features")
def faiss_search_features(type: str | None = None, feature_key: str | None = None, feature_value: str | None = None, limit: int = 200) -> dict:
    """Search FAISS entries by metadata `type` and feature key/value.

    - `type`: optional metadata type filter (e.g., 'log', 'news', 'agent_report')
    - `feature_key`: feature field name to match (only relevant for items with `features` metadata)
    - `feature_value`: substring to match against the feature value (optional)
    """
    try:
        from rag.vector_db import list_vectors, get_vector_by_id

        items = list_vectors(limit=10000)
        results: list[dict] = []
        count = 0
        for it in items:
            if type and (it.get("type") or "") != type:
                continue
            doc_id = it.get("id")
            if not doc_id:
                continue
            doc = get_vector_by_id(str(doc_id)) or {}
            meta = doc.get("metadata", {}) or {}
            features = meta.get("features", {}) or {}

            # If feature_key provided, require it exists
            if feature_key:
                if not isinstance(features, dict) or feature_key not in features:
                    continue
                if feature_value:
                    fv = features.get(feature_key)
                    if fv is None:
                        continue
                    # match substring (case-insensitive)
                    if str(feature_value).lower() not in str(fv).lower():
                        continue

            results.append({
                "id": str(doc_id),
                "type": it.get("type"),
                "product": it.get("product"),
                "snippet": it.get("snippet"),
                "features": features,
            })
            count += 1
            if count >= limit:
                break

        return {"status": "ok", "count": len(results), "items": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/ws/faiss")
async def websocket_faiss_updates(websocket: WebSocket):
    await websocket.accept()
    last_sent = 0
    last_signature = None
    try:
        while True:
            snapshot = state.snapshot()
            events = snapshot.get("vector_events", []) or []
            to_send = []
            for ev in reversed(events):
                ts = ev.get("timestamp") or ""
                try:
                    tval = int(float(ts.replace(".", ""))) if ts else 0
                except Exception:
                    tval = 0
                if tval > last_sent:
                    to_send.append(ev)
                    last_sent = max(last_sent, tval)

            signature = (
                (events[0].get("timestamp") if events else None),
                (snapshot.get("agent_activity_log", [{}])[0].get("timestamp") if snapshot.get("agent_activity_log") else None),
                snapshot.get("last_news_prompt_input_time"),
                snapshot.get("last_log_prompt_input_time"),
                snapshot.get("last_news_briefing_time"),
                snapshot.get("last_log_briefing_time"),
                snapshot.get("last_news_time"),
                snapshot.get("last_new_item_time"),
                snapshot.get("news_crawl_running"),
                snapshot.get("news_crawl_success_count"),
                snapshot.get("news_crawl_failure_count"),
                snapshot.get("last_news_crawl_time"),
            )

            for ev in to_send:
                try:
                    payload = {
                        "type": "vector_event",
                        "event": ev,
                        "snapshot": build_ws_snapshot(snapshot),
                    }
                except Exception:
                    payload = {"type": "vector_event", "event": ev}
                await websocket.send_json(payload)

            if signature != last_signature:
                last_signature = signature
                await websocket.send_json(
                    {
                        "type": "state_update",
                        "snapshot": build_ws_snapshot(snapshot),
                    }
                )
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return


@app.post("/worker/start", response_model=GenericMessage)
def worker_start(payload: WorkerConfigRequest) -> GenericMessage:
    worker.update_interval(payload.interval_seconds)
    started = worker.start()
    detail = f"worker interval={worker.interval_seconds}s"
    return GenericMessage(
        status="started" if started else "already_running", detail=detail
    )


@app.post("/worker/stop", response_model=GenericMessage)
def worker_stop() -> GenericMessage:
    stopped = worker.stop()
    return GenericMessage(status="stopped" if stopped else "already_stopped")
