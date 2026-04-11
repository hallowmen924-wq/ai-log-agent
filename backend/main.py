from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import sys
import pathlib

# 프로젝트 루트를 sys.path에 추가하여 상위 폴더의 모듈들을 가져올 수 있게 함
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer.log_analyzer import analyze_logs
from analyzer.risk_analyzer import calculate_risk
from agent.news_agent import collect_news, analyze_news
from rag.vector_db import build_vector_db, get_vector_count
from agent.strategy_chat import strategy_chat

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = {"running": False, "results": [], "issues": [], "file_count": 0, "total_time": 0.0}


def load_all_logs(log_dir="data/logs"):
    logs = ""
    count = 0
    if not os.path.exists(log_dir):
        return logs, count
    for f in os.listdir(log_dir):
        if f.endswith(".txt") or f.endswith(".log"):
            with open(os.path.join(log_dir, f), encoding="utf-8") as file:
                logs += file.read()
                count += 1
    return logs, count


def run_analysis():
    state["running"] = True
    start = time.time()

    raw, file_count = load_all_logs()
    state["file_count"] = file_count

    results = analyze_logs(raw)
    news = collect_news()
    issues = analyze_news(news)

    # build vector DB (may be a no-op in tests)
    build_vector_db(results, news)

    state["results"] = results
    state["issues"] = issues
    state["total_time"] = time.time() - start
    state["running"] = False


@app.post("/run-analysis")
def start_analysis(background_tasks: BackgroundTasks):
    if state["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(run_analysis)
    return {"status": "started"}


@app.get("/status")
def status():
    return {
        "running": state["running"],
        "file_count": state["file_count"],
        "vector_count": get_vector_count(),
        "total_time": state["total_time"],
    }


@app.get("/results")
def get_results():
    def safe_serialize(o):
        # primitives
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        # dict
        if isinstance(o, dict):
            return {k: safe_serialize(v) for k, v in o.items()}
        # list/tuple
        if isinstance(o, (list, tuple)):
            return [safe_serialize(v) for v in o]
        # fallback to str
        try:
            return str(o)
        except:
            return None

    serialized = []
    for r in state.get("results", []):
        try:
            # try to extract fields expected by risk calculation
            in_fields = r.get("in_fields", {}) if isinstance(r, dict) else {}
            out_fields = r.get("out_fields", {}) if isinstance(r, dict) else {}
            in_mapping = r.get("in_mapping", {}) if isinstance(r, dict) else {}
            out_mapping = r.get("out_mapping", {}) if isinstance(r, dict) else {}

            risk = calculate_risk(in_fields, out_fields, in_mapping, out_mapping)

            serialized.append({
                "product": r.get("product") if isinstance(r, dict) else None,
                "in_fields": safe_serialize(in_fields),
                "out_fields": safe_serialize(out_fields),
                "in_mapping": safe_serialize(in_mapping),
                "out_mapping": safe_serialize(out_mapping),
                "risk": safe_serialize(risk)
            })
        except Exception as e:
            serialized.append({"error": str(e)})

    return {"results": serialized}


@app.get("/vector-count")
def vector_count():
    return {"count": get_vector_count()}


@app.post("/chat")
def chat(payload: dict):
    q = payload.get("question") if isinstance(payload, dict) else None
    if not q:
        return {"answer": ""}
    ans = strategy_chat(q)
    return {"answer": ans}


@app.post("/populate-sample")
def populate_sample():
    """테스트용 샘플 결과를 생성합니다. 외부 의존성(FAISS 등)을 호출하지 않습니다."""
    sample_logs = []

    # 제품별 샘플 생성: C6, C9, C11, C12을 골고루 생성
    products = ["C6", "C9", "C11", "C12"]
    for i in range(8):
        prod = products[i % len(products)]
        in_fields = {
            "A2001": 5000000 * (i + 1),
            "CREDIT_SCORE": 650 - (i % 4) * 40,
            "INCOME": 2500000 + (i % 5) * 400000,
            "LOAN_CNT": (i % 5),
            "JOB_TYPE": "정규직" if i % 2 == 0 else "프리랜서"
        }

        out_fields = {
            "R0003": 25 + (i % 5) * 8,
            "OVERDUE_YN": "Y" if i % 5 == 3 else "N"
        }

        sample_logs.append({
            "product": prod,
            "in_fields": in_fields,
            "out_fields": out_fields,
            "in_mapping": {k: k for k in in_fields.keys()},
            "out_mapping": {k: k for k in out_fields.keys()}
        })

    # 상태에 반영
    state["results"] = sample_logs
    state["file_count"] = len(sample_logs)
    state["issues"] = []
    state["total_time"] = 0.1

    # 반환은 기존 /results와 동일한 형태로 직렬화
    def safe_serialize(o):
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        if isinstance(o, dict):
            return {k: safe_serialize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [safe_serialize(v) for v in o]
        try:
            return str(o)
        except:
            return None

    serialized = []
    for r in state.get("results", []):
        in_fields = r.get("in_fields", {})
        out_fields = r.get("out_fields", {})
        in_mapping = r.get("in_mapping", {})
        out_mapping = r.get("out_mapping", {})
        risk = calculate_risk(in_fields, out_fields, in_mapping, out_mapping)

        serialized.append({
            "product": r.get("product"),
            "in_fields": safe_serialize(in_fields),
            "out_fields": safe_serialize(out_fields),
            "in_mapping": safe_serialize(in_mapping),
            "out_mapping": safe_serialize(out_mapping),
            "risk": safe_serialize(risk)
        })

    state["results"] = serialized
    return {"status": "ok", "results": serialized}
