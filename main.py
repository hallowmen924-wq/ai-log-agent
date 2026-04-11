from fastapi import FastAPI
from pydantic import BaseModel
import time

from analyzer.log_analyzer import analyze_logs
from analyzer.risk_analyzer import calculate_risk
from agent.news_agent import collect_news, analyze_news
from rag.vector_db import build_vector_db, get_vector_count
from agent.strategy_chat import strategy_chat

app = FastAPI()

# -------------------------------
# 📥 요청 모델
# -------------------------------
class ChatRequest(BaseModel):
    question: str

# -------------------------------
# 📊 전체 분석 API
# -------------------------------
@app.get("/analyze")
def analyze():

    start = time.time()

    # 로그 읽기
    with open("data/logs/sample.txt", encoding="utf-8") as f:
        raw = f.read()

    results = analyze_logs(raw)

    news = collect_news()
    issues = analyze_news(news)

    build_vector_db(results, news)

    # 리스크 계산
    risk_list = []
    for r in results:
        risk = calculate_risk(
            r["in_fields"],
            r["out_fields"],
            r["in_mapping"],
            r["out_mapping"]
        )
        risk_list.append(risk)

    return {
        "results": results,
        "risk": risk_list,
        "vector_count": get_vector_count(),
        "time": time.time() - start
    }

# -------------------------------
# 💬 AI 채팅 API
# -------------------------------
@app.post("/chat")
def chat(req: ChatRequest):

    answer = strategy_chat(req.question)

    return {"answer": answer}