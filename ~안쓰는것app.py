import os
import time
import streamlit as st
import pandas as pd

import ui
from agent.strategy_chat import strategy_chat
from analyzer.log_analyzer import analyze_logs
from agent.news_agent import collect_news, analyze_news

# rag.vector_db may load heavy ML models (FAISS / HuggingFace). Provide a
# lightweight fallback when those dependencies are not available so the
# Streamlit UI can start quickly for local testing.
try:
    from rag.vector_db import build_vector_db, get_vector_count
except Exception:
    def build_vector_db(results, news):
        print("[경량모드] build_vector_db 호출이 무시되었습니다.")

    def get_vector_count():
        return 0

st.set_page_config(page_title="AI 대출 심사", layout="wide")

# 레이아웃 컬럼
col_left, col_main = st.columns([1, 3])

# -------------------------------
# 유틸: 로그 로딩
# -------------------------------
def load_all_logs(log_dir="data/logs"):
    logs = ""
    count = 0
    if not os.path.exists(log_dir):
        return logs, count
    for f in os.listdir(log_dir):
        if f.endswith(".txt") or f.endswith(".log"):
            try:
                with open(os.path.join(log_dir, f), encoding="utf-8") as file:
                    logs += file.read()
                    count += 1
            except Exception:
                continue
    return logs, count

# -------------------------------
# LEFT PANEL
# -------------------------------
with col_left:
    st.header("⚙️ 실행")

    if st.button("📂 전체 분석 실행"):
        progress = st.progress(0)
        status = st.empty()
        start = time.time()

        raw, file_count = load_all_logs()
        progress.progress(10)

        status.info("🔍 로그 분석 중...")
        results = analyze_logs(raw)
        progress.progress(40)

        status.info("📰 뉴스 분석 중...")
        news = collect_news()
        issues = analyze_news(news)
        progress.progress(70)

        status.info("🧠 FAISS 생성 중...")
        try:
            build_vector_db(results, news)
        except Exception:
            pass
        progress.progress(100)

        status.success("✅ 완료!")

        st.session_state.results = results
        st.session_state.issues = issues
        st.session_state.file_count = file_count
        st.session_state.total_time = time.time() - start



    # 자동 새로고침 설정
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "auto_interval" not in st.session_state:
        st.session_state.auto_interval = 3


    # 벡터 히스토리 초기화
    if "vector_history" not in st.session_state:
        st.session_state.vector_history = []

    st.divider()

    if "results" in st.session_state:
        st.subheader("📊 상태")
        st.metric("📁 파일 수", st.session_state.file_count)
        st.metric("📊 로그 수", len(st.session_state.results))
        st.metric("🧠 벡터 수", get_vector_count())
        st.metric("⏱️ 시간", f"{st.session_state.total_time:.1f}s")

# -------------------------------
# MAIN
# -------------------------------
with col_main:
    st.header("💬 AI 심사 전략")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("질문 입력 (예: 승인율 올리는 방법?)")
    if user_input:
        with st.spinner("🧠 전략 생성 중..."):
            answer = strategy_chat(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", answer))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    st.divider()

    # 결과 대시보드 (ui.py로 위임)
    ui.render_dashboard()

    # 자동 새로고침 처리
    if st.session_state.get("auto_refresh"):
        time.sleep(max(1, int(st.session_state.get("auto_interval", 3))))
        try:
            params = {"_autorefresh": int(time.time())}
            st.experimental_set_query_params(**params)
        except Exception:
            try:
                st.stop()
            except Exception:
                pass


        # 자동 새로고침 처리
        if st.session_state.get("auto_refresh"):
            time.sleep(max(1, int(st.session_state.get("auto_interval", 3))))
            try:
                params = {"_autorefresh": int(time.time())}
                st.experimental_set_query_params(**params)
            except Exception:
                try:
                    st.stop()
                except Exception:
                    pass