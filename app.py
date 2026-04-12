import os
import time
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px

from agent.strategy_chat import strategy_chat
from analyzer.log_analyzer import analyze_logs
from analyzer.risk_analyzer import calculate_risk
from agent.news_agent import collect_news, analyze_news
from rag.vector_db import build_vector_db, get_vector_count

# -------------------------------
# 🔥 자동 새로고침 (10초)
# -------------------------------
st.set_page_config(page_title="AI 대출 심사", layout="wide")

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# 10초마다 새로고침
if time.time() - st.session_state.last_refresh > 10:
    st.session_state.last_refresh = time.time()
    st.rerun()

st.title("🔥 AI 대출 심사 실시간 대시보드")

# -------------------------------
# 📁 로그 로딩
# -------------------------------
def load_all_logs(log_dir="data/logs"):
    logs = ""
    count = 0

    if not os.path.exists(log_dir):
        return "", 0

    for f in os.listdir(log_dir):
        if f.endswith(".txt") or f.endswith(".log"):
            with open(os.path.join(log_dir, f), encoding="utf-8") as file:
                logs += file.read()
                count += 1

    return logs, count


def run_full_analysis(show_progress: bool = False):
    start = time.time()
    progress = st.progress(0) if show_progress else None
    status = st.empty()

    raw, file_count = load_all_logs()
    if progress is not None:
        progress.progress(10)

    status.info("🔍 로그 분석 중...")
    results = analyze_logs(raw)
    if progress is not None:
        progress.progress(40)

    status.info("📰 뉴스 분석 중...")
    news = collect_news()
    issues = analyze_news(news)
    if progress is not None:
        progress.progress(70)

    status.info("🧠 FAISS 생성 중...")
    try:
        build_vector_db(results, news)
    except Exception:
        pass
    if progress is not None:
        progress.progress(100)

    status.success("✅ 완료!")

    st.session_state.results = results
    st.session_state.issues = issues
    st.session_state.news = news
    st.session_state.file_count = file_count
    st.session_state.total_time = time.time() - start
    st.session_state.last_news_time = datetime.datetime.now()
    st.session_state.initial_analysis_done = True


if "initial_analysis_done" not in st.session_state:
    with st.spinner("초기 분석 실행 중..."):
        run_full_analysis(show_progress=False)


# -------------------------------
# 📊 레이아웃
# -------------------------------
col_left, col_main = st.columns([1, 3])

# -------------------------------
# 🧭 LEFT PANEL
# -------------------------------
with col_left:

    st.header("⚙️ 실행")

    if st.button("📂 전체 분석 실행"):
        run_full_analysis(show_progress=True)

    st.divider()

    # 상태 표시
    if "results" in st.session_state:
        st.subheader("📊 상태")

        st.metric("📁 파일 수", st.session_state.file_count)
        st.metric("📊 로그 수", len(st.session_state.results))
        st.metric("🧠 벡터 수", get_vector_count())
        st.metric("⏱️ 시간", f"{st.session_state.total_time:.1f}s")

        if "last_news_time" in st.session_state:
            st.caption(f"🕒 뉴스 업데이트: {st.session_state.last_news_time}")

    st.divider()

    # PDF 규제 문서 뷰어 (왼쪽 탭)
    st.markdown("### 📋 금감원/여신협회 가이드라인 분석")
    with st.expander("📄 PDF 첨부", expanded=False):
        uploaded_pdf = st.file_uploader("PDF 파일", type=["pdf"], key="left_regulation_pdf")
        
        if uploaded_pdf is not None:
            try:
                import pdfplumber
                with pdfplumber.open(uploaded_pdf) as pdf:
                    st.caption(f"📖 {len(pdf.pages)}쪽 | {round(uploaded_pdf.size / 1024, 1)}KB")
                    page_num = st.slider("페이지", 1, len(pdf.pages), 1, key="left_pdf_page")
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text()
                    if text:
                        st.text_area("내용", text[:300], height=100, disabled=True)
                    tables = page.extract_tables()
                    if tables:
                        for table in tables[:1]:
                            try:
                                df_t = pd.DataFrame(table[1:], columns=table[0])
                                st.dataframe(df_t, use_container_width=True)
                            except:
                                pass
            except Exception as e:
                st.error(f"오류: {str(e)[:80]}")
        else:
            st.caption("PDF 선택 후 조회")


# -------------------------------
# 🧠 MAIN
# -------------------------------
with col_main:

    # -------------------------------
    # 📰 뉴스 자동 갱신
    # -------------------------------
    if "last_news_time" not in st.session_state:
        st.session_state.last_news_time = None

    now = datetime.datetime.now()

    if (
        "news" in st.session_state and
        (
            st.session_state.last_news_time is None or
            (now - st.session_state.last_news_time).seconds > 10
        )
    ):
        with st.spinner("📰 뉴스 자동 업데이트 중..."):
            news = collect_news()
            issues = analyze_news(news)

            st.session_state.news = news
            st.session_state.issues = issues
            st.session_state.last_news_time = now

    # -------------------------------
    # 📰 뉴스 애니메이션 출력
    # -------------------------------
    if "news" in st.session_state:
        st.subheader("📰 실시간 뉴스")

        placeholder = st.empty()

        for n in st.session_state.news[:5]:
            placeholder.info(f"📰 {n.get('title','')}")
            time.sleep(0.3)

    st.divider()

    # -------------------------------
    # 💬 AI 채팅
    # -------------------------------
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

    # -------------------------------
    # 📊 리스크 + FAISS 통합 차트
    # -------------------------------
    if "results" in st.session_state:

        results = st.session_state.results

        st.subheader("📊 실시간 리스크 & 벡터 분석")

        risk_data = []

        for i, r in enumerate(results):
            risk = calculate_risk(
                r["in_fields"],
                r["out_fields"],
                r["in_mapping"],
                r["out_mapping"]
            )

            risk_data.append({
                "index": i,
                "score": risk["score"],
                "grade": risk["grade"],
                "financial": risk["details"]["financial"],
                "credit": risk["details"]["credit"],
                "behavior": risk["details"]["behavior"],
                "regulation": risk["details"]["regulation"]
            })

        df = pd.DataFrame(risk_data)

        # 🔥 애니메이션 데이터 생성
        frames = []
        for t in range(1, len(df)+1):
            temp = df.copy()
            temp["time"] = t
            temp["score"] = temp["score"] * (t / len(df))
            frames.append(temp)

        anim_df = pd.concat(frames)

        col1, col2 = st.columns([2, 1])

        # -------------------------------
        # 📈 리스크 애니메이션
        # -------------------------------
        with col1:

            fig = px.line(
                anim_df,
                x="index",
                y="score",
                animation_frame="time",
                markers=True,
                range_y=[0, max(df["score"]) + 20],
            )

            st.plotly_chart(fig, use_container_width=True, key="risk_line_chart")

            if not df.empty:

                fig_bar = px.bar(
                 df,
                  x="index",
                  y=["financial", "credit", "behavior", "regulation"],
                  barmode="stack"
                )
                fig_bar.update_layout(
                 transition_duration=500
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="risk_bar")

        # -------------------------------
        # 🧠 FAISS 상태
        # -------------------------------
        with col2:

            vec_placeholder = st.empty()

            total = max(1, len(results) * 2)

            for i in range(1, 6):

                current = int((i / 5) * get_vector_count())

                fig_donut = px.pie(
                    names=["Stored", "Remaining"],
                    values=[current, max(1, total - current)],
                    hole=0.6
                )

                vec_placeholder.plotly_chart(
                    fig_donut,
                    use_container_width=True,
                    key=f"faiss_{i}"
                )

                time.sleep(0.2)

            st.success(f"총 벡터: {get_vector_count()}")

# ================================
# 백그라운드: 10초마다 뉴스 수집 & 벡터 DB 학습
# ================================
if "last_auto_vector_time" not in st.session_state:
    st.session_state.last_auto_vector_time = None

now = datetime.datetime.now()

# 10초 이상 지났으면 자동 갱신
if (st.session_state.last_auto_vector_time is None or 
    (now - st.session_state.last_auto_vector_time).total_seconds() >= 10):
    try:
        # 뉴스 수집
        news = collect_news()
        
        # 기존 결과가 있으면 벡터 DB 빌드
        if "results" in st.session_state and st.session_state.results and news:
            try:
                build_vector_db(st.session_state.results, news)
            except Exception:
                pass
        
        st.session_state.last_auto_vector_time = now
        
        # 약간의 지연 후 페이지 새로고침
        time.sleep(0.1)
        st.rerun()
    except Exception:
        pass