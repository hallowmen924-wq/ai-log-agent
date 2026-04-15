import datetime
import html
import threading
import time
import pandas as pd
import plotly.express as px
import streamlit as st

from agent.strategy_chat import regulation_agent
from backend.streamlit_client import BackendClient
from rag.vector_db import get_vector_count, ingest_files, search_context

# 백그라운드 작업 결과 저장소 (스레드 -> 메인 폴링으로 전달)
_background_results: dict = {}
_background_lock = threading.Lock()

# 이 파일은 최종 Streamlit 진입점입니다.
# 핵심 역할은 "직접 분석하지 않고" FastAPI 백엔드에서 준비한 데이터를 받아
# 화면에 보여주는 것입니다.


# -------------------------------
# 🔥 자동 새로고침 (10초)
# -------------------------------
st.set_page_config(page_title="AI 대출 심사", layout="wide")

HAS_FRAGMENT_REFRESH = hasattr(st, "fragment")


def fragment_decorator(*args, **kwargs):
    if HAS_FRAGMENT_REFRESH:
        return st.fragment(*args, **kwargs)
    return lambda func: func


if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# 10초마다 새로고침
if (not HAS_FRAGMENT_REFRESH) and time.time() - st.session_state.last_refresh > 10:
    st.session_state.last_refresh = time.time()
    st.rerun()

st.title("🔥 AI 대출 심사 실시간 대시보드")


def render_loading_styles():
    st.markdown(
        """
        <style>
        .loading-panel {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            padding: 18px;
            background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(241,245,249,0.94));
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 14px;
        }
        .loading-title {
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 10px;
        }
        .loading-sub {
            font-size: 13px;
            color: #475569;
            margin-bottom: 14px;
            line-height: 1.5;
        }
        .loading-step {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            margin-bottom: 8px;
            background: rgba(255,255,255,0.7);
            border: 1px solid rgba(226,232,240,0.9);
        }
        .loading-step.active {
            background: rgba(224,242,254,0.95);
            border-color: rgba(56,189,248,0.55);
        }
        .loading-step.done {
            background: rgba(220,252,231,0.9);
            border-color: rgba(74,222,128,0.45);
        }
        .loading-badge {
            width: 26px;
            height: 26px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            color: white;
            background: #94a3b8;
        }
        .loading-step.active .loading-badge { background: #0284c7; }
        .loading-step.done .loading-badge { background: #16a34a; }
        .loading-meta {
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.04);
            color: #334155;
            font-size: 13px;
        }
        .skeleton-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }
        .skeleton-card {
            min-height: 136px;
            border-radius: 16px;
            background: linear-gradient(90deg, #e2e8f0 25%, #f8fafc 37%, #e2e8f0 63%);
            background-size: 400% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
            border: 1px solid rgba(226,232,240,0.95);
        }
        .skeleton-wide {
            min-height: 208px;
            margin-top: 14px;
            border-radius: 16px;
            background: linear-gradient(90deg, #e2e8f0 25%, #f8fafc 37%, #e2e8f0 63%);
            background-size: 400% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
            border: 1px solid rgba(226,232,240,0.95);
        }
        @keyframes shimmer {
            0% { background-position: 100% 0; }
            100% { background-position: 0 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_loading_checklist(
    target, active_step: int, eta_text: str, elapsed_text: str = ""
):
    steps = [
        "백엔드 연결 상태 확인",
        "로그 파일 분석 요청",
        "뉴스 수집 및 이슈 분석",
        "FAISS 벡터 생성 및 결과 동기화",
    ]
    rows = []
    for index, label in enumerate(steps):
        class_name = "loading-step"
        badge = str(index + 1)
        if index < active_step:
            class_name += " done"
            badge = "OK"
        elif index == active_step:
            class_name += " active"
            badge = ".."
        rows.append(
            f"<div class='{class_name}'><div class='loading-badge'>{badge}</div><div>{label}</div></div>"
        )

    meta = f"예상 소요 시간: {eta_text}"
    if elapsed_text:
        meta += f"<br>경과 시간: {elapsed_text}"

    html = (
        "<div class='loading-panel'>"
        "<div class='loading-title'>초기 데이터 준비 중</div>"
        "<div class='loading-sub'>첫 실행은 로그 분석, 뉴스 수집, 임베딩 모델 준비 때문에 평소보다 더 오래 걸릴 수 있습니다.</div>"
        + "".join(rows)
        + f"<div class='loading-meta'>{meta}</div>"
        + "</div>"
    )
    target.markdown(html, unsafe_allow_html=True)


def render_loading_skeleton(target):
    target.markdown(
        """
        <div class='loading-panel'>
            <div class='loading-title'>대시보드 미리보기</div>
            <div class='loading-sub'>차트와 카드 영역을 준비하고 있습니다.</div>
            <div class='skeleton-grid'>
                <div class='skeleton-card'></div>
                <div class='skeleton-card'></div>
                <div class='skeleton-card'></div>
                <div class='skeleton-card'></div>
            </div>
            <div class='skeleton-wide'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_backend_client() -> BackendClient:
    # 왼쪽 패널에서 주소를 바꾸면 같은 함수가 새 백엔드 주소를 사용합니다.
    base_url = st.session_state.get("backend_url", "http://127.0.0.1:18000")
    return BackendClient(base_url)


def get_backend_health() -> dict:
    try:
        return get_backend_client().health()
    except Exception as error:
        return {"status": "down", "detail": str(error)}


def sync_session_from_backend(payload: dict):
    # 백엔드 응답을 Streamlit 세션 상태로 옮겨서 화면 어디서든 재사용합니다.
    st.session_state.results = payload.get("results", [])
    st.session_state.issues = payload.get("issues", [])
    st.session_state.news = payload.get("news", [])
    st.session_state.file_count = payload.get("file_count", 0)
    st.session_state.vector_count = payload.get("vector_count", 0)
    st.session_state.total_time = payload.get("total_time", 0.0)
    st.session_state.last_news_time = payload.get("last_news_time")
    st.session_state.last_new_item_time = payload.get("last_new_item_time")
    st.session_state.latest_strategy_question = payload.get("latest_strategy_question")
    st.session_state.last_strategy_time = payload.get("last_strategy_time")
    st.session_state.last_log_ingest_time = payload.get("last_log_ingest_time")
    st.session_state.latest_log_briefing = payload.get("latest_log_briefing")
    st.session_state.last_log_briefing_time = payload.get("last_log_briefing_time")
    st.session_state.latest_log_prompt_input = payload.get(
        "latest_log_prompt_input", {}
    )
    st.session_state.last_log_prompt_input_time = payload.get(
        "last_log_prompt_input_time"
    )
    st.session_state.latest_news_briefing = payload.get("latest_news_briefing")
    st.session_state.last_news_briefing_time = payload.get("last_news_briefing_time")
    st.session_state.latest_news_prompt_input = payload.get(
        "latest_news_prompt_input", {}
    )
    st.session_state.last_news_prompt_input_time = payload.get(
        "last_news_prompt_input_time"
    )
    st.session_state.agent_statuses = payload.get("agent_statuses", {})
    st.session_state.agent_activity_log = payload.get("agent_activity_log", [])
    st.session_state.vector_events = payload.get("vector_events", [])


def format_status_time(value) -> str:
    if not value:
        return "-"
    parsed = parse_status_time(value)
    if parsed is not None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def parse_status_time(value):
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def render_news_freshness_badge(last_news_time, last_new_item_time):
    collected_at = parse_status_time(last_news_time)
    new_item_at = parse_status_time(last_new_item_time)

    if collected_at is None:
        label = "수집 대기"
        background = "#e2e8f0"
        color = "#334155"
    elif new_item_at is not None and collected_at == new_item_at:
        label = "신규 기사 유입"
        background = "#dcfce7"
        color = "#166534"
    elif new_item_at is not None:
        label = "중복 기사만 수집"
        background = "#fef3c7"
        color = "#92400e"
    else:
        label = "신규 기사 이력 없음"
        background = "#e0f2fe"
        color = "#075985"

    st.markdown(
        f"""
        <div style=\"margin: 8px 0 10px 0;\">
            <span style=\"
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                background: {background};
                color: {color};
                border: 1px solid rgba(15, 23, 42, 0.08);
            \">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_strategy_response(message):
    if isinstance(message, str):
        st.write(message)
        return

    if not isinstance(message, dict):
        st.write(str(message))
        return

    sections = message.get("sections", {})
    answer = message.get("answer", "")
    vector_update = message.get("vector_update", {})

    final_decision = sections.get("final_decision", answer or "분석 결과가 없습니다.")
    decision_label = "판단 대기"
    decision_background = "#e2e8f0"
    decision_color = "#334155"

    if "조건부 승인" in final_decision:
        decision_label = "조건부 승인"
        decision_background = "#fef3c7"
        decision_color = "#92400e"
    elif "승인" in final_decision:
        decision_label = "승인"
        decision_background = "#dcfce7"
        decision_color = "#166534"
    elif "거절" in final_decision:
        decision_label = "거절"
        decision_background = "#fee2e2"
        decision_color = "#991b1b"

    log_text = sections.get("log_analysis", "분석 결과가 없습니다.")
    news_text = sections.get("news_analysis", "분석 결과가 없습니다.")
    regulation_text = sections.get("regulation_analysis", "분석 결과가 없습니다.")

    summary_cols = st.columns(4)
    summary_items = [
        ("📄 로그 분석", log_text, "#eff6ff", "#1d4ed8"),
        ("📰 뉴스 영향", news_text, "#ecfeff", "#0f766e"),
        ("⚖️ 규제 판단", regulation_text, "#fff7ed", "#c2410c"),
        ("🧠 최종 결론", final_decision, decision_background, decision_color),
    ]

    for column, (title, body, background, color) in zip(summary_cols, summary_items):
        preview = body.replace("\n", " ").strip()[:130]
        if not preview:
            preview = "분석 결과가 없습니다."
        column.markdown(
            f"""
            <div style=\"
                height: 168px;
                border-radius: 16px;
                padding: 14px;
                background: {background};
                border: 1px solid rgba(15, 23, 42, 0.08);
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            \">
                <div style=\"font-size: 13px; font-weight: 700; color: {color}; margin-bottom: 10px;\">{title}</div>
                <div style=\"font-size: 13px; line-height: 1.55; color: #0f172a;\">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style=\"
            margin: 14px 0 10px 0;
            padding: 14px 16px;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(248,250,252,0.96), rgba(241,245,249,0.96));
            border: 1px solid rgba(148, 163, 184, 0.22);
        \">
            <div style=\"font-size: 12px; font-weight: 700; color: {decision_color}; margin-bottom: 6px;\">최종 심사 판단</div>
            <div style=\"font-size: 20px; font-weight: 800; color: #0f172a; margin-bottom: 8px;\">{decision_label}</div>
            <div style=\"font-size: 13px; line-height: 1.6; color: #334155;\">{final_decision.replace(chr(10), '<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if vector_update:
        vector_cols = st.columns(3)
        vector_cols[0].metric("적재 전 벡터", vector_update.get("before_count", 0))
        vector_cols[1].metric("적재 후 벡터", vector_update.get("after_count", 0))
        vector_cols[2].metric("이번 추가량", vector_update.get("added_count", 0))

    tab_log, tab_news, tab_regulation, tab_decision, tab_context = st.tabs(
        [
            "📄 로그 분석",
            "📰 뉴스 영향",
            "⚖️ 규제 판단",
            "🧠 최종 결론",
            "📚 참고 컨텍스트",
        ]
    )

    with tab_log:
        st.write(log_text)

    with tab_news:
        st.write(news_text)

    with tab_regulation:
        st.write(regulation_text)

    with tab_decision:
        st.write(final_decision)

    with tab_context:
        context = message.get("context", {})
        context_log, context_news, context_rules = st.columns(3)
        with context_log:
            st.markdown("##### 로그")
            for item in context.get("logs", []):
                st.info(item)
        with context_news:
            st.markdown("##### 뉴스")
            for item in context.get("news", []):
                st.info(item)
        with context_rules:
            st.markdown("##### 규제")
            for item in context.get("rules", []):
                st.info(item)


# `get_vector_count` is provided by `rag.vector_db` import; avoid redefining it here.


def get_chart_snapshots() -> dict:
    try:
        payload = get_backend_client().get_charts()
        return payload.get("charts", {})
    except Exception:
        return {}


def render_runtime_dashboard():
    st.subheader("🤖 에이전트 실시간 작업 현황")

    # 백그라운드 작업 결과를 메인 스레드로 폴링하여 session_state로 반영
    try:
        with _background_lock:
            tasks = list(_background_results.items())
        for task_id, payload in tasks:
            # 반영 처리
            if payload.get("status") == "completed":
                result = payload.get("result")
                done_time = payload.get("updated_at")
                statuses = st.session_state.get("agent_statuses", {})
                statuses["regulation_agent"] = {
                    "status": "completed",
                    "updated_at": done_time,
                    "detail": "규제 분석 완료",
                }
                st.session_state.agent_statuses = statuses
                st.session_state.latest_regulation_analysis = result
                st.session_state.last_regulation_time = done_time
                # 벡터 카운트 및 이벤트 갱신
                vector_count = payload.get("vector_count")
                added = payload.get("added")
                if vector_count is not None:
                    st.session_state.vector_count = vector_count
                    ve = st.session_state.get("vector_events", [])
                    ve.insert(
                        0,
                        {
                            "time": done_time,
                            "added_count": added or 0,
                            "source": "regulation_upload",
                        },
                    )
                    st.session_state.vector_events = ve
                log = st.session_state.get("agent_activity_log", [])
                log.insert(
                    0,
                    {
                        "agent": "regulation",
                        "title": "uploaded regulation analysis",
                        "content": result,
                        "time": done_time,
                    },
                )
                st.session_state.agent_activity_log = log
                with _background_lock:
                    del _background_results[task_id]
            elif payload.get("status") == "failed":
                err = payload.get("error")
                err_time = payload.get("updated_at")
                statuses = st.session_state.get("agent_statuses", {})
                statuses["regulation_agent"] = {
                    "status": "failed",
                    "updated_at": err_time,
                    "detail": f"분석 실패: {err}",
                }
                st.session_state.agent_statuses = statuses
                with _background_lock:
                    del _background_results[task_id]
    except Exception:
        pass

    latest_question = st.session_state.get("latest_strategy_question")
    last_strategy_time = st.session_state.get("last_strategy_time")
    last_log_ingest_time = parse_status_time(
        st.session_state.get("last_log_ingest_time")
    )
    if latest_question:
        st.caption(
            f"최근 질문: {latest_question} | 마지막 실행: {format_status_time(last_strategy_time)}"
        )

    status_map = {
        "pending": ("대기", "#e2e8f0", "#334155"),
        "running": ("실행 중", "#dbeafe", "#1d4ed8"),
        "completed": ("완료", "#dcfce7", "#166534"),
        "failed": ("실패", "#fee2e2", "#991b1b"),
    }
    display_names = [
        ("orchestrator", "Orchestrator"),
        ("log_agent", "Log Agent"),
        ("news_agent", "News Agent"),
        ("regulation_agent", "Regulation Agent"),
        ("decision_agent", "Decision Agent"),
        ("vector_store", "Vector Store"),
    ]
    statuses = st.session_state.get("agent_statuses", {})
    last_news_time = parse_status_time(st.session_state.get("last_news_time"))
    last_new_item_time = parse_status_time(st.session_state.get("last_new_item_time"))
    has_fresh_news_cycle = (
        last_news_time is not None
        and last_new_item_time is not None
        and last_news_time == last_new_item_time
    )

    if has_fresh_news_cycle:
        st.markdown(
            """
            <div style="
                display:inline-flex;
                align-items:center;
                gap:10px;
                margin: 6px 0 14px 0;
                padding: 10px 14px;
                border-radius: 999px;
                background: linear-gradient(90deg, rgba(220,252,231,0.95), rgba(187,247,208,0.95));
                border: 1px solid rgba(34,197,94,0.22);
                box-shadow: 0 10px 20px rgba(34,197,94,0.10);
            ">
                <span style="
                    width: 10px;
                    height: 10px;
                    border-radius: 999px;
                    background: #16a34a;
                    box-shadow: 0 0 0 rgba(22,163,74,0.6);
                    animation: newsPulse 1.3s infinite;
                "></span>
                <span style="font-size:13px; font-weight:800; color:#166534;">신규 뉴스 유입 감지 · 뉴스 에이전트가 최신 브리핑을 반영했습니다</span>
            </div>
            <style>
            @keyframes newsPulse {
                0% { box-shadow: 0 0 0 0 rgba(22,163,74,0.60); }
                70% { box-shadow: 0 0 0 10px rgba(22,163,74,0.00); }
                100% { box-shadow: 0 0 0 0 rgba(22,163,74,0.00); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if (
        last_log_ingest_time is not None
        and (datetime.datetime.now() - last_log_ingest_time).total_seconds() <= 12
    ):
        st.markdown(
            """
            <div style="
                display:inline-flex;
                align-items:center;
                gap:10px;
                margin: 0 0 14px 10px;
                padding: 10px 14px;
                border-radius: 999px;
                background: linear-gradient(90deg, rgba(254,243,199,0.96), rgba(253,230,138,0.96));
                border: 1px solid rgba(245,158,11,0.22);
                box-shadow: 0 10px 20px rgba(245,158,11,0.10);
            ">
                <span style="
                    width: 10px;
                    height: 10px;
                    border-radius: 999px;
                    background: #d97706;
                    box-shadow: 0 0 0 rgba(217,119,6,0.6);
                    animation: logPulse 1.3s infinite;
                "></span>
                <span style="font-size:13px; font-weight:800; color:#92400e;">신규 테스트 로그 유입 감지 · 로그 에이전트가 최신 브리핑을 반영했습니다</span>
            </div>
            <style>
            @keyframes logPulse {
                0% { box-shadow: 0 0 0 0 rgba(217,119,6,0.60); }
                70% { box-shadow: 0 0 0 10px rgba(217,119,6,0.00); }
                100% { box-shadow: 0 0 0 0 rgba(217,119,6,0.00); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    agent_cols = st.columns(3)
    for index, (agent_key, title) in enumerate(display_names):
        info = statuses.get(agent_key, {})
        status_code = info.get("status", "pending")
        label, background, color = status_map.get(
            status_code, (status_code, "#e2e8f0", "#334155")
        )
        updated_at = format_status_time(info.get("updated_at"))
        detail = info.get("detail", "아직 실행 이력이 없습니다.")
        agent_cols[index % 3].markdown(
            f"""
            <div style=\"
                min-height: 152px;
                border-radius: 16px;
                padding: 14px;
                margin-bottom: 12px;
                background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.98));
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            \">
                <div style=\"display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;\">
                    <div style=\"font-size:14px; font-weight:800; color:#0f172a;\">{title}</div>
                    <span style=\"padding:4px 8px; border-radius:999px; font-size:11px; font-weight:700; background:{background}; color:{color};\">{label}</span>
                </div>
                <div style=\"font-size:12px; color:#64748b; margin-bottom:8px;\">업데이트: {updated_at}</div>
                <div style=\"font-size:13px; line-height:1.55; color:#334155;\">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    briefing_text = st.session_state.get("latest_news_briefing")
    briefing_time = st.session_state.get("last_news_briefing_time")
    if briefing_text:
        st.markdown("#### 📰 최신 뉴스 에이전트 브리핑")
        st.markdown(
            f"""
            <div style="
                margin-bottom: 14px;
                padding: 16px 18px;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(236,254,255,0.98), rgba(240,249,255,0.98));
                border: 1px solid rgba(34,211,238,0.22);
                box-shadow: 0 12px 28px rgba(14,116,144,0.08);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; gap:10px;">
                    <div style="font-size:14px; font-weight:800; color:#0f172a;">시장 리스크 브리핑</div>
                    <div style="font-size:12px; color:#0f766e; font-weight:700;">업데이트: {format_status_time(briefing_time)}</div>
                </div>
                <div style="font-size:13px; line-height:1.7; color:#334155; white-space:pre-wrap;">{briefing_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    log_briefing_text = st.session_state.get("latest_log_briefing")
    log_briefing_time = st.session_state.get("last_log_briefing_time")
    if log_briefing_text:
        st.markdown("#### 📄 최신 로그 에이전트 브리핑")
        st.markdown(
            f"""
            <div style="
                margin-bottom: 14px;
                padding: 16px 18px;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(255,251,235,0.98), rgba(254,243,199,0.98));
                border: 1px solid rgba(245,158,11,0.22);
                box-shadow: 0 12px 28px rgba(146,64,14,0.08);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; gap:10px;">
                    <div style="font-size:14px; font-weight:800; color:#0f172a;">로그 유입 리스크 브리핑</div>
                    <div style="font-size:12px; color:#92400e; font-weight:700;">업데이트: {format_status_time(log_briefing_time)}</div>
                </div>
                <div style="font-size:13px; line-height:1.7; color:#334155; white-space:pre-wrap;">{log_briefing_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 규제 에이전트 최신 결과 표시
        reg_text = st.session_state.get("latest_regulation_analysis")
        reg_time = st.session_state.get("last_regulation_time")
        if reg_text:
            st.markdown("#### ⚖️ 최신 규제 에이전트 분석")
            st.markdown(
                f"""
                <div style="
                    margin-bottom: 14px;
                    padding: 16px 18px;
                    border-radius: 18px;
                    background: linear-gradient(135deg, rgba(255,250,240,0.98), rgba(255,247,237,0.98));
                    border: 1px solid rgba(245,158,11,0.22);
                    box-shadow: 0 12px 28px rgba(146,64,14,0.06);
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; gap:10px;">
                        <div style="font-size:14px; font-weight:800; color:#0f172a;">규제 문서 분석 결과</div>
                        <div style="font-size:12px; color:#c2410c; font-weight:700;">업데이트: {format_status_time(reg_time)}</div>
                    </div>
                    <div style="font-size:13px; line-height:1.7; color:#334155; white-space:pre-wrap;">{reg_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### 🧠 벡터 DB 적재 현황")
    vector_events = st.session_state.get("vector_events", [])
    latest_vector_event = vector_events[0] if vector_events else {}
    vector_metric_cols = st.columns(3)
    vector_metric_cols[0].metric("현재 벡터 수", get_vector_count())
    vector_metric_cols[1].metric(
        "마지막 증감", latest_vector_event.get("added_count", 0)
    )
    vector_metric_cols[2].metric(
        "최근 적재 소스", latest_vector_event.get("source", "-")
    )

    event_col, vector_col = st.columns([1.2, 1])
    with event_col:
        st.markdown("#### 실행 타임라인")
        activity_log = st.session_state.get("agent_activity_log", [])
        if not activity_log:
            st.info("아직 기록된 에이전트 실행 이력이 없습니다.")
        for event in activity_log[:10]:
            st.markdown(
                f"""
                <div style=\"
                    border-left: 4px solid #38bdf8;
                    padding: 10px 12px;
                    margin-bottom: 10px;
                    background: rgba(248,250,252,0.95);
                    border-radius: 0 12px 12px 0;
                \">
                    <div style=\"font-size:12px; font-weight:700; color:#0f172a;\">{event.get('source', '-')} · {event.get('status', '-')}</div>
                    <div style=\"font-size:12px; color:#64748b; margin:4px 0 6px 0;\">{format_status_time(event.get('timestamp'))}</div>
                    <div style=\"font-size:13px; color:#334155; line-height:1.55;\">{event.get('detail', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with vector_col:
        st.markdown("#### 적재 이벤트")
        if not vector_events:
            st.info("아직 기록된 벡터 적재 이벤트가 없습니다.")
        else:
            chart_rows = []
            for event in reversed(vector_events[:20]):
                timestamp = parse_status_time(event.get("timestamp"))
                chart_rows.append(
                    {
                        "time": (
                            timestamp
                            if timestamp is not None
                            else event.get("timestamp")
                        ),
                        "after_count": event.get("after_count", 0),
                        "added_count": event.get("added_count", 0),
                        "source": event.get("source", "-"),
                    }
                )

            df_vector = pd.DataFrame(chart_rows)
            if not df_vector.empty:
                fig_vector = px.line(
                    df_vector,
                    x="time",
                    y="after_count",
                    markers=True,
                    color="source",
                )
                fig_vector.update_layout(
                    height=260,
                    margin=dict(l=16, r=16, t=20, b=16),
                    legend_title_text="소스",
                    xaxis_title="시간",
                    yaxis_title="누적 벡터 수",
                )
                st.plotly_chart(
                    fig_vector, width="stretch", key="vector_event_timeline"
                )

                fig_delta = px.bar(
                    df_vector,
                    x="time",
                    y="added_count",
                    color="source",
                )
                fig_delta.update_layout(
                    height=180,
                    margin=dict(l=16, r=16, t=16, b=16),
                    showlegend=False,
                    xaxis_title="시간",
                    yaxis_title="추가량",
                )
                st.plotly_chart(fig_delta, width="stretch", key="vector_event_delta")

            st.markdown("##### 최근 적재 내역")
            for event in vector_events[:5]:
                st.markdown(
                    f"""
                    <div style="
                        padding: 12px 14px;
                        margin-bottom: 10px;
                        border-radius: 14px;
                        background: linear-gradient(180deg, rgba(239,246,255,0.96), rgba(224,242,254,0.92));
                        border: 1px solid rgba(56, 189, 248, 0.2);
                    ">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <div style="font-size:12px; font-weight:800; color:#0f172a;">{event.get('source', '-')} · {event.get('action', '-')}</div>
                            <div style="font-size:11px; color:#0369a1; font-weight:700;">{event.get('before_count', 0)} → {event.get('after_count', 0)}</div>
                        </div>
                        <div style="font-size:12px; color:#0f766e; font-weight:700; margin-bottom:6px;">증감: {event.get('added_count', 0)}</div>
                        <div style="font-size:13px; color:#334155; line-height:1.55; margin-bottom:6px;">{event.get('detail', '')}</div>
                        <div style="font-size:11px; color:#64748b;">{format_status_time(event.get('timestamp'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_agent_prompt_panel(
    agent_key: str, title: str, accent_color: str, soft_background: str
):
    prompt_input = st.session_state.get(f"latest_{agent_key}_prompt_input", {}) or {}
    updated_at = st.session_state.get(f"last_{agent_key}_prompt_input_time")

    st.subheader(title)
    if not prompt_input:
        st.info("아직 표시할 프롬프트 입력값이 없습니다.")
        return

    source = prompt_input.get("source", "-")
    user_input = prompt_input.get("user_input", "-")
    context_text = prompt_input.get("context", "관련 데이터가 없습니다.")
    prompt_text = prompt_input.get("prompt", "-")

    metric_cols = st.columns(3)
    metric_cols[0].metric("최근 갱신", format_status_time(updated_at))
    metric_cols[1].metric("실행 소스", source)
    metric_cols[2].metric("컨텍스트 길이", len(context_text))

    st.markdown(
        f"""
        <div style="
            margin: 10px 0 14px 0;
            padding: 16px 18px;
            border-radius: 18px;
            background: {soft_background};
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px;">
                <div style="font-size:14px; font-weight:800; color:#0f172a;">현재 프롬프트 입력 상태</div>
                <span style="padding:4px 10px; border-radius:999px; font-size:11px; font-weight:800; background:rgba(255,255,255,0.8); color:{accent_color}; border:1px solid rgba(15,23,42,0.08);">{html.escape(source)}</span>
            </div>
            <div style="font-size:12px; font-weight:700; color:{accent_color}; margin-bottom:6px;">사용자 지시 / 작업 문장</div>
            <div style="font-size:13px; line-height:1.65; color:#334155; white-space:pre-wrap;">{html.escape(user_input)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt_col, context_col = st.columns([1.1, 1])
    with prompt_col:
        st.markdown("#### 실제 프롬프트 본문")
        st.code(prompt_text, language="text")
    with context_col:
        st.markdown("#### 투입 컨텍스트")
        st.code(context_text, language="text")


def render_chart_dashboard():
    # 메인 차트 4개는 별도 탭으로 분리해서 필요할 때만 보게 합니다.
    if "results" not in st.session_state:
        st.info("차트에 표시할 분석 결과가 없습니다.")
        return

    st.subheader("📊 실시간 비동기 4차트 대시보드")
    _charts = get_chart_snapshots()

    top_left_chart, top_right_chart = st.columns(2)
    bottom_left_chart, bottom_right_chart = st.columns(2)

    with top_left_chart:
        st.markdown("#### 리스크 점수 추이")
        # trend data not used directly here

        def render_sidebar_news_cards():
            news_items = st.session_state.get("news", [])
            st.subheader("📰 실시간 뉴스 (최대 2개)")

            if not news_items:
                st.info("표시할 뉴스가 없습니다.")
                return

            latest_news_time = parse_status_time(st.session_state.get("last_news_time"))
            latest_new_item_time = parse_status_time(
                st.session_state.get("last_new_item_time")
            )
            has_fresh_news_cycle = (
                latest_news_time is not None
                and latest_new_item_time is not None
                and latest_news_time == latest_new_item_time
            )

            header_badge = "신규 유입" if has_fresh_news_cycle else "동기화 완료"
            header_background = "#dcfce7" if has_fresh_news_cycle else "#e0f2fe"
            header_color = "#166534" if has_fresh_news_cycle else "#075985"
            st.markdown(
                f"""
                <div style="margin-bottom: 12px;">
                    <span style="
                        display:inline-block;
                        padding:6px 10px;
                        border-radius:999px;
                        font-size:12px;
                        font-weight:800;
                        background:{header_background};
                        color:{header_color};
                        border:1px solid rgba(15,23,42,0.08);
                    ">{header_badge}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for index, news_item in enumerate(news_items[:2]):
                title = str(news_item.get("title", "")).strip() or "제목 없음"
                summary = str(news_item.get("summary", "")).strip()
                summary = summary.replace("<b>", "").replace("</b>", "")
                summary = summary.replace("<br>", " ").replace("<br/>", " ")
                preview = (
                    summary[:110] + ("..." if len(summary) > 110 else "")
                    if summary
                    else "요약 정보가 없습니다."
                )
                published = news_item.get("published") or st.session_state.get(
                    "last_news_time"
                )
                link = str(news_item.get("link", "")).strip()

                badge_label = (
                    "NEW" if has_fresh_news_cycle and index == 0 else f"#{index + 1}"
                )
                badge_background = "#16a34a" if badge_label == "NEW" else "#0f172a"
                safe_title = html.escape(title)
                safe_preview = html.escape(preview)

                card_html = f"""
                    <div style="
                        margin-bottom: 12px;
                        padding: 14px;
                        border-radius: 18px;
                        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
                        border: 1px solid rgba(148,163,184,0.18);
                        box-shadow: 0 10px 24px rgba(15,23,42,0.05);
                    ">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:8px;">
                            <div style="font-size:13px; font-weight:800; color:#0f172a; line-height:1.5;">{safe_title}</div>
                            <span style="flex-shrink:0; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:800; background:{badge_background}; color:white;">{badge_label}</span>
                        </div>
                        <div style="font-size:12px; color:#64748b; margin-bottom:8px;">{format_status_time(published)}</div>
                        <div style="font-size:13px; line-height:1.6; color:#334155; margin-bottom:10px;">{safe_preview}</div>
                    </div>
                """

                if link:
                    safe_link = html.escape(link)
                    wrapped = f'<a href="{safe_link}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:inherit;">{card_html}</a>'
                    st.markdown(wrapped, unsafe_allow_html=True)
                else:
                    st.markdown(card_html, unsafe_allow_html=True)

            # 나머지 항목은 접이식으로 제공
            remaining = news_items[2:5]
            if remaining:
                with st.expander(f"더보기 ({len(remaining)}건)"):
                    for i, news_item in enumerate(remaining, start=3):
                        title = str(news_item.get("title", "")).strip() or "제목 없음"
                        summary = str(news_item.get("summary", "")).strip()
                        summary = (
                            summary.replace("<b>", "")
                            .replace("</b>", "")
                            .replace("<br>", " ")
                            .replace("<br/>", " ")
                        )
                        preview = (
                            summary[:200] + ("..." if len(summary) > 200 else "")
                            if summary
                            else "요약 정보가 없습니다."
                        )
                        published = news_item.get("published") or st.session_state.get(
                            "last_news_time"
                        )
                        safe_title = html.escape(title)
                        safe_preview = html.escape(preview)
                        badge_label = f"#{i}"
                        badge_background = "#0f172a"

                        card_html = f"""
                            <div style="margin-bottom:12px; padding:14px; border-radius:18px; background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)); border:1px solid rgba(148,163,184,0.18); box-shadow:0 10px 24px rgba(15,23,42,0.05);">
                                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:8px;">
                                    <div style="font-size:13px; font-weight:800; color:#0f172a; line-height:1.5;">{safe_title}</div>
                                    <span style="flex-shrink:0; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:800; background:{badge_background}; color:white;">{badge_label}</span>
                                </div>
                                <div style="font-size:12px; color:#64748b; margin-bottom:8px;">{format_status_time(published)}</div>
                                <div style="font-size:13px; line-height:1.6; color:#334155; margin-bottom:10px;">{safe_preview}</div>
                            </div>
                            """

                        link = str(news_item.get("link", "")).strip()
                        if link:
                            wrapped = f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:inherit;">{card_html}</a>'
                            st.markdown(wrapped, unsafe_allow_html=True)
                        else:
                            st.markdown(card_html, unsafe_allow_html=True)


@fragment_decorator(run_every="3s")
def render_live_operations_fragment():
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
    except Exception:
        pass
    render_runtime_dashboard()


def render_sidebar_news_cards():
    news_items = st.session_state.get("news", [])
    st.subheader("📰 실시간 뉴스 (최대 2개)")

    if not news_items:
        st.info("표시할 뉴스가 없습니다.")
        return

    latest_news_time = parse_status_time(st.session_state.get("last_news_time"))
    latest_new_item_time = parse_status_time(st.session_state.get("last_new_item_time"))
    has_fresh_news_cycle = (
        latest_news_time is not None
        and latest_new_item_time is not None
        and latest_news_time == latest_new_item_time
    )

    header_badge = "신규 유입" if has_fresh_news_cycle else "동기화 완료"
    header_background = "#dcfce7" if has_fresh_news_cycle else "#e0f2fe"
    header_color = "#166534" if has_fresh_news_cycle else "#075985"
    st.markdown(
        f"""
        <div style=\"margin-bottom: 12px;\">
            <span style=\"
                display:inline-block;
                padding:6px 10px;
                border-radius:999px;
                font-size:12px;
                font-weight:800;
                background:{header_background};
                color:{header_color};
                border:1px solid rgba(15,23,42,0.08);
            \">{header_badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, news_item in enumerate(news_items[:2]):
        title = str(news_item.get("title", "")).strip() or "제목 없음"
        summary = str(news_item.get("summary", "")).strip()
        summary = summary.replace("<b>", "").replace("</b>", "")
        summary = summary.replace("<br>", " ").replace("<br/>", " ")
        preview = (
            summary[:110] + ("..." if len(summary) > 110 else "")
            if summary
            else "요약 정보가 없습니다."
        )
        published = news_item.get("published") or st.session_state.get("last_news_time")
        badge_label = "NEW" if has_fresh_news_cycle and index == 0 else f"#{index + 1}"
        badge_background = "#16a34a" if badge_label == "NEW" else "#0f172a"
        safe_title = html.escape(title)
        safe_preview = html.escape(preview)
        # 외부 링크는 표시하지 않음 (보안/프라이버시 이유)
        link_html = '<span style="font-size:12px; color:#94a3b8; font-weight:700;">원문 링크 생략</span>'

        st.markdown(
            f"""
            <div style=\"
                margin-bottom: 12px;
                padding: 14px;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
                border: 1px solid rgba(148,163,184,0.18);
                box-shadow: 0 10px 24px rgba(15,23,42,0.05);
            \">
                <div style=\"display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:8px;\">
                    <div style="font-size:13px; font-weight:800; color:#0f172a; line-height:1.5;">{safe_title}</div>
                    <span style=\"
                        flex-shrink:0;
                        padding:4px 8px;
                        border-radius:999px;
                        font-size:11px;
                        font-weight:800;
                        background:{badge_background};
                        color:white;
                    \">{badge_label}</span>
                </div>
                <div style=\"font-size:12px; color:#64748b; margin-bottom:8px;\">{format_status_time(published)}</div>
                <div style="font-size:13px; line-height:1.6; color:#334155; margin-bottom:10px;">{safe_preview}</div>
                <div>{link_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 나머지 항목은 접이식으로 제공
    remaining = news_items[2:5]
    if remaining:
        with st.expander(f"더보기 ({len(remaining)}건)"):
            for i, news_item in enumerate(remaining, start=3):
                title = str(news_item.get("title", "")).strip() or "제목 없음"
                summary = str(news_item.get("summary", "")).strip()
                summary = (
                    summary.replace("<b>", "")
                    .replace("</b>", "")
                    .replace("<br>", " ")
                    .replace("<br/>", " ")
                )
                preview = (
                    summary[:200] + ("..." if len(summary) > 200 else "")
                    if summary
                    else "요약 정보가 없습니다."
                )
                published = news_item.get("published") or st.session_state.get(
                    "last_news_time"
                )
                safe_title = html.escape(title)
                safe_preview = html.escape(preview)
                badge_label = f"#{i}"
                badge_background = "#0f172a"
                st.markdown(
                    f"""
                    <div style=\"margin-bottom:12px; padding:14px; border-radius:18px; background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)); border:1px solid rgba(148,163,184,0.18); box-shadow:0 10px 24px rgba(15,23,42,0.05);\">\n                        <div style=\"display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:8px;\">\n                            <div style=\"font-size:13px; font-weight:800; color:#0f172a; line-height:1.5;\">{safe_title}</div>\n                            <span style=\"flex-shrink:0; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:800; background:{badge_background}; color:white;\">{badge_label}</span>\n                        </div>\n                        <div style=\"font-size:12px; color:#64748b; margin-bottom:8px;\">{format_status_time(published)}</div>\n                        <div style=\"font-size:13px; line-height:1.6; color:#334155; margin-bottom:10px;\">{safe_preview}</div>\n                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_sidebar_news_compact():
    news_items = st.session_state.get("news", [])
    st.subheader("📰 실시간 뉴스 (최대 2개)")

    if not news_items:
        st.info("표시할 뉴스가 없습니다.")
        return

    latest_news_time = parse_status_time(st.session_state.get("last_news_time"))
    latest_new_item_time = parse_status_time(st.session_state.get("last_new_item_time"))
    has_fresh_news_cycle = (
        latest_news_time is not None
        and latest_new_item_time is not None
        and latest_news_time == latest_new_item_time
    )

    header_badge = "신규 유입" if has_fresh_news_cycle else "동기화 완료"
    header_background = "#dcfce7" if has_fresh_news_cycle else "#e0f2fe"
    header_color = "#166534" if has_fresh_news_cycle else "#075985"
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <span style="
                display:inline-block;
                padding:6px 10px;
                border-radius:999px;
                font-size:12px;
                font-weight:800;
                background:{header_background};
                color:{header_color};
                border:1px solid rgba(15,23,42,0.08);
            ">{header_badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, news_item in enumerate(news_items[:2]):
        title = str(news_item.get("title", "")).strip() or "제목 없음"
        link = str(news_item.get("link", "")).strip()
        badge_label = "NEW" if has_fresh_news_cycle and index == 0 else f"#{index + 1}"
        badge_background = "#16a34a" if badge_label == "NEW" else "#0f172a"
        safe_title = html.escape(title)

        card_html = f"""
            <div style="margin-bottom:12px; padding:10px 12px; border-radius:14px; background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)); border:1px solid rgba(148,163,184,0.12);">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                    <div style="font-size:13px; font-weight:800; color:#0f172a;">{safe_title}</div>
                    <span style="flex-shrink:0; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:800; background:{badge_background}; color:white;">{badge_label}</span>
                </div>
            </div>
        """

        if link:
            wrapped = f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:inherit;">{card_html}</a>'
            st.markdown(wrapped, unsafe_allow_html=True)
        else:
            st.markdown(card_html, unsafe_allow_html=True)

    remaining = news_items[2:5]
    if remaining:
        with st.expander(f"더보기 ({len(remaining)}건)"):
            for i, news_item in enumerate(remaining, start=3):
                title = str(news_item.get("title", "")).strip() or "제목 없음"
                link = str(news_item.get("link", "")).strip()
                safe_title = html.escape(title)
                badge_label = f"#{i}"
                badge_background = "#0f172a"
                card_html = f"""
                    <div style="margin-bottom:12px; padding:10px 12px; border-radius:14px; background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)); border:1px solid rgba(148,163,184,0.12);">
                        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                            <div style="font-size:13px; font-weight:800; color:#0f172a;">{safe_title}</div>
                            <span style="flex-shrink:0; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:800; background:{badge_background}; color:white;">{badge_label}</span>
                        </div>
                    </div>
                """
                if link:
                    wrapped = f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:inherit;">{card_html}</a>'
                    st.markdown(wrapped, unsafe_allow_html=True)
                else:
                    st.markdown(card_html, unsafe_allow_html=True)

    # 사이드바 하단: 규제 문서 업로드
    st.markdown(
        "#### 규제 문서 업로드 — 금감원/여신협회 문서를 업로드하면 규제 에이전트가 분석합니다."
    )
    uploaded = st.file_uploader(
        "규제 문서 업로드 (PDF/TXT/MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="sidebar_reg_upload",
    )
    if uploaded:
        if st.button("규제 문서 분석 실행", key="sidebar_reg_run"):
            # 메인 스레드에서 파일 바이트를 읽고 상태를 'running'으로 표시
            files_data = []
            for f in uploaded:
                try:
                    raw = f.read()
                except Exception:
                    raw = b""
                files_data.append((getattr(f, "name", "unknown"), raw))

            now = datetime.datetime.now().isoformat()
            statuses = st.session_state.get("agent_statuses", {})
            statuses["regulation_agent"] = {
                "status": "running",
                "updated_at": now,
                "detail": "규제 문서 분석 실행 중...",
            }
            st.session_state.agent_statuses = statuses

            def _run_and_store(files_data, task_id):
                try:
                    # 1) FAISS에 파일 청킹/임베딩해서 저장
                    before = get_vector_count()
                    new_count = ingest_files(files_data, doc_type="regulation")
                    added = new_count - before

                    # 2) FAISS에서 업로드한 문서와 관련된 규제 문맥을 검색
                    #    질의는 간단히 '규제'로 하되, 필요하면 파일명 기반으로 확장할 수 있습니다.
                    query = "규제"
                    logs_found, news_found, rules_found = search_context(query, k=6)
                    rule_context = "\n\n".join(rules_found)

                    # 3) 규제 에이전트를 호출 (검색된 규제 문맥을 전달)
                    result = regulation_agent(
                        rule_context, "", "업로드된 규제 문서 분석 및 요약을 작성하라"
                    )

                    done_time = datetime.datetime.now().isoformat()
                    with _background_lock:
                        _background_results[task_id] = {
                            "status": "completed",
                            "updated_at": done_time,
                            "result": result,
                            "vector_count": new_count,
                            "added": added,
                        }
                except Exception as e:
                    err_time = datetime.datetime.now().isoformat()
                    with _background_lock:
                        _background_results[task_id] = {
                            "status": "failed",
                            "updated_at": err_time,
                            "error": str(e),
                        }

            # 고유 task id 생성
            task_id = f"reg_{int(time.time() * 1000)}"
            thread = threading.Thread(
                target=_run_and_store, args=(files_data, task_id), daemon=True
            )
            thread.start()
            st.success(
                "규제 문서 분석을 백그라운드에서 시작했습니다. 상태를 대시보드에서 확인하세요."
            )


def render_faiss_tab():
    """FAISS 상태와 최근 적재 이벤트, 간단 검색 UI를 제공한다."""
    st.header("🧠 FAISS 벡터 DB 현황")

    try:
        status = get_backend_client().get_status()
    except Exception:
        st.error("백엔드에 연결할 수 없습니다.")
        return

    vector_count = status.get("vector_count", 0)
    last_ingest = status.get("last_log_ingest_time") or status.get("last_run_time")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric("벡터 수", vector_count)
        if last_ingest:
            st.caption(f"마지막 적재: {last_ingest}")
    with c2:
        if st.button("새로고침 FAISS 상태"):
            try:
                _ = get_backend_client().get_status()
                st.experimental_rerun()
            except Exception:
                st.error("새로고침 실패")

    st.markdown("---")

    st.subheader("최근 벡터 이벤트")
    events = status.get("vector_events") or []
    if not events:
        st.info("최근 벡터 이벤트가 없습니다.")
    else:
        for ev in events:
            label = f"{ev.get('timestamp')} — {ev.get('source')} ({ev.get('action')})"
            with st.expander(label, expanded=False):
                st.write({
                    "before_count": ev.get("before_count"),
                    "after_count": ev.get("after_count"),
                    "added": ev.get("added_count"),
                    "detail": ev.get("detail"),
                })

    st.markdown("---")

    st.subheader("실시간 FAISS 검색")
    q = st.text_input("검색어 입력 (예: 대출 한도)")
    k = st.number_input("k", min_value=1, max_value=20, value=5)
    if st.button("검색 실행") and q.strip():
        with st.spinner("검색 중..."):
            try:
                resp = get_backend_client().search_faiss(q, int(k))
                logs = resp.get("logs") or []
                news = resp.get("news") or []
                rules = resp.get("rules") or []

                if logs:
                    st.markdown("**Logs (유사 로그)**")
                    for item in logs:
                        st.write(item)

                if news:
                    st.markdown("**News (유사 뉴스)**")
                    for item in news:
                        st.write(item)

                if rules:
                    st.markdown("**Rules / 기타**")
                    for item in rules:
                        st.write(item)

                if not (logs or news or rules):
                    st.info("검색 결과가 없습니다.")
            except Exception as e:
                st.error(f"검색 실패: {e}")

    st.markdown("---")
    st.subheader("FAISS 저장된 벡터 항목 보기 / 내보내기")
    if st.button("목록 불러오기 (최대 200)"):
        try:
            resp = get_backend_client().get_faiss_entries(limit=200)
            items = resp.get("items", [])
            if items:
                import pandas as _pd

                df = _pd.DataFrame(items)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("CSV로 다운로드", csv, file_name="faiss_entries.csv", mime="text/csv")
            else:
                st.info("목록이 비어있습니다.")
        except Exception as e:
            st.error(f"목록 불러오기 실패: {e}")


@fragment_decorator(run_every="3s")
def render_live_news_fragment():
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
    except Exception:
        pass
    render_sidebar_news_compact()
    # (deprecated) original render kept for compatibility


@fragment_decorator(run_every="3s")
def render_live_news_prompt_fragment():
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
    except Exception:
        pass
    render_agent_prompt_panel(
        "news",
        "📰 뉴스 에이전트 프롬프트 입력값",
        "#0f766e",
        "linear-gradient(135deg, rgba(236,254,255,0.98), rgba(240,249,255,0.98))",
    )


@fragment_decorator(run_every="3s")
def render_live_log_prompt_fragment():
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
    except Exception:
        pass
    render_agent_prompt_panel(
        "log",
        "📄 로그 에이전트 프롬프트 입력값",
        "#92400e",
        "linear-gradient(135deg, rgba(255,251,235,0.98), rgba(254,243,199,0.98))",
    )


@fragment_decorator(run_every="5s")
def render_live_faiss_fragment():
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
    except Exception:
        pass
    render_faiss_tab()


def run_full_analysis(show_progress: bool = False, initial_load: bool = False):
    # 최초 진입 또는 수동 재실행 시 전체 분석을 백엔드에 요청합니다.
    # 실제 로그 파싱, 뉴스 수집, FAISS 생성은 모두 서버에서 처리됩니다.
    start = time.time()
    progress = st.progress(0) if show_progress else None
    status = st.empty()
    checklist_box = None
    skeleton_box = None
    summary_box = None

    if initial_load:
        render_loading_styles()
        loading_left, loading_right = st.columns([1.1, 1.4])
        with loading_left:
            checklist_box = st.empty()
            summary_box = st.empty()
        with loading_right:
            skeleton_box = st.empty()
        render_loading_checklist(
            checklist_box, active_step=0, eta_text="20~40초", elapsed_text="0초"
        )
        render_loading_skeleton(skeleton_box)

    if progress is not None:
        progress.progress(10)

    status.info("🔌 백엔드 연결 및 분석 요청 준비 중...")
    if checklist_box is not None:
        render_loading_checklist(
            checklist_box,
            active_step=1,
            eta_text="15~35초",
            elapsed_text=f"{int(time.time() - start)}초",
        )
    if progress is not None:
        progress.progress(40)

    status.info("🔍 로그 분석, 뉴스 수집, FAISS 생성을 백엔드에서 처리 중...")
    if checklist_box is not None:
        render_loading_checklist(
            checklist_box,
            active_step=2,
            eta_text="10~25초",
            elapsed_text=f"{int(time.time() - start)}초",
        )
    if progress is not None:
        progress.progress(70)

    status.info("🧠 결과 수신 및 화면 데이터 반영 중...")
    if checklist_box is not None:
        render_loading_checklist(
            checklist_box,
            active_step=3,
            eta_text="5~10초",
            elapsed_text=f"{int(time.time() - start)}초",
        )
    try:
        payload = get_backend_client().run_full_analysis(log_dir="data/logs")
    except Exception:
        payload = {}
    if progress is not None:
        progress.progress(100)

    if payload:
        sync_session_from_backend(payload)
        st.session_state.total_time = payload.get("total_time", time.time() - start)
        st.session_state.initial_analysis_done = True
        if checklist_box is not None:
            checklist_box.empty()
        if summary_box is not None:
            summary_box.empty()
        if skeleton_box is not None:
            skeleton_box.empty()
        status.empty()
    else:
        status.error("백엔드 호출에 실패했습니다. FastAPI 서버 상태를 확인하세요.")
        if summary_box is not None:
            summary_box.error(
                "초기 화면 준비에 실패했습니다. 백엔드 서버 상태 또는 포트를 확인하세요."
            )


if "initial_analysis_done" not in st.session_state:
    # 앱을 처음 열었을 때 한 번만 초기 데이터 준비를 수행합니다.
    startup_header = st.empty()
    startup_header.subheader("⏳ 초기 데이터 로딩")
    startup_status = st.empty()
    startup_status.info("백엔드 워커를 시작하고 초기 분석을 준비하고 있습니다...")
    try:
        get_backend_client().start_worker(interval_seconds=10)
        startup_status.info(
            "백엔드 워커 시작 완료. 로그/뉴스/FAISS 초기 분석을 진행합니다..."
        )
    except Exception:
        startup_status.warning(
            "백엔드 워커 시작에 실패했습니다. 초기 분석만 시도합니다."
        )
    run_full_analysis(show_progress=True, initial_load=True)
    startup_header.empty()
    startup_status.empty()


# -------------------------------
# 📊 레이아웃
# -------------------------------
col_left, col_main = st.columns([1, 3])

# -------------------------------
# 🧭 LEFT PANEL
# -------------------------------
with col_left:
    # 왼쪽 패널: 뉴스/상태 요약 (간단 호출로 교체)
    try:
        if HAS_FRAGMENT_REFRESH:
            render_live_news_fragment()
        else:
            render_sidebar_news_compact()
    except Exception:
        # 예외가 발생해도 사이드바 렌더링 실패만 처리
        st.warning("사이드바 뉴스를 불러오는 중 문제가 발생했습니다.")


# -------------------------------
# 🧠 MAIN
# -------------------------------
with col_main:
    if not HAS_FRAGMENT_REFRESH:
        # fragment 미지원 환경에서는 왼쪽 뉴스 패널이 상태 동기화만 사용하므로 메인에서 한 번 가져옵니다.
        if "last_news_time" not in st.session_state:
            st.session_state.last_news_time = None

        now = datetime.datetime.now()
        last_news_at = parse_status_time(st.session_state.get("last_news_time"))

        if "news" in st.session_state and (
            st.session_state.last_news_time is None
            or (last_news_at is not None and (now - last_news_at).seconds > 10)
        ):
            try:
                payload = get_backend_client().get_status()
                sync_session_from_backend(payload)
                st.session_state.last_news_time = payload.get("last_news_time", now)
            except Exception:
                pass

    operations_tab, strategy_tab, news_prompt_tab, log_prompt_tab, charts_tab, faiss_tab = st.tabs(
        [
            "🤖 운영 현황",
            "💬 AI 심사 전략",
            "📰 뉴스 에이전트 입력",
            "📄 로그 에이전트 입력",
            "📊 차트",
            "🧠 FAISS",
        ]
    )

    with operations_tab:
        if HAS_FRAGMENT_REFRESH:
            render_live_operations_fragment()
        else:
            render_runtime_dashboard()

    with strategy_tab:
        st.header("💬 AI 심사 전략")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.chat_input("질문 입력 (예: 승인율 올리는 방법?)")

        if user_input:
            with st.spinner("🧠 전략 생성 중..."):
                try:
                    response_payload = get_backend_client().strategy_chat(user_input)
                    latest_status = get_backend_client().get_status()
                    sync_session_from_backend(latest_status)
                except Exception:
                    response_payload = {
                        "answer": "백엔드 전략 챗 호출에 실패했습니다.",
                        "sections": {},
                    }

            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("ai", response_payload))

        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").write(msg)
            else:
                with st.chat_message("assistant"):
                    render_strategy_response(msg)

    with news_prompt_tab:
        if HAS_FRAGMENT_REFRESH:
            render_live_news_prompt_fragment()
        else:
            render_agent_prompt_panel(
                "news",
                "📰 뉴스 에이전트 프롬프트 입력값",
                "#0f766e",
                "linear-gradient(135deg, rgba(236,254,255,0.98), rgba(240,249,255,0.98))",
            )

        st.info(
            "규제 문서 업로드는 왼쪽 사이드바 '실시간 뉴스 더보기' 아래로 이동했습니다."
        )

    with log_prompt_tab:
        if HAS_FRAGMENT_REFRESH:
            render_live_log_prompt_fragment()
        else:
            render_agent_prompt_panel(
                "log",
                "📄 로그 에이전트 프롬프트 입력값",
                "#92400e",
                "linear-gradient(135deg, rgba(255,251,235,0.98), rgba(254,243,199,0.98))",
            )

    with charts_tab:
        render_chart_dashboard()

    with faiss_tab:
        if HAS_FRAGMENT_REFRESH:
            render_live_faiss_fragment()
        else:
            render_faiss_tab()
            auto = st.checkbox("자동 새로고침 (5초)", key="faiss_auto_refresh")
            if auto:
                try:
                    time.sleep(5)
                    params = {"_autorefresh": int(time.time())}
                    try:
                        st.experimental_set_query_params(**params)
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
                except Exception:
                    pass

# ================================
# 백엔드 상태 동기화: 10초마다 갱신
# ================================
if "last_backend_sync_time" not in st.session_state:
    st.session_state.last_backend_sync_time = None

now = datetime.datetime.now()

# fragment 미지원 환경에서만 전체 상태를 주기 동기화합니다.
if not HAS_FRAGMENT_REFRESH:
    if (
        st.session_state.last_backend_sync_time is None
        or (now - st.session_state.last_backend_sync_time).total_seconds() >= 10
    ):
        try:
            status_payload = get_backend_client().get_status()
            sync_session_from_backend(status_payload)
            st.session_state.last_backend_sync_time = now
        except Exception:
            pass
