import datetime
import html
import os
import threading
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent.strategy_chat import regulation_agent
from backend.streamlit_client import BackendClient
from rag.vector_db import get_vector_count, ingest_files, search_context

# 백그라운드 작업 결과 저장소 (스레드 -> 메인 폴링으로 전달)
_background_results: dict = {}
_background_lock = threading.Lock()
_ws_snapshot_buffer: dict = {}
_ws_snapshot_lock = threading.Lock()

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


# Start a background WebSocket listener to receive FAISS updates from backend
def _start_faiss_ws():
    if st.session_state.get("faiss_ws_started"):
        return
    st.session_state.faiss_ws_started = True

    def _run_ws():
        try:
            import json
            try:
                from websocket import WebSocketApp
            except Exception:
                return

            while True:
                try:
                    base = st.session_state.get("backend_url", "http://127.0.0.1:18000")
                    if base.startswith("https://"):
                        ws_url = "wss://" + base[len("https://") :]
                    elif base.startswith("http://"):
                        ws_url = "ws://" + base[len("http://") :]
                    else:
                        ws_url = base
                    if not ws_url.endswith("/ws/faiss"):
                        ws_url = ws_url.rstrip("/") + "/ws/faiss"

                    def on_message(ws, message):
                        try:
                            payload = json.loads(message)
                        except Exception:
                            return
                        try:
                            ev = payload.get("event") or payload
                            snap = payload.get("snapshot") or {}
                            with _ws_snapshot_lock:
                                if snap:
                                    _ws_snapshot_buffer["snapshot"] = snap
                                if ev:
                                    _ws_snapshot_buffer["event"] = ev
                        except Exception:
                            pass

                    def on_error(ws, err):
                        return

                    def on_close(ws, code, reason):
                        return

                    def on_open(ws):
                        return

                    ws = WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
                    ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception:
                    pass

                time.sleep(3)
        except Exception:
            return

    thread = threading.Thread(target=_run_ws, daemon=True)
    thread.start()


# start websocket listener (non-fatal if websocket-client not installed)
try:
    if "faiss_ws_started" not in st.session_state:
        st.session_state.faiss_ws_started = False
    _start_faiss_ws()
except Exception:
    pass


def consume_ws_snapshot_buffer() -> bool:
    try:
        with _ws_snapshot_lock:
            if not _ws_snapshot_buffer:
                return False
            payload = dict(_ws_snapshot_buffer)
            _ws_snapshot_buffer.clear()

        snapshot = payload.get("snapshot") or {}
        event = payload.get("event") or {}
        if snapshot:
            sync_session_from_backend(snapshot)
        elif event:
            events = st.session_state.get("vector_events", []) or []
            events.insert(0, event)
            st.session_state.vector_events = events

        if event:
            st.session_state.faiss_toast = {
                "msg": f"FAISS 업데이트: {event.get('source','?')} {event.get('added_count',0)}건 추가",
                "ts": time.time(),
            }
            st.session_state.faiss_last_event_time = time.time()
        return True
    except Exception:
        return False

def render_dashboard_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');

        :root {
            --bg-0: #06131f;
            --bg-1: #0b2234;
            --bg-2: #113c52;
            --ink-0: #f7fbff;
            --ink-1: #d9ecfb;
            --ink-2: #88a8be;
            --accent-cyan: #61f4de;
            --accent-amber: #ffbf69;
            --accent-red: #ff6b6b;
            --panel-border: rgba(151, 196, 225, 0.16);
            --panel-bg: rgba(8, 26, 39, 0.82);
            --panel-bg-soft: rgba(10, 34, 50, 0.68);
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(97, 244, 222, 0.12), transparent 28%),
                radial-gradient(circle at 100% 10%, rgba(255, 191, 105, 0.11), transparent 26%),
                linear-gradient(180deg, #07131e 0%, #0a1d2d 45%, #081723 100%);
            color: var(--ink-0);
            font-family: 'IBM Plex Sans KR', sans-serif;
        }

        .stApp [data-testid="stHeader"] {
            background: rgba(7, 19, 30, 0.0);
        }

        .stApp [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6,19,31,0.96), rgba(9,28,43,0.94));
            border-right: 1px solid rgba(151, 196, 225, 0.12);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }

        h1, h2, h3 {
            color: var(--ink-0);
            font-family: 'Space Grotesk', 'IBM Plex Sans KR', sans-serif;
            letter-spacing: -0.02em;
        }

        .dashboard-hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 30px 30px 26px 30px;
            background:
                radial-gradient(circle at 16% 24%, rgba(97, 244, 222, 0.20), transparent 24%),
                radial-gradient(circle at 88% 18%, rgba(255, 191, 105, 0.16), transparent 20%),
                linear-gradient(135deg, rgba(9, 31, 46, 0.98), rgba(12, 49, 67, 0.94));
            border: 1px solid rgba(151, 196, 225, 0.18);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
            margin-bottom: 18px;
            animation: riseIn 0.8s ease-out both;
        }

        .dashboard-hero::after {
            content: '';
            position: absolute;
            inset: auto -15% -38% auto;
            width: 320px;
            height: 320px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(97, 244, 222, 0.10), transparent 62%);
            pointer-events: none;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: var(--ink-1);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 16px;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 700;
            line-height: 1.1;
            color: var(--ink-0);
            max-width: 760px;
            margin-bottom: 12px;
        }

        .hero-subtitle {
            max-width: 840px;
            font-size: 15px;
            line-height: 1.7;
            color: var(--ink-1);
            margin-bottom: 18px;
        }

        .hero-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }

        .hero-chip {
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(14px);
        }

        .hero-chip-label {
            font-size: 12px;
            color: var(--ink-2);
            margin-bottom: 6px;
            font-weight: 600;
        }

        .hero-chip-value {
            font-size: 24px;
            color: var(--ink-0);
            font-weight: 700;
            font-family: 'Space Grotesk', 'IBM Plex Sans KR', sans-serif;
        }

        .hero-chip-detail {
            margin-top: 6px;
            font-size: 12px;
            color: var(--ink-1);
        }

        .metric-card {
            position: relative;
            overflow: hidden;
            min-height: 144px;
            border-radius: 24px;
            padding: 18px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.20);
            animation: riseIn 0.8s ease-out both;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: -48px;
            right: -48px;
            width: 120px;
            height: 120px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255,255,255,0.14), transparent 70%);
        }

        .metric-tone-cyan { background: linear-gradient(180deg, rgba(8,33,43,0.95), rgba(8,27,40,0.92)); }
        .metric-tone-amber { background: linear-gradient(180deg, rgba(45,28,10,0.92), rgba(29,20,7,0.90)); }
        .metric-tone-red { background: linear-gradient(180deg, rgba(53,16,20,0.92), rgba(34,11,15,0.90)); }
        .metric-tone-blue { background: linear-gradient(180deg, rgba(12,24,52,0.92), rgba(8,18,35,0.90)); }

        .metric-eyebrow {
            font-size: 12px;
            color: var(--ink-2);
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 34px;
            color: var(--ink-0);
            font-weight: 700;
            line-height: 1;
            font-family: 'Space Grotesk', 'IBM Plex Sans KR', sans-serif;
        }

        .metric-detail {
            margin-top: 12px;
            font-size: 13px;
            line-height: 1.6;
            color: var(--ink-1);
        }

        .metric-pill {
            margin-top: 12px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            color: var(--ink-0);
            font-size: 11px;
            font-weight: 700;
        }

        .debate-hero {
            position: relative;
            overflow: hidden;
            border-radius: 26px;
            padding: 22px;
            background: linear-gradient(135deg, rgba(9,31,46,0.98), rgba(16,48,65,0.94));
            border: 1px solid rgba(97,244,222,0.16);
            box-shadow: 0 18px 44px rgba(0,0,0,0.22);
            margin-bottom: 16px;
        }

        .debate-hero::after {
            content: '';
            position: absolute;
            inset: auto -22px -42px auto;
            width: 180px;
            height: 180px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255,191,105,0.18), transparent 66%);
            pointer-events: none;
        }

        .debate-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(97,244,222,0.10);
            color: #61f4de;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .debate-title {
            font-size: 28px;
            font-weight: 800;
            color: #f7fbff;
            margin-bottom: 8px;
            line-height: 1.18;
        }

        .debate-subtitle {
            font-size: 14px;
            color: #d9ecfb;
            line-height: 1.7;
            max-width: 860px;
        }

        .debate-wave {
            display: flex;
            align-items: flex-end;
            gap: 6px;
            margin-top: 14px;
            height: 28px;
        }

        .debate-wave span {
            width: 8px;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(180deg, #61f4de, #ffbf69);
            animation: debateWave 1.2s ease-in-out infinite;
            transform-origin: bottom;
        }

        .debate-wave span:nth-child(2) { animation-delay: 0.15s; }
        .debate-wave span:nth-child(3) { animation-delay: 0.30s; }
        .debate-wave span:nth-child(4) { animation-delay: 0.45s; }
        .debate-wave span:nth-child(5) { animation-delay: 0.60s; }

        .reviewer-card {
            position: relative;
            min-height: 198px;
            padding: 18px 18px 56px 18px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88));
            border: 1px solid rgba(151,196,225,0.14);
            box-shadow: 0 14px 30px rgba(0,0,0,0.18);
            overflow: hidden;
            margin-bottom: 10px;
            transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
        }

        .reviewer-card.conservative.active {
            border-color: rgba(255,143,143,0.28);
            box-shadow: 0 18px 34px rgba(0,0,0,0.20), inset 0 0 0 1px rgba(255,143,143,0.10);
        }

        .reviewer-card.sales.active {
            border-color: rgba(97,244,222,0.26);
            box-shadow: 0 18px 34px rgba(0,0,0,0.20), inset 0 0 0 1px rgba(97,244,222,0.10);
        }

        .reviewer-card.product.active {
            border-color: rgba(255,191,105,0.28);
            box-shadow: 0 18px 34px rgba(0,0,0,0.20), inset 0 0 0 1px rgba(255,191,105,0.10);
        }

        .reviewer-avatar-wrap {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 12px;
        }

        .reviewer-avatar {
            position: relative;
            width: 82px;
            height: 92px;
            flex-shrink: 0;
            animation: reviewerFloat 3.2s ease-in-out infinite;
        }

        .reviewer-avatar-head {
            position: absolute;
            top: 10px;
            left: 20px;
            width: 42px;
            height: 46px;
            border-radius: 46% 46% 42% 42%;
            background: #ffd8b5;
            box-shadow: inset 0 -3px 0 rgba(0,0,0,0.06);
            z-index: 2;
        }

        .reviewer-avatar-body {
            position: absolute;
            left: 12px;
            top: 52px;
            width: 58px;
            height: 34px;
            border-radius: 18px 18px 12px 12px;
            z-index: 1;
        }

        .reviewer-avatar-hair {
            position: absolute;
            top: 4px;
            left: 16px;
            width: 50px;
            height: 24px;
            border-radius: 20px 20px 10px 10px;
            z-index: 3;
        }

        .reviewer-avatar-eye {
            position: absolute;
            top: 30px;
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #0f172a;
            z-index: 4;
        }

        .reviewer-avatar-eye.left { left: 31px; }
        .reviewer-avatar-eye.right { left: 45px; }

        .reviewer-avatar-mouth {
            position: absolute;
            left: 34px;
            top: 42px;
            width: 12px;
            height: 6px;
            border-bottom: 2px solid #7c2d12;
            border-radius: 0 0 14px 14px;
            z-index: 4;
            transform-origin: center top;
        }

        .reviewer-avatar.badge-speaking::after {
            content: '';
            position: absolute;
            right: 4px;
            top: 6px;
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: #61f4de;
            box-shadow: 0 0 0 rgba(97,244,222,0.42);
            animation: reviewerPulse 1.6s infinite;
        }

        .reviewer-avatar.badge-speaking::before {
            content: '';
            position: absolute;
            right: -6px;
            top: 18px;
            width: 18px;
            height: 26px;
            border-right: 3px solid rgba(97,244,222,0.65);
            border-radius: 0 14px 14px 0;
            filter: drop-shadow(0 0 6px rgba(97,244,222,0.28));
            animation: voiceWave 1.1s ease-in-out infinite;
        }

        .reviewer-avatar.conservative .reviewer-avatar-body {
            background: linear-gradient(180deg, #334155, #1e293b);
        }

        .reviewer-avatar.conservative .reviewer-avatar-hair {
            background: #1f2937;
        }

        .reviewer-avatar.conservative .reviewer-avatar-mouth {
            width: 10px;
            border-radius: 0;
            border-bottom-color: #7f1d1d;
        }

        .reviewer-avatar.conservative.active .reviewer-avatar-eye.left {
            transform: rotate(16deg) scaleY(0.92);
        }

        .reviewer-avatar.conservative.active .reviewer-avatar-eye.right {
            transform: rotate(-16deg) scaleY(0.92);
        }

        .reviewer-avatar.sales .reviewer-avatar-body {
            background: linear-gradient(180deg, #0f766e, #115e59);
        }

        .reviewer-avatar.sales .reviewer-avatar-hair {
            background: #111827;
        }

        .reviewer-avatar.sales .reviewer-avatar-mouth {
            width: 14px;
            left: 33px;
            border-bottom-color: #14532d;
        }

        .reviewer-avatar.sales.active .reviewer-avatar-mouth {
            animation-duration: 0.72s;
        }

        .reviewer-avatar.sales .reviewer-avatar-head::after {
            content: '';
            position: absolute;
            left: 4px;
            right: 4px;
            top: 18px;
            height: 8px;
            border: 2px solid rgba(15,23,42,0.78);
            border-top: 0;
            border-radius: 8px;
            opacity: 0.9;
        }

        .reviewer-avatar.product .reviewer-avatar-body {
            background: linear-gradient(180deg, #7c3aed, #5b21b6);
        }

        .reviewer-avatar.product .reviewer-avatar-hair {
            background: #312e81;
        }

        .reviewer-avatar.product .reviewer-avatar-mouth {
            width: 12px;
            border-bottom-color: #92400e;
        }

        .reviewer-avatar.product.active .reviewer-avatar-eye.left,
        .reviewer-avatar.product.active .reviewer-avatar-eye.right {
            transform: translateY(-1px) scale(1.08);
        }

        .reviewer-avatar.active .reviewer-avatar-mouth {
            animation: reviewerTalk 0.9s ease-in-out infinite;
        }

        .reviewer-avatar.active .reviewer-avatar-eye {
            animation: reviewerBlink 4.4s ease-in-out infinite;
        }

        .reviewer-meta {
            flex: 1;
            min-width: 0;
        }

        .reviewer-card.active {
            transform: translateY(-4px);
        }

        .reviewer-card::before {
            content: '';
            position: absolute;
            top: -24px;
            right: -24px;
            width: 110px;
            height: 110px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
        }

        .reviewer-card::after {
            content: '';
            position: absolute;
            inset: auto 18px 14px 18px;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
            opacity: 0.55;
        }

        .reviewer-card.conservative::after {
            background: linear-gradient(90deg, transparent, rgba(255,143,143,0.18), transparent);
        }

        .reviewer-card.sales::after {
            background: linear-gradient(90deg, transparent, rgba(97,244,222,0.18), transparent);
        }

        .reviewer-card.product::after {
            background: linear-gradient(90deg, transparent, rgba(255,191,105,0.18), transparent);
        }

        .reviewer-role {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #61f4de;
            margin-bottom: 8px;
        }

        .reviewer-name {
            font-size: 22px;
            font-weight: 800;
            color: #f7fbff;
            margin-bottom: 6px;
        }

        .reviewer-dept {
            font-size: 13px;
            color: #d9ecfb;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .reviewer-tone {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: #f7fbff;
            font-size: 11px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .reviewer-desc {
            font-size: 13px;
            line-height: 1.65;
            color: #d9ecfb;
            min-height: 64px;
        }

        .reviewer-select-note {
            margin-top: 12px;
            font-size: 12px;
            color: #8fb9d6;
            font-weight: 700;
        }

        [class*="st-key-edit_reviewer_prompt_"] {
            margin-top: -264px;
            margin-bottom: 12px;
            position: relative;
            z-index: 8;
            padding-right: 0;
            height: 254px;
        }

        [class*="st-key-edit_reviewer_prompt_"] button {
            position: relative;
            width: 100%;
            height: 254px;
            min-height: 254px;
            padding: 0;
            justify-content: flex-end;
            align-items: flex-end;
            background: linear-gradient(180deg, rgba(97,244,222,0.01), rgba(97,244,222,0.04));
            border: 1px solid transparent;
            border-radius: 22px;
            box-shadow: none;
            color: transparent;
            font-size: 0;
            transition: background 0.22s ease, border-color 0.22s ease, transform 0.22s ease;
            cursor: pointer;
        }

        [class*="st-key-edit_reviewer_prompt_"] button::after {
            content: '클릭해서 프롬프트 편집';
            position: absolute;
            right: 16px;
            bottom: 14px;
            padding: 9px 12px;
            border-radius: 999px;
            background: rgba(97,244,222,0.12);
            border: 1px solid rgba(97,244,222,0.22);
            box-shadow: 0 12px 24px rgba(0,0,0,0.16);
            color: #a8fff2;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.02em;
            opacity: 0;
            transform: translateY(8px);
            transition: opacity 0.22s ease, transform 0.22s ease, background 0.22s ease;
            pointer-events: none;
        }

        [class*="st-key-edit_reviewer_prompt_"] button:hover {
            background: linear-gradient(180deg, rgba(97,244,222,0.04), rgba(97,244,222,0.08));
            border: 1px solid rgba(97,244,222,0.20);
            transform: translateY(-2px);
            box-shadow: 0 18px 36px rgba(0,0,0,0.12), inset 0 0 0 1px rgba(97,244,222,0.06);
        }

        [class*="st-key-edit_reviewer_prompt_conservative"] button:hover {
            background: linear-gradient(180deg, rgba(255,143,143,0.05), rgba(255,143,143,0.10));
            border-color: rgba(255,143,143,0.24);
            box-shadow: 0 18px 36px rgba(0,0,0,0.12), 0 0 26px rgba(255,143,143,0.14), inset 0 0 0 1px rgba(255,143,143,0.06);
        }

        [class*="st-key-edit_reviewer_prompt_sales"] button:hover {
            background: linear-gradient(180deg, rgba(97,244,222,0.05), rgba(97,244,222,0.10));
            border-color: rgba(97,244,222,0.24);
            box-shadow: 0 18px 36px rgba(0,0,0,0.12), 0 0 26px rgba(97,244,222,0.14), inset 0 0 0 1px rgba(97,244,222,0.06);
        }

        [class*="st-key-edit_reviewer_prompt_product"] button:hover {
            background: linear-gradient(180deg, rgba(255,191,105,0.05), rgba(255,191,105,0.10));
            border-color: rgba(255,191,105,0.24);
            box-shadow: 0 18px 36px rgba(0,0,0,0.12), 0 0 26px rgba(255,191,105,0.14), inset 0 0 0 1px rgba(255,191,105,0.06);
        }

        [class*="st-key-edit_reviewer_prompt_"] button:hover::after,
        [class*="st-key-edit_reviewer_prompt_"] button:focus-visible::after {
            opacity: 1;
            transform: translateY(0);
        }

        [class*="st-key-edit_reviewer_prompt_"] button:hover::before,
        [class*="st-key-edit_reviewer_prompt_"] button:focus-visible::before {
            opacity: 1;
        }

        [class*="st-key-edit_reviewer_prompt_"] button::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 22px;
            background: radial-gradient(circle at top, rgba(255,255,255,0.10), transparent 54%), linear-gradient(180deg, transparent 35%, rgba(97,244,222,0.10) 100%);
            opacity: 0;
            transition: opacity 0.22s ease;
            pointer-events: none;
        }

        [class*="st-key-edit_reviewer_prompt_conservative"] button::before {
            background: radial-gradient(circle at top, rgba(255,255,255,0.10), transparent 54%), linear-gradient(180deg, transparent 35%, rgba(255,143,143,0.12) 100%);
        }

        [class*="st-key-edit_reviewer_prompt_sales"] button::before {
            background: radial-gradient(circle at top, rgba(255,255,255,0.10), transparent 54%), linear-gradient(180deg, transparent 35%, rgba(97,244,222,0.12) 100%);
        }

        [class*="st-key-edit_reviewer_prompt_product"] button::before {
            background: radial-gradient(circle at top, rgba(255,255,255,0.10), transparent 54%), linear-gradient(180deg, transparent 35%, rgba(255,191,105,0.12) 100%);
        }

        [class*="st-key-edit_reviewer_prompt_conservative"] button::after {
            background: rgba(255,143,143,0.14);
            border-color: rgba(255,143,143,0.22);
            color: #ffd3d3;
        }

        [class*="st-key-edit_reviewer_prompt_sales"] button::after {
            background: rgba(97,244,222,0.14);
            border-color: rgba(97,244,222,0.22);
            color: #a8fff2;
        }

        [class*="st-key-edit_reviewer_prompt_product"] button::after {
            background: rgba(255,191,105,0.14);
            border-color: rgba(255,191,105,0.22);
            color: #ffe0b3;
        }

        [class*="st-key-edit_reviewer_prompt_"] button:focus,
        [class*="st-key-edit_reviewer_prompt_"] button:focus-visible {
            box-shadow: none;
            outline: none;
            border: 1px solid rgba(97,244,222,0.26);
        }

        [class*="st-key-edit_reviewer_prompt_conservative"] button:focus,
        [class*="st-key-edit_reviewer_prompt_conservative"] button:focus-visible {
            border-color: rgba(255,143,143,0.28);
        }

        [class*="st-key-edit_reviewer_prompt_sales"] button:focus,
        [class*="st-key-edit_reviewer_prompt_sales"] button:focus-visible {
            border-color: rgba(97,244,222,0.28);
        }

        [class*="st-key-edit_reviewer_prompt_product"] button:focus,
        [class*="st-key-edit_reviewer_prompt_product"] button:focus-visible {
            border-color: rgba(255,191,105,0.28);
        }

        [class*="st-key-save_reviewer_prompt_"] button {
            border-radius: 14px;
            min-height: 44px;
            font-weight: 800;
            border: 1px solid transparent;
            box-shadow: 0 12px 24px rgba(0,0,0,0.16);
        }

        [class*="st-key-save_reviewer_prompt_conservative"] button {
            background: linear-gradient(135deg, rgba(255,143,143,0.94), rgba(220,38,38,0.94));
            border-color: rgba(255,143,143,0.28);
            color: #fff7f7;
        }

        [class*="st-key-save_reviewer_prompt_conservative"] button:hover {
            background: linear-gradient(135deg, rgba(255,164,164,0.96), rgba(239,68,68,0.96));
        }

        [class*="st-key-save_reviewer_prompt_sales"] button {
            background: linear-gradient(135deg, rgba(45,212,191,0.94), rgba(13,148,136,0.94));
            border-color: rgba(97,244,222,0.28);
            color: #f4fffe;
        }

        [class*="st-key-save_reviewer_prompt_sales"] button:hover {
            background: linear-gradient(135deg, rgba(94,234,212,0.96), rgba(20,184,166,0.96));
        }

        [class*="st-key-save_reviewer_prompt_product"] button {
            background: linear-gradient(135deg, rgba(255,191,105,0.96), rgba(217,119,6,0.96));
            border-color: rgba(255,191,105,0.30);
            color: #fffaf2;
        }

        [class*="st-key-save_reviewer_prompt_product"] button:hover {
            background: linear-gradient(135deg, rgba(253,224,71,0.96), rgba(245,158,11,0.96));
        }

        .prompt-panel {
            border-radius: 22px;
            padding: 18px;
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88));
            border: 1px solid rgba(151,196,225,0.14);
            box-shadow: 0 14px 30px rgba(0,0,0,0.18);
            margin-bottom: 14px;
        }

        .prompt-panel-title {
            font-size: 18px;
            font-weight: 800;
            color: #f7fbff;
            margin-bottom: 6px;
        }

        .dialog-reviewer-hero {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 14px;
        }

        .dialog-reviewer-avatar {
            position: relative;
            width: 70px;
            height: 78px;
            flex-shrink: 0;
        }

        .dialog-reviewer-avatar .reviewer-avatar-head {
            left: 16px;
        }

        .dialog-reviewer-avatar .reviewer-avatar-body {
            left: 9px;
        }

        .dialog-reviewer-avatar .reviewer-avatar-hair {
            left: 12px;
        }

        .dialog-reviewer-avatar .reviewer-avatar-eye.left { left: 27px; }
        .dialog-reviewer-avatar .reviewer-avatar-eye.right { left: 41px; }
        .dialog-reviewer-avatar .reviewer-avatar-mouth { left: 30px; }

        .dialog-reviewer-meta {
            flex: 1;
            min-width: 0;
        }

        .dialog-reviewer-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            font-size: 11px;
            font-weight: 800;
            color: #d9ecfb;
            margin-bottom: 10px;
        }

        .dialog-reviewer-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            margin-top: 8px;
        }

        .dialog-reviewer-badge.conservative {
            background: rgba(255,143,143,0.12);
            color: #ffb4b4;
        }

        .dialog-reviewer-badge.sales {
            background: rgba(97,244,222,0.12);
            color: #8ef8e9;
        }

        .dialog-reviewer-badge.product {
            background: rgba(255,191,105,0.12);
            color: #ffd08c;
        }

        .dialog-save-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 0.04em;
            animation: reviewerPulse 1.2s ease-in-out 2;
        }

        .dialog-save-status.conservative {
            background: rgba(255,143,143,0.14);
            color: #ffd3d3;
        }

        .dialog-save-status.sales {
            background: rgba(97,244,222,0.14);
            color: #a8fff2;
        }

        .dialog-save-status.product {
            background: rgba(255,191,105,0.14);
            color: #ffe0b3;
        }

        .prompt-panel-subtitle {
            font-size: 13px;
            line-height: 1.65;
            color: #d9ecfb;
            margin-bottom: 10px;
        }

        .selected-reviewer-stage {
            min-height: 242px;
        }

        .selected-reviewer-stage.conservative {
            border-color: rgba(255,143,143,0.18);
            background: linear-gradient(180deg, rgba(32,17,20,0.96), rgba(48,22,28,0.88));
        }

        .selected-reviewer-stage.sales {
            border-color: rgba(97,244,222,0.18);
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(8,52,50,0.88));
        }

        .selected-reviewer-stage.product {
            border-color: rgba(255,191,105,0.18);
            background: linear-gradient(180deg, rgba(29,16,48,0.94), rgba(55,28,79,0.88));
        }

        .selected-reviewer-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }

        .selected-reviewer-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(97,244,222,0.10);
            border: 1px solid rgba(97,244,222,0.16);
            color: #61f4de;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .selected-reviewer-chip.conservative {
            background: rgba(255,143,143,0.10);
            border-color: rgba(255,143,143,0.20);
            color: #ffb4b4;
        }

        .selected-reviewer-chip.sales {
            background: rgba(97,244,222,0.10);
            border-color: rgba(97,244,222,0.18);
            color: #61f4de;
        }

        .selected-reviewer-chip.product {
            background: rgba(255,191,105,0.10);
            border-color: rgba(255,191,105,0.18);
            color: #ffd08c;
        }

        .selected-reviewer-bubble {
            position: relative;
            padding: 16px 18px;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(97,244,222,0.14), rgba(255,191,105,0.16));
            border: 1px solid rgba(151,196,225,0.16);
            color: #f7fbff;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.65;
            margin-bottom: 14px;
            animation: bubbleIn 0.35s ease-out both;
            overflow: hidden;
        }

        .selected-reviewer-bubble.conservative {
            background: linear-gradient(135deg, rgba(255,143,143,0.18), rgba(127,29,29,0.12));
            border-color: rgba(255,143,143,0.20);
        }

        .selected-reviewer-bubble.sales {
            background: linear-gradient(135deg, rgba(97,244,222,0.15), rgba(12,74,65,0.15));
            border-color: rgba(97,244,222,0.18);
        }

        .selected-reviewer-bubble.product {
            background: linear-gradient(135deg, rgba(255,191,105,0.16), rgba(124,58,237,0.12));
            border-color: rgba(255,191,105,0.18);
        }

        .selected-reviewer-bubble::before {
            content: '';
            position: absolute;
            left: 22px;
            bottom: -10px;
            width: 18px;
            height: 18px;
            background: rgba(97,244,222,0.14);
            border-right: 1px solid rgba(151,196,225,0.16);
            border-bottom: 1px solid rgba(151,196,225,0.16);
            transform: rotate(45deg);
        }

        .selected-reviewer-bubble::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(100deg, transparent 15%, rgba(255,255,255,0.12) 50%, transparent 85%);
            transform: translateX(-120%);
            animation: speakingSweep 2.8s ease-in-out infinite;
            pointer-events: none;
        }

        .selected-reviewer-preview {
            margin-top: 18px;
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .selected-reviewer-preview.conservative {
            background: rgba(127,29,29,0.12);
            border-color: rgba(255,143,143,0.14);
        }

        .selected-reviewer-preview.sales {
            background: rgba(8,145,118,0.10);
            border-color: rgba(97,244,222,0.12);
        }

        .selected-reviewer-preview.product {
            background: rgba(124,58,237,0.10);
            border-color: rgba(255,191,105,0.14);
        }

        .selected-reviewer-preview-label {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8fb9d6;
            margin-bottom: 8px;
        }

        .selected-reviewer-preview-text {
            font-size: 13px;
            line-height: 1.7;
            color: #d9ecfb;
        }

        .debate-status {
            margin: 12px 0 16px 0;
            padding: 14px 16px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(9,31,46,0.96), rgba(16,48,65,0.92));
            border: 1px solid rgba(97,244,222,0.16);
        }

        .debate-status-title {
            font-size: 13px;
            font-weight: 800;
            color: #f7fbff;
            margin-bottom: 6px;
        }

        .debate-status-text {
            font-size: 12px;
            line-height: 1.6;
            color: #d9ecfb;
        }

        .debate-transcript {
            border-radius: 22px;
            padding: 18px;
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88));
            border: 1px solid rgba(151,196,225,0.14);
            box-shadow: 0 14px 30px rgba(0,0,0,0.18);
            min-height: 250px;
        }

        .debate-bubble {
            margin-bottom: 12px;
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            position: relative;
            animation: bubbleIn 0.35s ease-out both;
        }

        .debate-bubble::before {
            content: '';
            position: absolute;
            left: 18px;
            top: -8px;
            width: 16px;
            height: 16px;
            background: rgba(255,255,255,0.05);
            border-left: 1px solid rgba(255,255,255,0.08);
            border-top: 1px solid rgba(255,255,255,0.08);
            transform: rotate(45deg);
        }

        .debate-bubble-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }

        .debate-bubble-avatar {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .debate-bubble-mini-avatar {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
        }

        .debate-bubble-name {
            font-size: 13px;
            font-weight: 800;
            color: #f7fbff;
        }

        .debate-bubble-badge {
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            color: #06131f;
        }

        .debate-bubble-text {
            font-size: 13px;
            line-height: 1.68;
            color: #d9ecfb;
            white-space: pre-wrap;
        }

        .consensus-card {
            border-radius: 22px;
            padding: 18px;
            background: linear-gradient(135deg, rgba(13,45,62,0.98), rgba(18,68,84,0.94));
            border: 1px solid rgba(97,244,222,0.18);
            box-shadow: 0 16px 36px rgba(0,0,0,0.22);
        }

        .consensus-label {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #61f4de;
            margin-bottom: 8px;
        }

        .consensus-title {
            font-size: 22px;
            font-weight: 800;
            color: #f7fbff;
            margin-bottom: 8px;
        }

        .consensus-body {
            font-size: 13px;
            line-height: 1.7;
            color: #d9ecfb;
            white-space: pre-wrap;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: linear-gradient(180deg, rgba(8,26,39,0.78), rgba(10,34,50,0.66));
            border: 1px solid rgba(151, 196, 225, 0.14);
            border-radius: 20px;
            padding: 10px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 12px 28px rgba(0,0,0,0.16);
            margin-bottom: 14px;
            overflow-x: auto;
            scrollbar-width: thin;
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 54px;
            padding: 0 18px;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border: 1px solid rgba(151, 196, 225, 0.12);
            color: #d9ecfb;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: -0.01em;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(180deg, rgba(97,244,222,0.10), rgba(34,211,238,0.06));
            border-color: rgba(97,244,222,0.22);
            color: #f7fbff;
            transform: translateY(-1px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(13,45,62,0.98), rgba(18,68,84,0.94)) !important;
            border-color: rgba(97,244,222,0.28) !important;
            color: #f7fbff !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.18), inset 0 0 0 1px rgba(255,255,255,0.04);
        }

        .stTabs [aria-selected="true"] p,
        .stTabs [data-baseweb="tab"] p {
            font-size: 14px;
            font-weight: 800;
            color: inherit !important;
            margin: 0;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, #61f4de, #ffbf69) !important;
            height: 3px !important;
            border-radius: 999px !important;
        }

        .section-shell {
            border-radius: 26px;
            padding: 22px;
            background: var(--panel-bg-soft);
            border: 1px solid var(--panel-border);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.16);
            margin-bottom: 18px;
        }

        .section-shell-tight {
            border-radius: 22px;
            padding: 18px;
            background: var(--panel-bg-soft);
            border: 1px solid var(--panel-border);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.16);
            margin-bottom: 18px;
        }

        .section-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }

        .section-kicker {
            font-size: 11px;
            color: var(--accent-cyan);
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .section-title {
            font-size: 24px;
            color: var(--ink-0);
            font-weight: 700;
            font-family: 'Space Grotesk', 'IBM Plex Sans KR', sans-serif;
        }

        .section-detail {
            font-size: 13px;
            color: var(--ink-2);
            line-height: 1.6;
            max-width: 520px;
            word-break: keep-all;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            align-items: stretch;
        }

        .workflow-card {
            position: relative;
            padding: 16px;
            min-height: 0;
            border-radius: 22px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            overflow-wrap: anywhere;
        }

        .workflow-index {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 800;
            color: #06131f;
            background: var(--accent-cyan);
            margin-bottom: 16px;
            box-shadow: 0 0 0 rgba(97,244,222,0.40);
            animation: nodePulse 1.8s infinite;
        }

        .workflow-name {
            font-size: 16px;
            color: var(--ink-0);
            font-weight: 700;
            margin-bottom: 8px;
        }

        .workflow-state {
            display: inline-flex;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .workflow-text {
            font-size: 13px;
            line-height: 1.65;
            color: var(--ink-1);
            word-break: keep-all;
        }

        .insight-card {
            border-radius: 22px;
            padding: 18px;
            min-height: 206px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .telemetry-card {
            border-radius: 18px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            min-height: 116px;
            margin-bottom: 10px;
        }

        .telemetry-label {
            font-size: 11px;
            color: var(--ink-2);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        .telemetry-value {
            font-size: 24px;
            color: var(--ink-0);
            font-weight: 700;
            font-family: 'Space Grotesk', 'IBM Plex Sans KR', sans-serif;
            margin-bottom: 8px;
        }

        .telemetry-detail {
            font-size: 12px;
            color: var(--ink-1);
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .insight-label {
            font-size: 12px;
            color: var(--ink-2);
            font-weight: 700;
            margin-bottom: 8px;
        }

        .insight-title {
            font-size: 18px;
            color: var(--ink-0);
            font-weight: 700;
            margin-bottom: 10px;
        }

        .insight-body {
            font-size: 13px;
            line-height: 1.75;
            color: var(--ink-1);
            white-space: pre-wrap;
        }

        .event-stack {
            display: grid;
            gap: 10px;
        }

        .event-card {
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .event-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 6px;
        }

        .event-source {
            font-size: 12px;
            font-weight: 800;
            color: var(--ink-0);
        }

        .event-time {
            font-size: 11px;
            color: var(--ink-2);
        }

        .event-body {
            font-size: 13px;
            line-height: 1.6;
            color: var(--ink-1);
        }

        .node-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }

        .node-legend span {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 11px;
            color: var(--ink-1);
            font-weight: 700;
        }

        .node-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
        }

        .upload-shell {
            margin-top: 18px;
            margin-bottom: 0;
            padding: 16px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88));
            border: 1px solid rgba(97, 244, 222, 0.14);
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22);
        }

        .upload-shell-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 10px;
        }

        .upload-shell-copy {
            flex: 1;
            min-width: 0;
        }

        .upload-doc-cluster {
            display: flex;
            align-items: flex-end;
            gap: 10px;
            flex-shrink: 0;
        }

        .upload-doc-card {
            position: relative;
            width: 58px;
            height: 72px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(248,250,252,0.98), rgba(226,232,240,0.96));
            border: 1px solid rgba(255,255,255,0.45);
            box-shadow: 0 18px 28px rgba(0,0,0,0.18);
            animation: docFloat 3.4s ease-in-out infinite;
            overflow: hidden;
        }

        .upload-shell.completed .upload-doc-card {
            background: linear-gradient(180deg, rgba(236,253,245,0.98), rgba(209,250,229,0.96));
            border-color: rgba(110,231,183,0.55);
            box-shadow: 0 18px 28px rgba(0,0,0,0.18), 0 0 24px rgba(110,231,183,0.18);
        }

        .upload-shell.running .upload-doc-card {
            border-color: rgba(97,244,222,0.34);
            box-shadow: 0 18px 28px rgba(0,0,0,0.18), 0 0 20px rgba(97,244,222,0.12);
        }

        .upload-shell.running .upload-doc-card::before {
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(191,219,254,0.92));
        }

        .upload-shell.completed .upload-doc-card .upload-doc-label {
            color: #166534;
        }

        .upload-shell.completed .upload-doc-card .upload-doc-lines span {
            background: rgba(22,101,52,0.22);
        }

        .upload-doc-check {
            position: absolute;
            right: 8px;
            bottom: 8px;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #22c55e, #86efac);
            color: #052e16;
            font-size: 11px;
            font-weight: 900;
            box-shadow: 0 8px 16px rgba(34,197,94,0.25);
            opacity: 0;
            transform: scale(0.72);
            transition: opacity 0.28s ease, transform 0.28s ease;
        }

        .upload-doc-progress {
            position: absolute;
            left: 8px;
            bottom: 8px;
            padding: 4px 7px;
            border-radius: 999px;
            background: rgba(15,23,42,0.84);
            color: #c9fdf5;
            font-size: 9px;
            font-weight: 900;
            letter-spacing: 0.08em;
            opacity: 0;
            transform: translateY(4px);
            transition: opacity 0.28s ease, transform 0.28s ease;
        }

        .upload-doc-orbit {
            position: absolute;
            inset: -5px;
            border-radius: 20px;
            border: 1px dashed rgba(97,244,222,0.26);
            opacity: 0;
            pointer-events: none;
            transform: scale(0.94);
        }

        .upload-shell.running .upload-doc-progress {
            opacity: 1;
            transform: translateY(0);
        }

        .upload-shell.running .upload-doc-orbit {
            opacity: 1;
            animation: orbitRing 2.2s linear infinite;
        }

        .upload-shell.running .upload-doc-check {
            opacity: 0;
            transform: scale(0.72);
        }

        .upload-shell.running .upload-doc-card.pdf {
            animation: docFloat 3.0s ease-in-out infinite, docRunningPulse 1.45s ease-in-out infinite;
        }

        .upload-shell.running .upload-doc-card.word {
            animation: docFloat 3.0s ease-in-out infinite, docRunningPulse 1.45s ease-in-out infinite 0.24s;
        }

        .upload-shell.completed .upload-doc-card .upload-doc-check {
            opacity: 1;
            transform: scale(1);
        }

        .upload-shell.completed .upload-doc-card.pdf {
            animation: docFloat 3.4s ease-in-out infinite, docSuccessPulse 1.6s ease-in-out infinite;
        }

        .upload-shell.completed .upload-doc-card.word {
            animation: docFloat 3.4s ease-in-out infinite, docSuccessPulse 1.6s ease-in-out infinite 0.22s;
        }

        .upload-doc-card::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(203,213,225,0.95));
            clip-path: polygon(0 0, 100% 0, 100% 100%);
        }

        .upload-doc-card::after {
            content: '';
            position: absolute;
            left: 10px;
            right: 10px;
            top: 34px;
            height: 20px;
            border-radius: 10px;
            opacity: 0.32;
        }

        .upload-doc-card.pdf {
            --doc-rotate: -8deg;
        }

        .upload-doc-card.word {
            --doc-rotate: 7deg;
            margin-bottom: -8px;
            animation-delay: 0.45s;
        }

        .upload-doc-card.pdf::after {
            background: linear-gradient(90deg, rgba(239,68,68,0.95), rgba(248,113,113,0.65));
        }

        .upload-doc-card.word::after {
            background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(96,165,250,0.65));
        }

        .upload-doc-label {
            position: absolute;
            left: 10px;
            top: 12px;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 0.08em;
            color: #0f172a;
        }

        .upload-doc-lines {
            position: absolute;
            left: 10px;
            right: 10px;
            top: 24px;
            display: grid;
            gap: 5px;
        }

        .upload-doc-lines span {
            display: block;
            height: 4px;
            border-radius: 999px;
            background: rgba(148,163,184,0.55);
        }

        .upload-doc-lines span:nth-child(2) { width: 85%; }
        .upload-doc-lines span:nth-child(3) { width: 62%; }

        .upload-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(97, 244, 222, 0.10);
            color: #61f4de;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .upload-title {
            font-size: 18px;
            font-weight: 800;
            color: #f7fbff;
            margin-bottom: 8px;
            line-height: 1.35;
        }

        .upload-subtitle {
            font-size: 13px;
            line-height: 1.65;
            color: #d9ecfb;
            margin-bottom: 12px;
        }

        .upload-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
        }

        .upload-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            color: #d9ecfb;
            font-size: 11px;
            font-weight: 700;
        }

        .upload-selected-box {
            margin: 10px 0 12px 0;
            padding: 12px 14px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(15,46,36,0.92), rgba(12,36,30,0.90));
            border: 1px solid rgba(34,197,94,0.18);
        }

        .upload-selected-title {
            font-size: 12px;
            font-weight: 800;
            color: #86efac;
            margin-bottom: 8px;
        }

        .upload-selected-item {
            font-size: 12px;
            line-height: 1.55;
            color: #dcfce7;
            margin-bottom: 4px;
        }

        .upload-steps {
            display: grid;
            gap: 8px;
            margin-top: 12px;
        }

        .upload-step {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .upload-step.active {
            background: rgba(97,244,222,0.10);
            border-color: rgba(97,244,222,0.22);
        }

        .upload-step.done {
            background: rgba(110,231,183,0.10);
            border-color: rgba(110,231,183,0.22);
        }

        .upload-step-badge {
            width: 24px;
            height: 24px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 800;
            color: #06131f;
            background: #8fb9d6;
            flex-shrink: 0;
        }

        .upload-step.active .upload-step-badge { background: #61f4de; }
        .upload-step.done .upload-step-badge { background: #6ee7b7; }

        .upload-step-text {
            font-size: 12px;
            line-height: 1.5;
            color: #d9ecfb;
        }

        .upload-learning-box {
            position: relative;
            overflow: hidden;
            margin: 12px 0;
            padding: 14px 14px 12px 14px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(8,33,43,0.98), rgba(12,49,67,0.95));
            border: 1px solid rgba(97,244,222,0.18);
            box-shadow: 0 14px 34px rgba(8,33,43,0.22);
        }

        .upload-learning-box::after {
            content: '';
            position: absolute;
            inset: auto -30px -36px auto;
            width: 130px;
            height: 130px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(97,244,222,0.18), transparent 68%);
            pointer-events: none;
        }

        .upload-learning-head {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }

        .upload-learning-core {
            position: relative;
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: radial-gradient(circle, #61f4de 0%, #22d3ee 55%, rgba(34,211,238,0.25) 100%);
            box-shadow: 0 0 0 rgba(97,244,222,0.45);
            animation: uploadPulse 1.8s infinite;
            flex-shrink: 0;
        }

        .upload-learning-core::before,
        .upload-learning-core::after {
            content: '';
            position: absolute;
            inset: -8px;
            border-radius: 999px;
            border: 1px solid rgba(97,244,222,0.30);
            animation: orbitRing 2.4s linear infinite;
        }

        .upload-learning-core::after {
            inset: -14px;
            animation-duration: 3.1s;
            border-color: rgba(255,191,105,0.28);
        }

        .upload-learning-title {
            font-size: 14px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 4px;
        }

        .upload-learning-text {
            font-size: 12px;
            line-height: 1.6;
            color: #d9ecfb;
        }

        .upload-learning-bar {
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
            margin-top: 10px;
        }

        .upload-learning-bar > span {
            display: block;
            width: 42%;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #61f4de, #ffbf69, #61f4de);
            background-size: 200% 100%;
            animation: uploadBarMove 1.4s linear infinite;
        }

        .upload-status-box {
            margin: 10px 0 12px 0;
            padding: 12px 14px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(11,35,52,0.92), rgba(9,28,43,0.88));
            border: 1px solid rgba(56,189,248,0.18);
        }

        .upload-status-box.success {
            background: linear-gradient(180deg, rgba(13,45,35,0.92), rgba(11,34,28,0.88));
            border-color: rgba(34,197,94,0.18);
            box-shadow: 0 16px 30px rgba(5,46,22,0.18), inset 0 0 0 1px rgba(134,239,172,0.05);
            position: relative;
            overflow: hidden;
        }

        .upload-status-box.success::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(100deg, transparent 15%, rgba(255,255,255,0.10) 50%, transparent 85%);
            transform: translateX(-120%);
            animation: speakingSweep 3.4s ease-in-out infinite;
            pointer-events: none;
        }

        .upload-status-box.error {
            background: linear-gradient(180deg, rgba(56,18,22,0.92), rgba(40,14,18,0.88));
            border-color: rgba(239,68,68,0.18);
        }

        .upload-status-title {
            font-size: 12px;
            font-weight: 800;
            color: #f7fbff;
            margin-bottom: 6px;
        }

        .upload-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 10px;
            padding: 5px 9px;
            border-radius: 999px;
            background: rgba(134,239,172,0.12);
            color: #86efac;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .upload-status-summary {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(110,231,183,0.18);
            font-size: 12px;
            line-height: 1.65;
            color: #dcfce7;
        }

        .upload-status-detail {
            font-size: 12px;
            line-height: 1.55;
            color: #d9ecfb;
        }

        .stApp [data-testid="stFileUploader"] {
            border: 1px dashed rgba(14,165,233,0.28);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(9,28,43,0.92), rgba(8,26,39,0.96));
            padding: 8px;
            margin-bottom: 10px;
        }

        [class*="st-key-sidebar_reg_upload"] {
            margin-top: -2px;
            margin-bottom: 12px;
            padding: 0 16px 16px 16px;
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88));
            border-left: 1px solid rgba(97,244,222,0.14);
            border-right: 1px solid rgba(97,244,222,0.14);
            border-bottom: 1px solid rgba(97,244,222,0.14);
            border-radius: 0 0 20px 20px;
            box-shadow: 0 18px 38px rgba(0,0,0,0.22);
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] {
            margin-bottom: 0;
            padding: 12px;
            border-radius: 18px;
            border: 1px dashed rgba(97,244,222,0.28);
            background: linear-gradient(180deg, rgba(9,28,43,0.80), rgba(6,23,35,0.96));
            transition: border-color 0.22s ease, background 0.22s ease, box-shadow 0.22s ease;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] section {
            min-height: 132px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            background: radial-gradient(circle at top, rgba(97,244,222,0.08), transparent 58%);
            transition: background 0.22s ease;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] button {
            border-radius: 999px;
            min-height: 40px;
            padding: 0 16px;
            background: linear-gradient(135deg, rgba(97,244,222,0.16), rgba(255,191,105,0.16));
            border: 1px solid rgba(97,244,222,0.24);
            color: #f7fbff;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(0,0,0,0.18);
            opacity: 0;
            transform: translateY(10px);
            transition: opacity 0.22s ease, transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] button:hover {
            border-color: rgba(97,244,222,0.34);
            background: linear-gradient(135deg, rgba(97,244,222,0.22), rgba(255,191,105,0.22));
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:hover,
        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within {
            border-color: rgba(97,244,222,0.38);
            background: linear-gradient(180deg, rgba(10,33,49,0.86), rgba(7,26,39,0.98));
            box-shadow: 0 16px 32px rgba(0,0,0,0.18), inset 0 0 0 1px rgba(97,244,222,0.06);
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:hover section,
        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within section {
            background: radial-gradient(circle at top, rgba(97,244,222,0.14), transparent 58%);
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:hover button,
        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within button {
            opacity: 1;
            transform: translateY(0);
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]::after {
            content: 'Hover to reveal upload';
            position: absolute;
            top: 14px;
            right: 14px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: #9cc4df;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.04em;
            pointer-events: none;
            transition: opacity 0.22s ease, transform 0.22s ease;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:hover::after,
        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within::after {
            opacity: 0;
            transform: translateY(-4px);
        }

        [class*="st-key-sidebar_reg_upload"] small {
            color: #9cc4df !important;
            font-size: 12px !important;
        }

        .stApp [data-testid="stFileUploader"] section {
            padding: 0.45rem 0.35rem;
        }

        .stApp [data-testid="stFileUploader"] small,
        .stApp [data-testid="stFileUploader"] label {
            color: #d9ecfb !important;
        }

        [data-testid="stSidebar"] .stButton button {
            width: 100%;
            border-radius: 14px;
            min-height: 46px;
            font-weight: 800;
        }

        .status-running { background: rgba(97, 244, 222, 0.14); color: #61f4de; }
        .status-completed { background: rgba(52, 211, 153, 0.16); color: #6ee7b7; }
        .status-failed { background: rgba(255, 107, 107, 0.16); color: #ff8f8f; }
        .status-pending { background: rgba(191, 219, 254, 0.12); color: #bfdbfe; }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes nodePulse {
            0% { box-shadow: 0 0 0 0 rgba(97,244,222,0.35); }
            70% { box-shadow: 0 0 0 12px rgba(97,244,222,0); }
            100% { box-shadow: 0 0 0 0 rgba(97,244,222,0); }
        }

        @keyframes debateWave {
            0%, 100% { transform: scaleY(0.42); opacity: 0.55; }
            50% { transform: scaleY(1); opacity: 1; }
        }

        @keyframes reviewerFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
        }

        @keyframes reviewerPulse {
            0% { box-shadow: 0 0 0 0 rgba(97,244,222,0.42); }
            70% { box-shadow: 0 0 0 12px rgba(97,244,222,0); }
            100% { box-shadow: 0 0 0 0 rgba(97,244,222,0); }
        }

        @keyframes bubbleIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes reviewerTalk {
            0%, 100% { transform: scaleX(1) scaleY(1); }
            30% { transform: scaleX(1.18) scaleY(1.55); }
            60% { transform: scaleX(0.92) scaleY(0.72); }
        }

        @keyframes reviewerBlink {
            0%, 44%, 48%, 100% { transform: scaleY(1); opacity: 1; }
            46% { transform: scaleY(0.12); opacity: 0.65; }
        }

        @keyframes voiceWave {
            0%, 100% { opacity: 0.2; transform: scaleY(0.65); }
            50% { opacity: 1; transform: scaleY(1.05); }
        }

        @keyframes speakingSweep {
            0% { transform: translateX(-120%); opacity: 0; }
            20% { opacity: 1; }
            60% { opacity: 0.8; }
            100% { transform: translateX(130%); opacity: 0; }
        }

        @keyframes uploadPulse {
            0% { box-shadow: 0 0 0 0 rgba(97,244,222,0.35); transform: scale(1); }
            70% { box-shadow: 0 0 0 14px rgba(97,244,222,0); transform: scale(1.04); }
            100% { box-shadow: 0 0 0 0 rgba(97,244,222,0); transform: scale(1); }
        }

        @keyframes orbitRing {
            from { transform: rotate(0deg) scale(0.98); opacity: 0.85; }
            to { transform: rotate(360deg) scale(1.02); opacity: 0.35; }
        }

        @keyframes uploadBarMove {
            0% { transform: translateX(-35%); background-position: 0% 50%; }
            100% { transform: translateX(170%); background-position: 100% 50%; }
        }

        @keyframes docFloat {
            0%, 100% { transform: translateY(0) rotate(var(--doc-rotate, 0deg)); }
            50% { transform: translateY(-6px) rotate(var(--doc-rotate, 0deg)); }
        }

        @keyframes docSuccessPulse {
            0%, 100% { box-shadow: 0 18px 28px rgba(0,0,0,0.18), 0 0 0 0 rgba(110,231,183,0); }
            50% { box-shadow: 0 18px 28px rgba(0,0,0,0.18), 0 0 0 10px rgba(110,231,183,0), 0 0 28px rgba(110,231,183,0.26); }
        }

        @keyframes docRunningPulse {
            0%, 100% { box-shadow: 0 18px 28px rgba(0,0,0,0.18), 0 0 0 0 rgba(97,244,222,0); }
            50% { box-shadow: 0 18px 28px rgba(0,0,0,0.18), 0 0 0 8px rgba(97,244,222,0), 0 0 24px rgba(97,244,222,0.24); }
        }

        @media (max-width: 1200px) {
            .hero-strip {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }

            .hero-title {
                font-size: 28px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


render_dashboard_theme()
st.title("카드론 AI 인사이트 대시보드")


def render_initial_analysis_badge():
    done = bool(st.session_state.get("initial_analysis_done"))
    started = bool(st.session_state.get("initial_analysis_started"))
    failed = bool(st.session_state.get("initial_analysis_failed"))

    if done:
        label = "초기 분석 완료"
        background = "#dcfce7"
        color = "#166534"
        detail = "기본 로그, 뉴스, FAISS 상태가 준비되었습니다."
    elif failed:
        label = "초기 분석 지연"
        background = "#fee2e2"
        color = "#991b1b"
        detail = "백그라운드 분석이 지연되고 있습니다. 화면은 계속 사용할 수 있습니다."
    elif started:
        label = "초기 분석 진행 중"
        background = "#dbeafe"
        color = "#1d4ed8"
        detail = "화면은 먼저 표시되고, 초기 분석은 백그라운드에서 진행됩니다."
    else:
        label = "초기 준비 대기"
        background = "#e2e8f0"
        color = "#334155"
        detail = "백엔드 연결과 워커 준비를 확인하는 중입니다."

    st.markdown(
        f"""
        <div style="margin: 8px 0 14px 0;">
            <span style="display:inline-block; padding:6px 10px; border-radius:999px; background:{background}; color:{color}; font-size:12px; font-weight:800; border:1px solid rgba(15,23,42,0.08);">{label}</span>
            <span style="margin-left:10px; font-size:12px; color:#64748b;">{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_faiss_toast():
    toast = st.session_state.get("faiss_toast")
    if not toast:
        return
    try:
        age = time.time() - float(toast.get("ts", 0))
    except Exception:
        age = 9999
    # only show recent toasts (5s)
    if age > 5:
        return

    css = """
    <style>
    .faiss-toast {
      position: fixed;
      right: 20px;
      top: 20px;
      z-index: 9999;
      background: linear-gradient(90deg,#06b6d4,#3b82f6);
      color: white;
      padding: 12px 16px;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(15,23,42,0.2);
      font-weight:700;
      animation: faissToastIn 0.5s ease-out;
    }
    @keyframes faissToastIn {
      from { transform: translateY(-8px); opacity: 0 }
      to { transform: translateY(0); opacity: 1 }
    }
    </style>
    """

    st.markdown(css + f"<div class='faiss-toast'>{html.escape(str(toast.get('msg','')))}</div>", unsafe_allow_html=True)


# render toast (if any) early so it appears above content
try:
    render_faiss_toast()
except Exception:
    pass


def render_faiss_product_stats():
    st.subheader("FAISS 상품별 실시간 통계")
    # Try backend stats first, fall back to session snapshot if unavailable
    products = {}
    try:
        client = get_backend_client()
        resp = client.get_faiss_stats()
        if resp.get("status") == "ok":
            products = resp.get("products", {}) or {}
            if not products:
                st.info("FAISS에 저장된 벡터가 없어 통계를 생성할 수 없습니다.")
                return
        else:
            raise Exception("bad status")
    except Exception:
        # Build simple per-product counts from cached session snapshot
        try:
            items = st.session_state.get("full_faiss_items", []) or []
            for it in items:
                prod = it.get("product") or "UNKNOWN"
                prod_stats = products.setdefault(prod, {"count": 0})
                prod_stats["count"] = prod_stats.get("count", 0) + 1
        except Exception:
            items = []
        if not products:
            st.info("FAISS 통계 조회 실패(백엔드 연결 필요)")
            return

    # 카드형으로 표시 (fallback may only have counts)
    # compute highlight if recent event
    highlight = False
    try:
        last_ev = st.session_state.get("faiss_last_event_time")
        if last_ev and (time.time() - float(last_ev)) < 5:
            highlight = True
    except Exception:
        highlight = False

    cols = st.columns(max(1, len(products)))
    idx = 0
    for prod, s in products.items():
        col = cols[idx % len(cols)]
        idx += 1
        prod_name = prod
        if prod == "C6":
            prod_name = "신용대출"
        elif prod == "C9":
            prod_name = "카드론"
        elif prod == "C11":
            prod_name = "개인사업자대출"
        elif prod == "C12":
            prod_name = "대환대출"

        count = s.get("count", 0)
        avg_rate = s.get("avg_applied_rate")
        avg_limit = s.get("avg_available_amount")
        approval_rate = s.get("approval_rate")
        avg_kcb = s.get("avg_kcb_grade")
        avg_score = s.get("avg_credit_score")

        # card highlight style when recent update
        card_style = ""
        if highlight:
            card_style = "border: 2px solid rgba(59,130,246,0.35); box-shadow: 0 6px 22px rgba(59,130,246,0.06); padding:10px; border-radius:8px;"
            col.markdown(f"<div style=\"{card_style}\">", unsafe_allow_html=True)

        col.markdown(f"#### {prod_name} ({prod})")
        col.metric("벡터 수", count)
        col.markdown(f"- 평균 금리: **{(round(avg_rate,2) if avg_rate is not None else '-') }**")
        col.markdown(f"- 평균 한도: **{(int(avg_limit) if avg_limit is not None else '-') }원**")
        col.markdown(f"- 승인율(추정): **{(str(round(approval_rate*100,1))+'%' if approval_rate is not None else '-') }**")
        col.markdown(f"- 평균 KCB등급(숫자 또는 매핑): **{(round(avg_kcb,2) if avg_kcb is not None else '-') }**")
        col.markdown(f"- 평균 신용점수: **{(round(avg_score,1) if avg_score is not None else '-') }**")
        top_reasons = s.get("top_reject_reasons") or []
        if top_reasons:
            col.markdown("- 주요 탈락 사유(상위):")
            for r, c in top_reasons[:3]:
                col.markdown(f"  - {r} ({c}건)")

        if highlight:
            col.markdown("</div>", unsafe_allow_html=True)


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
                f"{persona['name']} 프롬프트 편집",
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
    base_url = st.session_state.get(
        "backend_url",
        os.environ.get("BACKEND_URL", "http://127.0.0.1:18000"),
    )
    return BackendClient(base_url)


def get_backend_health() -> dict:
    # Try configured URL first, then fallback to common dev port 8000.
    configured = st.session_state.get(
        "backend_url", os.environ.get("BACKEND_URL", None)
    )
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(["http://127.0.0.1:18000", "http://127.0.0.1:8000"])
    tried = []
    for base in candidates:
        if base in tried:
            continue
        tried.append(base)
        try:
            client = BackendClient(base)
            h = client.health()
            # if health OK, persist to session and return
            st.session_state.backend_url = base
            return h
        except Exception:
            continue
    return {"status": "down", "detail": "백엔드에 연결할 수 없습니다. 포트 18000 또는 8000에서 서버가 실행 중인지 확인하세요."}


def sync_session_from_backend(payload: dict):
    # 백엔드 응답을 Streamlit 세션 상태로 옮겨서 화면 어디서든 재사용합니다.
    st.session_state.results = payload.get("results", st.session_state.get("results", []))
    st.session_state.issues = payload.get("issues", st.session_state.get("issues", []))
    st.session_state.news = payload.get("news", st.session_state.get("news", []))
    st.session_state.file_count = payload.get("file_count", st.session_state.get("file_count", 0))
    st.session_state.vector_count = payload.get("vector_count", st.session_state.get("vector_count", 0))
    st.session_state.total_time = payload.get("total_time", st.session_state.get("total_time", 0.0))
    st.session_state.last_news_time = payload.get("last_news_time", st.session_state.get("last_news_time"))
    st.session_state.last_new_item_time = payload.get("last_new_item_time", st.session_state.get("last_new_item_time"))
    st.session_state.latest_strategy_question = payload.get("latest_strategy_question", st.session_state.get("latest_strategy_question"))
    st.session_state.last_strategy_time = payload.get("last_strategy_time", st.session_state.get("last_strategy_time"))
    st.session_state.last_log_ingest_time = payload.get("last_log_ingest_time", st.session_state.get("last_log_ingest_time"))
    st.session_state.latest_log_briefing = payload.get("latest_log_briefing", st.session_state.get("latest_log_briefing"))
    st.session_state.last_log_briefing_time = payload.get("last_log_briefing_time", st.session_state.get("last_log_briefing_time"))
    st.session_state.latest_log_prompt_input = payload.get(
        "latest_log_prompt_input", st.session_state.get("latest_log_prompt_input", {})
    )
    st.session_state.last_log_prompt_input_time = payload.get(
        "last_log_prompt_input_time", st.session_state.get("last_log_prompt_input_time")
    )
    st.session_state.latest_news_briefing = payload.get("latest_news_briefing", st.session_state.get("latest_news_briefing"))
    st.session_state.last_news_briefing_time = payload.get("last_news_briefing_time", st.session_state.get("last_news_briefing_time"))
    st.session_state.latest_news_prompt_input = payload.get(
        "latest_news_prompt_input", st.session_state.get("latest_news_prompt_input", {})
    )
    st.session_state.last_news_prompt_input_time = payload.get(
        "last_news_prompt_input_time", st.session_state.get("last_news_prompt_input_time")
    )
    st.session_state.agent_statuses = payload.get("agent_statuses", st.session_state.get("agent_statuses", {}))
    st.session_state.agent_activity_log = payload.get("agent_activity_log", st.session_state.get("agent_activity_log", []))
    st.session_state.vector_events = payload.get("vector_events", st.session_state.get("vector_events", []))
    incoming_faiss_items = payload.get("full_faiss_items")
    if incoming_faiss_items:
        st.session_state.full_faiss_items = incoming_faiss_items
    elif payload.get("vector_count", 0):
        st.session_state.full_faiss_items = st.session_state.get(
            "full_faiss_items", []
        )
    else:
        st.session_state.full_faiss_items = []
    st.session_state.news_crawl_running = payload.get("news_crawl_running", st.session_state.get("news_crawl_running", False))
    st.session_state.news_crawl_target_count = payload.get("news_crawl_target_count", st.session_state.get("news_crawl_target_count", 0))
    st.session_state.news_crawl_success_count = payload.get("news_crawl_success_count", st.session_state.get("news_crawl_success_count", 0))
    st.session_state.news_crawl_failure_count = payload.get("news_crawl_failure_count", st.session_state.get("news_crawl_failure_count", 0))
    st.session_state.last_news_crawl_time = payload.get("last_news_crawl_time", st.session_state.get("last_news_crawl_time"))
    st.session_state.last_news_crawl_error = payload.get("last_news_crawl_error", st.session_state.get("last_news_crawl_error"))
    has_bootstrap_data = bool(
        payload.get("results") or payload.get("news") or payload.get("vector_count")
    )
    if has_bootstrap_data:
        st.session_state.initial_analysis_done = True
        st.session_state.initial_analysis_failed = False


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


def get_agent_status_palette(status_code: str) -> tuple[str, str, str]:
    palette = {
        "running": ("실행 중", "#61f4de", "status-running"),
        "completed": ("완료", "#6ee7b7", "status-completed"),
        "failed": ("실패", "#ff8f8f", "status-failed"),
        "pending": ("대기", "#bfdbfe", "status-pending"),
    }
    return palette.get(status_code or "pending", (status_code or "unknown", "#bfdbfe", "status-pending"))


def get_relative_minutes(value) -> str:
    parsed = parse_status_time(value)
    if parsed is None:
        return "-"
    delta_seconds = max(0, int((datetime.datetime.now() - parsed).total_seconds()))
    if delta_seconds < 60:
        return f"{delta_seconds}초 전"
    minutes = delta_seconds // 60
    if minutes < 60:
        return f"{minutes}분 전"
    hours = minutes // 60
    return f"{hours}시간 전"


def get_latest_failure_summary() -> tuple[str, str]:
    statuses = st.session_state.get("agent_statuses", {}) or {}
    failed = []
    for agent_key, info in statuses.items():
        if (info or {}).get("status") == "failed":
            failed.append((agent_key, info))
    if not failed:
        return ("없음", "현재 실패 상태의 Agent가 없습니다.")
    failed.sort(key=lambda item: parse_status_time(item[1].get("updated_at")) or datetime.datetime.min, reverse=True)
    agent_key, info = failed[0]
    return (agent_key, str(info.get("detail") or "실패 상세가 없습니다.")[:140])


def build_agent_flow_telemetry() -> dict:
    statuses = st.session_state.get("agent_statuses", {}) or {}
    vector_events = st.session_state.get("vector_events", []) or []
    latest_vector = vector_events[0] if vector_events else {}
    latest_vector_added = int(latest_vector.get("added_count", 0) or 0)
    latest_vector_after = int(latest_vector.get("after_count", st.session_state.get("vector_count", 0)) or 0)
    latest_vector_time = get_relative_minutes(latest_vector.get("timestamp")) if latest_vector else "-"

    updated_times = [
        parse_status_time((info or {}).get("updated_at"))
        for info in statuses.values()
        if parse_status_time((info or {}).get("updated_at")) is not None
    ]
    freshest_update = max(updated_times) if updated_times else None
    oldest_update = min(updated_times) if updated_times else None
    freshness_label = get_relative_minutes(freshest_update.isoformat()) if freshest_update else "-"
    lag_label = get_relative_minutes(oldest_update.isoformat()) if oldest_update else "-"
    failure_agent, failure_detail = get_latest_failure_summary()

    return {
        "latest_vector_added": latest_vector_added,
        "latest_vector_after": latest_vector_after,
        "latest_vector_time": latest_vector_time,
        "freshness_label": freshness_label,
        "lag_label": lag_label,
        "failure_agent": failure_agent,
        "failure_detail": failure_detail,
        "results_count": len(st.session_state.get("results", []) or []),
        "news_count": len(st.session_state.get("news", []) or []),
    }


def build_overview_metrics() -> dict:
    agent_statuses = st.session_state.get("agent_statuses", {}) or {}
    running_agents = sum(
        1 for info in agent_statuses.values() if (info or {}).get("status") == "running"
    )
    failed_agents = sum(
        1 for info in agent_statuses.values() if (info or {}).get("status") == "failed"
    )
    completed_agents = sum(
        1 for info in agent_statuses.values() if (info or {}).get("status") == "completed"
    )
    return {
        "results_count": len(st.session_state.get("results", []) or []),
        "news_count": len(st.session_state.get("news", []) or []),
        "issues_count": len(st.session_state.get("issues", []) or []),
        "vector_count": int(st.session_state.get("vector_count", 0) or 0),
        "running_agents": running_agents,
        "failed_agents": failed_agents,
        "completed_agents": completed_agents,
        "vector_events": len(st.session_state.get("vector_events", []) or []),
        "activity_events": len(st.session_state.get("agent_activity_log", []) or []),
        "last_news_time": format_status_time(st.session_state.get("last_news_time")),
        "last_log_ingest_time": format_status_time(st.session_state.get("last_log_ingest_time")),
    }


def render_dashboard_metric_card(title: str, value: str, detail: str, pill: str, tone: str):
    st.markdown(
        f"""
        <div class="metric-card metric-tone-{tone}">
            <div class="metric-eyebrow">{html.escape(title)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-detail">{html.escape(detail)}</div>
            <div class="metric-pill">{html.escape(pill)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_hero(metrics: dict):
    latest_question = st.session_state.get("latest_strategy_question") or "최근 전략 질의 없음"
    hero_html = f"""
    <div class="dashboard-hero">
        <div class="hero-kicker">AI Review Control Tower</div>
        <div class="hero-title">실시간 심사 운영, Agent 협업, 벡터 적재 흐름을 한 화면에서 통합 모니터링합니다.</div>
        <div class="hero-subtitle">
            로그 유입, 뉴스 리스크, 규제 분석, 전략 질의, FAISS 적재 이벤트를 운영자 관점으로 재구성했습니다.
            신규 이상 징후와 병목 구간을 먼저 보이고, 상세 분석은 아래 패널에서 이어집니다.
        </div>
        <div class="hero-strip">
            <div class="hero-chip">
                <div class="hero-chip-label">현재 심사 케이스</div>
                <div class="hero-chip-value">{metrics['results_count']}</div>
                <div class="hero-chip-detail">최근 적재 시각 {html.escape(metrics['last_log_ingest_time'])}</div>
            </div>
            <div class="hero-chip">
                <div class="hero-chip-label">활성 Agent</div>
                <div class="hero-chip-value">{metrics['running_agents']}</div>
                <div class="hero-chip-detail">완료 {metrics['completed_agents']} · 실패 {metrics['failed_agents']}</div>
            </div>
            <div class="hero-chip">
                <div class="hero-chip-label">최근 전략 질문</div>
                <div class="hero-chip-value" style="font-size:18px; line-height:1.35;">{html.escape(str(latest_question)[:54])}</div>
                <div class="hero-chip-detail">뉴스 동기화 {html.escape(metrics['last_news_time'])}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def render_dashboard_workflow(metrics: dict):
    statuses = st.session_state.get("agent_statuses", {}) or {}
    workflow_items = [
        ("01", "신규 로그 수집", statuses.get("log_agent", {}).get("status", "pending"), "실시간 로그와 생성 로그를 수집하고 대표 케이스를 추립니다."),
        ("02", "뉴스 리스크 반영", statuses.get("news_agent", {}).get("status", "pending"), "시장 뉴스와 이슈 태그를 묶어 심사 영향도를 갱신합니다."),
        ("03", "규제/근거 결합", statuses.get("regulation_agent", {}).get("status", "pending"), "규제 문서와 검색 결과를 결합해 준수 여부를 해석합니다."),
        ("04", "심사 전략 생성", statuses.get("decision_agent", {}).get("status", "pending"), "전략 질의와 Agent 요약을 합쳐 최종 판단 초안을 만듭니다."),
        ("05", "FAISS 동기화", statuses.get("vector_store", {}).get("status", "pending"), f"누적 벡터 {metrics['vector_count']}건 · 최근 이벤트 {metrics['vector_events']}건"),
    ]

    cards = []
    for index, name, status_code, detail in workflow_items:
        label, _, css_class = get_agent_status_palette(status_code)
        cards.append(
            f"""
            <div class="workflow-card">
                <div class="workflow-index">{index}</div>
                <div class="workflow-name">{html.escape(name)}</div>
                <div class="workflow-state {css_class}">{html.escape(label)}</div>
                <div class="workflow-text">{html.escape(detail)}</div>
            </div>
            """
        )

    st.markdown(
        """
        <div class="section-shell">
            <div class="section-header">
                <div>
                    <div class="section-kicker">Process View</div>
                    <div class="section-title">심사 플로우 보드</div>
                </div>
                <div class="section-detail">신규 유입부터 전략 응답과 벡터 저장까지의 단계를 상태 배지와 함께 한 줄 플로우로 보여줍니다.</div>
            </div>
            <div class="workflow-grid">
        """
        + "".join(cards)
        + """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_agent_flow_figure() -> go.Figure:
    statuses = st.session_state.get("agent_statuses", {}) or {}
    vector_count = int(st.session_state.get("vector_count", 0) or 0)
    latest_question = st.session_state.get("latest_strategy_question") or "전략 질문 없음"
    latest_log = st.session_state.get("latest_log_briefing") or "로그 브리핑 없음"
    latest_news = st.session_state.get("latest_news_briefing") or "뉴스 브리핑 없음"
    telemetry = build_agent_flow_telemetry()

    def status_detail(agent_key: str, fallback: str) -> str:
        info = statuses.get(agent_key, {}) or {}
        parts = [fallback]
        if info.get("updated_at"):
            parts.append(f"갱신 {get_relative_minutes(info.get('updated_at'))}")
        if info.get("detail"):
            parts.append(str(info.get("detail"))[:90])
        return " | ".join(parts)

    nodes = [
        {"id": "source_logs", "label": "Logs", "x": 0.02, "y": 0.66, "status": "completed", "detail": f"유입 로그 {telemetry['results_count']}건 | 최신 처리 {get_relative_minutes(st.session_state.get('last_log_ingest_time'))}"},
        {"id": "source_news", "label": "News", "x": 0.02, "y": 0.24, "status": "completed", "detail": f"수집 뉴스 {telemetry['news_count']}건 | 최신 수집 {get_relative_minutes(st.session_state.get('last_news_time'))}"},
        {"id": "log_agent", "label": "Log Agent", "x": 0.27, "y": 0.76, "status": statuses.get("log_agent", {}).get("status", "pending"), "detail": status_detail("log_agent", str(latest_log)[:90])},
        {"id": "news_agent", "label": "News Agent", "x": 0.27, "y": 0.14, "status": statuses.get("news_agent", {}).get("status", "pending"), "detail": status_detail("news_agent", str(latest_news)[:90])},
        {"id": "regulation_agent", "label": "Regulation", "x": 0.51, "y": 0.46, "status": statuses.get("regulation_agent", {}).get("status", "pending"), "detail": status_detail("regulation_agent", "업로드 문서와 규제 문맥 통합")},
        {"id": "orchestrator", "label": "Orchestrator", "x": 0.72, "y": 0.46, "status": statuses.get("orchestrator", {}).get("status", "pending"), "detail": status_detail("orchestrator", str(latest_question)[:90])},
        {"id": "decision_agent", "label": "Decision", "x": 0.92, "y": 0.62, "status": statuses.get("decision_agent", {}).get("status", "pending"), "detail": status_detail("decision_agent", "최종 심사 응답과 전략 판단")},
        {"id": "vector_store", "label": "Vector DB", "x": 0.92, "y": 0.24, "status": statuses.get("vector_store", {}).get("status", "pending"), "detail": status_detail("vector_store", f"누적 {vector_count} vectors | 최근 +{telemetry['latest_vector_added']}")},
    ]
    edges = [
        ("source_logs", "log_agent"),
        ("source_news", "news_agent"),
        ("log_agent", "regulation_agent"),
        ("news_agent", "regulation_agent"),
        ("regulation_agent", "orchestrator"),
        ("orchestrator", "decision_agent"),
        ("orchestrator", "vector_store"),
        ("decision_agent", "vector_store"),
    ]
    lookup = {node["id"]: node for node in nodes}
    color_map = {
        "running": "#61f4de",
        "completed": "#6ee7b7",
        "failed": "#ff8f8f",
        "pending": "#8fb9d6",
    }

    fig = go.Figure()
    for start, end in edges:
        start_node = lookup[start]
        end_node = lookup[end]
        fig.add_trace(
            go.Scatter(
                x=[start_node["x"], end_node["x"]],
                y=[start_node["y"], end_node["y"]],
                mode="lines",
                line={"width": 2.5, "color": "rgba(151,196,225,0.38)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[node["x"] for node in nodes],
            y=[node["y"] for node in nodes],
            mode="markers+text",
            text=[node["label"] for node in nodes],
            textposition="bottom center",
            textfont={"color": "#e7f4ff", "size": 12, "family": "IBM Plex Sans KR"},
            marker={
                "size": [36, 36, 48, 48, 54, 58, 50, 50],
                "color": [color_map.get(node["status"], "#8fb9d6") for node in nodes],
                "line": {"width": 2, "color": "rgba(7,19,30,0.95)"},
                "symbol": ["diamond", "diamond", "circle", "circle", "hexagon", "hexagon", "square", "square"],
            },
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            customdata=[html.escape(node["detail"]) for node in nodes],
            showlegend=False,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 20, "b": 10},
        xaxis={"visible": False, "range": [-0.04, 1.02]},
        yaxis={"visible": False, "range": [0.0, 0.92]},
        height=360,
    )
    return fig


def render_agent_flow_section():
    telemetry = build_agent_flow_telemetry()
    st.markdown(
        """
        <div class="section-shell-tight">
            <div class="section-header">
                <div>
                    <div class="section-kicker">Graph View</div>
                    <div class="section-title">Agent 간 데이터 흐름 시각화</div>
                </div>
                <div class="section-detail">노드별 최근 갱신 시각, 실패 상태, 벡터 증감량을 함께 보여주는 운영 그래프입니다.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    graph_col, telemetry_col = st.columns([1.45, 0.85])
    with graph_col:
        st.plotly_chart(build_agent_flow_figure(), width="stretch", key="agent_flow_graph")
    with telemetry_col:
        st.markdown(
            f"""
            <div class="telemetry-card">
                <div class="telemetry-label">최근 벡터 적재</div>
                <div class="telemetry-value">+{telemetry['latest_vector_added']}</div>
                <div class="telemetry-detail">누적 {telemetry['latest_vector_after']}건\n업데이트 {telemetry['latest_vector_time']}</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">최신 Agent 갱신</div>
                <div class="telemetry-value">{html.escape(telemetry['freshness_label'])}</div>
                <div class="telemetry-detail">가장 오래된 상태는 {html.escape(telemetry['lag_label'])}\n오래된 상태면 병목 가능성이 있습니다.</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">최근 실패 요약</div>
                <div class="telemetry-value">{html.escape(telemetry['failure_agent'])}</div>
                <div class="telemetry-detail">{html.escape(telemetry['failure_detail'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="node-legend">
            <span><i class="node-dot" style="background:#61f4de"></i>실행 중</span>
            <span><i class="node-dot" style="background:#6ee7b7"></i>완료</span>
            <span><i class="node-dot" style="background:#ff8f8f"></i>실패</span>
            <span><i class="node-dot" style="background:#8fb9d6"></i>대기</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_charts():
    charts = get_chart_snapshots()
    grade_counts = ((charts.get("grade_distribution") or {}).get("grades") or {})
    vector_status = charts.get("vector_status") or {}
    trend = charts.get("score_trend") or {}
    labels = trend.get("labels") or []
    scores = trend.get("scores") or []

    shell_left, shell_right = st.columns([1.1, 1])
    with shell_left:
        st.markdown(
            """
            <div class="section-shell-tight">
                <div class="section-header">
                    <div>
                        <div class="section-kicker">Risk Radar</div>
                        <div class="section-title">심사 리스크 분포</div>
                    </div>
                    <div class="section-detail">등급 분포와 최근 점수 흐름으로 운영 강도를 빠르게 파악합니다.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chart_col_a, chart_col_b = st.columns(2)
        with chart_col_a:
            if grade_counts:
                fig_grade = px.pie(
                    names=list(grade_counts.keys()),
                    values=list(grade_counts.values()),
                    hole=0.58,
                    color_discrete_sequence=["#61f4de", "#ffbf69", "#8fb9d6", "#ff6b6b", "#6ee7b7"],
                )
                fig_grade.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e7f4ff"},
                    height=280,
                    margin={"l": 20, "r": 20, "t": 10, "b": 10},
                    legend={"orientation": "h", "y": -0.12},
                )
                st.plotly_chart(fig_grade, width="stretch", key="grade_distribution_showcase")
            else:
                st.info("등급 분포 데이터가 아직 없습니다.")
        with chart_col_b:
            if labels and scores:
                trimmed_labels = labels[-8:]
                trimmed_scores = scores[-8:]
                fig_trend = go.Figure(
                    data=[
                        go.Scatter(
                            x=trimmed_labels,
                            y=trimmed_scores,
                            mode="lines+markers",
                            line={"width": 3, "color": "#61f4de"},
                            marker={"size": 9, "color": "#ffbf69"},
                            fill="tozeroy",
                            fillcolor="rgba(97,244,222,0.10)",
                        )
                    ]
                )
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e7f4ff"},
                    height=280,
                    margin={"l": 20, "r": 20, "t": 10, "b": 20},
                    xaxis={"tickangle": -20, "gridcolor": "rgba(151,196,225,0.10)"},
                    yaxis={"gridcolor": "rgba(151,196,225,0.12)", "title": "리스크 점수"},
                )
                st.plotly_chart(fig_trend, width="stretch", key="risk_trend_showcase")
            else:
                st.info("점수 추이 데이터가 아직 없습니다.")

    with shell_right:
        st.markdown(
            """
            <div class="section-shell-tight">
                <div class="section-header">
                    <div>
                        <div class="section-kicker">Ops Pulse</div>
                        <div class="section-title">운영 볼륨과 이슈 밀도</div>
                    </div>
                    <div class="section-detail">벡터 적재, 뉴스 수집, 이슈 탐지를 한 묶음으로 비교합니다.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pulse_df = pd.DataFrame(
            [
                {"bucket": "Vectors", "value": int(vector_status.get("vector_count", st.session_state.get("vector_count", 0)) or 0)},
                {"bucket": "News", "value": int(vector_status.get("news_count", len(st.session_state.get("news", []) or [])) or 0)},
                {"bucket": "Issues", "value": int(vector_status.get("issues_count", len(st.session_state.get("issues", []) or [])) or 0)},
                {"bucket": "Events", "value": len(st.session_state.get("agent_activity_log", []) or [])},
            ]
        )
        fig_pulse = px.bar(
            pulse_df,
            x="bucket",
            y="value",
            color="bucket",
            color_discrete_map={
                "Vectors": "#61f4de",
                "News": "#8fb9d6",
                "Issues": "#ff6b6b",
                "Events": "#ffbf69",
            },
        )
        fig_pulse.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e7f4ff"},
            height=320,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            xaxis={"title": ""},
            yaxis={"gridcolor": "rgba(151,196,225,0.12)", "title": "건수"},
            showlegend=False,
        )
        st.plotly_chart(fig_pulse, width="stretch", key="ops_pulse_showcase")


def render_live_insight_sections():
    latest_log = st.session_state.get("latest_log_briefing") or "아직 로그 브리핑이 없습니다."
    latest_news = st.session_state.get("latest_news_briefing") or "아직 뉴스 브리핑이 없습니다."
    activity_log = st.session_state.get("agent_activity_log", []) or []
    vector_events = st.session_state.get("vector_events", []) or []

    insight_left, insight_mid, insight_right = st.columns([1, 1, 1.05])
    with insight_left:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Live Briefing</div>
                <div class="insight-title">로그 에이전트 브리핑</div>
                <div class="insight-body">{html.escape(str(latest_log)[:720])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_mid:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Market Watch</div>
                <div class="insight-title">뉴스 에이전트 브리핑</div>
                <div class="insight-body">{html.escape(str(latest_news)[:720])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_right:
        cards = []
        for event in (activity_log[:3] or []):
            cards.append(
                f"""
                <div class="event-card">
                    <div class="event-head">
                        <div class="event-source">{html.escape(str(event.get('source', '-')))} · {html.escape(str(event.get('status', '-')))}</div>
                        <div class="event-time">{html.escape(format_status_time(event.get('timestamp')))}</div>
                    </div>
                    <div class="event-body">{html.escape(str(event.get('detail', ''))[:170])}</div>
                </div>
                """
            )
        if not cards:
            for event in (vector_events[:3] or []):
                cards.append(
                    f"""
                    <div class="event-card">
                        <div class="event-head">
                            <div class="event-source">{html.escape(str(event.get('source', '-')))} · {html.escape(str(event.get('action', '-')))}</div>
                            <div class="event-time">{html.escape(format_status_time(event.get('timestamp')))}</div>
                        </div>
                        <div class="event-body">누적 {event.get('after_count', 0)} · 추가 {event.get('added_count', 0)} · {html.escape(str(event.get('detail', ''))[:120])}</div>
                    </div>
                    """
                )
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-label">Recent Timeline</div>
                <div class="insight-title">최근 운영 이벤트</div>
                <div class="event-stack">
            """
            + "".join(cards or ["<div class='event-card'><div class='event-body'>표시할 이벤트가 없습니다.</div></div>"])
            + """
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_operations_showcase():
    metrics = build_overview_metrics()
    render_initial_analysis_badge()
    render_dashboard_hero(metrics)
    render_agent_flow_section()

    metric_row_top = st.columns(2)
    metric_row_bottom = st.columns(2)
    with metric_row_top[0]:
        render_dashboard_metric_card(
            "심사 대상", str(metrics["results_count"]), "현재 화면에 반영된 분석 케이스 수", f"Activity {metrics['activity_events']}건", "cyan"
        )
    with metric_row_top[1]:
        render_dashboard_metric_card(
            "FAISS 벡터", str(metrics["vector_count"]), "로그, 뉴스, 규제 문서가 적재된 총 벡터 수", f"Events {metrics['vector_events']}건", "blue"
        )
    with metric_row_bottom[0]:
        render_dashboard_metric_card(
            "리스크 이슈", str(metrics["issues_count"]), "뉴스 기반 경보 및 이슈 탐지 결과", f"News {metrics['news_count']}건", "red"
        )
    with metric_row_bottom[1]:
        render_dashboard_metric_card(
            "Agent 상태", str(metrics["running_agents"]), "현재 실행 중인 Agent 수와 운영 온도", f"Failed {metrics['failed_agents']}건", "amber"
        )

    render_dashboard_workflow(metrics)

    st.markdown(
        """
        <div class="section-shell">
            <div class="section-header">
                <div>
                    <div class="section-kicker">Signal Deck</div>
                    <div class="section-title">실시간 브리핑과 이벤트 스트림</div>
                </div>
                <div class="section-detail">운영자가 바로 읽어야 하는 로그 브리핑, 뉴스 브리핑, 이벤트 타임라인을 카드형으로 모았습니다.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_live_insight_sections()


def summarize_log_case_text(context_text: str) -> str:
    text = str(context_text or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    picked: list[str] = []
    for line in lines:
        if line.startswith("[") or "case_id=" in line or line.startswith("입력필드=") or line.startswith("출력필드="):
            picked.append(line)
    if not picked:
        picked = lines[:3]
    return "\n".join(picked[:4])


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


def get_reviewer_personas() -> list[dict[str, str]]:
    return [
        {
            "id": "conservative",
            "emoji": "🧑‍💼",
            "name": "보수적 심사관",
            "display": "신용기획부 직원",
            "tone": "리스크 우선",
            "accent": "#ff8f8f",
            "avatar_class": "conservative",
            "tagline": "부실 조짐이 보이면 바로 제동을 겁니다.",
            "description": "연체 가능성, 규제 위반, 부실화 신호를 먼저 보는 성향입니다. 승인보다 방어 논리를 우선합니다.",
            "default_prompt": "너는 신용기획부 소속의 매우 보수적인 심사관이다. 대출 심사에서 손실 가능성, 부도 가능성, 규제 위반, 한도 과다, 금리 리스크를 최우선으로 본다. 승인 가능성이 있더라도 먼저 거절 사유와 보완 필요사항을 제시하라. 최종 의견은 위험요인 중심으로 단호하게 작성하라.",
        },
        {
            "id": "sales",
            "emoji": "😎",
            "name": "영업 심사관",
            "display": "금융영업부 직원",
            "tone": "승인 우선",
            "accent": "#61f4de",
            "avatar_class": "sales",
            "tagline": "조건만 맞추면 고객은 놓치지 않습니다.",
            "description": "고객 유지, 승인율, 매출 확대를 중요하게 보는 성향입니다. 거절보다 조건부 승인과 완화책을 찾습니다.",
            "default_prompt": "너는 금융영업부 소속의 공격적인 영업 심사관이다. 고객 유지, 승인율 개선, 한도 제안, 조건부 승인 전략을 우선한다. 리스크가 있더라도 대안을 찾아 승인 가능한 구조를 제시하라. 최종 의견은 승인 방향의 논리와 실무적 완화 조건을 중심으로 작성하라.",
        },
        {
            "id": "product",
            "emoji": "⚖️",
            "name": "상품기획 심사관",
            "display": "금융솔루션부 직원",
            "tone": "상품 전략",
            "accent": "#ffbf69",
            "avatar_class": "product",
            "tagline": "이 건을 어떤 상품으로 설계할지가 핵심입니다.",
            "description": "개별 건 승인 여부보다 어떤 상품 구조와 정책이 맞는지 보는 성향입니다. 신상품 적합성과 전략적 확장성을 따집니다.",
            "default_prompt": "너는 금융솔루션부 소속의 상품기획 담당자다. 이 건이 어떤 금융상품 구조와 맞는지, 기존 정책을 어떻게 조정하면 더 적합한지, 신상품 기획 관점에서 어떤 실험이 가능한지를 제안하라. 최종 의견은 상품 전략과 구조적 개선 방향을 중심으로 작성하라.",
        },
    ]


def ensure_strategy_debate_state() -> None:
    personas = get_reviewer_personas()
    if "reviewer_prompts" not in st.session_state:
        st.session_state.reviewer_prompts = {
            persona["id"]: persona["default_prompt"] for persona in personas
        }
    if "selected_reviewer_id" not in st.session_state:
        st.session_state.selected_reviewer_id = personas[0]["id"]
    if "reviewer_debate_round" not in st.session_state:
        st.session_state.reviewer_debate_round = []
    if "strategy_debate_question" not in st.session_state:
        st.session_state.strategy_debate_question = "이 고객을 어떤 조건으로 승인 또는 보류해야 하는지 토론해줘"
    if "strategy_debate_status" not in st.session_state:
        st.session_state.strategy_debate_status = ""
    if "reviewer_prompt_dialog_open" not in st.session_state:
        st.session_state.reviewer_prompt_dialog_open = False
    if "reviewer_prompt_saved_feedback" not in st.session_state:
        st.session_state.reviewer_prompt_saved_feedback = {}


def open_reviewer_prompt_dialog(reviewer_id: str) -> None:
    st.session_state.selected_reviewer_id = reviewer_id
    persona_map = {persona["id"]: persona for persona in get_reviewer_personas()}
    persona = persona_map.get(reviewer_id)
    if persona:
        editor_key = f"reviewer_prompt_editor_dialog_{reviewer_id}"
        st.session_state[editor_key] = st.session_state.reviewer_prompts.get(reviewer_id, persona["default_prompt"])
    st.session_state.reviewer_prompt_dialog_open = True


def close_reviewer_prompt_dialog() -> None:
    st.session_state.reviewer_prompt_dialog_open = False


def _render_reviewer_prompt_dialog_body(persona: dict[str, str]) -> None:
    reviewer_id = persona["id"]
    editor_key = f"reviewer_prompt_editor_dialog_{reviewer_id}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = st.session_state.reviewer_prompts.get(reviewer_id, persona["default_prompt"])
    save_feedback = st.session_state.get("reviewer_prompt_saved_feedback", {}) or {}
    feedback_at = save_feedback.get(reviewer_id)
    show_saved_feedback = False
    if feedback_at:
        try:
            saved_at = datetime.datetime.fromisoformat(str(feedback_at))
            show_saved_feedback = (datetime.datetime.now() - saved_at).total_seconds() < 4.0
        except Exception:
            show_saved_feedback = False

    st.markdown(
        f"""
        <div class="prompt-panel">
            <div class="dialog-reviewer-hero">
                <div class="dialog-reviewer-avatar reviewer-avatar {persona['avatar_class']} active badge-speaking">
                    <div class="reviewer-avatar-hair"></div>
                    <div class="reviewer-avatar-head"></div>
                    <div class="reviewer-avatar-eye left"></div>
                    <div class="reviewer-avatar-eye right"></div>
                    <div class="reviewer-avatar-mouth"></div>
                    <div class="reviewer-avatar-body"></div>
                </div>
                <div class="dialog-reviewer-meta">
                    <div class="dialog-reviewer-kicker">{persona['emoji']} {persona['display']}</div>
                    <div class="prompt-panel-title">{persona['name']} 프롬프트 편집</div>
                    <div class="prompt-panel-subtitle">현재 선택된 심사관은 {persona['display']} 역할로 Ollama에 질의합니다. 아래 문장을 수정하면 이 심사관의 성향을 바로 바꿀 수 있습니다.</div>
                    <div class="dialog-reviewer-badge {persona['avatar_class']}">{persona['tone']} · {persona['tagline']}</div>
                    {f'<div class="dialog-save-status {persona["avatar_class"]}">저장 완료 · 프롬프트가 즉시 반영됩니다</div>' if show_saved_feedback else ''}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.text_area(
        "심사관 역할 프롬프트",
        key=editor_key,
        height=220,
    )

    with st.expander("실제 Ollama 질의 프롬프트 미리보기"):
        st.code(
            build_reviewer_question(
                persona,
                st.session_state.get("strategy_debate_question", ""),
                st.session_state.get(editor_key, persona["default_prompt"]),
            ),
            language="text",
        )

    action_col_a, action_col_b, action_col_c = st.columns(3)
    with action_col_a:
        if st.button("저장", key=f"save_reviewer_prompt_{reviewer_id}", use_container_width=True, type="primary"):
            st.session_state.reviewer_prompts[reviewer_id] = st.session_state.get(editor_key, persona["default_prompt"])
            st.session_state.strategy_debate_status = f"{persona['name']} 프롬프트를 수정했습니다."
            st.session_state.reviewer_prompt_saved_feedback = {
                **(st.session_state.get("reviewer_prompt_saved_feedback", {}) or {}),
                reviewer_id: datetime.datetime.now().isoformat(),
            }
            st.rerun()
    with action_col_b:
        if st.button("기본값 복원", key=f"reset_reviewer_prompt_{reviewer_id}", use_container_width=True):
            st.session_state.reviewer_prompts[reviewer_id] = persona["default_prompt"]
            st.session_state[editor_key] = persona["default_prompt"]
            st.rerun()
    with action_col_c:
        if st.button("닫기", key=f"close_reviewer_prompt_{reviewer_id}", use_container_width=True):
            close_reviewer_prompt_dialog()
            st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("심사관 프롬프트 편집")
    def render_reviewer_prompt_dialog(persona: dict[str, str]) -> None:
        _render_reviewer_prompt_dialog_body(persona)
else:
    def render_reviewer_prompt_dialog(persona: dict[str, str]) -> None:
        _render_reviewer_prompt_dialog_body(persona)


def build_reviewer_question(persona: dict[str, str], user_question: str, custom_prompt: str) -> str:
    return (
        f"[역할]\n{custom_prompt.strip()}\n\n"
        "[출력 지시]\n"
        f"너는 반드시 {persona['display']} 입장에서만 판단해야 한다. "
        "최종 결론에서는 자신의 부서 성향이 드러나야 하고, 승인/보류/거절 중 하나의 스탠스를 분명히 드러내라. "
        "리스크, 기회, 보완조건, 한줄 결론을 포함해 답하라.\n\n"
        f"[사용자 질문]\n{user_question.strip()}"
    )


def summarize_debate_result(response_payload: dict) -> tuple[str, str, str]:
    sections = response_payload.get("sections", {}) if isinstance(response_payload, dict) else {}
    final_decision = sections.get("final_decision") or response_payload.get("answer") or "분석 결과가 없습니다."
    log_text = sections.get("log_analysis", "")
    news_text = sections.get("news_analysis", "")
    regulation_text = sections.get("regulation_analysis", "")
    preview = " ".join(str(final_decision).split())[:180]
    evidence = " / ".join(
        [text for text in [log_text[:60], news_text[:60], regulation_text[:60]] if text]
    )
    if not evidence:
        evidence = "근거 요약이 없습니다."
    if "거절" in final_decision:
        verdict = "거절/보수"
    elif "조건부 승인" in final_decision:
        verdict = "조건부 승인"
    elif "승인" in final_decision:
        verdict = "승인"
    else:
        verdict = "판단 대기"
    return verdict, preview, evidence[:220]


def build_debate_consensus(personas: list[dict[str, str]], round_results: list[dict]) -> str:
    if not round_results:
        return "아직 토론 결과가 없습니다."
    lines = []
    for item in round_results:
        lines.append(f"{item['persona']['name']}: {item['verdict']} · {item['preview']}")
    stances = [item["verdict"] for item in round_results]
    if any("거절" in stance for stance in stances) and any("승인" in stance for stance in stances):
        title = "의견 충돌: 리스크와 성장 관점이 정면으로 갈렸습니다."
    elif all("승인" in stance or "조건부" in stance for stance in stances):
        title = "대체로 승인 기조이지만, 조건 정교화가 필요합니다."
    else:
        title = "보수적 의견이 우세하며 추가 보완 자료가 필요합니다."
    return title + "\n\n" + "\n".join(lines)


def render_role_based_strategy_tab():
    ensure_strategy_debate_state()
    personas = get_reviewer_personas()
    persona_map = {persona["id"]: persona for persona in personas}
    selected_id = st.session_state.get("selected_reviewer_id", personas[0]["id"])
    selected_persona = persona_map.get(selected_id, personas[0])

    st.markdown(
        """
        <div class="debate-hero">
            <div class="debate-kicker">Role-based Multi-Agent</div>
            <div class="debate-title">심사관 흉내내는 AI 토론실</div>
            <div class="debate-subtitle">세 명의 서로 다른 성향의 심사관이 같은 안건을 두고 각자 의견을 제시합니다. 카드를 누르면 해당 심사관의 Ollama 질의 프롬프트를 팝업에서 바로 수정할 수 있습니다.</div>
            <div class="debate-wave"><span></span><span></span><span></span><span></span><span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reviewer_cols = st.columns(3)
    for column, persona in zip(reviewer_cols, personas):
        is_active = persona["id"] == selected_id
        with column:
            column.markdown(
                f"""
                <div class="reviewer-card {persona['avatar_class']}{' active' if is_active else ''}">
                    <div class="reviewer-role">{persona['emoji']} Reviewer</div>
                    <div class="reviewer-avatar-wrap">
                        <div class="reviewer-avatar {persona['avatar_class']}{' active badge-speaking' if is_active else ''}">
                            <div class="reviewer-avatar-hair"></div>
                            <div class="reviewer-avatar-head"></div>
                            <div class="reviewer-avatar-eye left"></div>
                            <div class="reviewer-avatar-eye right"></div>
                            <div class="reviewer-avatar-mouth"></div>
                            <div class="reviewer-avatar-body"></div>
                        </div>
                        <div class="reviewer-meta">
                            <div class="reviewer-name">{persona['name']}</div>
                            <div class="reviewer-dept">{persona['display']}</div>
                            <div class="reviewer-tone">{persona['tone']}</div>
                        </div>
                    </div>
                    <div class="reviewer-desc">{persona['description']}</div>
                    <div class="reviewer-select-note">"{persona['tagline']}"</div>
                    <div class="reviewer-select-note cta">프롬프트 편집을 눌러 이 심사관의 성향을 조정하세요</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "프롬프트 편집 열기",
                key=f"edit_reviewer_prompt_{persona['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                open_reviewer_prompt_dialog(persona["id"])
                st.rerun()

    if st.session_state.get("reviewer_prompt_dialog_open"):
        render_reviewer_prompt_dialog(selected_persona)

    spotlight_col, control_col = st.columns([1.05, 0.95])
    with spotlight_col:
        prompt_preview = st.session_state.reviewer_prompts.get(selected_id, selected_persona["default_prompt"])
        prompt_preview = " ".join(str(prompt_preview).split())[:210]
        st.markdown(
            f"""
            <div class="prompt-panel selected-reviewer-stage {selected_persona['avatar_class']}">
                <div class="selected-reviewer-head">
                    <div>
                        <div class="prompt-panel-title">{selected_persona['name']} 발언 준비</div>
                        <div class="prompt-panel-subtitle">선택된 심사관이 어떤 기준으로 판단하는지 먼저 확인하고, 필요하면 팝업에서 역할 프롬프트를 수정할 수 있습니다.</div>
                    </div>
                    <div class="selected-reviewer-chip {selected_persona['avatar_class']}">{selected_persona['display']} · {selected_persona['tone']}</div>
                </div>
                <div class="selected-reviewer-bubble {selected_persona['avatar_class']}">“{selected_persona['tagline']}”</div>
                <div class="selected-reviewer-preview {selected_persona['avatar_class']}">
                    <div class="selected-reviewer-preview-label">Current Prompt Snapshot</div>
                    <div class="selected-reviewer-preview-text">{html.escape(prompt_preview)}...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("프롬프트 수정은 상단 심사관 카드의 편집 영역에서 실행됩니다.")

    with control_col:
        strategy_question = st.text_area(
            "토론 안건",
            value=st.session_state.get("strategy_debate_question", ""),
            height=190,
            key="strategy_debate_question_editor",
        )
        st.session_state.strategy_debate_question = strategy_question
        action_col_a, action_col_b = st.columns(2)
        with action_col_a:
            run_clicked = st.button("3인 토론 시작", use_container_width=True, type="primary")
        with action_col_b:
            if st.button("선택 프롬프트 기본값 복원", use_container_width=True):
                st.session_state.reviewer_prompts[selected_id] = selected_persona["default_prompt"]
                st.session_state[f"reviewer_prompt_editor_dialog_{selected_id}"] = selected_persona["default_prompt"]
                st.rerun()

    if run_clicked and strategy_question.strip():
        round_results: list[dict] = []
        status_placeholder = st.empty()
        for persona in personas:
            status_placeholder.markdown(
                f"""
                <div class="debate-status">
                    <div class="debate-status-title">{persona['name']} 의견 생성 중</div>
                    <div class="debate-status-text">{persona['display']} 관점에서 문맥 검색과 심사 판단을 수행하고 있습니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            composed_question = build_reviewer_question(
                persona,
                strategy_question,
                st.session_state.reviewer_prompts.get(persona["id"], persona["default_prompt"]),
            )
            try:
                response_payload = get_backend_client().strategy_chat(composed_question)
                latest_status = get_backend_client().get_status()
                sync_session_from_backend(latest_status)
            except Exception:
                response_payload = {
                    "answer": f"{persona['name']} 의견 생성에 실패했습니다.",
                    "sections": {},
                }
            verdict, preview, evidence = summarize_debate_result(response_payload)
            round_results.append(
                {
                    "persona": persona,
                    "response": response_payload,
                    "verdict": verdict,
                    "preview": preview,
                    "evidence": evidence,
                    "generated_at": datetime.datetime.now().isoformat(),
                }
            )
        st.session_state.reviewer_debate_round = round_results
        st.session_state.strategy_debate_status = f"최근 토론 완료 · {format_status_time(datetime.datetime.now().isoformat())}"
        status_placeholder.empty()

    if st.session_state.get("strategy_debate_status"):
        st.markdown(
            f"""
            <div class="debate-status">
                <div class="debate-status-title">최근 토론 상태</div>
                <div class="debate-status-text">{html.escape(st.session_state.get('strategy_debate_status', ''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    round_results = st.session_state.get("reviewer_debate_round", []) or []
    if round_results:
        st.markdown(
            f"""
            <div class="consensus-card">
                <div class="consensus-label">Debate Consensus</div>
                <div class="consensus-title">3인 토론 종합 메모</div>
                <div class="consensus-body">{html.escape(build_debate_consensus(personas, round_results))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        transcript_col, detail_col = st.columns([1.05, 0.95])
        with transcript_col:
            st.markdown('<div class="debate-transcript">', unsafe_allow_html=True)
            for item in round_results:
                persona = item["persona"]
                bubble_color = persona["accent"]
                st.markdown(
                    f"""
                    <div class="debate-bubble">
                        <div class="debate-bubble-head">
                            <div class="debate-bubble-avatar">
                                <div class="debate-bubble-mini-avatar" style="background:{bubble_color};">{persona['emoji']}</div>
                                <div class="debate-bubble-name">{persona['name']} · {persona['display']}</div>
                            </div>
                            <span class="debate-bubble-badge" style="background:{bubble_color};">{html.escape(item['verdict'])}</span>
                        </div>
                        <div class="debate-bubble-text">{html.escape(item['preview'])}\n\n근거: {html.escape(item['evidence'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with detail_col:
            detail_tabs = st.tabs([persona["name"] for persona in personas])
            for tab, item in zip(detail_tabs, round_results):
                with tab:
                    render_strategy_response(item["response"])
    else:
        st.info("토론 안건을 입력하고 '3인 토론 시작'을 누르면 세 심사관이 각자 의견을 생성합니다.")


# `get_vector_count` is provided by `rag.vector_db` import; avoid redefining it here.


def get_chart_snapshots() -> dict:
    try:
        payload = get_backend_client().get_charts()
        return payload.get("charts", {})
    except Exception:
        return {}


def get_live_faiss_items(limit: int = 1000) -> list[dict]:
    items = st.session_state.get("full_faiss_items", []) or []
    if items:
        return items[:limit]

    try:
        entries_resp = get_backend_client().get_faiss_entries(limit=limit)
        items = entries_resp.get("items", []) if isinstance(entries_resp, dict) else []
        if items:
            st.session_state.full_faiss_items = items
            return items
    except Exception:
        pass

    try:
        from rag.vector_db import list_vectors

        items = list_vectors(limit=limit)
        if items:
            st.session_state.full_faiss_items = items
        return items
    except Exception:
        return []


def render_vector_db_panel():
    st.subheader("🧠 Vector DB 실시간 적재 현황")

    vector_events = st.session_state.get("vector_events", []) or []
    latest_vector_event = vector_events[0] if vector_events else {}
    items = get_live_faiss_items(limit=1000)
    recent_items = list(reversed(items[-10:])) if items else []

    vector_metric_cols = st.columns(4)
    vector_metric_cols[0].metric(
        "현재 벡터 수", st.session_state.get("vector_count", 0)
    )
    vector_metric_cols[1].metric(
        "마지막 증감", latest_vector_event.get("added_count", 0)
    )
    vector_metric_cols[2].metric(
        "최근 적재 소스", latest_vector_event.get("source", "-")
    )
    vector_metric_cols[3].metric(
        "실시간 표시 항목", len(items)
    )

    st.markdown("#### 실시간 적재값 미리보기")
    if not recent_items:
        st.info("실시간으로 표시할 Vector DB 항목이 아직 없습니다.")
    else:
        preview_df = pd.DataFrame(
            [
                {
                    "id": it.get("id"),
                    "type": it.get("type"),
                    "product": it.get("product"),
                    "source": it.get("source"),
                    "name": it.get("name"),
                    "snippet": (it.get("snippet") or "")[:180],
                }
                for it in recent_items
            ]
        )
        st.dataframe(
            preview_df,
            height=280,
            width="stretch",
            hide_index=True,
        )

        selected_entry_id = st.selectbox(
            "상세 확인할 벡터 항목",
            options=[""] + [str(it.get("id") or "") for it in recent_items],
            key="vector_live_entry_select",
        )
        if selected_entry_id:
            selected_item = next(
                (it for it in recent_items if str(it.get("id")) == selected_entry_id),
                None,
            )
            if selected_item is not None:
                detail_col, meta_col = st.columns([1.2, 1])
                with detail_col:
                    st.markdown("##### 선택 항목 원문 미리보기")
                    st.code(selected_item.get("snippet", "")[:1200], language="text")
                with meta_col:
                    st.markdown("##### 선택 항목 메타데이터")
                    st.json(
                        {
                            "id": selected_item.get("id"),
                            "type": selected_item.get("type"),
                            "product": selected_item.get("product"),
                            "agent": selected_item.get("agent"),
                            "source": selected_item.get("source"),
                            "name": selected_item.get("name"),
                            "features": selected_item.get("features") or {},
                            "in_fields": selected_item.get("in_fields") or {},
                            "out_fields": selected_item.get("out_fields") or {},
                            "reject_reason_codes": selected_item.get("reject_reason_codes") or [],
                            "reject_reason_details": selected_item.get("reject_reason_details") or [],
                        }
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

    st.markdown("#### FAISS 전체 항목")
    if not items:
        st.info("FAISS에 저장된 항목이 없습니다.")
    else:
        st.caption("WebSocket으로 갱신된 세션 스냅샷을 우선 사용하고, 초기 로드 시에는 백엔드에서 목록을 보강합니다.")
        full_df = pd.DataFrame(
            [
                {
                    "id": it.get("id"),
                    "type": it.get("type"),
                    "product": it.get("product"),
                    "source": it.get("source"),
                    "name": it.get("name"),
                    "snippet": (it.get("snippet") or "")[:200],
                }
                for it in items
            ]
        )
        st.dataframe(full_df, height=360, width="stretch", hide_index=True)


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

    crawl_running = bool(st.session_state.get("news_crawl_running", False))
    crawl_target_count = int(st.session_state.get("news_crawl_target_count", 0) or 0)
    crawl_success_count = int(st.session_state.get("news_crawl_success_count", 0) or 0)
    crawl_failure_count = int(st.session_state.get("news_crawl_failure_count", 0) or 0)
    last_news_crawl_error = st.session_state.get("last_news_crawl_error")
    if crawl_running or crawl_failure_count > 0:
        crawl_background = "rgba(219,234,254,0.95)" if crawl_running else "rgba(254,226,226,0.95)"
        crawl_border = "rgba(59,130,246,0.28)" if crawl_running else "rgba(239,68,68,0.24)"
        crawl_color = "#1d4ed8" if crawl_running else "#991b1b"
        crawl_label = "뉴스 본문 크롤링 진행 중" if crawl_running else "뉴스 본문 크롤링 실패 건 존재"
        st.markdown(
            f"""
            <div style="display:inline-flex; align-items:center; gap:12px; margin: 0 0 14px 0; padding: 10px 14px; border-radius: 999px; background:{crawl_background}; border:1px solid {crawl_border};">
                <span style="font-size:13px; font-weight:800; color:{crawl_color};">{crawl_label}</span>
                <span style="font-size:12px; color:#334155;">대상 {crawl_target_count} · 성공 {crawl_success_count} · 실패 {crawl_failure_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if (not crawl_running) and last_news_crawl_error:
            st.caption(f"최근 뉴스 크롤링 오류: {last_news_crawl_error}")

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

    st.info("벡터 DB 적재 현황, 실행 타임라인, 적재 이벤트, 최근 적재 내역은 오른쪽의 Vector DB 탭으로 분리했습니다.")


def render_agent_prompt_panel(
    agent_key: str, title: str, accent_color: str, soft_background: str
):
    prompt_input = st.session_state.get(f"latest_{agent_key}_prompt_input", {}) or {}
    updated_at = st.session_state.get(f"last_{agent_key}_prompt_input_time")

    st.subheader(title)
    source = prompt_input.get("source", "-")
    user_input = prompt_input.get("user_input", "-")
    context_text = prompt_input.get("context", "관련 데이터가 없습니다.")
    prompt_text = prompt_input.get("prompt", "-")

    metric_cols = st.columns(3)
    metric_cols[0].metric("최근 갱신", format_status_time(updated_at))
    metric_cols[1].metric("실행 소스", source)
    metric_cols[2].metric("컨텍스트 길이", len(context_text))

    if agent_key == "news":
        news_items = st.session_state.get("news", []) or []
        crawled_items = [
            item for item in news_items if str(item.get("content", "")).strip()
        ]
        latest_crawled = crawled_items[0] if crawled_items else None
        crawl_running = bool(st.session_state.get("news_crawl_running", False))
        crawl_target_count = int(st.session_state.get("news_crawl_target_count", 0) or 0)
        crawl_success_count = int(st.session_state.get("news_crawl_success_count", 0) or 0)
        crawl_failure_count = int(st.session_state.get("news_crawl_failure_count", 0) or 0)
        crawl_updated_at = st.session_state.get("last_news_crawl_time")
        crawl_error = st.session_state.get("last_news_crawl_error")

        if crawl_running:
            badge_label = "본문 크롤링 진행 중"
            badge_background = "#dbeafe"
            badge_color = "#1d4ed8"
        elif crawl_failure_count > 0 and latest_crawled is None:
            badge_label = "본문 크롤링 실패"
            badge_background = "#fee2e2"
            badge_color = "#991b1b"
        elif latest_crawled:
            badge_label = "본문 크롤링 완료"
            badge_background = "#dcfce7"
            badge_color = "#166534"
        else:
            badge_label = "본문 크롤링 대기"
            badge_background = "#fef3c7"
            badge_color = "#92400e"

        st.markdown(
            f"""
            <div style="margin: 8px 0 12px 0;">
                <span style="display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:800; background:{badge_background}; color:{badge_color}; border:1px solid rgba(15,23,42,0.08);">{badge_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        crawl_cols = st.columns(4)
        crawl_cols[0].metric("대상 기사", crawl_target_count)
        crawl_cols[1].metric("성공", crawl_success_count)
        crawl_cols[2].metric("실패", crawl_failure_count)
        crawl_cols[3].metric("최근 갱신", format_status_time(crawl_updated_at))

        if crawl_running:
            st.info("뉴스 본문을 크롤링 중입니다. 완료되면 뉴스 에이전트 프롬프트가 자동으로 갱신됩니다.")
        elif crawl_failure_count > 0 and latest_crawled is None:
            st.warning(
                f"본문 크롤링에 실패해 뉴스 에이전트가 아직 브리핑을 만들지 못했습니다. 최근 오류: {crawl_error or '-'}"
            )

        if latest_crawled is not None:
            latest_title = str(latest_crawled.get("title", "")).strip() or "제목 없음"
            latest_content = str(latest_crawled.get("content", "")).strip()
            latest_link = str(latest_crawled.get("link", "")).strip()
            preview = latest_content[:320] + ("..." if len(latest_content) > 320 else "")
            link_html = (
                f'<a href="{html.escape(latest_link)}" target="_blank" rel="noopener noreferrer" style="font-size:12px; font-weight:700; color:{accent_color}; text-decoration:none;">원문 열기</a>'
                if latest_link
                else ""
            )
            st.markdown(
                f"""
                <div style="margin: 6px 0 16px 0; padding: 16px 18px; border-radius: 18px; background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.98)); border: 1px solid rgba(148,163,184,0.18); box-shadow: 0 10px 24px rgba(15,23,42,0.05);">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px;">
                        <div style="font-size:14px; font-weight:800; color:#0f172a;">최신 크롤링 뉴스 1건</div>
                        {link_html}
                    </div>
                    <div style="font-size:13px; font-weight:800; color:#0f172a; margin-bottom:8px;">{html.escape(latest_title)}</div>
                    <div style="font-size:13px; line-height:1.65; color:#334155; white-space:pre-wrap;">{html.escape(preview)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if agent_key == "log" and context_text and context_text != "관련 데이터가 없습니다.":
        preview = summarize_log_case_text(context_text)
        st.markdown(
            f"""
            <div style="margin: 6px 0 16px 0; padding: 16px 18px; border-radius: 18px; background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.98)); border: 1px solid rgba(148,163,184,0.18); box-shadow: 0 10px 24px rgba(15,23,42,0.05);">
                <div style="font-size:14px; font-weight:800; color:#0f172a; margin-bottom:10px;">현재 로그 에이전트 대표 케이스 1건</div>
                <div style="font-size:13px; line-height:1.65; color:#334155; white-space:pre-wrap;">{html.escape(preview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not prompt_input:
        st.info("아직 표시할 프롬프트 입력값이 없습니다.")
        return

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
    consume_ws_snapshot_buffer()
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
    st.subheader("📰 실시간 뉴스 ")

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

    def _strip_html_text(value: str) -> str:
        text = str(value or "")
        text = text.replace("<b>", "").replace("</b>", "")
        text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
        return " ".join(text.split()).strip()

    def _extract_first_sentence(value: str) -> str:
        text = _strip_html_text(value)
        if not text:
            return ""
        separators = [". ", "! ", "? ", "다. ", "요. ", "\n"]
        for separator in separators:
            if separator in text:
                return text.split(separator, 1)[0].strip() + (separator.strip() if separator.strip() in ["다.", "요."] else "")
        return text[:120].strip()

    def _resolve_news_title_and_preview(news_item: dict) -> tuple[str, str]:
        raw_title = _strip_html_text(news_item.get("title", ""))
        summary = _strip_html_text(news_item.get("summary", ""))
        content = _strip_html_text(news_item.get("content", ""))
        generic_titles = {"네이버뉴스", "네이버 뉴스", "기사 원문", "뉴스", "제목 없음"}

        fallback_title = _extract_first_sentence(content) or _extract_first_sentence(summary) or "제목 없음"
        if not raw_title or raw_title in generic_titles or len(raw_title) <= 4:
            title = fallback_title
        else:
            title = raw_title

        preview_source = content or summary
        preview = _extract_first_sentence(preview_source)
        if preview == title:
            remaining = preview_source[len(preview):].strip(" .") if preview_source.startswith(preview) else ""
            preview = _extract_first_sentence(remaining)
        if not preview:
            preview = summary or content or "요약 정보가 없습니다."

        return title[:90], preview[:140]

    for index, news_item in enumerate(news_items[:3]):
        title, preview = _resolve_news_title_and_preview(news_item)
        link = str(news_item.get("link", "")).strip()
        badge_label = "NEW" if has_fresh_news_cycle and index == 0 else f"#{index + 1}"
        badge_background = "#16a34a" if badge_label == "NEW" else "#0f172a"
        safe_title = html.escape(title)
        safe_preview = html.escape(preview)

        card_html = f"""
            <div style="margin-bottom:12px; padding:10px 12px; border-radius:14px; background:linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88)); border:1px solid rgba(151,196,225,0.14); box-shadow:0 12px 24px rgba(0,0,0,0.18);">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:6px;">
                    <div style="font-size:13px; font-weight:800; color:#f7fbff;">{safe_title}</div>
                    <span style="flex-shrink:0; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:800; background:{badge_background}; color:white;">{badge_label}</span>
                </div>
                <div style="font-size:12px; line-height:1.55; color:#d9ecfb;">{safe_preview}</div>
            </div>
        """

        if link:
            wrapped = f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:inherit;">{card_html}</a>'
            st.markdown(wrapped, unsafe_allow_html=True)
        else:
            st.markdown(card_html, unsafe_allow_html=True)

    def _format_upload_size(size_bytes: int) -> str:
        size = float(size_bytes or 0)
        units = ["B", "KB", "MB", "GB"]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024

    regulation_info = (st.session_state.get("agent_statuses", {}) or {}).get(
        "regulation_agent", {}
    ) or {}
    regulation_status = regulation_info.get("status", "pending")
    regulation_detail = str(
        regulation_info.get("detail")
        or "업로드된 문서를 벡터화하고 규제 요약을 생성합니다."
    )
    regulation_updated_at = format_status_time(regulation_info.get("updated_at"))
    upload_shell_class = ""
    if regulation_status == "running":
        upload_shell_class = " running"
    elif regulation_status == "completed":
        upload_shell_class = " completed"
    regulation_summary = str(st.session_state.get("latest_regulation_analysis") or "").strip()
    regulation_summary = regulation_summary[:320]
    regulation_steps_html = """
        <div class="upload-steps">
            <div class="upload-step done">
                <div class="upload-step-badge">1</div>
                <div class="upload-step-text">문서 분할: 업로드한 문서를 청킹해 규제 문맥 단위로 나눕니다.</div>
            </div>
            <div class="upload-step active">
                <div class="upload-step-badge">2</div>
                <div class="upload-step-text">벡터 학습: 임베딩 생성 후 FAISS에 적재하면서 검색 가능한 규제 근거로 변환합니다.</div>
            </div>
            <div class="upload-step">
                <div class="upload-step-badge">3</div>
                <div class="upload-step-text">규제 요약: AI가 실무 판단용 규제 브리핑과 핵심 준수 포인트를 생성합니다.</div>
            </div>
        </div>
    """

    st.markdown(
        f"""
        <div class="upload-shell{upload_shell_class}">
            <div class="upload-shell-head">
                <div class="upload-shell-copy">
                    <div class="upload-kicker">Regulation Intake</div>
                    <div class="upload-title">규제 문서를 실제 학습 데이터로 업로드하는 영역</div>
                    <div class="upload-subtitle">금감원, 여신협회, 내부 규정 문서를 올리면 AI가 문서를 청킹하고 벡터 DB에 적재한 뒤 규제 분석용 근거로 바로 사용합니다.</div>
                </div>
                <div class="upload-doc-cluster" aria-hidden="true">
                    <div class="upload-doc-card pdf">
                        <div class="upload-doc-orbit"></div>
                        <div class="upload-doc-label">PDF</div>
                        <div class="upload-doc-lines"><span></span><span></span><span></span></div>
                        <div class="upload-doc-progress">LIVE</div>
                        <div class="upload-doc-check">✓</div>
                    </div>
                    <div class="upload-doc-card word">
                        <div class="upload-doc-orbit"></div>
                        <div class="upload-doc-label">DOC</div>
                        <div class="upload-doc-lines"><span></span><span></span><span></span></div>
                        <div class="upload-doc-progress">LIVE</div>
                        <div class="upload-doc-check">✓</div>
                    </div>
                </div>
            </div>
            <div class="upload-chip-row">
                <span class="upload-chip">PDF / TXT / MD</span>
                <span class="upload-chip">다중 문서 업로드</span>
                <span class="upload-chip">FAISS + 규제 에이전트 연동</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "규제 문서 업로드 (PDF/TXT/MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="sidebar_reg_upload",
        label_visibility="collapsed",
    )

    if uploaded:
        total_bytes = sum(int(getattr(file, "size", 0) or 0) for file in uploaded)
        selected_rows = "".join(
            f'<div class="upload-selected-item">• {html.escape(getattr(file, "name", "unknown"))} <span style="color:#166534; font-weight:700;">({html.escape(_format_upload_size(getattr(file, "size", 0) or 0))})</span></div>'
            for file in uploaded[:4]
        )
        if len(uploaded) > 4:
            selected_rows += f'<div class="upload-selected-item">• 외 {len(uploaded) - 4}건 추가 선택됨</div>'
        st.markdown(
            f"""
            <div class="upload-selected-box">
                <div class="upload-selected-title">업로드 대기 문서 {len(uploaded)}건 · 총 {html.escape(_format_upload_size(total_bytes))}</div>
                {selected_rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if regulation_status == "running":
        st.markdown(
            f"""
            <div class="upload-learning-box">
                <div class="upload-learning-head">
                    <div class="upload-learning-core"></div>
                    <div>
                        <div class="upload-learning-title">AI가 규제 문서를 학습 중입니다</div>
                        <div class="upload-learning-text">문서 청킹, 벡터 적재, 규제 요약 생성을 순차적으로 수행하고 있습니다.<br>최근 업데이트: {html.escape(regulation_updated_at)}</div>
                    </div>
                </div>
                <div class="upload-learning-text">{html.escape(regulation_detail)}</div>
                <div class="upload-learning-bar"><span></span></div>
                {regulation_steps_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif regulation_status == "completed":
        st.markdown(
            f"""
            <div class="upload-status-box success">
                <div class="upload-status-pill">✓ Vectorized & Ready</div>
                <div class="upload-status-title">규제 문서 분석 완료</div>
                <div class="upload-status-detail">{html.escape(regulation_detail)}<br>완료 시각: {html.escape(regulation_updated_at)}</div>
                {f'<div class="upload-status-summary">최신 요약 미리보기<br>{html.escape(regulation_summary)}</div>' if regulation_summary else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif regulation_status == "failed":
        st.markdown(
            f"""
            <div class="upload-status-box error">
                <div class="upload-status-title">규제 문서 분석 실패</div>
                <div class="upload-status-detail">{html.escape(regulation_detail)}<br>업데이트: {html.escape(regulation_updated_at)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded:
        if st.button(
            "AI 규제 문서 학습 시작",
            key="sidebar_reg_run",
            type="primary",
            use_container_width=True,
            disabled=(regulation_status == "running"),
        ):
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
                    before = int(st.session_state.get("vector_count", 0) or 0)
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
    else:
        st.caption(
            "문서를 이 영역에 드롭하거나 클릭해 선택한 뒤, 아래 학습 버튼으로 규제 에이전트 분석을 시작하세요."
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
                # 이벤트로 추가된 실제 벡터 항목을 상세히 보여주는 기능
                try:
                    before = int(ev.get("before_count") or 0)
                    after = int(ev.get("after_count") or 0)
                except Exception:
                    before = 0
                    after = 0

                if after > before:
                    if st.button(f"이벤트에서 추가된 벡터 보기 ({after-before}건)", key=f"show_ev_{ev.get('timestamp')}_{before}_{after}"):
                        with st.spinner("FAISS 항목 불러오는 중..."):
                            try:
                                resp = get_backend_client().get_faiss_entries(limit=after)
                                items = resp.get("items", [])
                                added = items[before:after]
                                if not added:
                                    st.info("추가된 벡터 항목을 찾을 수 없습니다.")
                                else:
                                    for it in added:
                                        st.markdown(f"**ID:** {it.get('id')} — **type:** {it.get('type')} — **product:** {it.get('product')}")
                                        st.code(it.get("snippet", "")[:800], language="text")
                                    # CSV 다운로드
                                    import pandas as _pd

                                    df_added = _pd.DataFrame(added)
                                    csv_added = df_added.to_csv(index=False).encode("utf-8")
                                    st.download_button("이벤트 추가 벡터 CSV로 다운로드", csv_added, file_name=f"faiss_event_{before}_{after}.csv", mime="text/csv")
                            except Exception as e:
                                st.error(f"항목 불러오기 실패: {e}")

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
                # 초보자용: FAISS에 저장된 데이터 구조 요약
                st.markdown("#### 데이터 구조 요약 (초보자용)")
                try:
                    # 수집된 항목들을 순회하며 메타 필드 통계 수집
                    field_stats: dict = {}
                    sample_item = None
                    for it in items:
                        if sample_item is None:
                            sample_item = it
                        # 메타데이터가 dict로 있는 경우 우선 처리
                        meta = it.get("metadata") or it.get("meta") or {}
                        if isinstance(meta, dict):
                            for k, v in meta.items():
                                entry = field_stats.setdefault(k, {"count": 0, "types": {}, "samples": []})
                                entry["count"] += 1
                                t = type(v).__name__
                                entry["types"][t] = entry["types"].get(t, 0) + 1
                                if len(entry["samples"]) < 3:
                                    entry["samples"].append(v)
                        # 최상위 필드들도 취합 (id, page_content 길이 등)
                        for topk in ("id", "title", "page_content", "content"):
                            if topk in it:
                                entry = field_stats.setdefault(topk, {"count": 0, "types": {}, "samples": []})
                                entry["count"] += 1
                                v = it.get(topk)
                                t = type(v).__name__
                                entry["types"][t] = entry["types"].get(t, 0) + 1
                                if len(entry["samples"]) < 3:
                                    if topk == "page_content":
                                        entry["samples"].append((v or "")[:200])
                                    else:
                                        entry["samples"].append(v)

                    total = len(items)
                    st.markdown(f"- 총 항목: **{total}개**")
                    st.markdown("- 주요 메타 필드(출현 비율, 대표 타입, 예시값):")
                    for fname, info in sorted(field_stats.items(), key=lambda x: -x[1]["count"]):
                        pct = int(100 * info["count"] / total)
                        types_desc = ", ".join([f"{k}({v})" for k, v in info["types"].items()])
                        samples = ", ".join([str(s) for s in info["samples"]])
                        st.markdown(f"- **{fname}**: {info['count']}건 ({pct}%) · 타입: {types_desc} · 예시: `{samples}`")

                    if sample_item:
                        with st.expander("대표 항목 예시 (확장해서 보기)"):
                            st.json(sample_item)
                except Exception as e:
                    st.warning(f"구조 요약 생성 중 오류: {e}")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("CSV로 다운로드", csv, file_name="faiss_entries.csv", mime="text/csv")
                # ID 선택박스로 상세 조회
                ids = [str(x.get("id")) for x in items]
                if ids:
                    sel = st.selectbox("상세 조회할 ID 선택", options=[""] + ids)
                    if sel:
                        try:
                            resp = get_backend_client().get_faiss_entry(sel)
                            item = resp.get("item")
                            if item:
                                st.subheader("메타데이터")
                                st.json(item.get("metadata", {}))
                                st.subheader("원문 스니펫")
                                st.code((item.get("page_content") or "")[:2000], language="text")
                                st.download_button("JSON으로 다운로드", value=str(item), file_name=f"faiss_{sel}.json", mime="application/json")
                            else:
                                st.info("상세 항목 없음")
                        except Exception as e:
                            st.error(f"상세 조회 실패: {e}")
            else:
                st.info("목록이 비어있습니다.")
        except Exception as e:
            st.error(f"목록 불러오기 실패: {e}")


@fragment_decorator(run_every="3s")
def render_live_news_fragment():
    consume_ws_snapshot_buffer()
    render_sidebar_news_compact()
    # (deprecated) original render kept for compatibility


@fragment_decorator(run_every="3s")
def render_live_news_prompt_fragment():
    consume_ws_snapshot_buffer()
    render_agent_prompt_panel(
        "news",
        "📰 뉴스 에이전트 프롬프트 입력값",
        "#0f766e",
        "linear-gradient(135deg, rgba(236,254,255,0.98), rgba(240,249,255,0.98))",
    )


@fragment_decorator(run_every="3s")
def render_live_log_prompt_fragment():
    consume_ws_snapshot_buffer()
    render_agent_prompt_panel(
        "log",
        "📄 로그 에이전트 프롬프트 입력값",
        "#92400e",
        "linear-gradient(135deg, rgba(255,251,235,0.98), rgba(254,243,199,0.98))",
    )


@fragment_decorator(run_every="3s")
def render_live_vector_db_fragment():
    consume_ws_snapshot_buffer()
    render_vector_db_panel()


@fragment_decorator(run_every="5s")
def render_live_faiss_fragment():
    consume_ws_snapshot_buffer()
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
    st.session_state.initial_analysis_done = False

if "initial_analysis_started" not in st.session_state:
    st.session_state.initial_analysis_started = False

if "initial_analysis_failed" not in st.session_state:
    st.session_state.initial_analysis_failed = False

if "initial_analysis_autorun_disabled" not in st.session_state:
    st.session_state.initial_analysis_autorun_disabled = True

if not st.session_state.initial_analysis_done:
    startup_header = st.empty()
    startup_header.subheader("⏳ 초기 데이터 준비")
    startup_status = st.empty()
    startup_status.info("기존 백엔드 상태와 저장된 FAISS를 먼저 확인합니다.")

    try:
        get_backend_client().start_worker(interval_seconds=10)
    except Exception:
        pass

    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
        if status_payload.get("results") or status_payload.get("news") or status_payload.get("vector_count"):
            st.session_state.initial_analysis_done = True
            startup_header.empty()
            startup_status.empty()
    except Exception:
        pass

    if not st.session_state.initial_analysis_done:
        st.session_state.initial_analysis_started = False
        st.session_state.initial_analysis_failed = False
        startup_status.info(
            "자동 전체 분석은 비활성화되어 있습니다. 기존 데이터가 없으면 수동으로 전체 분석을 실행하세요."
        )

    with _background_lock:
        initial_task = _background_results.get("initial_analysis")
    if initial_task:
        if initial_task.get("status") == "completed":
            sync_session_from_backend(initial_task.get("result") or {})
            st.session_state.initial_analysis_done = True
            st.session_state.initial_analysis_failed = False
            startup_header.empty()
            startup_status.empty()
            with _background_lock:
                _background_results.pop("initial_analysis", None)
        elif initial_task.get("status") == "failed":
            st.session_state.initial_analysis_failed = True
            startup_status.warning("초기 분석이 지연되고 있습니다. 화면은 계속 사용할 수 있습니다.")


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

    operations_tab, strategy_tab, news_prompt_tab, log_prompt_tab, vector_db_tab = st.tabs(
        [
            "🤖 운영 현황",
            "💬 AI 심사 전략",
            "📰 뉴스 에이전트 입력",
            "📄 로그 에이전트 입력",
            "🧠 Vector DB",
        ]
    )

    with operations_tab:
        if HAS_FRAGMENT_REFRESH:
            consume_ws_snapshot_buffer()
            render_operations_showcase()
        else:
            render_operations_showcase()

        with st.expander("상세 운영 패널", expanded=False):
            if HAS_FRAGMENT_REFRESH:
                render_live_operations_fragment()
            else:
                render_runtime_dashboard()

    with strategy_tab:
        render_role_based_strategy_tab()

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

    with vector_db_tab:
        if HAS_FRAGMENT_REFRESH:
            render_live_vector_db_fragment()
        else:
            render_vector_db_panel()

    # charts and dedicated FAISS tab removed per UI simplification — FAISS stats shown in header

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
