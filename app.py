import datetime
import html
import json
import os
import re
import threading
import time
import warnings
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)

from agent.strategy_chat import regulation_agent
from agent.strategy_chat import DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE
from agent.strategy_chat import OLLAMA_LIGHTWEIGHT_MODEL
from backend.streamlit_client import BackendClient
from rag.product_pattern_summary import DEFAULT_SUMMARY_PATH, load_product_pattern_summary
from rag.vector_db import (
    FAISS_STORE_CUSTOMER,
    FAISS_STORE_DOCUMENT,
    FAISS_STORE_LOGS,
    FAISS_STORE_NEWS,
    get_vector_count,
    ingest_files,
    list_vectors,
    search_context,
)

# 백그라운드 작업 결과 저장소 (스레드 -> 메인 폴링으로 전달)
_background_results: dict = {}
_background_lock = threading.Lock()
_ws_snapshot_buffer: dict = {}
_ws_snapshot_lock = threading.Lock()
_shared_backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:18000")

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
    base_url = str(
        st.session_state.get(
            "backend_url",
            os.environ.get("BACKEND_URL", "http://127.0.0.1:18000"),
        )
    ).strip() or "http://127.0.0.1:18000"

    global _shared_backend_url
    _shared_backend_url = base_url

    def _run_ws():
        try:
            import json
            try:
                from websocket import WebSocketApp
            except Exception:
                return

            while True:
                try:
                    base = _shared_backend_url or "http://127.0.0.1:18000"
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

        .debate-hero-layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 250px;
            gap: 18px;
            align-items: stretch;
            margin-bottom: 16px;
        }

        .debate-hero-copy {
            min-width: 0;
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

        .debate-launch-panel {
            position: relative;
            overflow: hidden;
            min-height: 100%;
            border-radius: 26px;
            padding: 20px 18px;
            background: radial-gradient(circle at 30% 24%, rgba(97,244,222,0.18), transparent 34%), linear-gradient(160deg, rgba(8,26,39,0.96), rgba(12,39,56,0.94));
            border: 1px solid rgba(97,244,222,0.16);
            box-shadow: 0 18px 44px rgba(0,0,0,0.18);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 16px;
        }

        .debate-launch-panel.ready {
            border-color: rgba(255,191,105,0.22);
        }

        .debate-launch-panel::after {
            content: '';
            position: absolute;
            right: -36px;
            bottom: -42px;
            width: 140px;
            height: 140px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255,191,105,0.20), transparent 68%);
            pointer-events: none;
        }

        .debate-launch-kicker {
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #61f4de;
        }

        .debate-launch-title {
            font-size: 18px;
            font-weight: 900;
            line-height: 1.25;
            color: #f7fbff;
            margin-top: 8px;
        }

        .debate-launch-desc {
            font-size: 12px;
            line-height: 1.65;
            color: #d9ecfb;
            margin-top: 8px;
        }

        [class*="st-key-cardloan_debate_start"] {
            margin-top: 6px;
            margin-bottom: 0;
            position: relative;
            z-index: 8;
            display: flex;
            justify-content: center;
        }

        [class*="st-key-cardloan_debate_start"] button {
            position: relative;
            width: 154px;
            min-width: 154px;
            max-width: 154px;
            height: 154px;
            min-height: 154px;
            border-radius: 999px;
            border: 1px solid rgba(97,244,222,0.18);
            background: radial-gradient(circle at 50% 36%, rgba(255,255,255,0.16), rgba(255,255,255,0.06) 26%, rgba(10,34,50,0.24) 28%), linear-gradient(160deg, rgba(97,244,222,0.98), rgba(255,191,105,0.98));
            color: #082032;
            font-size: 16px;
            font-weight: 900;
            letter-spacing: 0.01em;
            box-shadow: 0 24px 42px rgba(8,32,50,0.24), inset 0 1px 0 rgba(255,255,255,0.22);
            transition: transform 0.22s ease, box-shadow 0.22s ease, filter 0.22s ease;
            padding: 0 24px;
            white-space: normal;
            line-height: 1.3;
        }

        [class*="st-key-cardloan_debate_start"] button::before,
        [class*="st-key-cardloan_debate_start"] button::after {
            content: '';
            position: absolute;
            inset: 12px;
            border-radius: 999px;
            border: 1px dashed rgba(8,32,50,0.18);
            animation: orbSpin 11s linear infinite;
            pointer-events: none;
        }

        [class*="st-key-cardloan_debate_start"] button::after {
            inset: 24px;
            border-style: solid;
            border-color: rgba(255,255,255,0.22);
            animation-duration: 8s;
            animation-direction: reverse;
        }

        [class*="st-key-cardloan_debate_start"] button:hover,
        [class*="st-key-cardloan_debate_start"] button:focus-visible {
            transform: translateY(-4px) scale(1.02);
            filter: saturate(1.08);
            box-shadow: 0 30px 48px rgba(8,32,50,0.28), inset 0 1px 0 rgba(255,255,255,0.28);
            outline: none;
        }

        [class*="st-key-cardloan_debate_start"] button:disabled {
            background: linear-gradient(135deg, rgba(148,163,184,0.92), rgba(203,213,225,0.9));
            color: rgba(15,23,42,0.72);
            border-color: rgba(148,163,184,0.24);
            box-shadow: none;
            transform: none;
        }

        [class*="st-key-cardloan_debate_start"] button p {
            font-size: 16px;
            font-weight: 900;
            line-height: 1.3;
            margin: 0;
        }

        @keyframes orbSpin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        @keyframes launchPulse {
            0%, 100% { transform: scale(0.94); opacity: 0.92; }
            50% { transform: scale(1.04); opacity: 1; }
        }

        @media (max-width: 1080px) {
            .debate-hero-layout {
                grid-template-columns: 1fr;
            }

            .debate-launch-panel {
                min-height: 240px;
            }
        }

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

        .reviewer-card.speaking {
            transform: translateY(-4px);
        }

        .reviewer-card.conservative.speaking {
            border-color: rgba(255,143,143,0.32);
            box-shadow: 0 20px 36px rgba(0,0,0,0.22), inset 0 0 0 1px rgba(255,143,143,0.12);
        }

        .reviewer-card.sales.speaking {
            border-color: rgba(97,244,222,0.30);
            box-shadow: 0 20px 36px rgba(0,0,0,0.22), inset 0 0 0 1px rgba(97,244,222,0.12);
        }

        .reviewer-card.product.speaking {
            border-color: rgba(255,191,105,0.32);
            box-shadow: 0 20px 36px rgba(0,0,0,0.22), inset 0 0 0 1px rgba(255,191,105,0.12);
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
            margin-top: -214px;
            margin-bottom: 12px;
            position: relative;
            z-index: 8;
            padding-right: 0;
            height: 206px;
        }

        [class*="st-key-edit_reviewer_prompt_"] button {
            position: relative;
            width: 100%;
            height: 206px;
            min-height: 206px;
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

        .debate-live-shell {
            margin: 12px 0 18px 0;
            padding: 18px 20px;
            border-radius: 24px;
            background: linear-gradient(145deg, rgba(10,21,33,0.98), rgba(18,35,52,0.94));
            color: white;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 22px 42px rgba(3,12,21,0.24);
            position: relative;
            overflow: hidden;
        }

        .debate-live-shell::before {
            content: '';
            position: absolute;
            inset: auto -10% 0 auto;
            width: 260px;
            height: 260px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(97,244,222,0.14), transparent 65%);
            pointer-events: none;
        }

        .debate-live-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 10px;
        }

        .debate-live-kicker {
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.08em;
            opacity: 0.7;
            text-transform: uppercase;
        }

        .debate-live-title {
            font-size: 20px;
            font-weight: 900;
            margin-top: 4px;
        }

        .debate-live-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }

        .debate-live-broadcast {
            margin-bottom: 12px;
            padding: 12px 14px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(97,244,222,0.10), rgba(125,211,252,0.08));
            border: 1px solid rgba(97,244,222,0.16);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
            position: relative;
            overflow: hidden;
        }

        .debate-live-broadcast::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(105deg, transparent 18%, rgba(255,255,255,0.08) 50%, transparent 82%);
            transform: translateX(-120%);
            animation: speakingSweep 2.8s ease-in-out infinite;
            pointer-events: none;
        }

        .debate-live-broadcast-title {
            position: relative;
            z-index: 1;
            font-size: 13px;
            font-weight: 900;
            color: #f7fbff;
            margin-bottom: 6px;
        }

        .debate-live-broadcast-body {
            position: relative;
            z-index: 1;
            font-size: 12px;
            line-height: 1.7;
            color: rgba(217,236,251,0.90);
        }

        .debate-live-helper {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            font-size: 11px;
            font-weight: 700;
            color: rgba(226,238,247,0.88);
        }

        .debate-live-body {
            position: relative;
            padding: 18px 18px 22px 18px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.08);
            min-height: 122px;
            backdrop-filter: blur(12px);
        }

        .debate-live-body.streaming {
            animation: liveBreathing 2.4s ease-in-out infinite;
        }

        .debate-live-message {
            font-size: 14px;
            line-height: 1.82;
            color: rgba(241,245,249,0.96);
            white-space: pre-wrap;
            position: relative;
            z-index: 1;
        }

        .debate-live-message.thinking {
            color: rgba(209,225,239,0.92);
        }

        .debate-live-cursor {
            display: inline-block;
            width: 9px;
            height: 1.05em;
            margin-left: 4px;
            border-radius: 2px;
            vertical-align: -2px;
            background: linear-gradient(180deg, rgba(97,244,222,0.96), rgba(255,191,105,0.92));
            animation: liveCursorBlink 1s steps(1, end) infinite;
            box-shadow: 0 0 10px rgba(97,244,222,0.22);
        }

        .debate-live-dots {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            margin-top: 14px;
            position: relative;
            z-index: 1;
        }

        .debate-live-dots span {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(97,244,222,0.95), rgba(148,163,184,0.7));
            animation: liveDots 1.2s ease-in-out infinite;
            box-shadow: 0 0 0 6px rgba(97,244,222,0.06);
        }

        .debate-live-dots span:nth-child(2) { animation-delay: 0.18s; }
        .debate-live-dots span:nth-child(3) { animation-delay: 0.36s; }

        .debate-live-caption {
            margin-top: 12px;
            font-size: 12px;
            line-height: 1.7;
            color: rgba(191,219,254,0.88);
        }

        @keyframes liveCursorBlink {
            0%, 48% { opacity: 1; }
            50%, 100% { opacity: 0; }
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

        @keyframes liveBreathing {
            0%, 100% { transform: translateY(0); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03); }
            50% { transform: translateY(-1px); box-shadow: inset 0 0 0 1px rgba(97,244,222,0.08); }
        }

        @keyframes liveDots {
            0%, 80%, 100% { transform: translateY(0) scale(0.92); opacity: 0.45; }
            40% { transform: translateY(-4px) scale(1.08); opacity: 1; }
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

        .reviewer-live-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin: 12px 0 10px 0;
        }

        .reviewer-status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
        }

        .reviewer-live-speech {
            position: relative;
            margin-top: 10px;
            padding: 14px 14px 15px 14px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
            border: 1px solid rgba(255,255,255,0.10);
            min-height: 96px;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }

        .reviewer-live-speech::before {
            content: '';
            position: absolute;
            left: 28px;
            bottom: -8px;
            width: 16px;
            height: 16px;
            background: rgba(255,255,255,0.08);
            border-right: 1px solid rgba(255,255,255,0.10);
            border-bottom: 1px solid rgba(255,255,255,0.10);
            transform: rotate(45deg);
        }

        .reviewer-live-speech.speaking::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(105deg, transparent 16%, rgba(255,255,255,0.12) 50%, transparent 84%);
            transform: translateX(-120%);
            animation: speakingSweep 2.8s ease-in-out infinite;
            pointer-events: none;
        }

        .reviewer-live-label {
            font-size: 10px;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(191,219,254,0.82);
            margin-bottom: 8px;
            position: relative;
            z-index: 1;
        }

        .reviewer-live-text {
            font-size: 13px;
            line-height: 1.68;
            color: #edf6ff;
            white-space: pre-wrap;
            position: relative;
            z-index: 1;
        }

        .reviewer-live-text.thinking {
            color: #d9ecfb;
        }

        .reviewer-live-dots {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-top: 10px;
            position: relative;
            z-index: 1;
        }

        .reviewer-live-dots span {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(97,244,222,0.95), rgba(255,255,255,0.7));
            animation: liveDots 1.2s ease-in-out infinite;
        }

        .reviewer-live-dots span:nth-child(2) { animation-delay: 0.18s; }
        .reviewer-live-dots span:nth-child(3) { animation-delay: 0.36s; }
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

        [class*="st-key-main_dashboard_section"] {
            margin: 2px 0 16px 0;
            padding: 10px 10px 12px;
            border-radius: 24px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01)),
                linear-gradient(180deg, rgba(8,26,39,0.78), rgba(10,34,50,0.66));
            border: 1px solid rgba(151, 196, 225, 0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 12px 28px rgba(0,0,0,0.16);
        }

        [class*="st-key-main_dashboard_section"] [role="radiogroup"] {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1.28fr) minmax(0, 1fr) minmax(0, 0.94fr);
            align-items: stretch;
            gap: 8px;
            width: 100%;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1.28fr) minmax(0, 1fr) minmax(0, 0.94fr);
            align-items: stretch;
            gap: 8px;
            width: 100%;
            background: transparent !important;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"] {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            position: relative;
            isolation: isolate;
            overflow: hidden;
            min-height: 68px;
            min-width: 0;
            width: 100%;
            padding: 25px 52px 12px 16px;
            border-radius: 18px 18px 14px 14px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)),
                linear-gradient(180deg, rgba(14,35,49,0.78), rgba(8,24,37,0.92));
            border: 1px solid rgba(151, 196, 225, 0.10) !important;
            color: #d9ecfb !important;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: -0.01em;
            transition: transform 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease, background 0.24s ease;
            box-shadow: inset 0 -1px 0 rgba(255,255,255,0.03) !important;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2),
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2) {
            min-height: 76px;
            padding-top: 27px;
            padding-right: 58px;
            border-radius: 20px 20px 16px 16px;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]::before {
            content: '';
            position: absolute;
            left: 14px;
            top: 13px;
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: rgba(156, 196, 223, 0.72);
            box-shadow: 0 0 0 0 rgba(156, 196, 223, 0.26);
            animation: dashboardSectionPulse 2.4s ease-in-out infinite;
            z-index: 2;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]::after {
            position: absolute;
            top: 9px;
            right: 10px;
            max-width: calc(100% - 34px);
            padding: 3px 7px;
            border-radius: 999px;
            font-size: 9px;
            font-weight: 900;
            letter-spacing: 0.07em;
            line-height: 1;
            color: #d9ecfb;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            white-space: nowrap;
            pointer-events: none;
            z-index: 2;
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1)::after {
            content: 'LIVE';
            color: #c9fdf5;
            background: rgba(97,244,222,0.12);
            border-color: rgba(97,244,222,0.18);
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2)::after {
            content: '3 AI';
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(3)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(3)::after {
            content: 'FEED';
            color: #d7fff9;
            background: rgba(14,165,233,0.12);
            border-color: rgba(14,165,233,0.18);
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(4)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(4)::after {
            content: 'TRACE';
            color: #ffe8c2;
            background: rgba(255,191,105,0.12);
            border-color: rgba(255,191,105,0.18);
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:hover,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:hover {
            background:
                linear-gradient(180deg, rgba(97,244,222,0.12), rgba(34,211,238,0.04)),
                linear-gradient(180deg, rgba(14,35,49,0.82), rgba(8,24,37,0.95));
            border-color: rgba(97,244,222,0.26) !important;
            color: #f7fbff !important;
            transform: translateY(-3px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.16), inset 0 -2px 0 rgba(97,244,222,0.22) !important;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:hover::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:hover::after {
            transform: translateY(-2px) scale(1.02);
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"],
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked) {
            background:
                linear-gradient(120deg, rgba(255,255,255,0.10), rgba(255,255,255,0.00) 42%),
                linear-gradient(135deg, rgba(13,45,62,0.98), rgba(18,68,84,0.94)) !important;
            border-color: rgba(97,244,222,0.34) !important;
            color: #f7fbff !important;
            box-shadow:
                0 14px 28px rgba(0,0,0,0.20),
                inset 0 0 0 1px rgba(255,255,255,0.04),
                inset 0 -3px 0 rgba(97,244,222,0.72) !important;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"]::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked)::before {
            background: #61f4de;
            box-shadow: 0 0 0 0 rgba(97,244,222,0.34);
            animation-duration: 1.4s;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"] span,
        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"] div,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked) span,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked) div {
            position: relative;
            z-index: 1;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"]::selection,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked)::selection {
            background: transparent;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"]::marker,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked)::marker {
            content: '';
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"]::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked)::after {
            box-shadow: 0 0 18px rgba(97,244,222,0.16);
            transform: translateY(-1px);
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)[aria-pressed="true"],
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1):has(input:checked) {
            border-color: rgba(97,244,222,0.42) !important;
            box-shadow:
                0 14px 28px rgba(0,0,0,0.22),
                0 0 26px rgba(97,244,222,0.16),
                inset 0 0 0 1px rgba(255,255,255,0.05) !important;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)[aria-pressed="true"]::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1):has(input:checked)::before {
            animation: dashboardOpsPulse 1.05s ease-in-out infinite;
            box-shadow: 0 0 0 0 rgba(97,244,222,0.42);
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)[aria-pressed="true"]::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1):has(input:checked)::after {
            box-shadow: 0 0 22px rgba(97,244,222,0.24);
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"] {
            background-image:
                linear-gradient(115deg, transparent 0%, transparent 38%, rgba(255,255,255,0.10) 50%, transparent 62%, transparent 100%),
                linear-gradient(135deg, rgba(13,45,62,0.98), rgba(18,68,84,0.94)) !important;
            background-size: 220% 100%, 100% 100% !important;
            animation: dashboardSectionSweep 3.6s linear infinite;
        }

        @keyframes dashboardSectionPulse {
            0% { box-shadow: 0 0 0 0 rgba(156,196,223,0.28); transform: scale(1); }
            65% { box-shadow: 0 0 0 9px rgba(156,196,223,0.0); transform: scale(1.08); }
            100% { box-shadow: 0 0 0 0 rgba(156,196,223,0.0); transform: scale(1); }
        }

        @keyframes dashboardSectionSweep {
            0% { background-position: 130% 0, 0 0; }
            100% { background-position: -90% 0, 0 0; }
        }

        @keyframes dashboardOpsPulse {
            0% { box-shadow: 0 0 0 0 rgba(97,244,222,0.42); transform: scale(1); }
            60% { box-shadow: 0 0 0 12px rgba(97,244,222,0.0); transform: scale(1.12); }
            100% { box-shadow: 0 0 0 0 rgba(97,244,222,0.0); transform: scale(1); }
        }

        @keyframes dashboardSectionTitleFloat {
            0% { transform: translateY(0); opacity: 0.94; }
            50% { transform: translateY(-2px); opacity: 1; }
            100% { transform: translateY(0); opacity: 0.94; }
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button p,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"] p,
        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button div,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"] div {
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            font-size: 13px !important;
            font-weight: 800 !important;
            color: inherit !important;
            line-height: 1.1 !important;
            text-align: left !important;
            white-space: normal !important;
            word-break: keep-all !important;
            text-wrap: balance;
            transform-origin: left center;
            transition: transform 0.24s ease, text-shadow 0.24s ease, opacity 0.24s ease;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button > div,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"] > div {
            padding-top: 4px;
        }

        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"] input {
            display: none;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"] p,
        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"] div,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked) p,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked) div {
            text-shadow: 0 0 18px rgba(151, 244, 222, 0.16);
            animation: dashboardSectionTitleFloat 2.8s ease-in-out infinite;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)[aria-pressed="true"] p,
        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)[aria-pressed="true"] div,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1):has(input:checked) p,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1):has(input:checked) div {
            animation-duration: 1.8s;
        }

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2) p,
        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2) div,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2) p,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2) div {
            font-size: 14px !important;
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
            margin-top: 10px;
            margin-bottom: 0;
            padding: 16px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88));
            border: 1px solid rgba(97, 244, 222, 0.14);
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22);
            transition: border-color 0.22s ease, box-shadow 0.22s ease, transform 0.22s ease;
        }

        .upload-shell:hover {
            border-color: rgba(97, 244, 222, 0.28);
            box-shadow: 0 22px 42px rgba(0, 0, 0, 0.24), 0 0 0 1px rgba(97, 244, 222, 0.08);
            transform: translateY(-1px);
        }

        .regulation-intake-anchor {
            width: 0;
            height: 0;
            overflow: hidden;
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
            margin-top: -4px;
            margin-bottom: 12px;
            padding: 12px 16px 16px 16px;
            background: linear-gradient(180deg, rgba(8,26,39,0.92), rgba(10,34,50,0.88));
            border-left: 1px solid rgba(97,244,222,0.14);
            border-right: 1px solid rgba(97,244,222,0.14);
            border-bottom: 1px solid rgba(97,244,222,0.14);
            border-radius: 0 0 20px 20px;
            box-shadow: 0 18px 38px rgba(0,0,0,0.22);
            position: relative;
            overflow: visible;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] {
            margin-bottom: 0;
            padding: 18px 12px 12px 12px;
            border-radius: 18px;
            border: 1px dashed rgba(97,244,222,0.28);
            background: linear-gradient(180deg, rgba(9,28,43,0.80), rgba(6,23,35,0.96));
            transition: border-color 0.22s ease, background 0.22s ease, box-shadow 0.22s ease;
            position: relative;
            overflow: visible;
            z-index: 1;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] section {
            min-height: 132px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            background: radial-gradient(circle at top, rgba(97,244,222,0.08), transparent 58%);
            transition: background 0.22s ease;
            padding-top: 14px;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] button {
            position: absolute;
            top: -18px;
            left: 50%;
            z-index: 3;
            border-radius: 999px;
            min-height: 40px;
            padding: 0 18px;
            background: linear-gradient(135deg, rgba(97,244,222,0.16), rgba(255,191,105,0.16));
            border: 1px solid rgba(97,244,222,0.24);
            color: #f7fbff;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(0,0,0,0.18);
            opacity: 0;
            transform: translate(-50%, 8px);
            transition: opacity 0.22s ease, transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] button:hover {
            border-color: rgba(97,244,222,0.34);
            background: linear-gradient(135deg, rgba(97,244,222,0.22), rgba(255,191,105,0.22));
        }

        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):hover .upload-shell,
        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):has([class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within) .upload-shell {
            border-color: rgba(97, 244, 222, 0.28);
            box-shadow: 0 22px 42px rgba(0, 0, 0, 0.24), 0 0 0 1px rgba(97, 244, 222, 0.08);
            transform: translateY(-1px);
        }

        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):hover [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"],
        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):has([class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within) [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] {
            border-color: rgba(97,244,222,0.38);
            background: linear-gradient(180deg, rgba(10,33,49,0.86), rgba(7,26,39,0.98));
            box-shadow: 0 16px 32px rgba(0,0,0,0.18), inset 0 0 0 1px rgba(97,244,222,0.06);
        }

        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):hover [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] section,
        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):has([class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within) [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] section {
            background: radial-gradient(circle at top, rgba(97,244,222,0.14), transparent 58%);
        }

        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):hover [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] button,
        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):has([class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within) [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"] button {
            opacity: 1;
            transform: translate(-50%, 0);
        }

        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):hover [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]::after,
        div[data-testid="stVerticalBlock"]:has(.regulation-intake-anchor):has([class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within) [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]::after {
            opacity: 0;
            transform: translate(-50%, -4px);
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
            transform: translate(-50%, 0);
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]::after {
            content: 'Hover to reveal upload';
            position: absolute;
            top: -18px;
            left: 50%;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: #9cc4df;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.04em;
            pointer-events: none;
            z-index: 2;
            transform: translateX(-50%);
            transition: opacity 0.22s ease, transform 0.22s ease;
        }

        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:hover::after,
        [class*="st-key-sidebar_reg_upload"] [data-testid="stFileUploader"]:focus-within::after {
            opacity: 0;
            transform: translate(-50%, -4px);
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


def _format_ollama_toast_agent_label(agent: str) -> str:
    normalized = str(agent or "").strip()
    if normalized == "log_agent":
        return "로그 에이전트"
    if normalized == "news_agent":
        return "뉴스 에이전트"
    if normalized == "credit_planning_agent":
        return "신용기획부"
    if normalized == "sales_strategy_agent":
        return "금융영업부"
    if normalized == "solution_planning_agent":
        return "금융솔루션부"
    if normalized == "decision_agent":
        return "의사결정 에이전트"
    if normalized == "regulation_agent":
        return "규정 에이전트"
    if normalized == "orchestrator":
        return "오케스트레이터"
    if not normalized:
        return "Ollama"
    return normalized.replace("_", " ").title()


def render_ollama_toast():
    toast = st.session_state.get("ollama_toast")
    if not toast:
        return
    try:
        age = time.time() - float(toast.get("ts", 0))
    except Exception:
        age = 9999
    if age > 5:
        return

    css = """
    <style>
        .ollama-toast {
      position: fixed;
      right: 20px;
      top: 20px;
      z-index: 9999;
            min-width: 280px;
            max-width: 420px;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(14,116,144,0.94));
      color: white;
            padding: 14px 16px 15px 16px;
            border-radius: 16px;
            border: 1px solid rgba(103,232,249,0.28);
            box-shadow: 0 14px 34px rgba(15,23,42,0.34);
            overflow: hidden;
            animation: ollamaToastIn 0.45s ease-out;
    }
        .ollama-toast::before {
            content: '';
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, #67e8f9, #fde68a, #67e8f9);
            background-size: 200% 100%;
            animation: ollamaToastSweep 1.8s linear infinite;
        }
        .ollama-toast-kicker {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #67e8f9;
            margin-bottom: 6px;
        }
        .ollama-toast-title {
            font-size: 14px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 4px;
        }
        .ollama-toast-body {
            font-size: 12px;
            line-height: 1.55;
            color: rgba(226,232,240,0.94);
        }
        @keyframes ollamaToastIn {
            from { transform: translateY(-12px) scale(0.96); opacity: 0 }
            to { transform: translateY(0) scale(1); opacity: 1 }
        }
        @keyframes ollamaToastSweep {
            from { background-position: 0% 0; }
            to { background-position: 200% 0; }
    }
    </style>
    """

    kicker = html.escape(str(toast.get("kicker", "OLLAMA COMPLETED")))
    title = html.escape(str(toast.get("title", "Ollama 작업 완료")))
    body = html.escape(str(toast.get("msg", "")))
    st.markdown(
        css
        + (
            "<div class='ollama-toast'>"
            f"<div class='ollama-toast-kicker'>{kicker}</div>"
            f"<div class='ollama-toast-title'>{title}</div>"
            f"<div class='ollama-toast-body'>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# render toast (if any) early so it appears above content
try:
    render_ollama_toast()
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
            box-shadow: 0 0 0 1px rgba(56,189,248,0.12), 0 10px 24px rgba(14,165,233,0.12);
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
        .loading-step.active .loading-badge {
            animation: loadingPulse 1.2s ease-in-out infinite;
        }
        .loading-meta {
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.04);
            color: #334155;
            font-size: 13px;
        }
        .loading-hero {
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            margin-bottom: 14px;
            border: 1px solid rgba(125, 211, 252, 0.28);
            background: radial-gradient(circle at 20% 10%, rgba(125,211,252,0.18), transparent 30%), linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,118,110,0.90));
            color: white;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.22);
        }
        .loading-orbit {
            position: absolute;
            top: -30px;
            right: -20px;
            width: 140px;
            height: 140px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
        }
        .loading-orbit::before,
        .loading-orbit::after {
            content: '';
            position: absolute;
            inset: 12px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.10);
            animation: orbitSpin 8s linear infinite;
        }
        .loading-orbit::after {
            inset: 28px;
            animation-duration: 5s;
            animation-direction: reverse;
        }
        .loading-progress-track {
            margin-top: 14px;
            height: 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            overflow: hidden;
        }
        .loading-progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #67e8f9, #a7f3d0, #fef08a);
            background-size: 200% 100%;
            animation: progressWave 2.4s linear infinite;
        }
        .loading-stage-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
            margin-top: 14px;
        }
        .loading-stage {
            position: relative;
            overflow: hidden;
            min-height: 42px;
            padding: 10px 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: rgba(255,255,255,0.72);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.04em;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .loading-stage.active {
            color: white;
            border-color: rgba(103,232,249,0.65);
            box-shadow: 0 0 0 1px rgba(103,232,249,0.12), 0 12px 24px rgba(34,211,238,0.16);
        }
        .loading-stage.active::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.20), transparent);
            animation: loadingSweep 1.7s linear infinite;
        }
        .loading-stage.done {
            color: #ecfeff;
            background: linear-gradient(135deg, rgba(34,197,94,0.34), rgba(45,212,191,0.22));
            border-color: rgba(134,239,172,0.42);
        }
        .loading-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }
        .loading-metric {
            padding: 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(6px);
        }
        .loading-metric-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(255,255,255,0.72);
            margin-bottom: 4px;
        }
        .loading-metric-value {
            font-size: 24px;
            font-weight: 800;
            color: white;
        }
        .loading-metric-sub {
            font-size: 12px;
            color: rgba(255,255,255,0.78);
            margin-top: 2px;
        }
        .loading-live-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }
        .loading-live-card {
            position: relative;
            overflow: hidden;
            padding: 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            min-height: 80px;
        }
        .loading-live-card.running {
            border-color: rgba(103,232,249,0.48);
            box-shadow: 0 10px 26px rgba(8,145,178,0.18);
        }
        .loading-live-card.running::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.14), transparent);
            animation: loadingSweep 1.6s linear infinite;
        }
        .loading-live-card.completed {
            border-color: rgba(134,239,172,0.34);
            background: linear-gradient(135deg, rgba(22,163,74,0.22), rgba(16,185,129,0.14));
        }
        .loading-live-label {
            position: relative;
            z-index: 1;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.74);
            margin-bottom: 8px;
        }
        .loading-live-detail {
            position: relative;
            z-index: 1;
            font-size: 13px;
            line-height: 1.55;
            color: rgba(255,255,255,0.92);
        }
        .loading-activity-panel {
            margin-top: 14px;
            border-radius: 16px;
            padding: 14px;
            background: rgba(5,16,28,0.34);
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(8px);
        }
        .loading-activity-head {
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(103,232,249,0.94);
            margin-bottom: 10px;
        }
        .loading-activity-row {
            display: grid;
            grid-template-columns: 12px minmax(0, 1fr) auto;
            gap: 10px;
            align-items: start;
            padding: 9px 0;
            border-top: 1px solid rgba(255,255,255,0.08);
        }
        .loading-activity-row:first-of-type {
            border-top: 0;
            padding-top: 0;
        }
        .loading-activity-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-top: 4px;
            background: rgba(148,163,184,0.88);
        }
        .loading-activity-dot.running {
            background: #67e8f9;
            box-shadow: 0 0 0 0 rgba(103,232,249,0.54);
            animation: loadingPulse 1.3s ease-in-out infinite;
        }
        .loading-activity-dot.completed {
            background: #86efac;
        }
        .loading-activity-dot.failed {
            background: #fca5a5;
        }
        .loading-activity-title {
            font-size: 12px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 2px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .loading-activity-detail {
            font-size: 12px;
            line-height: 1.5;
            color: rgba(226,232,240,0.92);
        }
        .loading-activity-time {
            font-size: 11px;
            color: rgba(191,219,254,0.72);
            white-space: nowrap;
            padding-top: 1px;
        }
        .loading-activity-empty {
            font-size: 12px;
            color: rgba(226,232,240,0.82);
            line-height: 1.6;
        }
        .backend-restart-overlay {
            position: fixed;
            inset: 0;
            z-index: 9998;
            background: linear-gradient(180deg, rgba(2,6,23,0.82), rgba(2,6,23,0.90));
            backdrop-filter: blur(14px);
            padding: 34px 26px;
            overflow-y: auto;
        }
        .backend-restart-shell {
            max-width: 1080px;
            margin: 0 auto;
        }
        .backend-restart-explain {
            margin-top: 14px;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .backend-restart-item {
            border-radius: 14px;
            padding: 12px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: rgba(241,245,249,0.92);
            min-height: 82px;
        }
        .backend-restart-item-title {
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #67e8f9;
            margin-bottom: 6px;
        }
        .backend-restart-item-detail {
            font-size: 12px;
            line-height: 1.55;
            color: rgba(226,232,240,0.92);
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
        @keyframes orbitSpin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes progressWave {
            0% { background-position: 0% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes loadingPulse {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(14,165,233,0.45); }
            50% { transform: scale(1.08); box-shadow: 0 0 0 8px rgba(14,165,233,0); }
        }
        @keyframes loadingSweep {
            0% { transform: translateX(-130%); }
            100% { transform: translateX(130%); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _start_initial_analysis_background(log_dir: str = "data/logs") -> bool:
    base_url = str(
        st.session_state.get(
            "backend_url",
            os.environ.get("BACKEND_URL", "http://127.0.0.1:18000"),
        )
    ).strip() or "http://127.0.0.1:18000"

    global _shared_backend_url
    _shared_backend_url = base_url

    with _background_lock:
        task = _background_results.get("initial_analysis") or {}
        if task.get("status") in {"running", "completed"}:
            return False
        _background_results["initial_analysis"] = {
            "status": "running",
            "started_at": datetime.datetime.now().isoformat(),
            "base_url": base_url,
        }

    def _run_initial_analysis() -> None:
        try:
            result = BackendClient(base_url).run_full_analysis(
                log_dir=log_dir,
                collect_news=False,
            )
            with _background_lock:
                _background_results["initial_analysis"] = {
                    "status": "completed",
                    "updated_at": datetime.datetime.now().isoformat(),
                    "result": result,
                }
        except Exception as error:
            with _background_lock:
                _background_results["initial_analysis"] = {
                    "status": "failed",
                    "updated_at": datetime.datetime.now().isoformat(),
                    "error": str(error),
                }

    threading.Thread(target=_run_initial_analysis, daemon=True).start()
    return True


def _is_initial_dashboard_ready(status_payload: dict[str, Any] | None) -> bool:
    payload = status_payload or {}
    return bool(payload.get("last_run_time") and payload.get("last_faiss_time"))


def _set_backend_bootstrap_mode(active: bool, reason: str | None = None) -> bool:
    previous = bool(st.session_state.get("initial_analysis_done", False))
    st.session_state.initial_analysis_done = not active
    if active:
        st.session_state.initial_analysis_failed = False
        if reason:
            st.session_state.initial_loading_reason = reason
        with _background_lock:
            existing_task = dict(_background_results.get("initial_analysis") or {})
            if existing_task.get("status") == "completed":
                _background_results.pop("initial_analysis", None)
                st.session_state.initial_analysis_started = False
    else:
        st.session_state.initial_loading_reason = None
    return previous != st.session_state.initial_analysis_done


def _sync_backend_bootstrap_state(status_payload: dict[str, Any] | None) -> bool:
    payload = status_payload or {}
    if not payload:
        return _set_backend_bootstrap_mode(
            True, reason="백엔드 재연결 및 파이프라인 재기동 대기"
        )
    if _is_initial_dashboard_ready(payload):
        return _set_backend_bootstrap_mode(False)
    loading_state = _derive_initial_loading_state(payload)
    return _set_backend_bootstrap_mode(
        True,
        reason=str(
            loading_state.get("latest_phase")
            or "백엔드 재기동 후 초기 파이프라인 재실행 중"
        ),
    )


def _derive_initial_loading_state(
    status_payload: dict[str, Any] | None,
    initial_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = status_payload or {}
    task = initial_task or {}
    diagnostics = payload.get("backend_diagnostics") or {}
    worker_runtime = diagnostics.get("worker_runtime") or {}
    agent_statuses = payload.get("agent_statuses") or {}
    activity_log = payload.get("agent_activity_log") or []
    ollama_runtime = payload.get("ollama_runtime") or {}
    results_count = len(payload.get("results") or [])
    news_count = len(payload.get("news") or [])
    vector_count = int(payload.get("vector_count", 0) or 0)
    backend_reachable = bool(payload)
    task_started = task.get("status") in {"running", "completed"}
    full_analysis_finished = bool(payload.get("last_run_time")) or task.get("status") == "completed"
    faiss_finished = bool(payload.get("last_faiss_time"))
    ready = _is_initial_dashboard_ready(payload)

    active_step = 0
    if backend_reachable:
        active_step = 1
    if task_started:
        active_step = 2
    if results_count > 0 or news_count > 0 or worker_runtime.get("log_analysis_ran") or worker_runtime.get("news_cycle_ran"):
        active_step = 3
    if full_analysis_finished or vector_count > 0 or worker_runtime.get("log_vector_append_ran"):
        active_step = 4
    if faiss_finished or worker_runtime.get("faiss_cycle_ran"):
        active_step = 5
    if ready:
        active_step = 5

    progress_percent = [10, 22, 38, 62, 84, 100][active_step]
    latest_phase = "백엔드 연결 대기"
    if faiss_finished or worker_runtime.get("faiss_cycle_ran"):
        latest_phase = "FAISS 로드 및 인덱스 동기화"
    elif full_analysis_finished:
        latest_phase = "초기 전체 분석 마무리"
    elif news_count > 0 or worker_runtime.get("news_cycle_ran") or payload.get("news_crawl_running"):
        latest_phase = "뉴스 수집 및 이슈 정리"
    elif results_count > 0 or worker_runtime.get("log_analysis_ran"):
        latest_phase = "로그 분석 진행"
    elif task_started:
        latest_phase = "초기 전체 분석 시작"
    elif backend_reachable:
        latest_phase = "백엔드 연결 확인"

    phase_items = [
        {"label": "백엔드 연결", "done": active_step > 1, "active": active_step == 1},
        {"label": "초기 분석 시작", "done": active_step > 2, "active": active_step == 2},
        {"label": "로그/뉴스 분석", "done": active_step > 3, "active": active_step == 3},
        {"label": "전체 분석 완료", "done": active_step > 4, "active": active_step == 4},
        {"label": "FAISS 로드", "done": ready, "active": active_step == 5 and not ready},
    ]

    live_cards: list[dict[str, str]] = []
    log_status = agent_statuses.get("log_analyzer") or {}
    news_status = agent_statuses.get("news_collector") or {}
    vector_status = agent_statuses.get("vector_store") or agent_statuses.get("startup_sequence") or {}
    if log_status or worker_runtime.get("log_analysis_ran"):
        live_cards.append(
            {
                "label": "로그 분석",
                "state": str(
                    log_status.get("status")
                    or ("running" if worker_runtime.get("log_analysis_ran") else "pending")
                ),
                "detail": str(
                    log_status.get("detail")
                    or f"최근 {int(worker_runtime.get('log_analysis_elapsed_ms', 0) or 0)}ms"
                ),
            }
        )
    if news_status or payload.get("news_crawl_running") or worker_runtime.get("news_cycle_ran"):
        news_detail = str(news_status.get("detail") or "뉴스 수집 사이클 점검 중")
        if payload.get("news_crawl_running"):
            news_detail = (
                f"본문 크롤링 {int(payload.get('news_crawl_success_count', 0) or 0)}/"
                f"{int(payload.get('news_crawl_target_count', 0) or 0)}"
            )
        live_cards.append(
            {
                "label": "뉴스 수집",
                "state": "running"
                if payload.get("news_crawl_running")
                else str(
                    news_status.get("status")
                    or ("running" if worker_runtime.get("news_cycle_ran") else "pending")
                ),
                "detail": news_detail,
            }
        )
    if vector_status or worker_runtime.get("faiss_cycle_ran") or worker_runtime.get("log_vector_append_ran"):
        live_cards.append(
            {
                "label": "벡터 동기화",
                "state": str(
                    vector_status.get("status")
                    or (
                        "running"
                        if worker_runtime.get("faiss_cycle_ran") or worker_runtime.get("log_vector_append_ran")
                        else "pending"
                    )
                ),
                "detail": str(vector_status.get("detail") or f"현재 {vector_count} vectors"),
            }
        )
    if str(ollama_runtime.get("status") or "") == "running":
        live_cards.append(
            {
                "label": "Ollama 생성",
                "state": "running",
                "detail": str(ollama_runtime.get("agent") or "에이전트 응답 생성 중"),
            }
        )

    activity_items: list[dict[str, str]] = []
    for event in list(activity_log)[:5]:
        activity_items.append(
            {
                "source": str(event.get("source") or "system"),
                "status": str(event.get("status") or "pending"),
                "detail": str(event.get("detail") or ""),
                "timestamp": format_status_time(event.get("timestamp")),
            }
        )
    return {
        "active_step": active_step,
        "progress_percent": progress_percent,
        "latest_phase": latest_phase,
        "results_count": results_count,
        "news_count": news_count,
        "vector_count": vector_count,
        "worker_runtime": worker_runtime,
        "phase_items": phase_items,
        "live_cards": live_cards,
        "activity_items": activity_items,
        "loading_reason": st.session_state.get("initial_loading_reason") or latest_phase,
        "ready": ready,
    }


def render_initial_loading_hero(
    target,
    loading_state: dict[str, Any],
    eta_text: str,
    elapsed_text: str,
) -> None:
    latest_phase = html.escape(str(loading_state.get("latest_phase") or "백엔드 준비"))
    progress_percent = int(loading_state.get("progress_percent", 18) or 18)
    results_count = int(loading_state.get("results_count", 0) or 0)
    news_count = int(loading_state.get("news_count", 0) or 0)
    vector_count = int(loading_state.get("vector_count", 0) or 0)
    worker_runtime = loading_state.get("worker_runtime") or {}
    phase_items = loading_state.get("phase_items") or []
    loading_reason = html.escape(str(loading_state.get("loading_reason") or latest_phase))
    live_cards = loading_state.get("live_cards") or []
    activity_items = loading_state.get("activity_items") or []
    phase_sub = html.escape(
        ", ".join(
            part
            for part in [
                f"로그 분석 {int(worker_runtime.get('log_analysis_elapsed_ms', 0) or 0)}ms" if worker_runtime.get("log_analysis_ran") else "",
                f"벡터 적재 {int(worker_runtime.get('log_vector_append_elapsed_ms', 0) or 0)}ms" if worker_runtime.get("log_vector_append_ran") else "",
                f"FAISS {int(worker_runtime.get('faiss_cycle_elapsed_ms', 0) or 0)}ms" if worker_runtime.get("faiss_cycle_ran") else "",
            ]
            if part
        ) or "진행 신호를 수집하는 중"
    )
    stage_html = "".join(
        f"<div class='loading-stage{' done' if item.get('done') else (' active' if item.get('active') else '')}'><span>{html.escape(str(item.get('label') or '-'))}</span></div>"
        for item in phase_items
    )
    cards_html = "".join(
        (
            f"<div class='loading-live-card {html.escape(str(card.get('state') or 'pending'))}'>"
            f"<div class='loading-live-label'>{html.escape(str(card.get('label') or '-'))}</div>"
            f"<div class='loading-live-detail'>{html.escape(str(card.get('detail') or '-'))}</div>"
            "</div>"
        )
        for card in live_cards[:4]
    )
    activity_html = "".join(
        (
            "<div class='loading-activity-row'>"
            f"<div class='loading-activity-dot {html.escape(str(item.get('status') or 'pending'))}'></div>"
            f"<div class='loading-activity-body'><div class='loading-activity-title'>{html.escape(str(item.get('source') or '-'))}</div>"
            f"<div class='loading-activity-detail'>{html.escape(str(item.get('detail') or '-'))}</div></div>"
            f"<div class='loading-activity-time'>{html.escape(str(item.get('timestamp') or '-'))}</div>"
            "</div>"
        )
        for item in activity_items
    )
    html_block = f"""
    <div class='loading-hero'>
        <div class='loading-orbit'></div>
        <div style='font-size:12px; font-weight:800; letter-spacing:0.08em; color:rgba(255,255,255,0.72); text-transform:uppercase;'>Live Bootstrap</div>
        <div style='font-size:28px; font-weight:900; margin-top:6px;'>화면 로딩 중 백엔드 파이프라인을 처리하고 있습니다</div>
        <div style='font-size:14px; line-height:1.65; margin-top:8px; color:rgba(255,255,255,0.84);'>현재 단계: {latest_phase}<br>{phase_sub}<br>상태 요약: {loading_reason}<br>예상 소요 {html.escape(eta_text)} · 경과 {html.escape(elapsed_text)}</div>
        <div class='loading-stage-strip'>{stage_html}</div>
        <div class='loading-progress-track'>
            <div class='loading-progress-fill' style='width:{progress_percent}%;'></div>
        </div>
        <div class='loading-metrics'>
            <div class='loading-metric'>
                <div class='loading-metric-label'>Analyzed Logs</div>
                <div class='loading-metric-value'>{results_count}</div>
                <div class='loading-metric-sub'>증분 로그 포함</div>
            </div>
            <div class='loading-metric'>
                <div class='loading-metric-label'>News Signals</div>
                <div class='loading-metric-value'>{news_count}</div>
                <div class='loading-metric-sub'>수집 및 요약 진행</div>
            </div>
            <div class='loading-metric'>
                <div class='loading-metric-label'>FAISS Vectors</div>
                <div class='loading-metric-value'>{vector_count}</div>
                <div class='loading-metric-sub'>검색 인덱스 동기화</div>
            </div>
        </div>
        <div class='loading-live-grid'>{cards_html or "<div class='loading-live-card pending'><div class='loading-live-label'>백엔드 부트스트랩</div><div class='loading-live-detail'>실시간 작업 이벤트를 기다리는 중</div></div>"}</div>
        <div class='loading-activity-panel'>
            <div class='loading-activity-head'>실시간 작업 로그</div>
            {activity_html or "<div class='loading-activity-empty'>백엔드가 재기동되면 수행 단계가 여기서 실시간으로 움직입니다.</div>"}
        </div>
    </div>
    """
    target.markdown(html_block, unsafe_allow_html=True)


def render_backend_restart_loading_overlay(
    loading_state: dict[str, Any],
    elapsed_text: str,
) -> None:
    render_loading_styles()
    hero_box = st.empty()
    latest_phase = html.escape(str(loading_state.get("latest_phase") or "백엔드 준비 중"))
    explain_items = [
        ("임베딩 워밍업", "로컬 sentence-transformers 모델과 임베딩 객체를 다시 깨웁니다."),
        ("로그 분석", "저장된 로그를 다시 읽고 구조화해 분석 결과를 복구합니다."),
        ("뉴스 수집", "신규 뉴스 수집과 본문 크롤링, 이슈 정리를 재가동합니다."),
        ("FAISS 동기화", "로그/뉴스 벡터를 add_documents로 붙이고 검색 인덱스를 다시 맞춥니다."),
    ]
    explain_html = "".join(
        (
            "<div class='backend-restart-item'>"
            f"<div class='backend-restart-item-title'>{html.escape(title)}</div>"
            f"<div class='backend-restart-item-detail'>{html.escape(detail)}</div>"
            "</div>"
        )
        for title, detail in explain_items
    )
    hero_html = f"""
    <div class='backend-restart-overlay'>
        <div class='backend-restart-shell'>
            <div style='font-size:13px; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; color:#67e8f9; margin-bottom:10px;'>Backend Restart Detected</div>
            <div style='font-size:32px; font-weight:900; color:white; line-height:1.2;'>백엔드가 재기동되어 준비가 끝날 때까지 화면을 보류합니다</div>
            <div style='font-size:14px; line-height:1.7; color:rgba(226,232,240,0.92); margin-top:10px;'>현재 단계: {latest_phase}<br>재기동 직후에는 이전 화면 데이터가 남아 있어도 그대로 보여주지 않고, 로그 분석과 벡터 동기화가 끝난 뒤에만 대시보드를 다시 엽니다.<br>경과 시간 {html.escape(elapsed_text)}</div>
            <div class='backend-restart-explain'>{explain_html}</div>
        </div>
    </div>
    """
    hero_box.markdown(hero_html, unsafe_allow_html=True)
    render_initial_loading_hero(hero_box, loading_state, eta_text="30~90초", elapsed_text=elapsed_text)


def render_initial_loading_screen() -> None:
    render_loading_styles()
    hero_box = st.empty()
    checklist_box = st.empty()
    skeleton_box = st.empty()
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
    except Exception:
        status_payload = {}

    with _background_lock:
        initial_task = dict(_background_results.get("initial_analysis") or {})
    started_at = parse_status_time(initial_task.get("started_at")) or datetime.datetime.now()
    elapsed_seconds = max(0, int((datetime.datetime.now() - started_at).total_seconds()))
    loading_state = _derive_initial_loading_state(status_payload, initial_task)
    render_initial_loading_hero(
        hero_box,
        loading_state,
        eta_text="30~90초",
        elapsed_text=f"{elapsed_seconds}초",
    )
    render_loading_checklist(
        checklist_box,
        active_step=min(int(loading_state.get("active_step", 0) or 0), 4),
        eta_text="30~90초",
        elapsed_text=f"{elapsed_seconds}초",
    )
    render_loading_skeleton(skeleton_box)


@fragment_decorator(run_every="2s")
def render_live_initial_loading_fragment():
    render_initial_loading_screen()


@fragment_decorator(run_every="2s")
def monitor_backend_bootstrap_fragment():
    changed = False
    status_payload: dict[str, Any] | None = None
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
        changed = _sync_backend_bootstrap_state(status_payload)
    except Exception:
        changed = _sync_backend_bootstrap_state(None)
    if not st.session_state.get("initial_analysis_done", False):
        try:
            get_backend_client().start_worker(interval_seconds=1)
        except Exception:
            pass
        with _background_lock:
            initial_task = dict(_background_results.get("initial_analysis") or {})
        if initial_task.get("status") != "running" and not bool(
            st.session_state.get("initial_analysis_started", False)
        ):
            launched = _start_initial_analysis_background(log_dir="data/logs")
            if launched:
                st.session_state.initial_analysis_started = True
            with _background_lock:
                initial_task = dict(_background_results.get("initial_analysis") or {})
    if not st.session_state.get("initial_analysis_done", False):
        started_at = parse_status_time(initial_task.get("started_at")) or datetime.datetime.now()
        elapsed_seconds = max(0, int((datetime.datetime.now() - started_at).total_seconds()))
        render_backend_restart_loading_overlay(
            _derive_initial_loading_state(status_payload, initial_task),
            elapsed_text=f"{elapsed_seconds}초",
        )
    if changed:
        st.rerun()


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
    base_url = st.session_state.get(
        "backend_url",
        os.environ.get("BACKEND_URL", "http://127.0.0.1:18000"),
    )
    global _shared_backend_url
    _shared_backend_url = str(base_url).strip() or "http://127.0.0.1:18000"
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
            global _shared_backend_url
            _shared_backend_url = base
            return h
        except Exception:
            continue
    return {"status": "down", "detail": "백엔드에 연결할 수 없습니다. 포트 18000 또는 8000에서 서버가 실행 중인지 확인하세요."}


def sync_session_from_backend(payload: dict):
    # 백엔드 응답을 Streamlit 세션 상태로 옮겨서 화면 어디서든 재사용합니다.
    previous_ollama_runtime = st.session_state.get("ollama_runtime", {}) or {}
    incoming_ollama_runtime = payload.get("ollama_runtime", previous_ollama_runtime) or {}
    previous_started_at = str(previous_ollama_runtime.get("started_at") or "").strip()
    previous_completed_at = str(previous_ollama_runtime.get("completed_at") or "").strip()
    incoming_started_at = str(incoming_ollama_runtime.get("started_at") or "").strip()
    incoming_completed_at = str(incoming_ollama_runtime.get("completed_at") or "").strip()
    incoming_status = str(incoming_ollama_runtime.get("status") or "").strip()
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
    st.session_state.log_prompt_template_override = payload.get(
        "log_prompt_template_override",
        st.session_state.get("log_prompt_template_override"),
    )
    st.session_state.latest_news_briefing = payload.get("latest_news_briefing", st.session_state.get("latest_news_briefing"))
    st.session_state.last_news_briefing_time = payload.get("last_news_briefing_time", st.session_state.get("last_news_briefing_time"))
    st.session_state.latest_news_prompt_input = payload.get(
        "latest_news_prompt_input", st.session_state.get("latest_news_prompt_input", {})
    )
    st.session_state.last_news_prompt_input_time = payload.get(
        "last_news_prompt_input_time", st.session_state.get("last_news_prompt_input_time")
    )
    st.session_state.news_prompt_template_override = payload.get(
        "news_prompt_template_override",
        st.session_state.get("news_prompt_template_override"),
    )
    st.session_state.agent_statuses = payload.get("agent_statuses", st.session_state.get("agent_statuses", {}))
    st.session_state.agent_activity_log = payload.get("agent_activity_log", st.session_state.get("agent_activity_log", []))
    st.session_state.vector_events = payload.get("vector_events", st.session_state.get("vector_events", []))
    st.session_state.ollama_runtime = incoming_ollama_runtime
    st.session_state.last_faiss_time = payload.get("last_faiss_time", st.session_state.get("last_faiss_time"))
    st.session_state.backend_diagnostics = payload.get(
        "backend_diagnostics", st.session_state.get("backend_diagnostics", {})
    )
    incoming_cardloan_debate = payload.get(
        "cardloan_debate", st.session_state.get("cardloan_debate", {})
    ) or {}
    current_cardloan_debate = st.session_state.get("cardloan_debate", {}) or {}
    incoming_cardloan_status = str(incoming_cardloan_debate.get("status") or "").strip()
    should_keep_cardloan_debate = (
        incoming_cardloan_status == "running"
        or bool(st.session_state.get("cardloan_debate_task_id"))
        or bool(current_cardloan_debate.get("round_results"))
        or str(current_cardloan_debate.get("status") or "").strip() in {"running", "completed"}
    )
    st.session_state.cardloan_debate = incoming_cardloan_debate if should_keep_cardloan_debate else {}
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
    agent_label = _format_ollama_toast_agent_label(
        str(incoming_ollama_runtime.get("agent") or "")
    )
    model_label = str(incoming_ollama_runtime.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)
    if (
        incoming_status == "running"
        and incoming_started_at
        and incoming_started_at != previous_started_at
    ):
        prompt_preview = str(incoming_ollama_runtime.get("prompt") or "").strip()
        prompt_preview = re.sub(r"\s+", " ", prompt_preview)
        if len(prompt_preview) > 72:
            prompt_preview = prompt_preview[:72].rstrip() + "..."
        if not prompt_preview:
            prompt_preview = "프롬프트를 구성하고 응답 생성을 시작했습니다."
        st.session_state.ollama_toast = {
            "kicker": "OLLAMA STARTED",
            "title": f"{agent_label} 분석 시작",
            "msg": f"{model_label} · {prompt_preview}",
            "ts": time.time(),
        }
    if (
        incoming_status == "completed"
        and incoming_completed_at
        and incoming_completed_at != previous_completed_at
    ):
        response_preview = str(incoming_ollama_runtime.get("response_text") or "").strip()
        response_preview = re.sub(r"\s+", " ", response_preview)
        if len(response_preview) > 96:
            response_preview = response_preview[:96].rstrip() + "..."
        if not response_preview:
            response_preview = "응답 생성을 완료했습니다."
        st.session_state.ollama_toast = {
            "kicker": "OLLAMA COMPLETED",
            "title": f"{agent_label} 작업 완료",
            "msg": f"{model_label} · {response_preview}",
            "ts": time.time(),
        }
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


CARDLOAN_DEBATE_DEFAULT_QUESTION = (
    "최신 뉴스 신호와 승인/거절 사례를 바탕으로 카드론 리스크 정책, 승인 전환 전략, 신규 상품 구조를 순차 토론하라."
)


def get_default_cardloan_debate_question() -> str:
    return CARDLOAN_DEBATE_DEFAULT_QUESTION


def get_cardloan_debate_agent_ids() -> set[str]:
    return {str(persona.get("id") or "").strip() for persona in get_reviewer_personas()}


def is_cardloan_debate_agent(agent: str) -> bool:
    return str(agent or "").strip() in get_cardloan_debate_agent_ids()


def get_cardloan_debate_thinking_message(current_stage: str, runtime: dict[str, Any], round_results: list[dict[str, Any]]) -> str:
    runtime_text = str(runtime.get("response_text") or "").strip()
    if runtime_text:
        return runtime_text

    stage_name = str(current_stage or "").strip()
    started_at = parse_status_time(runtime.get("started_at") or runtime.get("updated_at"))
    elapsed = 0 if started_at is None else max(0, int((datetime.datetime.now() - started_at).total_seconds()))
    turn = (elapsed // 2) % 3

    stage_scripts = {
        "신용기획부": [
            "음.. 잠시 데이터좀 확인하겠습니다. 최신 뉴스 신호에서 카드론 리스크가 먼저 커지는 지점을 보고 있습니다.",
            "승인 정책에 먼저 손봐야 할 구간을 정리하는 중입니다. 규제, 금리, 연체 징후를 같이 대조하고 있습니다.",
            "미래 리스크를 과하게 막지 않으면서도 선제 수정할 기준을 묶고 있습니다. 곧 첫 판단을 제시하겠습니다.",
        ],
        "금융영업부": [
            "음.. 승인과 거절 사례를 나란히 다시 보고 있습니다. 어떤 고객을 전환할 수 있을지 바로 추리고 있습니다.",
            "거절 고객을 승인 가능 고객으로 바꾸려면 조건을 얼마나 조정해야 하는지 비교 중입니다.",
            "채널, 한도, 금리 조합을 다시 맞추고 있습니다. 전환 가능한 시나리오를 곧 정리하겠습니다.",
        ],
        "금융솔루션부": [
            "잠시만요. 리스크 정책과 영업 전략이 부딪히는 부분을 상품 구조로 풀 수 있는지 보고 있습니다.",
            "두 부서 의견을 엮어서 카드론 구조를 설계하는 중입니다. 위험 통제와 매출 확대를 같이 맞추고 있습니다.",
            "상품 이름, 대상 고객, 수익 모델까지 한 번에 정리하고 있습니다. 마무리 제안을 곧 드리겠습니다.",
        ],
    }
    fallback_scripts = [
        "음.. 잠시 데이터좀 확인하겠습니다. 지금 관련 신호와 사례를 함께 정리하고 있습니다.",
        "단계별 근거를 다시 맞춰 보고 있습니다. 조금만 더 기다리시면 자연스럽게 이어서 말씀드리겠습니다.",
        "핵심 포인트를 압축하는 중입니다. 곧 정돈된 문장으로 이어서 답변드리겠습니다.",
    ]
    scripts = stage_scripts.get(stage_name, fallback_scripts)
    if not round_results and not stage_name:
        return "토론을 시작하면 최신 뉴스, 승인 사례, 거절 사례를 묶어서 세 부서가 순서대로 바로 이야기합니다."
    return scripts[turn]


def build_cardloan_live_broadcast(
    current_stage: str,
    runtime: dict[str, Any],
    round_results: list[dict[str, Any]],
    summary: str,
    is_streaming: bool,
) -> tuple[str, str]:
    stage_name = str(current_stage or "대기").strip() or "대기"
    runtime_text = str(runtime.get("response_text") or "").strip()
    completed_count = len(round_results)
    latest_name = str((round_results[-1] or {}).get("name") or "").strip() if round_results else ""

    if is_streaming:
        if runtime_text:
            return (
                f"{stage_name}가 근거를 묶어 실시간으로 발언 중입니다.",
                f"현재 {completed_count}개 부서 의견이 정리됐고, 새 문장이 도착하는 즉시 아래 LIVE OLLAMA 본문에 이어서 반영됩니다.",
            )
        return (
            f"{stage_name}가 데이터를 검토하며 다음 문장을 준비 중입니다.",
            "첫 문장이 아직 없어도 흐름이 끊기지 않도록 내부 추론 멘트를 부드럽게 이어서 중계하고 있습니다.",
        )

    if round_results:
        if str(summary or "").strip():
            return (
                f"{latest_name or stage_name} 단계까지 정리됐습니다.",
                str(summary).strip(),
            )
        return (
            "토론 라운드가 마무리됐습니다.",
            f"총 {completed_count}개 부서 의견이 정리됐고, 아래 종합 메모와 단계별 상세 결과에서 이어서 확인할 수 있습니다.",
        )

    return (
        "토론실이 대기 중입니다.",
        "시작 버튼을 누르면 신용기획부부터 순서대로 실행되고, LIVE OLLAMA 영역이 단계별 현황을 실시간으로 중계합니다.",
    )


def refresh_cardloan_debate_runtime() -> None:
    consume_ws_snapshot_buffer()
    try:
        payload = get_backend_client().get_status()
        sync_session_from_backend(payload)
    except Exception:
        pass
    _sync_cardloan_debate_background_tasks()


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


def _css_content_escape(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace("'", "\\'")


def _compact_badge_metric(value: int) -> str:
    number = int(value or 0)
    if number >= 1000:
        compact = f"{number / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}K"
    return str(number)


def render_main_section_status_styles() -> None:
    metrics = build_overview_metrics()
    round_results = st.session_state.get("reviewer_debate_round", []) or []
    main_sections = [
        "🤖 운영 현황",
        "💬 AI 카드론 토론실",
        "📄 대출상품 Dashboard",
        "🧠 Vector DB",
    ]
    selected_section = st.session_state.get("main_dashboard_section") or main_sections[0]
    active_index = main_sections.index(selected_section) if selected_section in main_sections else 0
    tab_weights = [1.0, 1.28, 1.0, 0.94]
    total_weight = sum(tab_weights)
    usable_width = "(100% - 20px - 24px)"

    def _tab_width(weight: float) -> str:
        return f"calc({usable_width} * {weight / total_weight:.8f})"

    tab_widths = [_tab_width(weight) for weight in tab_weights]
    tab_offsets = [
        "10px",
        f"calc(10px + {tab_widths[0]} + 8px)",
        f"calc(10px + {tab_widths[0]} + 8px + {tab_widths[1]} + 8px)",
        f"calc(10px + {tab_widths[0]} + 8px + {tab_widths[1]} + 8px + {tab_widths[2]} + 8px)",
    ]

    operations_badge = (
        f"{_compact_badge_metric(metrics['running_agents'])} RUN"
        if metrics["running_agents"] > 0
        else f"{_compact_badge_metric(metrics['activity_events'])} EVT"
    )
    strategy_badge = f"{_compact_badge_metric(len(round_results))} MEMO" if round_results else "3 AI"
    dashboard_badge = f"{_compact_badge_metric(metrics['results_count'])} LOG"
    vector_badge = f"{_compact_badge_metric(metrics['vector_count'])} VEC"

    operations_dot = "#61f4de" if metrics["running_agents"] > 0 else "#8fb9d6"
    strategy_dot = "#f9a8d4" if round_results else "#c4b5fd"
    dashboard_dot = "#ffbf69" if metrics["results_count"] > 0 else "#8fb9d6"
    vector_dot = "#a5b4fc" if metrics["vector_count"] > 0 else "#8fb9d6"

    active_backgrounds = [
        "linear-gradient(120deg, rgba(151,244,222,0.16), rgba(255,255,255,0.00) 42%), linear-gradient(135deg, rgba(13,45,62,0.98), rgba(18,68,84,0.94))",
        "linear-gradient(120deg, rgba(249,168,212,0.18), rgba(255,255,255,0.00) 42%), linear-gradient(135deg, rgba(59,24,67,0.98), rgba(111,45,87,0.94))",
        "linear-gradient(120deg, rgba(255,191,105,0.18), rgba(255,255,255,0.00) 42%), linear-gradient(135deg, rgba(71,40,20,0.98), rgba(132,77,27,0.94))",
        "linear-gradient(120deg, rgba(165,180,252,0.20), rgba(255,255,255,0.00) 42%), linear-gradient(135deg, rgba(34,33,79,0.98), rgba(60,61,133,0.94))",
    ]
    active_accents = ["#61f4de", "#f9a8d4", "#ffbf69", "#a5b4fc"]
    active_glows = [
        "rgba(97,244,222,0.28)",
        "rgba(249,168,212,0.30)",
        "rgba(255,191,105,0.30)",
        "rgba(165,180,252,0.32)",
    ]

    indicator_left = tab_offsets[active_index]
    indicator_width = tab_widths[active_index]
    active_background = active_backgrounds[active_index]
    active_accent = active_accents[active_index]
    active_glow = active_glows[active_index]

    st.markdown(
        f"""
        <style>
        :root {{
            --main-section-accent: {active_accent};
            --main-section-glow: {active_glow};
        }}

        [class*="st-key-main_dashboard_section"] {{
            position: relative;
        }}

        [class*="st-key-main_dashboard_section"] [role="radiogroup"],
        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1.28fr) minmax(0, 1fr) minmax(0, 0.94fr);
            gap: 8px;
            align-items: stretch;
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"] {{
            min-height: 70px;
            padding-top: 25px;
            padding-bottom: 12px;
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2),
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2) {{
            min-height: 78px;
            padding-top: 28px;
            padding-bottom: 14px;
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2) p,
        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2) div,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2) p,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2) div {{
            font-size: 14px !important;
        }}

        [class*="st-key-main_dashboard_section"]::after {{
            content: '';
            position: absolute;
            left: {indicator_left};
            bottom: 9px;
            width: {indicator_width};
            height: 4px;
            border-radius: 999px;
            background: linear-gradient(90deg, {active_accent}, rgba(255,255,255,0.92));
            box-shadow: 0 0 18px {active_glow};
            transition: left 0.34s ease, background 0.24s ease, box-shadow 0.24s ease;
            pointer-events: none;
            z-index: 3;
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1)::after {{
            content: '{_css_content_escape(operations_badge)}';
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2)::after {{
            content: '{_css_content_escape(strategy_badge)}';
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(3)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(3)::after {{
            content: '{_css_content_escape(dashboard_badge)}';
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(4)::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(4)::after {{
            content: '{_css_content_escape(vector_badge)}';
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(1)::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(1)::before {{
            background: {operations_dot};
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(2)::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(2)::before {{
            background: {strategy_dot};
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(3)::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(3)::before {{
            background: {dashboard_dot};
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button:nth-child(4)::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:nth-child(4)::before {{
            background: {vector_dot};
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"],
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked) {{
            background: {active_background} !important;
            border-color: {active_accent} !important;
            box-shadow:
                0 14px 28px rgba(0,0,0,0.20),
                0 0 28px {active_glow},
                inset 0 0 0 1px rgba(255,255,255,0.04),
                inset 0 -3px 0 {active_accent} !important;
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"]::before,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked)::before {{
            background: {active_accent};
            box-shadow: 0 0 0 0 {active_glow};
        }}

        [class*="st-key-main_dashboard_section"] [data-baseweb="button-group"] button[aria-pressed="true"]::after,
        [class*="st-key-main_dashboard_section"] label[data-baseweb="radio"]:has(input:checked)::after {{
            border-color: {active_accent};
            box-shadow: 0 0 18px {active_glow};
        }}

        .section-shell,
        .section-shell-tight {{
            position: relative;
            overflow: hidden;
            border-color: color-mix(in srgb, var(--main-section-accent) 34%, rgba(151, 196, 225, 0.18)) !important;
            box-shadow:
                0 18px 44px rgba(0, 0, 0, 0.16),
                0 0 0 1px rgba(255,255,255,0.02),
                0 0 28px var(--main-section-glow) !important;
            background:
                radial-gradient(circle at top right, color-mix(in srgb, var(--main-section-accent) 15%, transparent) 0%, transparent 42%),
                var(--panel-bg-soft) !important;
            animation: sectionPanelReveal 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
            will-change: transform, opacity;
        }}

        .section-shell::before,
        .section-shell-tight::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 18px;
            right: 18px;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, transparent, var(--main-section-accent), rgba(255,255,255,0.88), var(--main-section-accent), transparent);
            box-shadow: 0 0 20px var(--main-section-glow);
            opacity: 0.92;
            pointer-events: none;
        }}

        .section-shell-tight {{
            animation-duration: 0.5s;
        }}

        @keyframes sectionPanelReveal {{
            0% {{
                opacity: 0;
                transform: translateY(12px) scale(0.988);
                filter: saturate(0.88);
            }}
            60% {{
                opacity: 1;
                transform: translateY(-2px) scale(1);
                filter: saturate(1);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0) scale(1);
                filter: saturate(1);
            }}
        }}

        .section-kicker {{
            color: var(--main-section-accent) !important;
            text-shadow: 0 0 16px var(--main-section-glow);
        }}

        .section-title {{
            text-shadow: 0 0 22px color-mix(in srgb, var(--main-section-accent) 18%, transparent);
        }}

        .section-detail {{
            border-left: 2px solid color-mix(in srgb, var(--main-section-accent) 46%, transparent);
            padding-left: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
                <div class="hero-chip-detail">완료 {metrics['completed_agents']}{'' if int(metrics.get('failed_agents', 0) or 0) <= 0 else f' · 실패 {metrics["failed_agents"]}'}</div>
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
        ("01", "로그 브리핑 갱신", statuses.get("log_agent", {}).get("status", "pending"), "신규 로그를 요약하고 상품 패턴 기반 브리핑을 갱신합니다."),
        ("02", "뉴스 리스크 반영", statuses.get("news_agent", {}).get("status", "pending"), "시장 뉴스와 이슈 태그를 묶어 심사 영향도를 갱신합니다."),
        ("03", "규제/근거 결합", statuses.get("regulation_agent", {}).get("status", "pending"), "규제 문서와 검색 결과를 결합해 준수 여부를 해석합니다."),
        ("04", "전략 응답 합성", statuses.get("orchestrator", {}).get("status", "pending"), "별도 최종판단 Agent 없이 로그/뉴스/규제 결과를 코드에서 합성합니다."),
        ("05", "FAISS 동기화", statuses.get("vector_store", {}).get("status", "pending"), f"누적 벡터 {metrics['vector_count']}건 · 최근 이벤트 {metrics['vector_events']}건"),
    ]

    cards = []
    for index, name, status_code, detail in workflow_items:
        label, _, css_class = get_agent_status_palette(status_code)
        cards.append(
            f"""<div class="workflow-card">
<div class="workflow-index">{index}</div>
<div class="workflow-name">{html.escape(name)}</div>
<div class="workflow-state {css_class}">{html.escape(label)}</div>
<div class="workflow-text">{html.escape(detail)}</div>
</div>"""
        )

    st.markdown(
        """<div class="section-shell">
<div class="section-header">
<div>
<div class="section-kicker">Process View</div>
<div class="section-title">심사 플로우 보드</div>
</div>
<div class="section-detail">신규 유입부터 전략 응답과 벡터 저장까지의 단계를 상태 배지와 함께 한 줄 플로우로 보여줍니다.</div>
</div>
<div class="workflow-grid">"""
        + "".join(cards)
        + """</div>
</div>""",
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
        {"id": "orchestrator", "label": "Orchestrator", "x": 0.74, "y": 0.46, "status": statuses.get("orchestrator", {}).get("status", "pending"), "detail": status_detail("orchestrator", str(latest_question)[:90] + " | 최종 응답은 코드 합성")},
        {"id": "vector_store", "label": "Vector DB", "x": 0.92, "y": 0.32, "status": statuses.get("vector_store", {}).get("status", "pending"), "detail": status_detail("vector_store", f"누적 {vector_count} vectors | 최근 +{telemetry['latest_vector_added']}")},
    ]
    edges = [
        ("source_logs", "log_agent"),
        ("source_news", "news_agent"),
        ("log_agent", "regulation_agent"),
        ("news_agent", "regulation_agent"),
        ("regulation_agent", "orchestrator"),
        ("orchestrator", "vector_store"),
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
                "size": [36, 36, 48, 48, 54, 58, 50],
                "color": [color_map.get(node["status"], "#8fb9d6") for node in nodes],
                "line": {"width": 2, "color": "rgba(7,19,30,0.95)"},
                "symbol": ["diamond", "diamond", "circle", "circle", "hexagon", "hexagon", "square"],
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
                f"""<div class="event-card">
<div class="event-head">
<div class="event-source">{html.escape(str(event.get('source', '-')))} · {html.escape(str(event.get('status', '-')))}</div>
<div class="event-time">{html.escape(format_status_time(event.get('timestamp')))}</div>
</div>
<div class="event-body">{html.escape(str(event.get('detail', ''))[:170])}</div>
</div>"""
            )
        if not cards:
            for event in (vector_events[:3] or []):
                cards.append(
                    f"""<div class="event-card">
<div class="event-head">
<div class="event-source">{html.escape(str(event.get('source', '-')))} · {html.escape(str(event.get('action', '-')))}</div>
<div class="event-time">{html.escape(format_status_time(event.get('timestamp')))}</div>
</div>
<div class="event-body">누적 {event.get('after_count', 0)} · 추가 {event.get('added_count', 0)} · {html.escape(str(event.get('detail', ''))[:120])}</div>
</div>"""
                )
        st.markdown(
            """<div class="insight-card">
<div class="insight-label">Recent Timeline</div>
<div class="insight-title">최근 운영 이벤트</div>
<div class="event-stack">"""
            + "".join(cards or ["<div class='event-card'><div class='event-body'>표시할 이벤트가 없습니다.</div></div>"])
            + """</div>
</div>""",
            unsafe_allow_html=True,
        )


def render_operations_showcase():
    metrics = build_overview_metrics()
    failed_agents_detail = (
        f"Failed {metrics['failed_agents']}건"
        if int(metrics.get("failed_agents", 0) or 0) > 0
        else f"Completed {metrics['completed_agents']}건"
    )
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
            "Agent 상태", str(metrics["running_agents"]), "현재 실행 중인 Agent 수와 운영 온도", failed_agents_detail, "amber"
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


def extract_product_summary_block(prompt_text: str) -> str:
    text = str(prompt_text or "")
    if not text.strip():
        return ""

    match = re.search(
        r"\[상품별 승인/거절 패턴 요약\]\s*(.*?)(?=\n\[[^\n]+\]|\Z)",
        text,
        re.S,
    )
    return match.group(1).strip() if match else ""


def parse_product_summary_cards(prompt_text: str) -> list[dict[str, object]]:
    block = extract_product_summary_block(prompt_text)
    if not block or block == "상품별 패턴 요약이 없습니다.":
        return []

    cards: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for raw_line in block.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue

        product_match = re.match(r"\[상품:\s*([A-Za-z0-9]+)\]\s*(.*)", line)
        if product_match:
            if current is not None:
                cards.append(current)
            current = {
                "product_code": product_match.group(1).strip().upper(),
                "product_name": product_match.group(2).strip() or product_match.group(1).strip().upper(),
                "approval_rate": None,
                "approval_cases": None,
                "rejection_rate": None,
                "rejection_cases": None,
                "decision_known_cases": None,
                "approval_patterns": [],
                "rejection_patterns": [],
                "reject_reasons": [],
            }
            continue

        if current is None:
            continue

        if line.startswith("- 결정 분포:"):
            stat_match = re.search(
                r"승인\s+([\d.]+)%\s*\((\d+)/(\d+)\),\s*거절\s+([\d.]+)%\s*\((\d+)/(\d+)\)",
                line,
            )
            if stat_match:
                current["approval_rate"] = stat_match.group(1)
                current["approval_cases"] = stat_match.group(2)
                current["decision_known_cases"] = stat_match.group(3)
                current["rejection_rate"] = stat_match.group(4)
                current["rejection_cases"] = stat_match.group(5)
            continue

        if line.startswith("- 승인 패턴:"):
            payload = line.split(":", 1)[1].strip()
            current["approval_patterns"] = [] if payload == "뚜렷한 패턴 없음" else [item.strip() for item in payload.split(";") if item.strip()]
            continue

        if line.startswith("- 거절 패턴:"):
            payload = line.split(":", 1)[1].strip()
            current["rejection_patterns"] = [] if payload == "뚜렷한 패턴 없음" else [item.strip() for item in payload.split(";") if item.strip()]
            continue

        if line.startswith("- 주요 거절사유:"):
            payload = line.split(":", 1)[1].strip()
            current["reject_reasons"] = [] if payload == "데이터 없음" else [item.strip() for item in payload.split(";") if item.strip()]

    if current is not None:
        cards.append(current)
    return cards


def infer_product_codes_from_text(text: str) -> list[str]:
    matches = re.findall(r"\bC\d{1,2}\b", str(text or "").upper())
    seen: set[str] = set()
    ordered: list[str] = []
    for match in matches:
        if match in seen:
            continue
        seen.add(match)
        ordered.append(match)
    return ordered


def load_product_summary_payload() -> dict[str, object]:
    try:
        payload = load_product_pattern_summary(DEFAULT_SUMMARY_PATH)
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_product_summary_cards_from_json(context_text: str = "") -> list[dict[str, object]]:
    payload = load_product_summary_payload()
    if not payload:
        return []

    products = ((payload or {}).get("products") or {})
    if not products:
        return []

    product_codes = infer_product_codes_from_text(context_text)
    ordered_codes = product_codes or sorted(products.keys())
    cards: list[dict[str, object]] = []
    for product_code in ordered_codes:
        item = products.get(product_code)
        if not item:
            continue
        totals = item.get("totals") or {}
        approval_patterns = item.get("approval_patterns") or []
        rejection_patterns = item.get("rejection_patterns") or []
        reject_reasons = item.get("top_reject_reason_codes") or []
        cards.append(
            {
                "product_code": product_code,
                "product_name": item.get("product_name") or product_code,
                "approval_rate": totals.get("approval_rate_percent", "-"),
                "approval_cases": totals.get("approval_cases", "-"),
                "rejection_rate": totals.get("rejection_rate_percent", "-"),
                "rejection_cases": totals.get("rejection_cases", "-"),
                "decision_known_cases": totals.get("decision_known_cases", "-"),
                "approval_patterns": [
                    (
                        f"{pattern.get('rule') or '-'} -> 승인 비율 "
                        f"{float(pattern.get('decision_rate_percent', 0)):.1f}% "
                        f"({pattern.get('decision_count', 0)}/{pattern.get('support', 0)})"
                    )
                    for pattern in approval_patterns[:3]
                ],
                "rejection_patterns": [
                    (
                        f"{pattern.get('rule') or '-'} -> 거절 비율 "
                        f"{float(pattern.get('decision_rate_percent', 0)):.1f}% "
                        f"({pattern.get('decision_count', 0)}/{pattern.get('support', 0)})"
                    )
                    for pattern in rejection_patterns[:3]
                ],
                "reject_reasons": [
                    (
                        f"{reason.get('code')} {str(reason.get('description') or '').strip()} "+
                        f"{float(reason.get('share_of_rejections_percent', 0)):.1f}%"
                    ).strip()
                    for reason in reject_reasons[:3]
                ],
            }
        )
    return cards


def _get_faiss_session_cache_version(store_name: str | None = None) -> str:
    last_faiss_time = st.session_state.get("last_faiss_time") or ""
    vector_count = st.session_state.get("vector_count") or ""
    store_key = store_name or "all"
    return f"{store_key}::{last_faiss_time}::{vector_count}"


def render_log_product_summary_panel(prompt_text: str, updated_at, context_text: str = "") -> bool:
    cards = load_product_summary_cards_from_json("")
    if not cards:
        return False

    payload = load_product_summary_payload()

    header_html = f"""
    <div style="
        margin: 6px 0 14px 0;
        padding: 18px 20px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(239,246,255,0.95), rgba(240,253,250,0.96));
        border: 1px solid rgba(14,116,144,0.16);
        box-shadow: 0 16px 34px rgba(14,116,144,0.08);
    ">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:8px;">
            <div>
                <div style="font-size:12px; font-weight:800; letter-spacing:0.08em; color:#0f766e; text-transform:uppercase;">FAISS Summary</div>
                <div style="font-size:18px; font-weight:900; color:#0f172a; margin-top:3px;">상품별 승인/거절 패턴이 이렇게 요약돼 들어갑니다</div>
            </div>
            <div style="padding:6px 10px; border-radius:999px; font-size:11px; font-weight:800; color:#155e75; background:rgba(255,255,255,0.72); border:1px solid rgba(14,116,144,0.14); white-space:nowrap;">최근 반영 {html.escape(format_status_time(updated_at))}</div>
        </div>
        <div style="font-size:13px; line-height:1.7; color:#334155;">data/product_pattern_summary.json을 읽어서 현재 로그 맥락에 맞는 상품 요약만 바로 보여주는 화면입니다.</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    for index in range(0, len(cards), 2):
        columns = st.columns(2)
        for col_index, card in enumerate(cards[index : index + 2]):
            with columns[col_index]:
                product_code = html.escape(str(card.get("product_code") or "-"))
                product_name = html.escape(str(card.get("product_name") or product_code))
                approval_rate = html.escape(str(card.get("approval_rate") or "-"))
                approval_cases = html.escape(str(card.get("approval_cases") or "-"))
                rejection_rate = html.escape(str(card.get("rejection_rate") or "-"))
                rejection_cases = html.escape(str(card.get("rejection_cases") or "-"))
                decision_known_cases = html.escape(str(card.get("decision_known_cases") or "-"))
                approval_patterns = card.get("approval_patterns") or []
                rejection_patterns = card.get("rejection_patterns") or []
                reject_reasons = card.get("reject_reasons") or []

                approval_items = "".join(
                    f'<li style="margin-bottom:6px;">{html.escape(str(item))}</li>'
                    for item in approval_patterns[:3]
                ) or '<li style="margin-bottom:6px; color:#64748b;">뚜렷한 승인 패턴 없음</li>'
                rejection_items = "".join(
                    f'<li style="margin-bottom:6px;">{html.escape(str(item))}</li>'
                    for item in rejection_patterns[:3]
                ) or '<li style="margin-bottom:6px; color:#64748b;">뚜렷한 거절 패턴 없음</li>'
                reject_reason_items = "".join(
                    f'<span style="display:inline-flex; align-items:center; padding:6px 10px; margin:0 8px 8px 0; border-radius:999px; background:rgba(254,242,242,0.92); color:#991b1b; border:1px solid rgba(239,68,68,0.12); font-size:12px; font-weight:700;">{html.escape(str(item))}</span>'
                    for item in reject_reasons[:3]
                ) or '<span style="display:inline-flex; align-items:center; padding:6px 10px; border-radius:999px; background:rgba(248,250,252,0.96); color:#64748b; border:1px solid rgba(148,163,184,0.16); font-size:12px; font-weight:700;">대표 거절사유 없음</span>'

                st.markdown(
                    f"""
                    <div style="
                        height: 100%;
                        margin-bottom: 16px;
                        padding: 18px 18px 16px 18px;
                        border-radius: 20px;
                        background: linear-gradient(160deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
                        border: 1px solid rgba(148,163,184,0.18);
                        box-shadow: 0 14px 28px rgba(15,23,42,0.06);
                    ">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:14px;">
                            <div>
                                <div style="font-size:12px; font-weight:900; color:#0891b2; letter-spacing:0.08em; text-transform:uppercase;">{product_code}</div>
                                <div style="font-size:18px; font-weight:900; color:#0f172a; margin-top:4px;">{product_name}</div>
                            </div>
                            <div style="padding:6px 10px; border-radius:999px; background:rgba(241,245,249,0.95); color:#334155; font-size:12px; font-weight:800; border:1px solid rgba(148,163,184,0.14);">의사결정 표본 {decision_known_cases}건</div>
                        </div>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px;">
                            <div style="padding:12px 14px; border-radius:16px; background:linear-gradient(135deg, rgba(236,253,245,0.98), rgba(209,250,229,0.98)); border:1px solid rgba(34,197,94,0.16);">
                                <div style="font-size:12px; font-weight:800; color:#166534; margin-bottom:6px;">승인 시그널</div>
                                <div style="font-size:22px; font-weight:900; color:#14532d;">{approval_rate}%</div>
                                <div style="font-size:12px; color:#166534; margin-top:4px;">{approval_cases}건 승인</div>
                            </div>
                            <div style="padding:12px 14px; border-radius:16px; background:linear-gradient(135deg, rgba(254,242,242,0.98), rgba(254,226,226,0.98)); border:1px solid rgba(239,68,68,0.16);">
                                <div style="font-size:12px; font-weight:800; color:#b91c1c; margin-bottom:6px;">거절 시그널</div>
                                <div style="font-size:22px; font-weight:900; color:#7f1d1d;">{rejection_rate}%</div>
                                <div style="font-size:12px; color:#b91c1c; margin-top:4px;">{rejection_cases}건 거절</div>
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px;">
                            <div style="padding:14px; border-radius:16px; background:rgba(240,253,244,0.7); border:1px solid rgba(34,197,94,0.12);">
                                <div style="font-size:13px; font-weight:900; color:#166534; margin-bottom:10px;">승인으로 기우는 패턴</div>
                                <ul style="margin:0; padding-left:18px; font-size:12px; line-height:1.6; color:#334155;">{approval_items}</ul>
                            </div>
                            <div style="padding:14px; border-radius:16px; background:rgba(254,242,242,0.65); border:1px solid rgba(239,68,68,0.12);">
                                <div style="font-size:13px; font-weight:900; color:#b91c1c; margin-bottom:10px;">거절로 기우는 패턴</div>
                                <ul style="margin:0; padding-left:18px; font-size:12px; line-height:1.6; color:#334155;">{rejection_items}</ul>
                            </div>
                        </div>
                        <div>
                            <div style="font-size:13px; font-weight:900; color:#0f172a; margin-bottom:10px;">자주 붙는 거절사유 코드</div>
                            <div>{reject_reason_items}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if payload:
        generated_at = format_status_time((payload.get("generated_at") or ""))
        with st.expander("상품 패턴 요약 전체 데이터 보기", expanded=False):
            st.caption(f"생성 시각: {generated_at}")
            st.json(payload, expanded=False)

    return True


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
            "id": "credit_planning_agent",
            "emoji": "🧑‍💼",
            "name": "신용기획부",
            "display": "신용기획부",
            "tone": "리스크 정책",
            "accent": "#ff8f8f",
            "avatar_class": "conservative",
            "tagline": "미래 리스크를 먼저 보고 심사 룰을 선제적으로 바꿉니다.",
            "description": "뉴스 신호와 시장 변화를 읽어 미래 리스크를 예측하고, 현재 심사 정책의 취약점과 보완 룰을 설계하는 역할입니다.",
            "default_prompt": "너는 신용기획부 리스크 정책 담당자다. 목표: 미래 리스크를 선제적으로 차단하고 카드론 심사 기준을 개선하라. 시장 신호 TOP5를 바탕으로 향후 발생할 주요 리스크를 예측하고, 현재 심사 정책의 취약점을 도출하고, 보완해야 할 심사 기준과 구체적 룰을 작성하라. 반드시 JSON만 출력하라.",
        },
        {
            "id": "sales_strategy_agent",
            "emoji": "😎",
            "name": "금융영업부",
            "display": "금융영업부",
            "tone": "전환 영업",
            "accent": "#61f4de",
            "avatar_class": "sales",
            "tagline": "거절 고객도 승인 가능한 구조로 다시 바꿔냅니다.",
            "description": "승인 사례와 거절 사례의 차이를 비교해 현재 고객의 거절 원인을 좁히고, 승인율과 수익, 영업 채널 전략을 함께 설계합니다.",
            "default_prompt": "너는 금융영업부 전략 담당자다. 목표: 거절된 고객을 승인 가능한 고객으로 전환하고 승인율과 수익, 영업 채널을 동시에 고려하라. 현재 고객, 고금액 승인 사례, 유사 거절 사례를 비교해 핵심 원인과 전환 조건, 실행 전략을 JSON으로 작성하라.",
        },
        {
            "id": "solution_planning_agent",
            "emoji": "⚖️",
            "name": "금융솔루션부",
            "display": "금융솔루션부",
            "tone": "상품 기획",
            "accent": "#ffbf69",
            "avatar_class": "product",
            "tagline": "리스크와 영업 충돌을 상품 구조로 해결합니다.",
            "description": "신용기획부의 리스크 정책과 금융영업부의 전환 전략 충돌을 풀고, 카드론 매출을 키우는 신상품 구조와 기존 상품 개선안을 설계합니다.",
            "default_prompt": "너는 금융솔루션부 상품 기획자다. 목표: 리스크를 통제하면서도 카드론 매출을 확대하는 상품을 설계하라. 리스크 정책과 영업 전략의 충돌 지점을 분석하고, 이를 해결할 상품 구조, 신상품 1개, 기존 상품 개선안을 반드시 JSON으로 작성하라.",
        },
    ]


REVIEWER_PROMPT_STORE_PATH = os.path.join("data", "reviewer_prompts.json")


def load_reviewer_prompt_store() -> dict[str, str]:
    personas = get_reviewer_personas()
    defaults = {persona["id"]: persona["default_prompt"] for persona in personas}
    try:
        if not os.path.exists(REVIEWER_PROMPT_STORE_PATH):
            return defaults
        with open(REVIEWER_PROMPT_STORE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        stored_prompts = payload if isinstance(payload, dict) else {}
    except Exception:
        return defaults

    merged = dict(defaults)
    for persona in personas:
        reviewer_id = persona["id"]
        saved_prompt = stored_prompts.get(reviewer_id)
        if isinstance(saved_prompt, str) and saved_prompt.strip():
            merged[reviewer_id] = saved_prompt.strip()
    return merged


def save_reviewer_prompt_store(prompts: dict[str, str]) -> None:
    personas = get_reviewer_personas()
    valid_ids = {persona["id"] for persona in personas}
    payload = {
        reviewer_id: str(prompt).strip()
        for reviewer_id, prompt in (prompts or {}).items()
        if reviewer_id in valid_ids and str(prompt).strip()
    }
    os.makedirs(os.path.dirname(REVIEWER_PROMPT_STORE_PATH), exist_ok=True)
    with open(REVIEWER_PROMPT_STORE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def ensure_strategy_debate_state() -> None:
    personas = get_reviewer_personas()
    if "reviewer_prompts" not in st.session_state:
        st.session_state.reviewer_prompts = load_reviewer_prompt_store()
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
    if "cardloan_debate" not in st.session_state:
        st.session_state.cardloan_debate = {}
    if "cardloan_debate_task_id" not in st.session_state:
        st.session_state.cardloan_debate_task_id = None


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
            save_reviewer_prompt_store(st.session_state.reviewer_prompts)
            st.session_state.strategy_debate_status = f"{persona['name']} 프롬프트를 수정했습니다."
            st.session_state.reviewer_prompt_saved_feedback = {
                **(st.session_state.get("reviewer_prompt_saved_feedback", {}) or {}),
                reviewer_id: datetime.datetime.now().isoformat(),
            }
    with action_col_b:
        if st.button("기본값 복원", key=f"reset_reviewer_prompt_{reviewer_id}", use_container_width=True):
            st.session_state.reviewer_prompts[reviewer_id] = persona["default_prompt"]
            save_reviewer_prompt_store(st.session_state.reviewer_prompts)
            st.session_state[editor_key] = persona["default_prompt"]
    with action_col_c:
        if st.button("닫기", key=f"close_reviewer_prompt_{reviewer_id}", use_container_width=True):
            close_reviewer_prompt_dialog()


if hasattr(st, "dialog"):
    @st.dialog("심사관 프롬프트 편집")
    def render_reviewer_prompt_dialog(persona: dict[str, str]) -> None:
        _render_reviewer_prompt_dialog_body(persona)
else:
    def render_reviewer_prompt_dialog(persona: dict[str, str]) -> None:
        _render_reviewer_prompt_dialog_body(persona)


def build_reviewer_question(persona: dict[str, str], user_question: str, custom_prompt: str) -> str:
    return (
        f"[부서 역할 프롬프트]\n{custom_prompt.strip()}\n\n"
        f"[토론 주제]\n{user_question.strip()}\n\n"
        "[실행 메모]\n"
        "실제 카드론 토론실 실행 시에는 이 역할 프롬프트 위에 FAISS에서 검색한 시장 신호, 승인/거절 사례, 앞단계 결과가 자동으로 주입됩니다. "
        "이 미리보기는 부서 기본 임무와 토론 주제만 보여줍니다."
    )


def summarize_debate_result(response_payload: dict) -> tuple[str, str, str]:
    if not isinstance(response_payload, dict):
        return "판단 대기", str(response_payload or "분석 결과가 없습니다."), "근거 요약이 없습니다."

    parsed = response_payload.get("parsed") or {}
    raw_text = str(response_payload.get("raw_text") or response_payload.get("answer") or "").strip()
    preview = " ".join(str(response_payload.get("preview") or raw_text or "분석 결과가 없습니다.").split())[:180]
    evidence = " ".join(str(response_payload.get("evidence") or raw_text or "근거 요약이 없습니다.").split())[:220]
    verdict = str(response_payload.get("verdict") or parsed.get("current_status") or "판단 대기").strip() or "판단 대기"
    return verdict, preview, evidence


def build_cardloan_stage_spoken_message(item: dict[str, Any]) -> str:
    preview = str(item.get("preview") or "").strip()
    evidence = str(item.get("evidence") or "").strip()
    if preview and evidence and evidence != preview:
        return f"{preview}\n{evidence}"
    return preview or evidence or "아직 발언이 정리되지 않았습니다."


def build_cardloan_live_display_text(
    current_stage: str,
    runtime_text: str,
    thinking_text: str,
    broadcast_body: str,
    round_results: list[dict[str, Any]],
    is_streaming: bool,
) -> str:
    text = str(runtime_text or "").strip()
    if not text:
        return thinking_text

    stripped = text.lstrip()
    looks_like_json = stripped.startswith("{") or stripped.startswith("[") or ('"' in stripped[:80] and ":" in stripped[:120])
    if looks_like_json:
        if is_streaming:
            return broadcast_body or thinking_text
        if round_results:
            return build_cardloan_stage_spoken_message(round_results[-1])
        return thinking_text
    return text


def build_cardloan_typewriter_text(
    text: str,
    message_key: str,
    *,
    is_streaming: bool,
) -> tuple[str, bool]:
    target_text = str(text or "")
    if not target_text:
        return "", False

    cache = dict(st.session_state.get("cardloan_live_typewriter", {}) or {})
    cached_key = str(cache.get("key") or "")
    cached_target = str(cache.get("target") or "")
    cached_visible_len = int(cache.get("visible_len") or 0)
    growth_step = 28 if is_streaming else 52

    if cached_key != message_key or cached_target != target_text:
        inherited_visible_len = 0
        if target_text.startswith(cached_target):
            inherited_visible_len = len(cached_target)
        elif cached_target.startswith(target_text):
            inherited_visible_len = min(len(target_text), cached_visible_len)
        visible_len = min(len(target_text), max(inherited_visible_len, growth_step))
    else:
        visible_len = min(len(target_text), cached_visible_len + growth_step)

    st.session_state.cardloan_live_typewriter = {
        "key": message_key,
        "target": target_text,
        "visible_len": visible_len,
    }
    return target_text[:visible_len], visible_len < len(target_text)


def build_debate_consensus(personas: list[dict[str, str]], round_results: list[dict]) -> str:
    if not round_results:
        return "아직 토론 결과가 없습니다."
    lines = []
    for item in round_results:
        lines.append(f"{item['name']}: {item['verdict']} · {item['preview']}")
    title = "신용기획부 → 금융영업부 → 금융솔루션부 순서로 카드론 전략이 정리되었습니다."
    return title + "\n\n" + "\n".join(lines)


def _sync_cardloan_debate_background_tasks() -> None:
    try:
        with _background_lock:
            tasks = list(_background_results.items())
        for task_id, payload in tasks:
            if not str(task_id).startswith("cardloan_debate_"):
                continue
            if payload.get("status") == "completed":
                result = payload.get("result") or {}
                st.session_state.reviewer_debate_round = result.get("round_results", []) or []
                st.session_state.strategy_debate_status = f"최근 카드론 토론 완료 · {format_status_time(result.get('completed_at'))}"
                st.session_state.cardloan_debate_task_id = None
                with _background_lock:
                    _background_results.pop(task_id, None)
            elif payload.get("status") == "failed":
                st.session_state.strategy_debate_status = f"카드론 토론실 실패 · {payload.get('error', '-')}"
                st.session_state.cardloan_debate_task_id = None
                with _background_lock:
                    _background_results.pop(task_id, None)
    except Exception:
        pass


def _start_cardloan_debate_background_task(question: str, reviewer_prompts: dict[str, str]) -> None:
    task_id = f"cardloan_debate_{int(time.time() * 1000)}"
    st.session_state.cardloan_debate_task_id = task_id
    st.session_state.reviewer_debate_round = []
    st.session_state.strategy_debate_status = "카드론 토론실을 시작했습니다. Ollama가 3개 부서를 순서대로 실행합니다."
    st.session_state.cardloan_debate = {
        "status": "running",
        "question": question,
        "summary": "신용기획부 단계 준비 중",
        "current_stage": "신용기획부",
        "round_results": [],
    }

    backend_url = str(st.session_state.get("backend_url") or os.environ.get("BACKEND_URL", "http://127.0.0.1:18000")).strip() or "http://127.0.0.1:18000"

    def _run_task(task_id: str, backend_url: str, question: str, reviewer_prompts: dict[str, str]) -> None:
        try:
            client = BackendClient(backend_url)
            result = client.start_cardloan_debate(question, reviewer_prompts=reviewer_prompts)
            with _background_lock:
                _background_results[task_id] = {
                    "status": "completed",
                    "updated_at": datetime.datetime.now().isoformat(),
                    "result": result,
                }
        except Exception as error:
            with _background_lock:
                _background_results[task_id] = {
                    "status": "failed",
                    "updated_at": datetime.datetime.now().isoformat(),
                    "error": str(error),
                }

    threading.Thread(
        target=_run_task,
        args=(task_id, backend_url, question, dict(reviewer_prompts or {})),
        daemon=True,
    ).start()


def render_cardloan_debate_stage_detail(item: dict[str, Any]) -> None:
    response = item.get("response") or {}
    parsed = response.get("parsed") or {}
    raw_text = str(response.get("raw_text") or response.get("answer") or "").strip()
    st.markdown(f"#### {item.get('stage_title') or item.get('name')}")
    st.caption(f"생성 시각: {format_status_time(item.get('generated_at'))}")
    if parsed:
        st.json(parsed)
    elif raw_text:
        st.code(raw_text, language="json")
    else:
        st.info("아직 결과가 없습니다.")
    if raw_text:
        with st.expander("원본 Ollama 응답", expanded=False):
            st.code(raw_text, language="json")


def render_cardloan_reviewer_cards(personas: list[dict[str, str]]) -> None:
    debate_state = st.session_state.get("cardloan_debate", {}) or {}
    round_results = debate_state.get("round_results") or st.session_state.get("reviewer_debate_round", []) or []
    current_stage = str(debate_state.get("current_stage") or "").strip()
    raw_runtime = st.session_state.get("ollama_runtime", {}) or {}
    runtime = raw_runtime if is_cardloan_debate_agent(raw_runtime.get("agent")) else {}
    thinking_text = get_cardloan_debate_thinking_message(current_stage, runtime, round_results)
    statuses = st.session_state.get("agent_statuses", {}) or {}
    result_map = {str(item.get("persona_id") or ""): item for item in round_results}

    reviewer_cols = st.columns(3)
    for column, persona in zip(reviewer_cols, personas):
        is_selected = persona["id"] == st.session_state.get("selected_reviewer_id", personas[0]["id"])
        is_speaking = persona["name"] == current_stage and str(debate_state.get("status") or "") == "running"
        info = statuses.get(persona["id"], {}) or {}
        status_code = str(
            info.get("status")
            or (
                "running"
                if is_speaking
                else "completed" if persona["id"] in result_map else "pending"
            )
        )
        status_label, status_color, _ = get_agent_status_palette(status_code)
        result = result_map.get(persona["id"], {})
        speech_text = str(result.get("preview") or info.get("detail") or persona["tagline"] or "대기 중").strip()
        if is_speaking and (not speech_text or speech_text == "대기 중"):
            speech_text = thinking_text
        is_thinking = is_speaking and not str(result.get("preview") or "").strip()

        with column:
            column.markdown(
                f"""
                <div class="reviewer-card {persona['avatar_class']}{' active' if is_selected else ''}{' speaking' if is_speaking else ''}">
                    <div class="reviewer-role">{persona['emoji']} Reviewer</div>
                    <div class="reviewer-avatar-wrap">
                        <div class="reviewer-avatar {persona['avatar_class']}{' active badge-speaking' if is_speaking else ''}">
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
                    <div class="reviewer-live-head">
                        <div class="reviewer-select-note">\"{html.escape(persona['tagline'])}\"</div>
                        <span class="reviewer-status-chip" style="color:{status_color};">{html.escape(status_label)}</span>
                    </div>
                    <div class="reviewer-live-speech{' speaking' if is_speaking else ''}">
                        <div class="reviewer-live-label">{html.escape('Now Speaking' if is_speaking else 'Latest Note')}</div>
                        <div class="reviewer-live-text{' thinking' if is_thinking else ''}">{html.escape(speech_text[:220] or '대기 중')}</div>
                        {'<div class="reviewer-live-dots"><span></span><span></span><span></span></div>' if is_speaking else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "프롬프트 편집 열기",
                key=f"edit_reviewer_prompt_{persona['id']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                open_reviewer_prompt_dialog(persona["id"])


def render_cardloan_debate_live_panel(personas: list[dict[str, str]]) -> None:
    debate_state = st.session_state.get("cardloan_debate", {}) or {}
    round_results = debate_state.get("round_results") or st.session_state.get("reviewer_debate_round", []) or []
    current_stage = str(debate_state.get("current_stage") or "대기").strip() or "대기"
    summary = str(debate_state.get("summary") or st.session_state.get("strategy_debate_status") or "").strip()
    raw_runtime = st.session_state.get("ollama_runtime", {}) or {}
    runtime = raw_runtime if is_cardloan_debate_agent(raw_runtime.get("agent")) else {}
    runtime_agent = _format_ollama_toast_agent_label(str(runtime.get("agent") or ""))
    runtime_text = str(runtime.get("response_text") or "").strip()
    runtime_status = str(runtime.get("status") or ("running" if str(debate_state.get("status") or "") == "running" else "idle"))
    runtime_label, runtime_color, _ = get_agent_status_palette(runtime_status)
    is_streaming = runtime_status == "running" or str(debate_state.get("status") or "") == "running"
    thinking_text = get_cardloan_debate_thinking_message(current_stage, runtime, round_results)
    live_text = runtime_text or thinking_text
    live_helper = "실시간 답변 생성 중" if is_streaming else "최근 생성 결과"
    broadcast_title, broadcast_body = build_cardloan_live_broadcast(
        current_stage,
        runtime,
        round_results,
        summary,
        is_streaming,
    )
    live_text = build_cardloan_live_display_text(
        current_stage,
        runtime_text,
        thinking_text,
        broadcast_body,
        round_results,
        is_streaming,
    )
    live_message_key = "|".join(
        [
            str(current_stage or ""),
            str(runtime.get("updated_at") or debate_state.get("updated_at") or ""),
            str(runtime_status or ""),
            str(len(round_results)),
            str(len(live_text)),
        ]
    )
    live_text_typed, is_typing_live_text = build_cardloan_typewriter_text(
        live_text,
        live_message_key,
        is_streaming=is_streaming,
    )
    live_text_html = html.escape(live_text_typed).replace("\n", "<br>")
    live_cursor_html = '<span class="debate-live-cursor"></span>' if is_typing_live_text or is_streaming else ""
    live_caption = (
        "JSON 원문 대신 현재 단계가 실제로 어떤 판단을 만들고 있는지 자연어 문장으로 중계합니다."
        if is_streaming
        else "완료된 뒤에도 화면에는 JSON 대신 읽기 쉬운 문장으로 정리된 발언을 우선 보여줍니다."
    )

    st.markdown(
        f"""
        <div class="debate-status">
            <div class="debate-status-title">현재 단계 · {html.escape(current_stage)}</div>
            <div class="debate-status-text">{html.escape(summary or '카드론 토론실 대기 중입니다.')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="debate-live-shell">
            <div class="debate-live-header">
                <div>
                    <div class="debate-live-kicker">Live Ollama</div>
                    <div class="debate-live-title">{html.escape(runtime_agent or current_stage or '카드론 토론실')}</div>
                </div>
                <span style="padding:6px 10px; border-radius:999px; font-size:11px; font-weight:800; background:rgba(255,255,255,0.12); color:{runtime_color}; border:1px solid rgba(255,255,255,0.12);">{html.escape(runtime_label)}</span>
            </div>
            <div class="debate-live-broadcast">
                <div class="debate-live-broadcast-title">{html.escape(broadcast_title)}</div>
                <div class="debate-live-broadcast-body">{html.escape(broadcast_body)}</div>
            </div>
            <div class="debate-live-meta">
                <span class="debate-live-helper">{html.escape(live_helper)}</span>
                <span class="debate-live-helper">현재 단계 {html.escape(current_stage)}</span>
                <span class="debate-live-helper">업데이트 {html.escape(format_status_time(runtime.get('updated_at') or debate_state.get('updated_at')))}</span>
            </div>
            <div class="debate-live-body{' streaming' if is_streaming else ''}">
                <div class="debate-live-message{' thinking' if live_text == thinking_text or live_text == broadcast_body else ''}">{live_text_html}{live_cursor_html}</div>
                {'<div class="debate-live-dots"><span></span><span></span><span></span></div>' if is_streaming else ''}
            </div>
            <div class="debate-live-caption">{html.escape(live_caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if round_results:
        st.markdown(
            f"""
            <div class="consensus-card">
                <div class="consensus-label">Cardloan Debate</div>
                <div class="consensus-title">3개 부서 토론 종합 메모</div>
                <div class="consensus-body">{html.escape(build_debate_consensus(personas, round_results))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        transcript_col, detail_col = st.columns([1.05, 0.95])
        with transcript_col:
            st.markdown('<div class="debate-transcript">', unsafe_allow_html=True)
            for item in round_results:
                persona = next((p for p in personas if p["id"] == item.get("persona_id")), personas[0])
                st.markdown(
                    f"""
                    <div class="debate-bubble">
                        <div class="debate-bubble-head">
                            <div class="debate-bubble-avatar">
                                <div class="debate-bubble-mini-avatar" style="background:{persona['accent']};">{persona['emoji']}</div>
                                <div class="debate-bubble-name">{item.get('name')} · {item.get('stage_title')}</div>
                            </div>
                            <span class="debate-bubble-badge" style="background:{persona['accent']};">{html.escape(str(item.get('verdict') or '-'))}</span>
                        </div>
                        <div class="debate-bubble-text">{html.escape(build_cardloan_stage_spoken_message(item))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with detail_col:
            detail_tabs = st.tabs([item.get("name") or f"stage-{index + 1}" for index, item in enumerate(round_results)])
            for tab, item in zip(detail_tabs, round_results):
                with tab:
                    render_cardloan_debate_stage_detail(item)


def should_render_cardloan_live_panel() -> bool:
    debate_state = st.session_state.get("cardloan_debate", {}) or {}
    round_results = debate_state.get("round_results") or st.session_state.get("reviewer_debate_round", []) or []
    runtime = st.session_state.get("ollama_runtime", {}) or {}
    runtime_text = str(runtime.get("response_text") or "").strip()
    summary = str(debate_state.get("summary") or st.session_state.get("strategy_debate_status") or "").strip()
    status = str(debate_state.get("status") or "").strip()
    current_stage = str(debate_state.get("current_stage") or "").strip()
    return bool(
        round_results
        or runtime_text
        or summary
        or status == "running"
        or current_stage not in {"", "대기"}
        or st.session_state.get("cardloan_debate_task_id")
    )


@fragment_decorator(run_every="800ms")
def render_live_cardloan_ollama_fragment() -> None:
    refresh_cardloan_debate_runtime()
    personas = get_reviewer_personas()
    if should_render_cardloan_live_panel():
        render_cardloan_debate_live_panel(personas)


def render_role_based_strategy_tab():
    ensure_strategy_debate_state()
    personas = get_reviewer_personas()
    persona_map = {persona["id"]: persona for persona in personas}
    selected_id = st.session_state.get("selected_reviewer_id", personas[0]["id"])
    selected_persona = persona_map.get(selected_id, personas[0])
    debate_state = st.session_state.get("cardloan_debate", {}) or {}
    debate_running = str(debate_state.get("status") or "").strip() == "running" or bool(st.session_state.get("cardloan_debate_task_id"))
    st.session_state.strategy_debate_question = get_default_cardloan_debate_question()

    hero_main, hero_action = st.columns([0.72, 0.28])
    with hero_main:
        st.markdown(
            """
            <div class="debate-hero">
                <div class="debate-hero-copy">
                    <div class="debate-kicker">Cardloan Strategy Room</div>
                    <div class="debate-title">AI 카드론 토론실</div>
                    <div class="debate-subtitle">신용기획부가 시장 리스크를 먼저 정리하고, 금융영업부가 승인 전환 전략을 만들고, 마지막으로 금융솔루션부가 카드론 상품 구조를 설계합니다.</div>
                    <div class="debate-wave"><span></span><span></span><span></span><span></span><span></span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_action:
        
        run_clicked = st.button(
            "토론시작",
            key="cardloan_debate_start",
            use_container_width=False,
            type="primary",
            disabled=debate_running,
        )

    if st.session_state.get("reviewer_prompt_dialog_open"):
        render_reviewer_prompt_dialog(selected_persona)

    if run_clicked:
        _start_cardloan_debate_background_task(
            get_default_cardloan_debate_question(),
            st.session_state.get("reviewer_prompts", {}) or {},
        )

    debate_state = st.session_state.get("cardloan_debate", {}) or {}
    debate_running = str(debate_state.get("status") or "").strip() == "running" or bool(st.session_state.get("cardloan_debate_task_id"))

    refresh_cardloan_debate_runtime()
    render_cardloan_reviewer_cards(personas)

    if HAS_FRAGMENT_REFRESH and debate_running:
        render_live_cardloan_ollama_fragment()
    else:
        if should_render_cardloan_live_panel():
            render_cardloan_debate_live_panel(personas)


# `get_vector_count` is provided by `rag.vector_db` import; avoid redefining it here.


def get_chart_snapshots() -> dict:
    try:
        payload = get_backend_client().get_charts()
        return payload.get("charts", {})
    except Exception:
        return {}


def get_faiss_store_options() -> list[tuple[str, str | None]]:
    return [
        ("전체 DB", None),
        ("심사 로그 DB", FAISS_STORE_LOGS),
        ("뉴스 신호 DB", FAISS_STORE_NEWS),
        ("규제 문서 DB", FAISS_STORE_DOCUMENT),
        ("고객 패턴 DB", FAISS_STORE_CUSTOMER),
    ]


def get_live_faiss_items(
    limit: int = 1000, store_name: str | None = None
) -> tuple[list[dict], int]:
    cache_key = f"full_faiss_items::{store_name or 'all'}"
    cache_meta_key = f"{cache_key}::meta"
    cache_version = _get_faiss_session_cache_version(store_name)
    if store_name is None:
        items = st.session_state.get("full_faiss_items", []) or []
        if items:
            return items[:limit], int(st.session_state.get("vector_count", len(items)) or 0)

    cached_items = st.session_state.get(cache_key, []) or []
    cached_meta = st.session_state.get(cache_meta_key, {}) or {}
    if cached_items and cached_meta.get("version") == cache_version:
        return cached_items[:limit], int(cached_meta.get("total_count", len(cached_items)) or len(cached_items))

    try:
        entries_resp = get_backend_client().get_faiss_entries(limit=limit, store_name=store_name)
        items = entries_resp.get("items", []) if isinstance(entries_resp, dict) else []
        total_count = int((entries_resp or {}).get("total_count", len(items)) or 0)
        if items:
            st.session_state[cache_key] = items
            st.session_state[cache_meta_key] = {
                "version": cache_version,
                "total_count": total_count,
            }
            if store_name is None:
                st.session_state.full_faiss_items = items
            return items, total_count
    except Exception:
        pass

    try:
        from rag.vector_db import list_vectors

        items = list_vectors(limit=limit, store_name=store_name)
        if items:
            st.session_state[cache_key] = items
            st.session_state[cache_meta_key] = {
                "version": cache_version,
                "total_count": len(items),
            }
            if store_name is None:
                st.session_state.full_faiss_items = items
        return items, int((st.session_state.get("vector_count") if store_name is None else len(items)) or len(items))
    except Exception:
        return [], 0


def render_live_signal_news_board() -> None:
    news_items, news_total_count = get_live_faiss_items(limit=1000, store_name=FAISS_STORE_NEWS)
    signal_items = [
        item
        for item in news_items
        if str(item.get("type") or "").strip().lower() == "signal_news"
    ]

    st.markdown("#### 실시간 signal_news 전체 보기")
    if not signal_items:
        st.info("현재 실시간으로 표시할 signal_news가 없습니다.")
        return

    summary_col_a, summary_col_b, summary_col_c = st.columns(3)
    summary_col_a.metric("signal_news 수", len(signal_items))
    summary_col_b.metric("뉴스 신호 DB 전체", news_total_count)
    latest_signal = signal_items[0]
    summary_col_c.metric("최근 source", latest_signal.get("source") or "-")

    board_rows: list[dict[str, Any]] = []
    for item in signal_items:
        features = item.get("features") or {}
        board_rows.append(
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "product": item.get("product"),
                "name": item.get("name"),
                "tags": ", ".join(features.get("tags") or []),
                "signal_summary": str(features.get("signal_summary") or "")[:140],
                "snippet": str(item.get("snippet") or "")[:180],
            }
        )
    st.dataframe(pd.DataFrame(board_rows), height=320, width="stretch", hide_index=True)

    for item in signal_items:
        features = item.get("features") or {}
        title = str(item.get("name") or item.get("title") or item.get("id") or "signal_news").strip()
        label = f"{title[:72]} · {str(item.get('id') or '')[:8]}"
        with st.expander(label, expanded=False):
            detail_col_a, detail_col_b = st.columns([1.1, 1])
            with detail_col_a:
                st.markdown("##### page_content / snippet")
                st.code(str(item.get("snippet") or item.get("page_content") or "")[:2500], language="text")
            with detail_col_b:
                st.markdown("##### metadata.features")
                st.json(features)


def _summarize_top_counts(values: list[Any], top_n: int = 4) -> tuple[dict[str, int], str]:
    counter: dict[str, int] = {}
    for value in values:
        text = str(value or "-").strip() or "-"
        counter[text] = counter.get(text, 0) + 1
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    summary = ", ".join(f"{key} {count}건" for key, count in ordered[:top_n])
    return counter, summary or "-"


def _format_customer_pattern_top_reason(features: dict[str, Any]) -> str:
    reasons = features.get("top_reject_reason_codes") or []
    if not reasons:
        return "-"
    top_reason = reasons[0] or {}
    description = str(top_reason.get("description") or "").strip()
    code = str(top_reason.get("code") or "").strip()
    count = int(top_reason.get("count") or 0)
    label = description or code or "-"
    return f"{label} ({count}건)" if count else label


def _classify_vector_event(event: dict[str, Any]) -> tuple[str, str, str]:
    source = str(event.get("source") or "-").strip().lower()
    detail = str(event.get("detail") or "").strip()
    combined = f"{source} {detail}"
    if "news" in combined or "뉴스" in detail:
        return "news", "뉴스 증분 적재", "news"
    if "log" in combined or "심사 로그" in detail or "구조화" in detail:
        return "log", "심사로그 증분 적재", "log"
    return "mixed", "FAISS 증분 적재", "mixed"


def render_live_vector_append_board(vector_events: list[dict[str, Any]]) -> None:
    recent_events = [
        event
        for event in vector_events[:8]
        if int(event.get("added_count", 0) or 0) > 0
    ]
    if not recent_events:
        return

    latest_event = recent_events[0]
    latest_group, latest_label, latest_css = _classify_vector_event(latest_event)
    now = datetime.datetime.now()
    aggregate = {
        "log": {"added": 0, "events": 0},
        "news": {"added": 0, "events": 0},
        "mixed": {"added": 0, "events": 0},
    }
    for event in recent_events:
        event_group, _, _ = _classify_vector_event(event)
        aggregate[event_group]["added"] += int(event.get("added_count", 0) or 0)
        aggregate[event_group]["events"] += 1

    pulse_cards = "".join(
        f"""
        <div class='vector-live-card {card_css}'>
            <div class='vector-live-card-label'>{html.escape(card_label)}</div>
            <div class='vector-live-card-value'>+{int(aggregate[card_key]['added'] or 0)}</div>
            <div class='vector-live-card-sub'>최근 이벤트 {int(aggregate[card_key]['events'] or 0)}회</div>
        </div>
        """
        for card_key, card_label, card_css in [
            ("log", "심사로그 add", "log"),
            ("news", "뉴스 add", "news"),
            ("mixed", "기타 add", "mixed"),
        ]
    )

    ticker = "".join(
        f"<span>{html.escape(str(event.get('source') or '-'))} +{int(event.get('added_count', 0) or 0)} · {html.escape(str(event.get('detail') or '-'))}</span>"
        for event in recent_events[:5]
    )
    event_time = parse_status_time(latest_event.get("timestamp"))
    age_seconds = max(
        0,
        int((now - event_time).total_seconds()) if event_time is not None else 0,
    )
    progress_width = min(100, max(18, int(latest_event.get("added_count", 0) or 0) * 12))

    st.markdown(
        f"""
        <style>
        .vector-live-board {{
            position: relative;
            overflow: hidden;
            margin: 8px 0 16px 0;
            padding: 18px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(8,26,39,0.94), rgba(12,49,67,0.92));
            border: 1px solid rgba(97,244,222,0.18);
            box-shadow: 0 20px 50px rgba(2, 12, 27, 0.28);
        }}
        .vector-live-board::after {{
            content: '';
            position: absolute;
            inset: auto -8% -45% auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(97,244,222,0.18), transparent 68%);
            pointer-events: none;
        }}
        .vector-live-head {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 14px;
            margin-bottom: 14px;
        }}
        .vector-live-kicker {{
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(217,236,251,0.72);
        }}
        .vector-live-title {{
            font-size: 27px;
            font-weight: 900;
            color: #f7fbff;
            margin-top: 5px;
            line-height: 1.15;
        }}
        .vector-live-sub {{
            font-size: 13px;
            line-height: 1.6;
            color: rgba(217,236,251,0.82);
            margin-top: 7px;
            max-width: 760px;
        }}
        .vector-live-badge {{
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #d9ecfb;
            font-size: 12px;
            font-weight: 800;
            white-space: nowrap;
        }}
        .vector-live-progress {{
            position: relative;
            overflow: hidden;
            height: 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            margin-bottom: 14px;
        }}
        .vector-live-progress-fill {{
            height: 100%;
            width: {progress_width}%;
            border-radius: 999px;
            background: linear-gradient(90deg, #61f4de, #7dd3fc, #ffbf69);
            background-size: 200% 100%;
            animation: vectorFlow 1.8s linear infinite;
            box-shadow: 0 0 24px rgba(97,244,222,0.35);
        }}
        .vector-live-grid {{
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 12px;
        }}
        .vector-live-card-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }}
        .vector-live-card {{
            position: relative;
            overflow: hidden;
            min-height: 108px;
            border-radius: 18px;
            padding: 14px;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.10);
        }}
        .vector-live-card::after {{
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent, rgba(255,255,255,0.14), transparent);
            transform: translateX(-140%);
            animation: vectorSweep 2.4s ease-in-out infinite;
        }}
        .vector-live-card.log {{ box-shadow: inset 0 0 0 1px rgba(245,158,11,0.18); }}
        .vector-live-card.news {{ box-shadow: inset 0 0 0 1px rgba(34,197,94,0.18); }}
        .vector-live-card.mixed {{ box-shadow: inset 0 0 0 1px rgba(125,211,252,0.18); }}
        .vector-live-card-label {{
            font-size: 12px;
            color: rgba(217,236,251,0.72);
            font-weight: 700;
        }}
        .vector-live-card-value {{
            font-size: 36px;
            line-height: 1;
            margin-top: 10px;
            font-weight: 900;
            color: #f7fbff;
            animation: vectorCountPop 0.6s ease-out;
        }}
        .vector-live-card-sub {{
            font-size: 12px;
            margin-top: 8px;
            color: rgba(217,236,251,0.74);
        }}
        .vector-live-event {{
            position: relative;
            border-radius: 18px;
            padding: 15px 16px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
        }}
        .vector-live-event.log {{ box-shadow: 0 0 0 1px rgba(245,158,11,0.16), 0 0 26px rgba(245,158,11,0.12); }}
        .vector-live-event.news {{ box-shadow: 0 0 0 1px rgba(34,197,94,0.16), 0 0 26px rgba(34,197,94,0.12); }}
        .vector-live-event.mixed {{ box-shadow: 0 0 0 1px rgba(125,211,252,0.16), 0 0 26px rgba(125,211,252,0.12); }}
        .vector-live-event-kicker {{
            font-size: 12px;
            color: rgba(217,236,251,0.74);
            font-weight: 800;
            margin-bottom: 8px;
        }}
        .vector-live-event-value {{
            font-size: 40px;
            font-weight: 900;
            color: #ffffff;
            line-height: 1;
            text-shadow: 0 0 20px rgba(255,255,255,0.12);
        }}
        .vector-live-event-sub {{
            margin-top: 10px;
            font-size: 13px;
            color: rgba(217,236,251,0.82);
            line-height: 1.55;
        }}
        .vector-live-ticker {{
            overflow: hidden;
            margin-top: 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 10px 0;
        }}
        .vector-live-ticker-track {{
            display: inline-flex;
            gap: 28px;
            white-space: nowrap;
            padding-left: 100%;
            animation: vectorTicker 18s linear infinite;
        }}
        .vector-live-ticker span {{
            color: rgba(217,236,251,0.82);
            font-size: 12px;
            font-weight: 700;
        }}
        @keyframes vectorFlow {{
            0% {{ background-position: 0% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        @keyframes vectorSweep {{
            0% {{ transform: translateX(-140%); }}
            100% {{ transform: translateX(140%); }}
        }}
        @keyframes vectorCountPop {{
            0% {{ opacity: 0; transform: translateY(12px) scale(0.94); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        @keyframes vectorTicker {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-100%); }}
        }}
        @media (max-width: 1080px) {{
            .vector-live-grid, .vector-live-card-grid {{
                grid-template-columns: 1fr;
            }}
            .vector-live-head {{
                flex-direction: column;
            }}
        }}
        </style>
        <div class='vector-live-board'>
            <div class='vector-live-head'>
                <div>
                    <div class='vector-live-kicker'>Live Vector Sync</div>
                    <div class='vector-live-title'>{html.escape(latest_label)} · +{int(latest_event.get('added_count', 0) or 0)}</div>
                    <div class='vector-live-sub'>{html.escape(str(latest_event.get('detail') or '-'))}</div>
                </div>
                <div class='vector-live-badge'>{html.escape(str(latest_event.get('source') or '-'))} · {age_seconds}초 전</div>
            </div>
            <div class='vector-live-progress'>
                <div class='vector-live-progress-fill'></div>
            </div>
            <div class='vector-live-grid'>
                <div class='vector-live-card-grid'>
                    {pulse_cards}
                </div>
                <div class='vector-live-event {latest_css}'>
                    <div class='vector-live-event-kicker'>최근 증분 적재</div>
                    <div class='vector-live-event-value'>+{int(latest_event.get('added_count', 0) or 0)}</div>
                    <div class='vector-live-event-sub'>누적 {int(latest_event.get('before_count', 0) or 0)} → {int(latest_event.get('after_count', 0) or 0)}<br>{html.escape(str(latest_event.get('source') or '-'))}</div>
                </div>
            </div>
            <div class='vector-live-ticker'><div class='vector-live-ticker-track'>{ticker}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_runtime_live_stage(
    statuses: dict[str, Any],
    vector_events: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    running_agents = [
        (agent_key, info or {})
        for agent_key, info in statuses.items()
        if str((info or {}).get("status") or "") == "running"
    ]
    latest_vector = vector_events[0] if vector_events else {}
    crawl_running = bool(st.session_state.get("news_crawl_running", False))
    has_vector_update = bool(int(latest_vector.get("added_count", 0) or 0) > 0)
    if not running_agents and not crawl_running and not has_vector_update:
        return

    title = "운영 파이프라인이 실시간으로 반응 중입니다"
    subtitle = "신호가 들어오면 에이전트 실행과 FAISS 반영 상태를 즉시 업데이트합니다."
    badge = "LIVE"
    if running_agents:
        lead_key, lead_info = running_agents[0]
        title = f"{lead_key} 실행 중 · 실시간 운영 동기화"
        subtitle = str(lead_info.get("detail") or "최신 입력을 처리하고 있습니다.")
        badge = f"RUN {len(running_agents)}"
    elif crawl_running:
        title = "뉴스 크롤링 진행 중 · 수집 결과가 이어서 반영됩니다"
        subtitle = (
            f"대상 {int(st.session_state.get('news_crawl_target_count', 0) or 0)}건 / "
            f"성공 {int(st.session_state.get('news_crawl_success_count', 0) or 0)} / "
            f"실패 {int(st.session_state.get('news_crawl_failure_count', 0) or 0)}"
        )
        badge = "CRAWL"
    elif has_vector_update:
        title = f"FAISS 업데이트 감지 · +{int(latest_vector.get('added_count', 0) or 0)}"
        subtitle = str(latest_vector.get("detail") or "신규 데이터가 벡터 스토어에 반영되었습니다.")
        badge = "SYNC"

    chip_html = "".join(
        f"<span class='runtime-stage-chip'>{html.escape(agent_key)} running</span>"
        for agent_key, _ in running_agents[:4]
    )
    if has_vector_update:
        chip_html += (
            f"<span class='runtime-stage-chip vector'>{html.escape(str(latest_vector.get('source') or '-'))} "
            f"+{int(latest_vector.get('added_count', 0) or 0)}</span>"
        )
    if crawl_running:
        chip_html += "<span class='runtime-stage-chip crawl'>news crawler active</span>"

    ticker_items = []
    for agent_key, info in running_agents[:4]:
        ticker_items.append(
            f"<span>{html.escape(agent_key)} · {html.escape(str(info.get('detail') or 'running'))}</span>"
        )
    for event in vector_events[:4]:
        ticker_items.append(
            f"<span>{html.escape(str(event.get('source') or '-'))} +{int(event.get('added_count', 0) or 0)} · {html.escape(str(event.get('detail') or '-'))}</span>"
        )
    ticker_html = "".join(ticker_items)

    st.markdown(
        f"""
        <style>
        .runtime-stage-board {{
            position: relative;
            overflow: hidden;
            margin: 8px 0 18px 0;
            border-radius: 24px;
            padding: 20px;
            background: linear-gradient(135deg, rgba(8,26,39,0.95), rgba(14,48,68,0.92));
            border: 1px solid rgba(125,211,252,0.18);
            box-shadow: 0 22px 52px rgba(2,12,27,0.26);
        }}
        .runtime-stage-board::before, .runtime-stage-board::after {{
            content: '';
            position: absolute;
            border-radius: 999px;
            pointer-events: none;
        }}
        .runtime-stage-board::before {{
            inset: -25% auto auto -8%;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(97,244,222,0.14), transparent 72%);
            animation: runtimeOrb 6s ease-in-out infinite;
        }}
        .runtime-stage-board::after {{
            inset: auto -10% -42% auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(255,191,105,0.12), transparent 72%);
            animation: runtimeOrb 7.5s ease-in-out infinite reverse;
        }}
        .runtime-stage-head {{ position: relative; z-index: 1; display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
        .runtime-stage-kicker {{ font-size:12px; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; color:rgba(217,236,251,0.72); }}
        .runtime-stage-title {{ margin-top:6px; font-size:28px; line-height:1.14; font-weight:900; color:#f7fbff; }}
        .runtime-stage-sub {{ margin-top:9px; max-width:760px; font-size:13px; line-height:1.68; color:rgba(217,236,251,0.82); }}
        .runtime-stage-pill {{ position:relative; z-index:1; padding:9px 13px; border-radius:999px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14); color:#f8fafc; font-size:12px; font-weight:900; white-space:nowrap; }}
        .runtime-stage-pulse {{ position:relative; z-index:1; margin-top:14px; height:12px; border-radius:999px; background:rgba(255,255,255,0.10); overflow:hidden; }}
        .runtime-stage-pulse-fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg, #61f4de, #7dd3fc, #ffbf69); background-size:200% 100%; animation:runtimeFlow 2.1s linear infinite; }}
        .runtime-stage-body {{ position:relative; z-index:1; display:grid; grid-template-columns:1.1fr 0.9fr; gap:12px; margin-top:14px; }}
        .runtime-stage-chip-row {{ display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start; }}
        .runtime-stage-chip {{ padding:8px 10px; border-radius:999px; font-size:12px; font-weight:800; color:#f8fafc; border:1px solid rgba(255,255,255,0.10); background:rgba(30,64,175,0.34); animation:runtimeChipPulse 1.8s ease-in-out infinite; }}
        .runtime-stage-chip.vector {{ background:rgba(14,116,144,0.34); }}
        .runtime-stage-chip.crawl {{ background:rgba(5,150,105,0.34); }}
        .runtime-stage-stat {{ padding:16px; border-radius:18px; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.10); }}
        .runtime-stage-stat-label {{ font-size:12px; color:rgba(217,236,251,0.70); font-weight:700; }}
        .runtime-stage-stat-value {{ margin-top:10px; font-size:38px; line-height:1; font-weight:900; color:#ffffff; animation:runtimeNumberPulse 0.65s ease-out; }}
        .runtime-stage-stat-sub {{ margin-top:10px; font-size:12px; line-height:1.55; color:rgba(217,236,251,0.78); }}
        .runtime-stage-ticker {{ position:relative; z-index:1; overflow:hidden; margin-top:13px; border-radius:14px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); padding:10px 0; }}
        .runtime-stage-ticker-track {{ display:inline-flex; gap:28px; white-space:nowrap; padding-left:100%; animation:runtimeTicker 20s linear infinite; }}
        .runtime-stage-ticker span {{ color:rgba(217,236,251,0.82); font-size:12px; font-weight:700; }}
        @keyframes runtimeFlow {{ 0% {{ background-position:0% 0; }} 100% {{ background-position:200% 0; }} }}
        @keyframes runtimeOrb {{ 0%,100% {{ transform:translate3d(0,0,0) scale(1); }} 50% {{ transform:translate3d(10px,-8px,0) scale(1.06); }} }}
        @keyframes runtimeChipPulse {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-1px); }} }}
        @keyframes runtimeNumberPulse {{ 0% {{ opacity:0; transform:translateY(10px) scale(0.96); }} 100% {{ opacity:1; transform:translateY(0) scale(1); }} }}
        @keyframes runtimeTicker {{ 0% {{ transform:translateX(0); }} 100% {{ transform:translateX(-100%); }} }}
        @media (max-width: 1080px) {{ .runtime-stage-head {{ flex-direction:column; }} .runtime-stage-body {{ grid-template-columns:1fr; }} }}
        </style>
        <div class='runtime-stage-board'>
            <div class='runtime-stage-head'>
                <div>
                    <div class='runtime-stage-kicker'>Live Ops Stage</div>
                    <div class='runtime-stage-title'>{html.escape(title)}</div>
                    <div class='runtime-stage-sub'>{html.escape(subtitle)}</div>
                </div>
                <div class='runtime-stage-pill'>{html.escape(badge)}</div>
            </div>
            <div class='runtime-stage-pulse'><div class='runtime-stage-pulse-fill'></div></div>
            <div class='runtime-stage-body'>
                <div class='runtime-stage-chip-row'>{chip_html}</div>
                <div class='runtime-stage-stat'>
                    <div class='runtime-stage-stat-label'>최근 FAISS 증분 반영</div>
                    <div class='runtime-stage-stat-value'>+{int(latest_vector.get('added_count', 0) or 0)}</div>
                    <div class='runtime-stage-stat-sub'>마지막 활동 {html.escape(str(diagnostics.get('last_activity_source') or '-'))}<br>실행 중 에이전트 {len(running_agents)}개</div>
                </div>
            </div>
            <div class='runtime-stage-ticker'><div class='runtime-stage-ticker-track'>{ticker_html}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vector_db_panel():
    st.subheader("🧠 Vector DB 실시간 적재 현황")

    vector_events = st.session_state.get("vector_events", []) or []
    latest_vector_event = vector_events[0] if vector_events else {}
    render_live_vector_append_board(vector_events)
    grouped_items: dict[str, list[dict[str, Any]]] = {
        FAISS_STORE_LOGS: [],
        FAISS_STORE_NEWS: [],
        FAISS_STORE_CUSTOMER: [],
        FAISS_STORE_DOCUMENT: [],
    }
    store_options = get_faiss_store_options()
    selected_store_label = st.selectbox(
        "조회할 DB",
        options=[label for label, _ in store_options],
        key="vector_db_panel_store_filter",
    )
    selected_store = dict(store_options).get(selected_store_label)
    items, total_count = get_live_faiss_items(limit=1000, store_name=selected_store)
    store_snapshots: list[dict[str, Any]] = []
    store_descriptions = {
        FAISS_STORE_LOGS: "구조화 심사로그와 generated_log",
        FAISS_STORE_NEWS: "원본 뉴스와 signal_news",
        FAISS_STORE_DOCUMENT: "업로드 규제문서와 규제 분석 결과",
        FAISS_STORE_CUSTOMER: "product_pattern_summary, sales_strategy, generated_customer",
    }
    for item in items:
        store_name = str(item.get("store") or "").strip().lower()
        if store_name in grouped_items:
            grouped_items[store_name].append(item)

    if selected_store is None:
        for snapshot_label, snapshot_store in store_options[1:]:
            snapshot_items = grouped_items.get(snapshot_store, [])
            _, type_summary = _summarize_top_counts([item.get("type") for item in snapshot_items])
            _, product_summary = _summarize_top_counts([item.get("product") for item in snapshot_items if item.get("product")])
            store_snapshots.append(
                {
                    "DB": snapshot_label,
                    "store": snapshot_store,
                    "현재 로드": len(snapshot_items),
                    "주요 type": type_summary,
                    "주요 product": product_summary,
                    "설명": store_descriptions.get(snapshot_store, "-"),
                }
            )
    else:
        _, type_summary = _summarize_top_counts([item.get("type") for item in items])
        _, product_summary = _summarize_top_counts([item.get("product") for item in items if item.get("product")])
        store_snapshots.append(
            {
                "DB": selected_store_label,
                "store": selected_store,
                "현재 로드": len(items),
                "주요 type": type_summary,
                "주요 product": product_summary,
                "설명": store_descriptions.get(selected_store, "-"),
            }
        )

    type_counts, _ = _summarize_top_counts([item.get("type") for item in items])
    product_counts, _ = _summarize_top_counts([item.get("product") for item in items if item.get("product")])

    vector_metric_cols = st.columns(4)
    vector_metric_cols[0].metric(
        f"{selected_store_label} 벡터 수", total_count
    )
    vector_metric_cols[1].metric(
        "문서 type 수", len(type_counts)
    )
    vector_metric_cols[2].metric(
        "상품 코드 수", len(product_counts)
    )
    vector_metric_cols[3].metric(
        "마지막 증감", latest_vector_event.get("added_count", 0)
    )

    st.caption(
        f"현재 화면은 {selected_store_label} 기준 단일 조회 결과로만 렌더링합니다. 로드된 항목 {len(items)}건"
        + (f" / 총 벡터 {total_count}건" if total_count else "")
    )

    st.markdown("#### 현재 FAISS 스토어 구조")
    st.dataframe(pd.DataFrame(store_snapshots), width="stretch", hide_index=True)

    if items:
        dist_col_a, dist_col_b = st.columns(2)
        with dist_col_a:
            st.markdown("#### 선택 DB type 분포")
            type_df = pd.DataFrame(
                [{"type": key, "count": value} for key, value in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))]
            )
            st.dataframe(type_df, width="stretch", hide_index=True)
        with dist_col_b:
            st.markdown("#### 선택 DB product 분포")
            if product_counts:
                product_df = pd.DataFrame(
                    [{"product": key, "count": value} for key, value in sorted(product_counts.items(), key=lambda item: (-item[1], item[0]))]
                )
                st.dataframe(product_df, width="stretch", hide_index=True)
            else:
                st.info("선택한 DB에는 product 메타데이터가 없습니다.")

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
        st.caption("현재 탭은 store/type/product/features 중심의 최신 FAISS shape를 보여줍니다. WebSocket 스냅샷을 우선 사용하고, 초기 로드 시에는 백엔드 목록으로 보강합니다.")
        full_df = pd.DataFrame(
            [
                {
                    "id": it.get("id"),
                    "store": it.get("store"),
                    "type": it.get("type"),
                    "product": it.get("product"),
                    "source": it.get("source"),
                    "name": it.get("name"),
                    "feature_keys": ", ".join(list((it.get("features") or {}).keys())[:6]),
                    "snippet": (it.get("snippet") or "")[:200],
                }
                for it in items
            ]
        )
        st.dataframe(full_df, height=360, width="stretch", hide_index=True)


def render_runtime_dashboard():
    st.subheader("🤖 에이전트 실시간 작업 현황")
    render_live_vector_append_board(st.session_state.get("vector_events", []) or [])

    # 백그라운드 작업 결과를 메인 스레드로 폴링하여 session_state로 반영
    try:
        with _background_lock:
            tasks = list(_background_results.items())
        for task_id, payload in tasks:
            if not str(task_id).startswith("reg_"):
                continue
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

    diagnostics = st.session_state.get("backend_diagnostics", {}) or {}
    statuses = st.session_state.get("agent_statuses", {}) or {}
    vector_events = st.session_state.get("vector_events", []) or []
    render_runtime_live_stage(statuses, vector_events, diagnostics)
    if diagnostics:
        st.markdown("#### 백엔드 진단")
        diag_cols = st.columns(4)
        worker_label = (
            f"ON / {diagnostics.get('worker_interval_seconds', '-')}s"
            if diagnostics.get("worker_running")
            else "OFF"
        )
        diag_cols[0].metric("Worker 루프", worker_label)
        diag_cols[1].metric(
            "최근 60초 활동",
            int(diagnostics.get("activity_events_last_60s", 0) or 0),
        )
        diag_cols[2].metric(
            "최근 60초 벡터 이벤트",
            int(diagnostics.get("vector_events_last_60s", 0) or 0),
        )
        diag_cols[3].metric(
            "뉴스 크롤링 백로그",
            int(diagnostics.get("news_crawl_backlog", 0) or 0),
        )

        hotspot_text = diagnostics.get("hotspots") or []
        if hotspot_text:
            st.warning("의심 구간: " + " | ".join(str(item) for item in hotspot_text))

        summary_col_a, summary_col_b = st.columns([1.1, 1])
        with summary_col_a:
            st.markdown(
                f"""
                <div style="padding:14px 16px; border-radius:16px; background:rgba(248,250,252,0.96); border:1px solid rgba(148,163,184,0.18); margin-bottom:12px;">
                    <div style="font-size:12px; font-weight:800; color:#0f172a; margin-bottom:8px;">최근 작업 시각</div>
                    <div style="font-size:13px; color:#334155; line-height:1.7;">
                        마지막 활동: {html.escape(format_status_time(diagnostics.get('last_activity_time')))}<br>
                        마지막 활동 소스: {html.escape(str(diagnostics.get('last_activity_source') or '-'))}<br>
                        마지막 FAISS 반영: {html.escape(format_status_time(diagnostics.get('last_faiss_time')))}<br>
                        마지막 벡터 이벤트: {html.escape(format_status_time(diagnostics.get('last_vector_event_time')))} · {html.escape(str(diagnostics.get('last_vector_event_source') or '-'))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with summary_col_b:
            top_activity_sources = diagnostics.get("top_activity_sources") or []
            top_vector_sources = diagnostics.get("top_vector_sources") or []
            st.markdown("##### 최근 빈도 상위 소스")
            if top_activity_sources:
                activity_df = pd.DataFrame(
                    [{"source": source, "count": count} for source, count in top_activity_sources]
                )
                st.dataframe(activity_df, width="stretch", hide_index=True)
            else:
                st.info("활동 로그가 아직 없습니다.")

            if top_vector_sources:
                vector_df = pd.DataFrame(
                    [{"source": source, "count": count} for source, count in top_vector_sources]
                )
                st.dataframe(vector_df, width="stretch", hide_index=True)

        worker_runtime = diagnostics.get("worker_runtime") or {}
        if worker_runtime:
            st.markdown("##### Worker 단계별 소요시간")
            cadence_cols = st.columns(4)
            cadence_cols[0].metric("기본 루프", f"{int(worker_runtime.get('base_interval_seconds', 0) or 0)}s")
            cadence_cols[1].metric("로그 cadence", f"{int(worker_runtime.get('log_cycle_seconds', 0) or 0)}s")
            cadence_cols[2].metric("뉴스 cadence", f"{int(worker_runtime.get('news_cycle_seconds', 0) or 0)}s")
            cadence_cols[3].metric("FAISS cadence", f"{int(worker_runtime.get('faiss_cycle_seconds', 0) or 0)}s")

            log_runtime_cols = st.columns(4)
            log_runtime_cols[0].metric("1회 로그 생성량", int(worker_runtime.get("log_burst_count", 0) or 0))
            log_runtime_cols[1].metric("로그 분석 cadence", f"{int(worker_runtime.get('log_analysis_seconds', 0) or 0)}s")
            log_runtime_cols[2].metric("로그 브리핑 cadence", f"{int(worker_runtime.get('log_agent_seconds', 0) or 0)}s")
            log_runtime_cols[3].metric("FAISS 재빌드 기준", f"+{int(worker_runtime.get('faiss_log_rebuild_threshold', 0) or 0)} logs")

            generated_products = str(worker_runtime.get("log_generated_products") or "-")
            if generated_products and generated_products != "-":
                st.caption(f"최근 burst 상품 분포: {generated_products}")

            phase_df = pd.DataFrame(
                [
                    {
                        "phase": "log_cycle",
                        "ran": bool(worker_runtime.get("log_cycle_ran")),
                        "elapsed_ms": int(worker_runtime.get("log_cycle_elapsed_ms", 0) or 0),
                    },
                    {
                        "phase": "log_analysis",
                        "ran": bool(worker_runtime.get("log_analysis_ran")),
                        "elapsed_ms": int(worker_runtime.get("log_analysis_elapsed_ms", 0) or 0),
                    },
                    {
                        "phase": "log_agent",
                        "ran": bool(worker_runtime.get("log_agent_ran")),
                        "elapsed_ms": int(worker_runtime.get("log_agent_elapsed_ms", 0) or 0),
                    },
                    {
                        "phase": "news_cycle",
                        "ran": bool(worker_runtime.get("news_cycle_ran")),
                        "elapsed_ms": int(worker_runtime.get("news_cycle_elapsed_ms", 0) or 0),
                    },
                    {
                        "phase": "faiss_cycle",
                        "ran": bool(worker_runtime.get("faiss_cycle_ran")),
                        "elapsed_ms": int(worker_runtime.get("faiss_cycle_elapsed_ms", 0) or 0),
                    },
                ]
            )
            st.dataframe(phase_df, width="stretch", hide_index=True)
            st.caption(
                "최근 worker loop 소요 "
                + f"{int(worker_runtime.get('last_loop_elapsed_ms', 0) or 0)}ms"
                + " · FAISS 재빌드 이유: "
                + str(worker_runtime.get("faiss_rebuild_reason") or "-")
                + f" · 뉴스 유입={bool(worker_runtime.get('faiss_rebuild_due_to_news'))}"
                + f" / 로그 누적 기준 충족={bool(worker_runtime.get('faiss_rebuild_due_to_logs'))}"
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
        pulse_dot = ""
        card_class = "idle"
        if status_code == "running":
            pulse_dot = "<span class='runtime-agent-dot running'></span>"
            card_class = "running"
        elif status_code == "completed":
            pulse_dot = "<span class='runtime-agent-dot completed'></span>"
            card_class = "completed"
        elif status_code == "failed":
            pulse_dot = "<span class='runtime-agent-dot failed'></span>"
            card_class = "failed"
        agent_cols[index % 3].markdown(
            f"""
            <style>
            .runtime-agent-card {{
                min-height: 152px;
                border-radius: 16px;
                padding: 14px;
                margin-bottom: 12px;
                background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.98));
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            }}
            .runtime-agent-card.running {{
                border-color: rgba(59,130,246,0.32);
                box-shadow: 0 14px 30px rgba(59,130,246,0.12);
                animation: runtimeAgentLift 1.8s ease-in-out infinite;
            }}
            .runtime-agent-card.completed {{
                border-color: rgba(34,197,94,0.24);
            }}
            .runtime-agent-card.failed {{
                border-color: rgba(239,68,68,0.24);
            }}
            .runtime-agent-dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
            .runtime-agent-dot.running {{ background:#2563eb; box-shadow: 0 0 0 rgba(37,99,235,0.55); animation: runtimeAgentPulse 1.25s infinite; }}
            .runtime-agent-dot.completed {{ background:#16a34a; }}
            .runtime-agent-dot.failed {{ background:#dc2626; }}
            @keyframes runtimeAgentPulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(37,99,235,0.52); }}
                70% {{ box-shadow: 0 0 0 12px rgba(37,99,235,0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(37,99,235,0); }}
            }}
            @keyframes runtimeAgentLift {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-3px); }}
            }}
            </style>
            <div class="runtime-agent-card {card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; gap:8px;">
                    <div style="display:flex; align-items:center; gap:8px; font-size:14px; font-weight:800; color:#0f172a;">{pulse_dot}{title}</div>
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
    if "news_prompt_template_editor" not in st.session_state:
        st.session_state.news_prompt_template_editor = ""

    prompt_input = st.session_state.get(f"latest_{agent_key}_prompt_input", {}) or {}
    updated_at = st.session_state.get(f"last_{agent_key}_prompt_input_time")

    if agent_key != "log":
        st.subheader(title)
    source = prompt_input.get("source", "-")
    user_input = prompt_input.get("user_input", "-")
    context_text = prompt_input.get("context", "관련 데이터가 없습니다.")
    prompt_text = prompt_input.get("prompt", "-")
    is_faiss_log_ingest = bool(
        agent_key == "log"
        and (
            prompt_input.get("mode") == "faiss_ingest"
            or "faiss_logs_db.py" in str(source)
        )
    )

    if agent_key == "news":
        active_news_template = str(
            st.session_state.get("news_prompt_template_override")
            or DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE
        )
        if not st.session_state.get("news_prompt_template_editor"):
            st.session_state.news_prompt_template_editor = active_news_template

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

        st.markdown(
            f"""
            <div style="margin: 8px 0 14px 0; padding: 18px 20px; border-radius: 22px; background: linear-gradient(135deg, rgba(240,253,250,0.98), rgba(236,254,255,0.98)); border: 1px solid rgba(20,184,166,0.18); box-shadow: 0 12px 30px rgba(15,118,110,0.08);">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:10px;">
                    <div style="font-size:15px; font-weight:900; color:#0f172a;">뉴스 Agent 프롬프트 편집</div>
                    <span style="padding:6px 10px; border-radius:999px; font-size:11px; font-weight:800; background:rgba(255,255,255,0.78); color:#0f766e; border:1px solid rgba(15,23,42,0.08);">placeholder: {{news_text}}</span>
                </div>
                <div style="font-size:13px; line-height:1.65; color:#335c67;">입력 탭에서 수정한 템플릿이 실제 Ollama 뉴스 에이전트 호출과 백그라운드 뉴스 신호 생성에 함께 적용됩니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        news_prompt_editor = st.text_area(
            "뉴스 Agent Prompt Template",
            key="news_prompt_template_editor",
            height=360,
            help="{news_text} placeholder를 포함하면 최신 크롤링 뉴스 본문이 그 자리에 주입됩니다.",
        )
        template_info_col, template_action_col, template_reset_col = st.columns([1.2, 0.9, 0.9])
        with template_info_col:
            mode_label = "커스텀 적용 중" if st.session_state.get("news_prompt_template_override") else "기본 템플릿 사용 중"
            st.markdown(
                f"""
                <div style="padding:10px 12px; border-radius:16px; background:rgba(255,255,255,0.82); border:1px solid rgba(148,163,184,0.16); font-size:13px; font-weight:700; color:#0f766e;">{mode_label}</div>
                """,
                unsafe_allow_html=True,
            )
        with template_action_col:
            if st.button("프롬프트 적용", key="apply_news_prompt_template", use_container_width=True):
                if "{news_text}" not in news_prompt_editor:
                    st.error("뉴스 프롬프트에는 {news_text} placeholder가 포함돼야 합니다.")
                else:
                    try:
                        result = get_backend_client().set_news_prompt_template(news_prompt_editor)
                        st.session_state.news_prompt_template_override = result.get("news_prompt_template_override")
                        st.success("뉴스 에이전트 프롬프트를 적용했습니다.")
                    except Exception as error:
                        st.error(f"프롬프트 적용 실패: {error}")
        with template_reset_col:
            if st.button("기본값 복원", key="reset_news_prompt_template", use_container_width=True):
                try:
                    result = get_backend_client().set_news_prompt_template(DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE)
                    st.session_state.news_prompt_template_override = result.get("news_prompt_template_override")
                    st.session_state.news_prompt_template_editor = DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE
                    st.success("기본 뉴스 프롬프트로 복원했습니다.")
                except Exception as error:
                    st.error(f"기본값 복원 실패: {error}")

    log_summary_rendered = False
    if agent_key == "log":
        log_summary_rendered = render_log_product_summary_panel(prompt_text, updated_at, context_text)
        if log_summary_rendered:
            return

    if not prompt_input:
        if agent_key != "log":
            st.info("아직 표시할 프롬프트 입력값이 없습니다.")
        return

    if agent_key == "log":
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
                <div style="font-size:14px; font-weight:800; color:#0f172a;">{"현재 FAISS 적재 입력 상태" if is_faiss_log_ingest else "현재 프롬프트 입력 상태"}</div>
                <span style="padding:4px 10px; border-radius:999px; font-size:11px; font-weight:800; background:rgba(255,255,255,0.8); color:{accent_color}; border:1px solid rgba(15,23,42,0.08);">{html.escape(source)}</span>
            </div>
            <div style="font-size:12px; font-weight:700; color:{accent_color}; margin-bottom:6px;">{"정제 기준 / 적재 설명" if is_faiss_log_ingest else "사용자 지시 / 작업 문장"}</div>
            <div style="font-size:13px; line-height:1.65; color:#334155; white-space:pre-wrap;">{html.escape(user_input)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if agent_key == "news":
        st.markdown("#### FAISS 적재 미리보기")
        latest_news_briefing = str(st.session_state.get("latest_news_briefing") or "").strip()
        latest_signal_payload: dict[str, Any] = {}
        try:
            latest_signal_payload = json.loads(latest_news_briefing) if latest_news_briefing else {}
        except Exception:
            latest_signal_payload = {}

        preview_title = "periodic news signal"
        preview_search_text = str(
            latest_signal_payload.get("search_text") or latest_news_briefing or "-"
        ).strip()
        preview_features = {
            "tags": latest_signal_payload.get("tags") or [],
            "signal_summary": latest_signal_payload.get("signal_summary") or "",
            "risk_signal": latest_signal_payload.get("risk_signal") or [],
            "opportunity_signal": latest_signal_payload.get("opportunity_signal") or [],
            "linked_decision": latest_signal_payload.get("linked_decision") or [],
        }

        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns: 0.9fr 1.1fr; gap:14px; margin: 10px 0 14px 0;">
                <div style="padding:16px 18px; border-radius:20px; background:linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,118,110,0.92)); color:white; box-shadow:0 16px 34px rgba(15,23,42,0.16);">
                    <div style="font-size:12px; letter-spacing:0.08em; text-transform:uppercase; opacity:0.72; margin-bottom:8px;">Stored Document Shape</div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
                        <span style="padding:5px 10px; border-radius:999px; background:rgba(255,255,255,0.12); font-size:11px; font-weight:800;">store = news</span>
                        <span style="padding:5px 10px; border-radius:999px; background:rgba(255,255,255,0.12); font-size:11px; font-weight:800;">type = signal_news</span>
                        <span style="padding:5px 10px; border-radius:999px; background:rgba(255,255,255,0.12); font-size:11px; font-weight:800;">agent = news</span>
                    </div>
                    <div style="font-size:13px; line-height:1.7; opacity:0.95; white-space:pre-wrap;">제목: {html.escape(preview_title)}\n내용: {html.escape(preview_search_text[:520] + ('...' if len(preview_search_text) > 520 else ''))}</div>
                </div>
                <div style="padding:16px 18px; border-radius:20px; background:linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)); border:1px solid rgba(148,163,184,0.18); box-shadow:0 12px 28px rgba(15,23,42,0.06);">
                    <div style="font-size:13px; font-weight:900; color:#0f172a; margin-bottom:10px;">metadata.features 에 저장되는 값</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.json(preview_features)
        return

    prompt_col, context_col = st.columns([1.1, 1])
    with prompt_col:
        st.markdown("#### 실제 프롬프트 본문")
        st.code(prompt_text, language="text")
    with context_col:
        st.markdown("#### 투입 컨텍스트")
        st.code(context_text, language="text")


def render_ollama_live_panel() -> None:
    statuses = st.session_state.get("agent_statuses", {}) or {}
    activity_log = st.session_state.get("agent_activity_log", []) or []
    ollama_runtime = st.session_state.get("ollama_runtime", {}) or {}
    relevant_sources = {"log_agent", "news_agent", "orchestrator"}
    running_sources = [
        source
        for source in ("log_agent", "news_agent")
        if (statuses.get(source, {}) or {}).get("status") == "running"
    ]
    recent_ollama_events = [
        event for event in activity_log if str(event.get("source") or "") in relevant_sources
    ]
    latest_event = recent_ollama_events[0] if recent_ollama_events else {}

    st.subheader("Ollama 실시간 실행 상태")
    st.markdown(
        f"""
        <div style="margin: 8px 0 16px 0; padding: 20px 22px; border-radius: 24px; background: linear-gradient(135deg, rgba(236,254,255,0.98), rgba(239,246,255,0.98)); border: 1px solid rgba(56,189,248,0.18); box-shadow: 0 16px 34px rgba(14,116,144,0.08);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:8px;">
                <div>
                    <div style="font-size:12px; font-weight:800; letter-spacing:0.08em; color:#0369a1; text-transform:uppercase;">Live Runtime</div>
                    <div style="font-size:22px; font-weight:900; color:#0f172a; margin-top:4px;">지금 Ollama가 무엇을 처리 중인지 바로 보여줍니다</div>
                </div>
                <div style="padding:7px 12px; border-radius:999px; font-size:11px; font-weight:800; background:rgba(255,255,255,0.86); color:#0f766e; border:1px solid rgba(15,23,42,0.08);">model {html.escape(str(OLLAMA_LIGHTWEIGHT_MODEL))}</div>
            </div>
            <div style="font-size:13px; line-height:1.7; color:#334155;">로그 Agent와 뉴스 Agent의 현재 상태, 마지막 실행 이벤트, 최근 프롬프트 입력을 실시간 상태 동기화로 묶어서 보여주는 탭입니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("실행 중 Agent", len(running_sources))
    metric_cols[1].metric("최근 이벤트", str(latest_event.get("source") or "-"))
    metric_cols[2].metric("최근 갱신", format_status_time(latest_event.get("timestamp")))
    metric_cols[3].metric("이벤트 수", len(recent_ollama_events[:12]))

    runtime_agent = str(ollama_runtime.get("agent") or "-")
    runtime_status = str(ollama_runtime.get("status") or "idle")
    runtime_model = str(ollama_runtime.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)
    runtime_text = str(ollama_runtime.get("response_text") or "").strip()
    runtime_error = str(ollama_runtime.get("error") or "").strip()
    runtime_updated_at = format_status_time(ollama_runtime.get("updated_at"))
    runtime_started_at = format_status_time(ollama_runtime.get("started_at"))
    runtime_label, runtime_color, _ = get_agent_status_palette(runtime_status)

    st.markdown(
        f"""
        <div style="padding:18px 20px; border-radius:24px; background:linear-gradient(135deg, rgba(15,23,42,0.98), rgba(14,116,144,0.92)); color:white; box-shadow:0 18px 36px rgba(15,23,42,0.16); margin: 8px 0 18px 0;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:10px;">
                <div>
                    <div style="font-size:12px; letter-spacing:0.08em; text-transform:uppercase; opacity:0.74; margin-bottom:8px;">Live Generation</div>
                    <div style="font-size:20px; font-weight:900;">현재 생성 중인 Ollama 출력</div>
                </div>
                <span style="padding:6px 10px; border-radius:999px; font-size:11px; font-weight:800; background:rgba(255,255,255,0.14); color:{runtime_color}; border:1px solid rgba(255,255,255,0.12);">{html.escape(runtime_label)}</span>
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
                <span style="padding:5px 10px; border-radius:999px; background:rgba(255,255,255,0.12); font-size:11px; font-weight:800;">agent = {html.escape(runtime_agent)}</span>
                <span style="padding:5px 10px; border-radius:999px; background:rgba(255,255,255,0.12); font-size:11px; font-weight:800;">model = {html.escape(runtime_model)}</span>
                <span style="padding:5px 10px; border-radius:999px; background:rgba(255,255,255,0.12); font-size:11px; font-weight:800;">started = {html.escape(runtime_started_at)}</span>
                <span style="padding:5px 10px; border-radius:999px; background:rgba(255,255,255,0.12); font-size:11px; font-weight:800;">updated = {html.escape(runtime_updated_at)}</span>
            </div>
            <div style="font-size:13px; line-height:1.7; white-space:pre-wrap; min-height:120px;">{html.escape(runtime_text or '아직 생성 중 텍스트가 없습니다.')}</div>
            {f'<div style="margin-top:12px; font-size:12px; color:#fecaca; font-weight:700;">오류: {html.escape(runtime_error)}</div>' if runtime_error else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    agent_specs = [
        ("log_agent", "로그 Agent", "#92400e", "rgba(255,251,235,0.98)", st.session_state.get("latest_log_prompt_input", {}) or {}),
        ("news_agent", "뉴스 Agent", "#0f766e", "rgba(236,254,255,0.98)", st.session_state.get("latest_news_prompt_input", {}) or {}),
    ]
    card_cols = st.columns(2)
    for column, (agent_key, label, accent, background, prompt_input) in zip(card_cols, agent_specs):
        info = statuses.get(agent_key, {}) or {}
        status_code = str(info.get("status") or "pending")
        status_label, _, _ = get_agent_status_palette(status_code)
        detail = str(info.get("detail") or "최근 실행 정보가 없습니다.")
        user_input = str(prompt_input.get("user_input") or "-").strip()
        context_preview = " ".join(str(prompt_input.get("context") or "-").split())[:220]
        updated_at = format_status_time(info.get("updated_at"))
        with column:
            st.markdown(
                f"""
                <div style="padding:18px 20px; border-radius:22px; background:{background}; border:1px solid rgba(148,163,184,0.16); box-shadow:0 14px 30px rgba(15,23,42,0.06); min-height:310px;">
                    <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:12px;">
                        <div>
                            <div style="font-size:18px; font-weight:900; color:#0f172a;">{label}</div>
                            <div style="font-size:12px; color:#64748b; margin-top:4px;">최근 업데이트 {html.escape(updated_at)}</div>
                        </div>
                        <span style="padding:6px 10px; border-radius:999px; font-size:11px; font-weight:800; background:rgba(255,255,255,0.88); color:{accent}; border:1px solid rgba(15,23,42,0.08);">{html.escape(status_label)}</span>
                    </div>
                    <div style="font-size:13px; font-weight:800; color:{accent}; margin-bottom:6px;">현재 작업</div>
                    <div style="font-size:13px; line-height:1.65; color:#334155; white-space:pre-wrap; margin-bottom:14px;">{html.escape(detail)}</div>
                    <div style="font-size:13px; font-weight:800; color:{accent}; margin-bottom:6px;">마지막 요청 문장</div>
                    <div style="font-size:13px; line-height:1.65; color:#334155; white-space:pre-wrap; margin-bottom:14px;">{html.escape(user_input[:220] or '-')}</div>
                    <div style="font-size:13px; font-weight:800; color:{accent}; margin-bottom:6px;">최근 컨텍스트 미리보기</div>
                    <div style="font-size:13px; line-height:1.65; color:#334155;">{html.escape(context_preview or '-')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### 최근 Ollama 실행 타임라인")
    if not recent_ollama_events:
        st.info("아직 기록된 Ollama 실행 이벤트가 없습니다.")
    else:
        for event in recent_ollama_events[:12]:
            source = str(event.get("source") or "-")
            status_code = str(event.get("status") or "pending")
            status_label, status_color, _ = get_agent_status_palette(status_code)
            st.markdown(
                f"""
                <div style="border-left:4px solid {status_color}; padding:12px 14px; margin-bottom:10px; background:rgba(248,250,252,0.95); border-radius:0 14px 14px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:6px;">
                        <div style="font-size:13px; font-weight:800; color:#0f172a;">{html.escape(source)} · {html.escape(status_label)}</div>
                        <div style="font-size:11px; color:#64748b; font-weight:700;">{html.escape(format_status_time(event.get('timestamp')))}</div>
                    </div>
                    <div style="font-size:13px; color:#334155; line-height:1.6;">{html.escape(str(event.get('detail') or ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    prompt_specs = [
        ("log_agent", "로그 Agent 전체 프롬프트", st.session_state.get("latest_log_prompt_input", {}) or {}),
        ("news_agent", "뉴스 Agent 전체 프롬프트", st.session_state.get("latest_news_prompt_input", {}) or {}),
    ]
    st.markdown("#### 전체 프롬프트 보기")
    for agent_key, label, prompt_input in prompt_specs:
        full_prompt = str(prompt_input.get("prompt") or "").strip()
        with st.expander(label, expanded=runtime_agent == agent_key and bool(full_prompt)):
            if not full_prompt:
                st.info("아직 저장된 프롬프트가 없습니다.")
            else:
                st.code(full_prompt, language="text")


@fragment_decorator(run_every="3s")
def render_live_ollama_fragment():
    consume_ws_snapshot_buffer()
    render_ollama_live_panel()


@fragment_decorator(run_every="1s")
def render_global_ollama_toast_fragment():
    render_ollama_toast()


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


@fragment_decorator(run_every="1s")
def render_live_operations_fragment():
    consume_ws_snapshot_buffer()
    render_runtime_dashboard()


@fragment_decorator(run_every="1s")
def render_live_operations_showcase_fragment():
    consume_ws_snapshot_buffer()
    render_operations_showcase()


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

    latest_news_time = parse_status_time(st.session_state.get("last_news_time"))
    latest_new_item_time = parse_status_time(st.session_state.get("last_new_item_time"))
    has_fresh_news_cycle = (
        latest_news_time is not None
        and latest_new_item_time is not None
        and latest_news_time == latest_new_item_time
    )

    if news_items:
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
    else:
        st.info("표시할 뉴스가 없습니다.")

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

    if news_items:
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

    with st.container():
        st.markdown(
            '<div class="regulation-intake-anchor"></div>',
            unsafe_allow_html=True,
        )
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
                        before = int(st.session_state.get("vector_count", 0) or 0)
                        new_count = ingest_files(files_data, doc_type="regulation")
                        added = new_count - before

                        query = "규제"
                        logs_found, news_found, rules_found = search_context(query, k=6)
                        rule_context = "\n\n".join(rules_found)

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
    store_options = get_faiss_store_options()
    selected_store_label = st.selectbox(
        "검색/조회할 DB",
        options=[label for label, _ in store_options],
        key="faiss_tab_store_filter",
    )
    selected_store = dict(store_options).get(selected_store_label)
    _, selected_total_count = get_live_faiss_items(limit=1, store_name=selected_store)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric(
            "벡터 수",
            selected_total_count if selected_store is not None else vector_count,
        )
        if last_ingest:
            st.caption(f"조회 DB: {selected_store_label} · 마지막 적재: {last_ingest}")
    with c2:
        if st.button("새로고침 FAISS 상태"):
            try:
                _ = get_backend_client().get_status()
                st.experimental_rerun()
            except Exception:
                st.error("새로고침 실패")

    st.markdown("---")

    st.subheader("뉴스 Signal 실제 적재 상세")
    try:
        news_store_resp = get_backend_client().get_faiss_entries(limit=1000, store_name=FAISS_STORE_NEWS)
        news_items = news_store_resp.get("items", []) or []
        signal_items = [item for item in news_items if str(item.get("type") or "").strip().lower() == "signal_news"]
        if not signal_items:
            st.info("현재 뉴스 신호 DB에는 signal_news 문서가 없습니다.")
        else:
            st.caption(f"현재 signal_news 전체 {len(signal_items)}건을 실시간으로 표시합니다.")
            for signal_item in signal_items:
                signal_detail = get_backend_client().get_faiss_entry(str(signal_item.get("id"))).get("item") or {}
                signal_meta = signal_detail.get("metadata", {}) or {}
                signal_features = signal_meta.get("features", {}) or {}
                signal_raw_content = signal_meta.get("raw_content", "")
                signal_page_content = signal_detail.get("page_content", "")
                signal_title = str(signal_item.get("name") or signal_item.get("title") or signal_item.get("id") or "signal_news")

                with st.expander(f"{signal_title[:72]} · {str(signal_item.get('id') or '')[:8]}", expanded=False):
                    detail_col_a, detail_col_b = st.columns([1.1, 1])
                    with detail_col_a:
                        st.markdown(
                            """
                            <div style="padding:16px 18px; border-radius:20px; background:linear-gradient(135deg, rgba(15,23,42,0.98), rgba(17,94,89,0.96)); color:white; box-shadow:0 16px 34px rgba(15,23,42,0.16); margin-bottom:12px;">
                                <div style="font-size:12px; letter-spacing:0.08em; text-transform:uppercase; opacity:0.72; margin-bottom:8px;">page_content</div>
                                <div style="font-size:13px; line-height:1.7; white-space:pre-wrap;">FAISS 임베딩에 직접 들어가는 문자열입니다.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.code(signal_page_content[:3000], language="text")
                    with detail_col_b:
                        st.markdown(
                            """
                            <div style="padding:16px 18px; border-radius:20px; background:linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)); border:1px solid rgba(148,163,184,0.18); box-shadow:0 12px 28px rgba(15,23,42,0.06); margin-bottom:12px;">
                                <div style="font-size:13px; font-weight:900; color:#0f172a; margin-bottom:8px;">metadata.features</div>
                                <div style="font-size:13px; line-height:1.65; color:#475569;">검색 필터링과 후속 설명에 쓰이는 구조화 필드입니다.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.json(signal_features)

                    with st.expander("raw_content 보기", expanded=False):
                        st.code(str(signal_raw_content)[:4000], language="json")
    except Exception as error:
        st.warning(f"signal_news 상세 조회 실패: {error}")

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
                                resp = get_backend_client().get_faiss_entries(limit=after, store_name=selected_store)
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
                resp = get_backend_client().search_faiss(q, int(k), store_name=selected_store)
                logs = resp.get("logs") or []
                news = resp.get("news") or []
                rules = resp.get("rules") or []
                customer = resp.get("customer") or []

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

                if customer:
                    st.markdown("**Customer / 영업 패턴**")
                    for item in customer:
                        st.write(item)

                if not (logs or news or rules or customer):
                    st.info("검색 결과가 없습니다.")
            except Exception as e:
                st.error(f"검색 실패: {e}")

    st.markdown("---")
    st.subheader("FAISS 저장된 벡터 항목 보기 / 내보내기")
    if st.button("목록 불러오기 (최대 200)"):
        try:
            resp = get_backend_client().get_faiss_entries(limit=200, store_name=selected_store)
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
        "📄 대출상품 Dashboard",
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
    st.session_state.initial_analysis_autorun_disabled = False

if "initial_loading_reason" not in st.session_state:
    st.session_state.initial_loading_reason = None

if HAS_FRAGMENT_REFRESH:
    monitor_backend_bootstrap_fragment()

if not st.session_state.initial_analysis_done:
    try:
        get_backend_client().start_worker(interval_seconds=1)
    except Exception:
        pass

    status_payload = {}
    try:
        status_payload = get_backend_client().get_status()
        sync_session_from_backend(status_payload)
        _sync_backend_bootstrap_state(status_payload)
    except Exception:
        _sync_backend_bootstrap_state(None)

    if not st.session_state.initial_analysis_done:
        with _background_lock:
            initial_task = dict(_background_results.get("initial_analysis") or {})
        if not st.session_state.initial_analysis_started and initial_task.get("status") != "running":
            launched = _start_initial_analysis_background(log_dir="data/logs")
            st.session_state.initial_analysis_started = launched or st.session_state.initial_analysis_started
        if HAS_FRAGMENT_REFRESH:
            render_live_initial_loading_fragment()
        else:
            render_initial_loading_screen()

    with _background_lock:
        initial_task = dict(_background_results.get("initial_analysis") or {})
    if initial_task:
        if initial_task.get("status") == "completed":
            sync_session_from_backend(initial_task.get("result") or {})
            st.session_state.initial_analysis_done = _is_initial_dashboard_ready(
                initial_task.get("result") or {}
            )
            st.session_state.initial_analysis_failed = False
            with _background_lock:
                _background_results.pop("initial_analysis", None)
        elif initial_task.get("status") == "failed":
            st.session_state.initial_analysis_failed = True
            st.warning("초기 분석이 지연되고 있습니다. 백엔드 상태를 확인하세요.")

    if not st.session_state.initial_analysis_done:
        st.stop()


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
    if HAS_FRAGMENT_REFRESH:
        render_global_ollama_toast_fragment()
    else:
        render_ollama_toast()

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

    main_sections = [
        "🤖 운영 현황",
        "💬 AI 카드론 토론실",
        "📄 대출상품 Dashboard",
        "🧠 Vector DB",
    ]
    if "main_dashboard_section" not in st.session_state:
        st.session_state.main_dashboard_section = main_sections[1]
    render_main_section_status_styles()

    if hasattr(st, "segmented_control"):
        selected_section = st.segmented_control(
            "메인 섹션",
            options=main_sections,
            selection_mode="single",
            key="main_dashboard_section",
            label_visibility="collapsed",
        )
    else:
        selected_section = st.radio(
            "메인 섹션",
            options=main_sections,
            horizontal=True,
            key="main_dashboard_section",
            label_visibility="collapsed",
        )

    if selected_section == "🤖 운영 현황":
        if HAS_FRAGMENT_REFRESH:
            render_live_operations_showcase_fragment()
        else:
            render_operations_showcase()

        with st.expander("상세 운영 패널", expanded=False):
            if HAS_FRAGMENT_REFRESH:
                render_live_operations_fragment()
            else:
                render_runtime_dashboard()

    elif selected_section == "💬 AI 카드론 토론실":
        render_role_based_strategy_tab()

    elif selected_section == "📄 대출상품 Dashboard":
        if HAS_FRAGMENT_REFRESH:
            render_live_log_prompt_fragment()
        else:
            render_agent_prompt_panel(
                "log",
                "📄 대출상품 Dashboard",
                "#92400e",
                "linear-gradient(135deg, rgba(255,251,235,0.98), rgba(254,243,199,0.98))",
            )

    elif selected_section == "🧠 Vector DB":
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
