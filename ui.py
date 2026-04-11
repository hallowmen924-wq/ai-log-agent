import time
import streamlit as st
import pandas as pd
import plotly.express as px
try:
    from streamlit_echarts import st_echarts
    _HAS_ECHARTS = True
except Exception:
    _HAS_ECHARTS = False

from analyzer.risk_analyzer import calculate_risk
from agent.news_agent import collect_news, analyze_news

# vector db helpers (optional)
try:
    from rag.vector_db import get_vector_count, get_top_keywords
except Exception:
    def get_vector_count():
        return 0

    def get_top_keywords(n=10):
        return []

# try to import streamlit-lottie (preferred) else fall back to html embed
_HAS_STREAMLIT_LOTTIE = False
try:
    from streamlit_lottie import st_lottie
    _HAS_STREAMLIT_LOTTIE = True
except Exception:
    _HAS_STREAMLIT_LOTTIE = False

# Palette & animation configuration (quick presets — 편하게 조정하세요)
PALETTE = {
    'bg_start': '#f0f9ff',
    'bg_end': '#e6f7ff',
    'card_bg': 'rgba(249,250,255,0.95)',
    'pulse': '#06b6d4',
    'activity_start': '#06b6f1',
    'activity_end': '#06b6d4',
    'chip_start':'#0f172a',
    'chip_end':'#334155'
}

# animation speeds (seconds)
ANIM = {
    'marquee_sec': 10,
    'pulse_sec': 1.2,
    'bar_sec': 0.9
}

# common lottie urls
HEADER_LOTTIE = 'https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json'
VECTOR_LOTTIE = 'https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json'


def render_lottie_url(url: str, height: int = 120, loop: bool = True, autoplay: bool = True, max_width: int | None = None):
    """Try to render Lottie using `streamlit-lottie` when available, otherwise embed via lottie-player HTML."""
    if _HAS_STREAMLIT_LOTTIE:
        try:
            st_lottie({'v': url}, height=height)  # st_lottie accepts dict or json; some versions accept url wrapped
            return
        except Exception:
            pass
    # fallback: embed lottie-player html
    try:
        loop_attr = 'loop' if loop else ''
        autoplay_attr = 'autoplay' if autoplay else ''
        max_w = f"max-width:{max_width}px;" if max_width else "width:100%;"
        player_html = f'''<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
<div style="display:flex;justify-content:center;align-items:center;{max_w}">
  <lottie-player src="{url}" background="transparent" speed="1" {loop_attr} {autoplay_attr} style="width:100%; height:{height}px;"></lottie-player>
</div>'''
        st.components.v1.html(player_html, height=height + 28)
    except Exception:
        st.write("")


def render_header():
    header_col, header_anim = st.columns([3, 1])
    with header_col:
        # 제목 + CSS 펄스 배지
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px'><h1 style=\'font-size:28px;margin:6px 0 12px 0;color:#0f172a;\'>🔥 AI 대출 심사 대시보드</h1>"
            "<div class=\'pulse-badge\'><div class=\'pulse-dot\'></div><div style=\'font-size:12px;color:#334155;\'>실시간 모니터링</div></div></div>",
            unsafe_allow_html=True,
        )
    with header_anim:
        # 작은 애니메이션: Lottie 시도, 실패 시 CSS 활동 표시
        try:
            render_lottie_url(HEADER_LOTTIE, height=110, loop=True, autoplay=True, max_width=140)
        except Exception:
            st.markdown("<div style='display:flex;justify-content:center;align-items:center; width:140px;'><div class='activity-bars'><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div></div></div>", unsafe_allow_html=True)


def render_background():
    st.markdown(
        """
                <style>
                html, body { height: 100%; }
                .streamlit-app-background {
                    position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;
                    background: linear-gradient(180deg, %s, %s);
                    background-size: 400%% 400%%; animation: gradientBG 28s ease infinite;
                    opacity: 0.14; filter: blur(6px);
                }
                @keyframes gradientBG { 0%% { background-position: 0%% 0%%; } 50%% { background-position: 100%% 50%%; } 100%% { background-position: 0%% 100%%; } }
                /* 카드 배경을 약간 톤 다운된 색으로 변경해 흰 배경처럼 보이지 않게 함 */
                .stApp > .main > div { background: %s; box-shadow: 0 6px 20px rgba(15,23,42,0.06); border-radius:10px; }
                /* 화면에 스크롤 없이 맞추기 위한 추가 CSS */
                .reportview-container .main .block-container{ padding-top:8px; padding-left:16px; padding-right:16px; padding-bottom:8px; }
                .stApp > .main { height: calc(100vh - 72px); overflow: hidden; }
                .stPlotlyChart > div { max-height: 100%; }
        /* 라이브 키워드 토스트/마키 스타일 */
        .keyword-toast { position: absolute; right: 24px; top: 84px; z-index: 9999; }
        .keyword-chip { display:inline-block; background: linear-gradient(90deg,%s,%s); color: #fff; padding:6px 12px; border-radius:14px; margin:6px; opacity:0.98; font-size:13px; box-shadow:0 6px 14px rgba(16,24,40,0.06); }
        .keyword-marquee { overflow: hidden; white-space: nowrap; width: 520px; }
        .keyword-marquee > div { display:inline-block; animation: marquee 12s linear infinite; }
        @keyframes marquee { 0% { transform: translateX(100%);} 100% { transform: translateX(-100%);} }
        /* 펄스 배지 (헤더 옆) */
        .pulse-badge { display:inline-flex; align-items:center; gap:8px; }
        .pulse-dot { width:12px; height:12px; border-radius:50%; background:%s; box-shadow:0 0 0 rgba(6,182,212,0.6); animation: pulse %ss infinite; }
        @keyframes pulse { 0%% { box-shadow: 0 0 0 0 rgba(6,182,212,0.6);} 70%% { box-shadow: 0 0 0 10px rgba(6,182,212,0);} 100%% { box-shadow: 0 0 0 0 rgba(6,182,212,0);} }
        /* 키워드 칩 애니메이션 */
        .keyword-chip { transition: transform 0.3s ease, opacity 0.4s ease; }
        .keyword-chip:hover { transform: translateY(-4px); opacity:1; }
        /* 벡터 활동 바 */
        .activity-bars { display:flex; gap:6px; align-items:end; height:48px; }
        .activity-bars .bar { width:8px; background:linear-gradient(180deg,%s,%s); animation: baranim %ss linear infinite; border-radius:4px 4px 2px 2px; }
        .activity-bars .bar:nth-child(1){ animation-delay:0s } .activity-bars .bar:nth-child(2){ animation-delay:0.12s }
        .activity-bars .bar:nth-child(3){ animation-delay:0.24s } .activity-bars .bar:nth-child(4){ animation-delay:0.36s }
        .activity-bars .bar:nth-child(5){ animation-delay:0.48s }
        @keyframes baranim { 0%%{height:6px}50%%{height:42px}100%%{height:6px} }
        </style>
        <div class="streamlit-app-background"></div>
        """ % (
            PALETTE['bg_start'], PALETTE['bg_end'], PALETTE['card_bg'], PALETTE['chip_start'], PALETTE['chip_end'], PALETTE['pulse'], ANIM['pulse_sec'], PALETTE['activity_start'], PALETTE['activity_end'], ANIM['bar_sec']
        ),
        unsafe_allow_html=True,
    )


def render_live_keywords(keywords:list[str] | list[tuple], height:int=36):
    """화면 우측 상단에 토스트 형태로 라이브 키워드 마키를 표시합니다.
       keywords: list of strings or list of (keyword,count) tuples
    """
    if not keywords:
        return
    # normalize to list of strings
    items = []
    for k in keywords:
        if isinstance(k, (list, tuple)) and len(k) >= 1:
            items.append(str(k[0]))
        elif isinstance(k, dict) and 'keyword' in k:
            items.append(str(k.get('keyword')))
        else:
            items.append(str(k))

    chips = "".join([f"<span class='keyword-chip'>{c}</span>" for c in items])
    html = f"""
    <div class='keyword-toast'>
      <div class='keyword-marquee'><div>{chips}</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_news_snippets(k=10):
    try:
        from rag.vector_db import search_context
        _, news_snips, _ = search_context("뉴스", k=k)
        return news_snips or []
    except Exception:
        try:
            news = collect_news()
            return [f"{item.get('title','')}: {item.get('content','')[:200]}" for item in (news or [])[:k]]
        except Exception:
            return []


def extract_keywords(snippets, top_n=10):
    import re
    text = "\n".join(snippets)
    kor = re.findall(r'[가-힣]{2,}', text)
    eng = re.findall(r'[A-Za-z]{2,}', text)
    words = [w for w in kor + eng if len(w) > 1]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return items


def render_dashboard():
    # results에서 제품별 요약 및 차트 표시
    if "results" not in st.session_state:
        return
    results = st.session_state.results

    st.subheader("📊 제품별 실시간 리스크 대시보드 (C6, C9, C11, C12)")
    products = ["C6", "C9", "C11", "C12"]
    per_prod = {p: [] for p in products}
    raw_by_product = {p: [] for p in products}
    for r in results:
        p = r.get("product") or ""
        if p in products:
            risk = calculate_risk(r.get("in_fields", {}), r.get("out_fields", {}), r.get("in_mapping", {}), r.get("out_mapping", {}), product=p)
            raw_by_product[p].append({"score": risk["score"], "grade": risk["grade"]})

    # layout
    top_left, top_right = st.columns(2)
    bot_left, bot_right = st.columns(2)

    # Top-left: 뉴스 키워드
    with top_left:
        st.markdown("### 실시간 뉴스 키워드")
        snippets = get_news_snippets(20)
        keywords = extract_keywords(snippets, top_n=12)
        # 우측 상단에 라이브 키워드 토스트 렌더
        try:
            render_live_keywords(keywords)
        except Exception:
            pass
        if not snippets:
            st.info("사용 가능한 뉴스가 없습니다. FAISS가 생성되었는지 확인하세요.")
        else:
            if keywords:
                kw_df = pd.DataFrame(keywords, columns=["keyword", "count"]) 
                if _HAS_ECHARTS:
                    kws = kw_df['keyword'].astype(str).tolist()
                    counts = kw_df['count'].astype(int).tolist()
                    option = {
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {"type": "category", "data": kws},
                        "yAxis": {"type": "value"},
                        "series": [{"data": counts, "type": "bar", "itemStyle": {"color": "#4f46e5"}}]
                    }
                    st_echarts(options=option, height="260px")
                else:
                    fig_kw = px.bar(kw_df, x="keyword", y="count", title="Top 뉴스 키워드")
                    fig_kw.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_kw, use_container_width=True)
            with st.expander("원문 뉴스/스니펫 보기", expanded=False):
                for s in snippets[:10]:
                    st.write(s)

    # Top-right: 제품별 등급 분포
    with top_right:
        st.markdown("### 제품별 등급 분포")
        rows = []
        for p, items in raw_by_product.items():
            for it in items:
                rows.append({"product": p, "grade": it.get("grade", "N/A")})
        if rows:
            df_grades = pd.DataFrame(rows)
            pivot = df_grades.pivot_table(index="product", columns="grade", aggfunc=len, fill_value=0)
            pivot = pivot.reindex(index=products, fill_value=0)
            if _HAS_ECHARTS:
                grades = pivot.columns.tolist()
                series = []
                colors = {"HIGH": "#ff4d4f", "MEDIUM": "#ffa940", "LOW": "#52c41a", "N/A": "#999"}
                for g in grades:
                    series.append({"name": g, "type": "bar", "stack": "grades", "data": pivot[g].tolist(), "itemStyle": {"color": colors.get(g, '#999')}})
                option = {
                    "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                    "legend": {"data": grades},
                    "xAxis": {"type": "category", "data": pivot.index.tolist()},
                    "yAxis": {"type": "value"},
                    "series": series
                }
                st_echarts(options=option, height="260px")
            else:
                fig_grade = px.bar(pivot, x=pivot.index, y=pivot.columns.tolist(), title="제품별 등급 분포", barmode="stack")
                fig_grade.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_grade, use_container_width=True)
        else:
            st.info("제품별 등급 분포를 계산할 데이터가 없습니다.")

    # Bottom-left: 제품 요약
    with bot_left:
        st.markdown("### 제품 요약")
        cols = st.columns(len(products))
        color_map = {"HIGH": "#ff4d4f", "MEDIUM": "#ffa940", "LOW": "#52c41a", "N/A": "#999"}
        anim_urls = {
            "HIGH": "https://assets6.lottiefiles.com/packages/lf20_tutvdkg0.json",
            "MEDIUM": "https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json",
            "LOW": "https://assets2.lottiefiles.com/packages/lf20_o1q1k5o9.json",
            "N/A": "https://assets2.lottiefiles.com/packages/lf20_sF8Qq3.json",
        }
        latest_rows = []
        for i, p in enumerate(products):
            last = per_prod[p][-1] if per_prod[p] else {"score": 0, "grade": "N/A"}
            with cols[i]:
                st.markdown(f"**{p}**")
                st.markdown(f"<div style='font-size:22px;font-weight:700;color:{color_map.get(last['grade'],'#333')}'>{int(last['score'])}</div>", unsafe_allow_html=True)
                st.markdown(f"등급: **{last['grade']}**")
                try:
                    render_lottie_url(anim_urls.get(last['grade'], anim_urls['N/A']), height=72, loop=False, autoplay=False, max_width=120)
                except Exception:
                    pass
            if per_prod[p]:
                latest_rows.append({"product": p, "score": per_prod[p][-1]["score"]})
        if latest_rows:
            df_latest = pd.DataFrame(latest_rows)
            if _HAS_ECHARTS:
                prod = df_latest['product'].tolist()
                scores = df_latest['score'].tolist()
                option = {
                    "xAxis": {"type": "category", "data": prod},
                    "yAxis": {"type": "value"},
                    "series": [{"data": scores, "type": "bar", "itemStyle": {"color": "#06b6d4"}}],
                    "tooltip": {"trigger": "axis"}
                }
                st_echarts(options=option, height="200px")
            else:
                fig_bar = px.bar(df_latest, x="product", y="score", title="제품별 최신 점수")
                fig_bar.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

    # Bottom-right: 벡터 DB 시각화
    with bot_right:
        st.markdown("### 벡터 DB: 주요 키워드 & 저장 상태")
        # 작은 Lottie 애니메이션으로 '활성' 느낌 추가
        try:
                render_lottie_url(VECTOR_LOTTIE, height=80, loop=True, autoplay=True, max_width=120)
        except Exception:
            pass
        top_keywords = []
        try:
            vb = __import__("rag.vector_db", fromlist=["get_top_keywords"]) 
            if hasattr(vb, "get_top_keywords"):
                top_keywords = vb.get_top_keywords(10)
        except Exception:
            top_keywords = []
        if not top_keywords:
            snippets_for_vec = get_news_snippets(50)
            kw_items = extract_keywords(snippets_for_vec, top_n=10)
            top_keywords = [{"keyword": k, "count": v} for k, v in kw_items]
        if top_keywords:
            kw_df = pd.DataFrame(top_keywords)
            if kw_df.shape[1] == 2 and 'keyword' in kw_df.columns:
                kw_df = kw_df
            else:
                kw_df = pd.DataFrame(top_keywords, columns=['keyword', 'count'])
            if _HAS_ECHARTS:
                kws = kw_df['keyword'].astype(str).tolist()
                counts = kw_df['count'].astype(int).tolist()
                option = {
                    "xAxis": {"type": "category", "data": kws},
                    "yAxis": {"type": "value"},
                    "series": [{"data": counts, "type": "bar", "itemStyle": {"color": "#06b6d4"}}],
                    "tooltip": {"trigger": "axis"}
                }
                st_echarts(options=option, height="220px")
            else:
                fig_vk = px.bar(kw_df, x='keyword', y='count', title='벡터 DB에 저장된 상위 키워드')
                fig_vk.update_layout(height=220, margin=dict(l=8, r=8, t=32, b=8))
                st.plotly_chart(fig_vk, use_container_width=True)
        else:
            st.info("벡터 키워드 정보를 불러올 수 없습니다.")
        vc = get_vector_count()
        vh = st.session_state.get("vector_history", [])
        vh.append(int(vc))
        if len(vh) > 60:
            vh = vh[-60:]
        st.session_state.vector_history = vh
        st.caption(f"벡터 카운트: {int(vc)}")
        df_v = pd.DataFrame({"vector_count": st.session_state.vector_history})
        df_v.index = pd.RangeIndex(start=0, stop=len(df_v))
        if _HAS_ECHARTS:
            option = {
                "xAxis": {"type": "category", "data": df_v.index.tolist()},
                "yAxis": {"type": "value"},
                "series": [{"data": df_v['vector_count'].tolist(), "type": "line", "smooth": True, "areaStyle": {}}],
                "tooltip": {"trigger": "axis"}
            }
            st_echarts(options=option, height="100px")
        else:
            v_fig = px.line(df_v, y="vector_count")
            v_fig.update_layout(height=90, margin=dict(l=6, r=6, t=6, b=6))
            st.plotly_chart(v_fig, use_container_width=True)
