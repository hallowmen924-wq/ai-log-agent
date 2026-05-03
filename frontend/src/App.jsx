import React, { useEffect, useState } from 'react';
import Plotly from 'plotly.js-basic-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
import {
  createFaissWebSocket,
  fetchCharts,
  fetchHealth,
  fetchProductSummary,
  fetchFaissEntriesByStore,
  fetchFaissStats,
  startCardloanDebate,
  uploadRegulationFiles,
} from './api';
import { DEFAULT_QUESTION, MAIN_SECTIONS, REVIEWER_PERSONAS, STORE_OPTIONS } from './constants';

const Plot = createPlotlyComponent(Plotly);

function initialStatus() {
  return {
    results: [],
    news: [],
    issues: [],
    vector_count: 0,
    latest_log_briefing: '',
    latest_news_briefing: '',
    latest_regulation_analysis: '',
    agent_statuses: {},
    agent_activity_log: [],
    vector_events: [],
    cardloan_debate: { status: 'idle', round_results: [] },
    ollama_runtime: {},
    backend_diagnostics: {},
    news_crawl_running: false,
    news_crawl_target_count: 0,
    news_crawl_success_count: 0,
    news_crawl_failure_count: 0,
    last_news_crawl_time: null,
    last_news_crawl_error: null,
    last_news_time: null,
    last_log_ingest_time: null,
  };
}

function formatTime(value) {
  if (!value) {
    return '-';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function relativeMinutes(value) {
  if (!value) {
    return '-';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '-';
  }
  const diffMinutes = Math.max(0, Math.floor((Date.now() - date.getTime()) / 60000));
  if (diffMinutes < 1) {
    return '방금 전';
  }
  if (diffMinutes < 60) {
    return `${diffMinutes}분 전`;
  }
  const hours = Math.floor(diffMinutes / 60);
  if (hours < 24) {
    return `${hours}시간 전`;
  }
  return `${Math.floor(hours / 24)}일 전`;
}

function truncate(value, size = 160) {
  const text = String(value || '').trim();
  if (text.length <= size) {
    return text || '-';
  }
  return `${text.slice(0, size)}...`;
}

function getStatusPalette(status) {
  switch (String(status || 'pending')) {
    case 'running':
      return { label: '실행 중', color: '#61f4de', tone: 'running' };
    case 'completed':
      return { label: '완료', color: '#6ee7b7', tone: 'completed' };
    case 'failed':
      return { label: '실패', color: '#ff8f8f', tone: 'failed' };
    default:
      return { label: '대기', color: '#8fb9d6', tone: 'pending' };
  }
}

function buildOverviewMetrics(status) {
  const agentStatuses = status.agent_statuses || {};
  const values = Object.values(agentStatuses);
  return {
    resultsCount: (status.results || []).length,
    newsCount: (status.news || []).length,
    issuesCount: (status.issues || []).length,
    vectorCount: Number(status.vector_count || 0),
    runningAgents: values.filter((item) => item?.status === 'running').length,
    failedAgents: values.filter((item) => item?.status === 'failed').length,
    completedAgents: values.filter((item) => item?.status === 'completed').length,
    activityEvents: (status.agent_activity_log || []).length,
    vectorEvents: (status.vector_events || []).length,
  };
}

function buildAgentFlowFigure(status) {
  const statuses = status.agent_statuses || {};
  const latestVector = (status.vector_events || [])[0] || {};
  const colorMap = {
    running: '#61f4de',
    completed: '#6ee7b7',
    failed: '#ff8f8f',
    pending: '#8fb9d6',
  };
  const nodes = [
    {
      id: 'source_logs',
      label: 'Logs',
      x: 0.02,
      y: 0.66,
      status: 'completed',
      detail: `유입 로그 ${(status.results || []).length}건`,
      symbol: 'diamond',
      size: 34,
    },
    {
      id: 'source_news',
      label: 'News',
      x: 0.02,
      y: 0.24,
      status: 'completed',
      detail: `수집 뉴스 ${(status.news || []).length}건`,
      symbol: 'diamond',
      size: 34,
    },
    {
      id: 'log_agent',
      label: 'Log Agent',
      x: 0.27,
      y: 0.76,
      status: statuses.log_agent?.status || 'pending',
      detail: truncate(status.latest_log_briefing || statuses.log_agent?.detail || '대기 중', 96),
      symbol: 'circle',
      size: 46,
    },
    {
      id: 'news_agent',
      label: 'News Agent',
      x: 0.27,
      y: 0.14,
      status: statuses.news_agent?.status || 'pending',
      detail: truncate(status.latest_news_briefing || statuses.news_agent?.detail || '대기 중', 96),
      symbol: 'circle',
      size: 46,
    },
    {
      id: 'regulation_agent',
      label: 'Regulation',
      x: 0.51,
      y: 0.46,
      status: statuses.regulation_agent?.status || 'pending',
      detail: truncate(status.latest_regulation_analysis || statuses.regulation_agent?.detail || '대기 중', 96),
      symbol: 'hexagon',
      size: 52,
    },
    {
      id: 'orchestrator',
      label: 'Orchestrator',
      x: 0.74,
      y: 0.46,
      status: statuses.orchestrator?.status || 'pending',
      detail: truncate(status.latest_strategy_question || statuses.orchestrator?.detail || '질문 대기', 96),
      symbol: 'hexagon',
      size: 56,
    },
    {
      id: 'vector_store',
      label: 'Vector DB',
      x: 0.92,
      y: 0.32,
      status: statuses.vector_store?.status || 'pending',
      detail: `누적 ${status.vector_count || 0}건 · 최근 +${latestVector.added_count || 0}`,
      symbol: 'square',
      size: 48,
    },
  ];
  const edges = [
    ['source_logs', 'log_agent'],
    ['source_news', 'news_agent'],
    ['log_agent', 'regulation_agent'],
    ['news_agent', 'regulation_agent'],
    ['regulation_agent', 'orchestrator'],
    ['orchestrator', 'vector_store'],
  ];
  const lookup = Object.fromEntries(nodes.map((node) => [node.id, node]));
  const traces = edges.map(([start, end]) => ({
    x: [lookup[start].x, lookup[end].x],
    y: [lookup[start].y, lookup[end].y],
    mode: 'lines',
    line: { width: 2.5, color: 'rgba(151,196,225,0.35)' },
    hoverinfo: 'skip',
    showlegend: false,
    type: 'scatter',
  }));
  traces.push({
    x: nodes.map((node) => node.x),
    y: nodes.map((node) => node.y),
    text: nodes.map((node) => node.label),
    customdata: nodes.map((node) => node.detail),
    mode: 'markers+text',
    textposition: 'bottom center',
    textfont: { color: '#e7f4ff', size: 12, family: 'IBM Plex Sans KR' },
    marker: {
      size: nodes.map((node) => node.size),
      color: nodes.map((node) => colorMap[node.status] || colorMap.pending),
      symbol: nodes.map((node) => node.symbol),
      line: { width: 2, color: 'rgba(7,19,30,0.95)' },
    },
    hovertemplate: '<b>%{text}</b><br>%{customdata}<extra></extra>',
    showlegend: false,
    type: 'scatter',
  });
  return {
    data: traces,
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 18, r: 18, t: 18, b: 8 },
      xaxis: { visible: false, range: [-0.04, 1.02] },
      yaxis: { visible: false, range: [0, 0.92] },
      height: 360,
    },
  };
}

function buildGradeFigure(charts) {
  const grades = charts?.grade_distribution?.grades || {};
  const keys = Object.keys(grades);
  if (!keys.length) {
    return null;
  }
  return {
    data: [
      {
        type: 'pie',
        labels: keys,
        values: keys.map((key) => grades[key]),
        hole: 0.58,
        marker: { colors: ['#61f4de', '#ffbf69', '#8fb9d6', '#ff6b6b', '#6ee7b7'] },
      },
    ],
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#e7f4ff' },
      height: 280,
      margin: { l: 16, r: 16, t: 8, b: 12 },
      legend: { orientation: 'h', y: -0.12 },
    },
  };
}

function buildTrendFigure(charts) {
  const labels = charts?.score_trend?.labels || [];
  const scores = charts?.score_trend?.scores || [];
  if (!labels.length || !scores.length) {
    return null;
  }
  const trimmedLabels = labels.slice(-8);
  const trimmedScores = scores.slice(-8);
  return {
    data: [
      {
        type: 'scatter',
        x: trimmedLabels,
        y: trimmedScores,
        mode: 'lines+markers',
        line: { width: 3, color: '#61f4de' },
        marker: { size: 9, color: '#ffbf69' },
        fill: 'tozeroy',
        fillcolor: 'rgba(97,244,222,0.10)',
      },
    ],
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#e7f4ff' },
      height: 280,
      margin: { l: 16, r: 16, t: 8, b: 20 },
      xaxis: { tickangle: -20, gridcolor: 'rgba(151,196,225,0.10)' },
      yaxis: { gridcolor: 'rgba(151,196,225,0.12)', title: '리스크 점수' },
    },
  };
}

function buildPulseFigure(status) {
  const rows = [
    { bucket: 'Vectors', value: Number(status.vector_count || 0), color: '#61f4de' },
    { bucket: 'News', value: (status.news || []).length, color: '#8fb9d6' },
    { bucket: 'Issues', value: (status.issues || []).length, color: '#ff6b6b' },
    { bucket: 'Events', value: (status.agent_activity_log || []).length, color: '#ffbf69' },
  ];
  return {
    data: [
      {
        type: 'bar',
        x: rows.map((item) => item.bucket),
        y: rows.map((item) => item.value),
        marker: { color: rows.map((item) => item.color) },
      },
    ],
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#e7f4ff' },
      height: 320,
      margin: { l: 16, r: 16, t: 8, b: 20 },
      showlegend: false,
      xaxis: { title: '' },
      yaxis: { gridcolor: 'rgba(151,196,225,0.12)', title: '건수' },
    },
  };
}

function buildVectorFigures(events) {
  const rows = [...(events || [])].slice(0, 20).reverse();
  if (!rows.length) {
    return { line: null, bar: null };
  }
  return {
    line: {
      data: [
        {
          type: 'scatter',
          x: rows.map((item) => formatTime(item.timestamp)),
          y: rows.map((item) => item.after_count || 0),
          mode: 'lines+markers',
          line: { color: '#61f4de', width: 3 },
          marker: { color: '#ffbf69', size: 8 },
        },
      ],
      layout: {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e7f4ff' },
        height: 260,
        margin: { l: 16, r: 16, t: 16, b: 16 },
        xaxis: { title: '시간', gridcolor: 'rgba(151,196,225,0.10)' },
        yaxis: { title: '누적 벡터 수', gridcolor: 'rgba(151,196,225,0.12)' },
      },
    },
    bar: {
      data: [
        {
          type: 'bar',
          x: rows.map((item) => formatTime(item.timestamp)),
          y: rows.map((item) => item.added_count || 0),
          marker: { color: '#8fb9d6' },
        },
      ],
      layout: {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e7f4ff' },
        height: 180,
        margin: { l: 16, r: 16, t: 16, b: 16 },
        xaxis: { title: '시간', gridcolor: 'rgba(151,196,225,0.10)' },
        yaxis: { title: '추가량', gridcolor: 'rgba(151,196,225,0.12)' },
      },
    },
  };
}

function countBy(items, getter) {
  return items.reduce((accumulator, item) => {
    const key = getter(item) || '-';
    accumulator[key] = (accumulator[key] || 0) + 1;
    return accumulator;
  }, {});
}

function loadStoredPrompts() {
  try {
    const raw = window.localStorage.getItem('reviewer_prompts');
    if (!raw) {
      return Object.fromEntries(REVIEWER_PERSONAS.map((persona) => [persona.id, persona.defaultPrompt]));
    }
    const parsed = JSON.parse(raw);
    return Object.fromEntries(
      REVIEWER_PERSONAS.map((persona) => [persona.id, parsed?.[persona.id] || persona.defaultPrompt]),
    );
  } catch {
    return Object.fromEntries(REVIEWER_PERSONAS.map((persona) => [persona.id, persona.defaultPrompt]));
  }
}

function saveStoredPrompts(prompts) {
  window.localStorage.setItem('reviewer_prompts', JSON.stringify(prompts));
}

function App() {
  const [status, setStatus] = useState(initialStatus);
  const [charts, setCharts] = useState({});
  const [productSummary, setProductSummary] = useState({});
  const [productStats, setProductStats] = useState({});
  const [selectedSection, setSelectedSection] = useState(MAIN_SECTIONS[1]);
  const [selectedStore, setSelectedStore] = useState('');
  const [vectorSummary, setVectorSummary] = useState({ items: [], total_count: 0 });
  const [fullVectorItems, setFullVectorItems] = useState([]);
  const [fullVectorLoaded, setFullVectorLoaded] = useState(false);
  const [regulationFiles, setRegulationFiles] = useState([]);
  const [regulationBusy, setRegulationBusy] = useState(false);
  const [debateBusy, setDebateBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [reviewerPrompts, setReviewerPrompts] = useState(loadStoredPrompts);

  useEffect(() => {
    saveStoredPrompts(reviewerPrompts);
  }, [reviewerPrompts]);

  useEffect(() => {
    let ignore = false;
    async function bootstrap() {
      try {
        const [health, chartPayload, summaryPayload, statsPayload] = await Promise.all([
          fetchHealth(),
          fetchCharts(),
          fetchProductSummary(),
          fetchFaissStats(),
        ]);
        if (ignore) {
          return;
        }
        setStatus((previous) => ({ ...previous, ...health }));
        setCharts(chartPayload?.charts || {});
        setProductSummary(summaryPayload?.payload || {});
        setProductStats(statsPayload?.products || {});
        setErrorMessage('');
      } catch (error) {
        if (!ignore) {
          setErrorMessage(String(error.message || error));
        }
      }
    }
    bootstrap();
    const intervalId = window.setInterval(bootstrap, 15000);
    return () => {
      ignore = true;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadVectorSummary() {
      try {
        const payload = await fetchFaissEntriesByStore(120, selectedStore);
        if (!cancelled) {
          setVectorSummary(payload || { items: [], total_count: 0 });
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(String(error.message || error));
        }
      }
    }
    loadVectorSummary();
    return () => {
      cancelled = true;
    };
  }, [selectedStore, status.last_faiss_time]);

  useEffect(() => {
    const socket = createFaissWebSocket((payload) => {
      if (payload?.snapshot) {
        setStatus((previous) => ({ ...previous, ...payload.snapshot }));
      }
      if (payload?.event) {
        setStatus((previous) => ({
          ...previous,
          vector_events: [payload.event, ...(previous.vector_events || [])].slice(0, 40),
        }));
      }
    });
    socket.onopen = () => {
      setErrorMessage('');
    };
    socket.onerror = () => {
      setErrorMessage('실시간 상태 연결이 잠시 끊겼습니다.');
    };
    return () => {
      socket.close();
    };
  }, []);

  async function handleLoadFullVectors() {
    try {
      const payload = await fetchFaissEntriesByStore(1000, selectedStore);
      setFullVectorItems(payload?.items || []);
      setFullVectorLoaded(true);
    } catch (error) {
      setErrorMessage(String(error.message || error));
    }
  }

  async function handleRegulationUpload() {
    if (!regulationFiles.length) {
      return;
    }
    try {
      setRegulationBusy(true);
      const result = await uploadRegulationFiles(regulationFiles);
      setStatus((previous) => ({
        ...previous,
        vector_count: result.vector_count,
        latest_regulation_analysis: result.summary,
        agent_statuses: {
          ...(previous.agent_statuses || {}),
          regulation_agent: {
            status: 'completed',
            detail: result.detail,
            updated_at: result.updated_at,
          },
        },
      }));
      setRegulationFiles([]);
      setErrorMessage('');
    } catch (error) {
      setErrorMessage(String(error.message || error));
    } finally {
      setRegulationBusy(false);
    }
  }

  async function handleStartDebate() {
    try {
      setDebateBusy(true);
      setStatus((previous) => ({
        ...previous,
        cardloan_debate: {
          ...(previous.cardloan_debate || {}),
          status: 'running',
          current_stage: '신용기획부',
          question: DEFAULT_QUESTION,
          round_results: [],
        },
      }));
      const result = await startCardloanDebate(DEFAULT_QUESTION, reviewerPrompts);
      setStatus((previous) => ({ ...previous, cardloan_debate: result }));
      setErrorMessage('');
    } catch (error) {
      setErrorMessage(String(error.message || error));
    } finally {
      setDebateBusy(false);
    }
  }

  const metrics = buildOverviewMetrics(status);
  const gradeFigure = buildGradeFigure(charts);
  const trendFigure = buildTrendFigure(charts);
  const pulseFigure = buildPulseFigure(status);
  const agentFlowFigure = buildAgentFlowFigure(status);
  const vectorFigures = buildVectorFigures(status.vector_events || []);
  const summaryProducts = Object.entries(productSummary?.products || {});
  const selectedStoreLabel = STORE_OPTIONS.find((item) => item.value === selectedStore)?.label || '전체 DB';
  const summaryItems = vectorSummary.items || [];
  const typeCounts = countBy(summaryItems, (item) => item.type);
  const productCounts = countBy(summaryItems, (item) => item.product);
  const groupedStoreCounts = STORE_OPTIONS.filter((item) => item.value).map((item) => {
    const storeItems = summaryItems.filter((entry) => entry.store === item.value);
    return {
      label: item.label,
      store: item.value,
      loaded: storeItems.length,
      topType: Object.entries(countBy(storeItems, (entry) => entry.type)).sort((left, right) => right[1] - left[1])[0]?.[0] || '-',
      topProduct: Object.entries(countBy(storeItems, (entry) => entry.product)).sort((left, right) => right[1] - left[1])[0]?.[0] || '-',
    };
  });

  return (
    <div className="app-shell">
      <div className="page-grid">
        <aside className="left-rail">
          <section className="rail-panel">
            <div className="panel-kicker">News Feed</div>
            <h2>실시간 뉴스 요약</h2>
            <p className="panel-copy">최근 수집 기사와 리스크 신호를 좌측 패널에서 바로 확인합니다.</p>
            <div className="news-stack">
              {(status.news || []).slice(0, 5).map((item, index) => (
                <article className="news-card" key={`${item.link || item.title || 'news'}-${index}`}>
                  <div className="news-card-meta">
                    <span>{truncate(item.source || 'news', 28)}</span>
                    <span>{formatTime(item.published_at || item.collected_at || status.last_news_time)}</span>
                  </div>
                  <h3>{truncate(item.title || '제목 없음', 92)}</h3>
                  <p>{truncate(item.content || item.summary || '본문 미리보기가 없습니다.', 150)}</p>
                  {item.link ? (
                    <a href={item.link} target="_blank" rel="noreferrer">원문 열기</a>
                  ) : null}
                </article>
              ))}
              {!(status.news || []).length ? <div className="empty-box">표시할 뉴스가 아직 없습니다.</div> : null}
            </div>
          </section>

          <section className="rail-panel upload-panel">
            <div className="panel-kicker">Regulation Intake</div>
            <h2>규제 문서 업로드</h2>
            <p className="panel-copy">PDF, TXT, MD 파일을 벡터 DB에 적재하고 규제 브리핑까지 바로 생성합니다.</p>
            <div className="status-pill-row">
              <span className={`status-pill ${getStatusPalette(status.agent_statuses?.regulation_agent?.status).tone}`}>
                {getStatusPalette(status.agent_statuses?.regulation_agent?.status).label}
              </span>
              <span className="status-pill neutral">최근 {formatTime(status.agent_statuses?.regulation_agent?.updated_at)}</span>
            </div>
            <label className="file-picker">
              <span>파일 선택</span>
              <input
                type="file"
                accept=".pdf,.txt,.md"
                multiple
                onChange={(event) => setRegulationFiles(Array.from(event.target.files || []))}
              />
            </label>
            {regulationFiles.length ? (
              <div className="selection-box">
                <strong>업로드 대기 {regulationFiles.length}건</strong>
                {regulationFiles.slice(0, 4).map((file) => (
                  <div key={`${file.name}-${file.size}`}>{file.name}</div>
                ))}
              </div>
            ) : null}
            <button className="primary-button" type="button" onClick={handleRegulationUpload} disabled={regulationBusy || !regulationFiles.length}>
              {regulationBusy ? 'AI가 문서를 학습 중입니다' : 'AI 규제 문서 학습 시작'}
            </button>
            <div className="summary-box">
              <div className="summary-box-title">최신 규제 요약</div>
              <p>{truncate(status.latest_regulation_analysis || status.agent_statuses?.regulation_agent?.detail || '업로드된 규제 요약이 없습니다.', 260)}</p>
            </div>
          </section>
        </aside>

        <main className="main-rail">
          <section className="hero-panel">
            <div>
              <div className="hero-kicker">AI Review Control Tower</div>
              <h1>핵심 운영 신호만 빠르게 읽을 수 있도록 React 구조로 재구성했습니다.</h1>
              <p>
                기존 Streamlit 화면의 좌측 뉴스 패널, 4개 메인 섹션, 실시간 상태 동기화 흐름을 유지하면서
                브라우저 프런트와 FastAPI 백엔드가 분리된 형태로 옮겼습니다.
              </p>
            </div>
            <div className="hero-chips">
              <div className="hero-chip">
                <span>현재 심사 케이스</span>
                <strong>{metrics.resultsCount}</strong>
                <small>최근 적재 {formatTime(status.last_log_ingest_time)}</small>
              </div>
              <div className="hero-chip">
                <span>활성 Agent</span>
                <strong>{metrics.runningAgents}</strong>
                <small>완료 {metrics.completedAgents} · 실패 {metrics.failedAgents}</small>
              </div>
              <div className="hero-chip">
                <span>FAISS 누적</span>
                <strong>{metrics.vectorCount}</strong>
                <small>최근 뉴스 {formatTime(status.last_news_time)}</small>
              </div>
            </div>
          </section>

          <div className="section-tabs">
            {MAIN_SECTIONS.map((item) => (
              <button
                key={item}
                type="button"
                className={`section-tab ${selectedSection === item ? 'active' : ''}`}
                onClick={() => setSelectedSection(item)}
              >
                <span>{item}</span>
              </button>
            ))}
          </div>

          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

          {selectedSection === '운영 현황' ? (
            <section className="content-stack">
              <div className="two-column">
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Graph View</div>
                      <h2>Agent 간 데이터 흐름 시각화</h2>
                    </div>
                    <p>로그, 뉴스, 규제, 오케스트레이션, 벡터 저장 흐름을 실시간 상태와 함께 보여줍니다.</p>
                  </div>
                  <Plot data={agentFlowFigure.data} layout={agentFlowFigure.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} />
                </div>
                <div className="panel telemetry-stack">
                  <div className="telemetry-card">
                    <span>최근 벡터 적재</span>
                    <strong>+{status.vector_events?.[0]?.added_count || 0}</strong>
                    <small>누적 {metrics.vectorCount}건</small>
                  </div>
                  <div className="telemetry-card">
                    <span>최신 Agent 갱신</span>
                    <strong>{relativeMinutes(status.agent_activity_log?.[0]?.timestamp)}</strong>
                    <small>오래된 상태는 병목 신호일 수 있습니다.</small>
                  </div>
                  <div className="telemetry-card">
                    <span>최근 실패 요약</span>
                    <strong>{Object.entries(status.agent_statuses || {}).find(([, value]) => value?.status === 'failed')?.[0] || '없음'}</strong>
                    <small>{truncate(Object.values(status.agent_statuses || {}).find((value) => value?.status === 'failed')?.detail || '현재 실패 상태의 Agent가 없습니다.', 120)}</small>
                  </div>
                </div>
              </div>

              <div className="metric-grid">
                <article className="metric-card cyan"><span>심사 대상</span><strong>{metrics.resultsCount}</strong><small>현재 화면에 반영된 분석 케이스 수</small></article>
                <article className="metric-card blue"><span>FAISS 벡터</span><strong>{metrics.vectorCount}</strong><small>로그, 뉴스, 규제 문서 총합</small></article>
                <article className="metric-card red"><span>리스크 이슈</span><strong>{metrics.issuesCount}</strong><small>뉴스 기반 경보와 이슈 탐지</small></article>
                <article className="metric-card amber"><span>Agent 상태</span><strong>{metrics.runningAgents}</strong><small>실행 중 Agent 수</small></article>
              </div>

              <div className="two-column">
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Risk Radar</div>
                      <h2>심사 리스크 분포</h2>
                    </div>
                    <p>등급 분포와 최근 점수 추이로 운영 강도를 한 번에 읽습니다.</p>
                  </div>
                  <div className="two-chart-grid">
                    <div className="chart-box">
                      {gradeFigure ? <Plot data={gradeFigure.data} layout={gradeFigure.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} /> : <div className="empty-box">등급 분포 데이터가 아직 없습니다.</div>}
                    </div>
                    <div className="chart-box">
                      {trendFigure ? <Plot data={trendFigure.data} layout={trendFigure.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} /> : <div className="empty-box">점수 추이 데이터가 아직 없습니다.</div>}
                    </div>
                  </div>
                </div>
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Ops Pulse</div>
                      <h2>운영 볼륨과 이슈 밀도</h2>
                    </div>
                    <p>벡터 적재, 뉴스 수집, 이슈 탐지를 한 묶음으로 비교합니다.</p>
                  </div>
                  <Plot data={pulseFigure.data} layout={pulseFigure.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} />
                </div>
              </div>

              <div className="two-column">
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Live Briefing</div>
                      <h2>로그 브리핑</h2>
                    </div>
                  </div>
                  <p className="long-copy">{truncate(status.latest_log_briefing || '아직 로그 브리핑이 없습니다.', 800)}</p>
                </div>
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Market Watch</div>
                      <h2>뉴스 브리핑</h2>
                    </div>
                  </div>
                  <p className="long-copy">{truncate(status.latest_news_briefing || '아직 뉴스 브리핑이 없습니다.', 800)}</p>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div>
                    <div className="panel-kicker">Recent Timeline</div>
                    <h2>최근 운영 이벤트</h2>
                  </div>
                </div>
                <div className="timeline-grid">
                  {(status.agent_activity_log || []).slice(0, 6).map((event, index) => (
                    <article className="event-card" key={`${event.timestamp || 'event'}-${index}`}>
                      <div className="event-card-head">
                        <strong>{event.source || '-'}</strong>
                        <span>{formatTime(event.timestamp)}</span>
                      </div>
                      <span className={`status-pill ${getStatusPalette(event.status).tone}`}>{getStatusPalette(event.status).label}</span>
                      <p>{truncate(event.detail || '-', 180)}</p>
                    </article>
                  ))}
                </div>
              </div>
            </section>
          ) : null}

          {selectedSection === 'AI 카드론 토론실' ? (
            <section className="content-stack">
              <section className="debate-hero">
                <div>
                  <div className="panel-kicker">Cardloan Strategy Room</div>
                  <h2>AI 카드론 토론실</h2>
                  <p>신용기획부가 리스크 정책을 먼저 정리하고, 금융영업부가 승인 전환 전략을 만들고, 금융솔루션부가 상품 구조를 설계합니다.</p>
                </div>
                <button className="primary-button" type="button" disabled={debateBusy || status.cardloan_debate?.status === 'running'} onClick={handleStartDebate}>
                  {debateBusy || status.cardloan_debate?.status === 'running' ? '토론 진행 중' : '토론시작'}
                </button>
              </section>

              <div className="reviewer-grid">
                {REVIEWER_PERSONAS.map((persona) => {
                  const result = (status.cardloan_debate?.round_results || []).find((item) => item.persona_id === persona.id) || {};
                  const info = status.agent_statuses?.[persona.id] || {};
                  const palette = getStatusPalette(info.status || (result.name ? 'completed' : 'pending'));
                  return (
                    <article className="reviewer-card" key={persona.id} style={{ '--persona-accent': persona.accent }}>
                      <div className="reviewer-topline">{persona.emoji} Reviewer</div>
                      <h3>{persona.name}</h3>
                      <div className="reviewer-tone">{persona.tone}</div>
                      <p>{persona.description}</p>
                      <div className="status-pill-row">
                        <span className={`status-pill ${palette.tone}`}>{palette.label}</span>
                        <span className="status-pill neutral">{truncate(result.preview || info.detail || persona.tagline, 48)}</span>
                      </div>
                      <label className="prompt-editor">
                        <span>프롬프트</span>
                        <textarea
                          value={reviewerPrompts[persona.id] || persona.defaultPrompt}
                          onChange={(event) => setReviewerPrompts((previous) => ({ ...previous, [persona.id]: event.target.value }))}
                        />
                      </label>
                    </article>
                  );
                })}
              </div>

              <div className="two-column">
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Live Broadcast</div>
                      <h2>실시간 토론 현황</h2>
                    </div>
                    <p>WebSocket 상태 동기화와 API 응답을 함께 사용해 현재 라운드 상태를 반영합니다.</p>
                  </div>
                  <div className="debate-status-card">
                    <strong>{status.cardloan_debate?.current_stage || '대기'}</strong>
                    <span className={`status-pill ${getStatusPalette(status.cardloan_debate?.status).tone}`}>
                      {getStatusPalette(status.cardloan_debate?.status).label}
                    </span>
                    <p>{truncate(status.cardloan_debate?.summary || '시작 버튼을 누르면 카드론 토론이 순차적으로 실행됩니다.', 240)}</p>
                  </div>
                  <div className="transcript-stack">
                    {(status.cardloan_debate?.round_results || []).map((item, index) => (
                      <article className="transcript-card" key={`${item.persona_id || 'round'}-${index}`}>
                        <div className="event-card-head">
                          <strong>{item.name || item.persona_id || `stage-${index + 1}`}</strong>
                          <span>{item.completed_at ? formatTime(item.completed_at) : '진행 중'}</span>
                        </div>
                        <p>{truncate(item.preview || item.summary || item.raw_text || '-', 240)}</p>
                      </article>
                    ))}
                    {!(status.cardloan_debate?.round_results || []).length ? <div className="empty-box">아직 정리된 토론 라운드가 없습니다.</div> : null}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Round Detail</div>
                      <h2>단계별 결과</h2>
                    </div>
                    <p>각 부서가 반환한 미리보기와 요약을 오른쪽 패널에서 확인합니다.</p>
                  </div>
                  <div className="round-detail-stack">
                    {(status.cardloan_debate?.round_results || []).map((item, index) => (
                      <section className="detail-card" key={`${item.persona_id || 'detail'}-${index}`}>
                        <h3>{item.name || `단계 ${index + 1}`}</h3>
                        <p>{truncate(item.preview || '-', 180)}</p>
                        <pre>{JSON.stringify(item.payload || item, null, 2)}</pre>
                      </section>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          ) : null}

          {selectedSection === '대출상품 Dashboard' ? (
            <section className="content-stack">
              <div className="panel">
                <div className="panel-head">
                  <div>
                    <div className="panel-kicker">FAISS Summary</div>
                    <h2>상품별 승인/거절 패턴이 이렇게 요약돼 들어갑니다</h2>
                  </div>
                  <p>data/product_pattern_summary.json을 백엔드 API로 읽어와 현재 로그 요약과 같은 화면 구조로 보여줍니다.</p>
                </div>
                <div className="product-grid">
                  {summaryProducts.map(([productCode, payload]) => {
                    const totals = payload?.totals || {};
                    const approvalPatterns = payload?.approval_patterns || [];
                    const rejectionPatterns = payload?.rejection_patterns || [];
                    const rejectReasons = payload?.top_reject_reason_codes || [];
                    const stat = productStats?.[productCode] || {};
                    return (
                      <article className="product-card" key={productCode}>
                        <div className="product-card-head">
                          <div>
                            <div className="product-code">{productCode}</div>
                            <h3>{payload?.product_name || productCode}</h3>
                          </div>
                          <span className="sample-pill">의사결정 표본 {totals.decision_known_cases || 0}건</span>
                        </div>
                        <div className="signal-grid">
                          <div className="signal-box approve">
                            <span>승인 시그널</span>
                            <strong>{totals.approval_rate_percent || 0}%</strong>
                            <small>{totals.approval_cases || 0}건 승인</small>
                          </div>
                          <div className="signal-box reject">
                            <span>거절 시그널</span>
                            <strong>{totals.rejection_rate_percent || 0}%</strong>
                            <small>{totals.rejection_cases || 0}건 거절</small>
                          </div>
                        </div>
                        <div className="product-lists">
                          <div>
                            <h4>승인으로 기우는 패턴</h4>
                            <ul>
                              {approvalPatterns.slice(0, 3).map((item, index) => <li key={`approve-${index}`}>{truncate(item.rule || '-', 74)}</li>)}
                              {!approvalPatterns.length ? <li>뚜렷한 승인 패턴 없음</li> : null}
                            </ul>
                          </div>
                          <div>
                            <h4>거절로 기우는 패턴</h4>
                            <ul>
                              {rejectionPatterns.slice(0, 3).map((item, index) => <li key={`reject-${index}`}>{truncate(item.rule || '-', 74)}</li>)}
                              {!rejectionPatterns.length ? <li>뚜렷한 거절 패턴 없음</li> : null}
                            </ul>
                          </div>
                        </div>
                        <div className="reason-chip-row">
                          {rejectReasons.slice(0, 3).map((item, index) => (
                            <span className="reason-chip" key={`reason-${index}`}>
                              {truncate(`${item.code || ''} ${item.description || ''}`.trim(), 42)}
                            </span>
                          ))}
                          {!rejectReasons.length ? <span className="reason-chip muted">대표 거절사유 없음</span> : null}
                        </div>
                        <div className="product-footer">평균 승인율 {stat.approval_rate ? `${Math.round(stat.approval_rate * 100)}%` : '-'} · 평균 금리 {stat.avg_applied_rate ? stat.avg_applied_rate.toFixed(2) : '-'}</div>
                      </article>
                    );
                  })}
                  {!summaryProducts.length ? <div className="empty-box">표시할 상품 패턴 요약이 없습니다.</div> : null}
                </div>
              </div>
            </section>
          ) : null}

          {selectedSection === 'Vector DB' ? (
            <section className="content-stack">
              <div className="panel">
                <div className="panel-head panel-head-spread">
                  <div>
                    <div className="panel-kicker">Vector Runtime</div>
                    <h2>Vector DB 실시간 적재 현황</h2>
                  </div>
                  <select className="store-select" value={selectedStore} onChange={(event) => setSelectedStore(event.target.value)}>
                    {STORE_OPTIONS.map((item) => (
                      <option key={item.value || 'all'} value={item.value}>{item.label}</option>
                    ))}
                  </select>
                </div>
                <div className="metric-grid compact">
                  <article className="metric-card blue"><span>{selectedStoreLabel} 벡터 수</span><strong>{vectorSummary.total_count || 0}</strong></article>
                  <article className="metric-card cyan"><span>문서 type 수</span><strong>{Object.keys(typeCounts).length}</strong></article>
                  <article className="metric-card amber"><span>상품 코드 수</span><strong>{Object.keys(productCounts).length}</strong></article>
                  <article className="metric-card red"><span>마지막 증감</span><strong>{status.vector_events?.[0]?.added_count || 0}</strong></article>
                </div>
                <div className="table-shell">
                  <table>
                    <thead>
                      <tr>
                        <th>DB</th>
                        <th>store</th>
                        <th>현재 로드</th>
                        <th>주요 type</th>
                        <th>주요 product</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(selectedStore ? [{ label: selectedStoreLabel, store: selectedStore, loaded: summaryItems.length, topType: Object.keys(typeCounts)[0] || '-', topProduct: Object.keys(productCounts)[0] || '-' }] : groupedStoreCounts).map((row) => (
                        <tr key={row.store}>
                          <td>{row.label}</td>
                          <td>{row.store}</td>
                          <td>{row.loaded}</td>
                          <td>{row.topType}</td>
                          <td>{row.topProduct}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="two-column">
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Execution Timeline</div>
                      <h2>실행 타임라인</h2>
                    </div>
                  </div>
                  <div className="timeline-grid single-column">
                    {(status.agent_activity_log || []).slice(0, 8).map((event, index) => (
                      <article className="event-card" key={`${event.timestamp || 'activity'}-${index}`}>
                        <div className="event-card-head">
                          <strong>{event.source || '-'}</strong>
                          <span>{formatTime(event.timestamp)}</span>
                        </div>
                        <span className={`status-pill ${getStatusPalette(event.status).tone}`}>{getStatusPalette(event.status).label}</span>
                        <p>{truncate(event.detail || '-', 160)}</p>
                      </article>
                    ))}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Vector Event</div>
                      <h2>적재 이벤트</h2>
                    </div>
                  </div>
                  {vectorFigures.line ? <Plot data={vectorFigures.line.data} layout={vectorFigures.line.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} /> : <div className="empty-box">아직 기록된 벡터 적재 이벤트가 없습니다.</div>}
                  {vectorFigures.bar ? <Plot data={vectorFigures.bar.data} layout={vectorFigures.bar.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} /> : null}
                </div>
              </div>

              <div className="two-column">
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Type Distribution</div>
                      <h2>선택 DB type 분포</h2>
                    </div>
                  </div>
                  <div className="table-shell compact-table">
                    <table>
                      <thead>
                        <tr><th>type</th><th>count</th></tr>
                      </thead>
                      <tbody>
                        {Object.entries(typeCounts).sort((left, right) => right[1] - left[1]).map(([key, value]) => (
                          <tr key={key}><td>{key}</td><td>{value}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="panel-kicker">Product Distribution</div>
                      <h2>선택 DB product 분포</h2>
                    </div>
                  </div>
                  <div className="table-shell compact-table">
                    <table>
                      <thead>
                        <tr><th>product</th><th>count</th></tr>
                      </thead>
                      <tbody>
                        {Object.entries(productCounts).sort((left, right) => right[1] - left[1]).map(([key, value]) => (
                          <tr key={key}><td>{key}</td><td>{value}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head panel-head-spread">
                  <div>
                    <div className="panel-kicker">Full Snapshot</div>
                    <h2>FAISS 전체 항목</h2>
                  </div>
                  <button className="secondary-button" type="button" onClick={handleLoadFullVectors}>전체 항목 불러오기</button>
                </div>
                {!fullVectorLoaded ? <div className="empty-box">초기 탭 전환 속도를 위해 전체 1000건 로드는 버튼 클릭 시에만 수행합니다.</div> : null}
                {fullVectorLoaded ? (
                  <div className="table-shell">
                    <table>
                      <thead>
                        <tr>
                          <th>id</th>
                          <th>store</th>
                          <th>type</th>
                          <th>product</th>
                          <th>source</th>
                          <th>name</th>
                          <th>snippet</th>
                        </tr>
                      </thead>
                      <tbody>
                        {fullVectorItems.map((item) => (
                          <tr key={item.id}>
                            <td>{truncate(item.id, 18)}</td>
                            <td>{item.store || '-'}</td>
                            <td>{item.type || '-'}</td>
                            <td>{item.product || '-'}</td>
                            <td>{item.source || '-'}</td>
                            <td>{truncate(item.name || '-', 32)}</td>
                            <td>{truncate(item.snippet || '-', 120)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </div>
            </section>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default App;