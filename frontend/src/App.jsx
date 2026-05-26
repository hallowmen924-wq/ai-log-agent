import React, { Suspense, lazy, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useRive } from '@rive-app/react-canvas';
import {
  createFaissWebSocket,
  fetchFaissEntry,
  fetchSimilarLogVectors,
  fetchCharts,
  fetchFeatureOntologyClusters,
  fetchHealth,
  fetchFaissEntriesByStore,
  fetchFaissStats,
  setLogAgentOllamaEnabled,
  setNewsAgentOllamaEnabled,
  setOllamaGpuEnabled,
  setOntologyQueryPriorityEnabled,
  setRegulationUploadSummaryEnabled,
  startCardloanDebate,
  uploadRegulationFiles,
} from './api';
import LazyPlot from './components/LazyPlot';
import { DEFAULT_QUESTION, REVIEWER_PERSONAS, STORE_OPTIONS } from './constants';

const OntologyWorkbench = lazy(() => import('./components/OntologyWorkbench'));
const APP_CHART_FONT = 'Toss Product Sans, Pretendard, Apple SD Gothic Neo, Noto Sans KR, system-ui, sans-serif';

const LEGACY_DEFAULT_PROMPTS = {
  credit_planning_agent:
    '너는 신용기획부 리스크 정책 담당자다. 목표: 미래 리스크를 선제적으로 차단하고 카드론 심사 기준을 개선하라. 시장 신호 TOP5를 바탕으로 향후 발생할 주요 리스크를 예측하고, 현재 심사 정책의 취약점을 도출하고, 보완해야 할 심사 기준과 구체적 룰을 작성하라. 반드시 JSON만 출력하라.',
  sales_strategy_agent:
    '너는 금융영업부 전략 담당자다. 목표: 거절된 고객을 승인 가능한 고객으로 전환하고 승인율과 수익, 영업 채널을 동시에 고려하라. 현재 고객, 고금액 승인 사례, 유사 거절 사례를 비교해 핵심 원인과 전환 조건, 실행 전략을 JSON으로 작성하라.',
};

const VECTOR_TYPE_OPTIONS = [
  { label: '전체 type', value: '' },
  { label: 'signal_news', value: 'signal_news' },
  { label: 'news', value: 'news' },
  { label: 'generated_news', value: 'generated_news' },
  { label: 'log', value: 'log' },
  { label: 'generated_log', value: 'generated_log' },
  { label: 'document', value: 'document' },
  { label: 'regulation', value: 'regulation' },
  { label: 'rule', value: 'rule' },
  { label: 'customer_pattern', value: 'customer_pattern' },
  { label: 'product_pattern_summary', value: 'product_pattern_summary' },
  { label: 'sales_strategy', value: 'sales_strategy' },
  { label: 'generated_decision', value: 'generated_decision' },
  { label: 'generated_customer', value: 'generated_customer' },
];

const LEGACY_THEME_OPTIONS = [
  { value: 'midnight', label: '미드나잇', description: '현재 다크 컨트롤 타워 톤' },
  { value: 'cute', label: '서머 블루', description: '밝고 부드러운 코파일럿 스타일 톤' },
];

const THEME_OPTIONS = [
  { value: 'cute', label: '금융솔루션부', description: '현재 블루 테마' },
  { value: 'mint', label: '신용기획부', description: '연한 초록 테마' },
  { value: 'lemon', label: '금융영업부', description: '연한 노랑 테마' },
  { value: 'orange', label: 'IT개발자', description: '연한 주황 테마' },
];

const SECTION_RAIL_ITEMS = [
  { id: '온톨로지', icon: 'spark', label: 'Bunny' },
  { id: '운영 현황', icon: 'pulse', label: 'Ops' },
  { id: 'AI 카드론 토론실', icon: 'chat', label: 'Debate' },
  { id: 'Vector DB', icon: 'database', label: 'Vector' },
];

const VECTOR_SCENE_STORE_META = {
  logs: { label: 'Log Cloud', color: '#61f4de', glow: 'rgba(97,244,222,0.22)', anchorX: -1.8 },
  news: { label: 'News Bubble', color: '#ff8ca8', glow: 'rgba(255,140,168,0.22)', anchorX: -0.55 },
  document: { label: 'Document Bloom', color: '#ffbf69', glow: 'rgba(255,191,105,0.22)', anchorX: 0.7 },
  customer: { label: 'Customer Spark', color: '#8ba8ff', glow: 'rgba(139,168,255,0.24)', anchorX: 1.95 },
  unknown: { label: 'Unknown', color: '#d8ebf8', glow: 'rgba(216,235,248,0.18)', anchorX: 0 },
};

const AGENT_FLOW_EDGES = [
  {
    id: 'source_logs__log_analyzer',
    start: 'source_logs',
    end: 'log_analyzer',
    label: 'Logs -> LOG_ANALYZER',
    description: '유입 로그를 바로 분석 가능한 심사 이벤트 묶음으로 정규화합니다.',
  },
  {
    id: 'log_analyzer__log_agent',
    start: 'log_analyzer',
    end: 'log_agent',
    label: 'LOG_ANALYZER -> Log Agent',
    description: '로그 분석 결과를 브리핑 가능한 리스크 요약으로 전달합니다.',
  },
  {
    id: 'source_news__news_collector',
    start: 'source_news',
    end: 'news_collector',
    label: 'News -> NEWS_COLLECTOR',
    description: '수집 대상 뉴스 피드를 크롤링 가능한 입력 단위로 모읍니다.',
  },
  {
    id: 'news_collector__news_agent',
    start: 'news_collector',
    end: 'news_agent',
    label: 'NEWS_COLLECTOR -> News Agent',
    description: '수집된 뉴스를 시장 신호와 리스크 관점으로 압축합니다.',
  },
  {
    id: 'log_agent__regulation_agent',
    start: 'log_agent',
    end: 'regulation_agent',
    label: 'Log Agent -> Regulation',
    description: '로그 브리핑과 심사 근거를 규제 판단 레이어로 전달합니다.',
  },
  {
    id: 'news_agent__regulation_agent',
    start: 'news_agent',
    end: 'regulation_agent',
    label: 'News Agent -> Regulation',
    description: '뉴스 신호를 규제/심사 룰 판단에 반영할 수 있도록 보냅니다.',
  },
  {
    id: 'regulation_agent__orchestrator',
    start: 'regulation_agent',
    end: 'orchestrator',
    label: 'Regulation -> Orchestrator',
    description: '규제 판단 결과를 오케스트레이터가 다음 질문과 전략으로 조립합니다.',
  },
  {
    id: 'orchestrator__credit_planning_agent',
    start: 'orchestrator',
    end: 'credit_planning_agent',
    label: 'Orchestrator -> 신용기획부',
    description: '오케스트레이터가 신용기획부에 리스크 정책 검토 단계를 지시합니다.',
    controlPoint: { x: 0.42, y: 0.22 },
  },
  {
    id: 'orchestrator__sales_strategy_agent',
    start: 'orchestrator',
    end: 'sales_strategy_agent',
    label: 'Orchestrator -> 금융영업부',
    description: '오케스트레이터가 금융영업부에 승인 전환 전략 검토를 지시합니다.',
    controlPoint: { x: 0.34, y: 0.18 },
  },
  {
    id: 'orchestrator__solution_planning_agent',
    start: 'orchestrator',
    end: 'solution_planning_agent',
    label: 'Orchestrator -> 금융솔루션부',
    description: '오케스트레이터가 금융솔루션부에 최종 상품 구조 설계를 지시합니다.',
    controlPoint: { x: 0.28, y: 0.16 },
  },
  {
    id: 'orchestrator__vector_store',
    start: 'orchestrator',
    end: 'vector_store',
    label: 'Orchestrator -> Vector DB',
    description: '최종 질문, 답변, 전략 신호를 FAISS 벡터 저장소로 적재합니다.',
  },
];

const AGENT_FLOW_PACKET_COLORS = {
  log: { midnight: '#ffbf69', cute: '#ffb774' },
  news: { midnight: '#61f4de', cute: '#5fd6d3' },
  regulation: { midnight: '#ff8f8f', cute: '#ff8ca8' },
  strategy: { midnight: '#8fb9d6', cute: '#8ba8ff' },
};

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.04,
    },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 22, scale: 0.985 },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.42, ease: 'easeOut' },
  },
};

const sectionTransition = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -18 },
  transition: { duration: 0.35, ease: 'easeOut' },
};

function MotionCard({ as = 'div', className = '', children, ...props }) {
  const Component = motion[as];
  return (
    <Component className={className} variants={cardVariants} {...props}>
      {children}
    </Component>
  );
}

function AmbientOrb({ className, delay = 0, duration = 10 }) {
  return (
    <motion.span
      aria-hidden="true"
      className={className}
      animate={{ y: [0, -16, 0], x: [0, 10, 0], scale: [1, 1.06, 1] }}
      transition={{ duration, delay, repeat: Infinity, ease: 'easeInOut' }}
    />
  );
}

function AnimatedSignalWave({ className = '' }) {
  return (
    <div className={`css-signal-wave ${className}`} aria-hidden="true">
      <span className="css-signal-ring ring-one" />
      <span className="css-signal-ring ring-two" />
      <span className="css-signal-ring ring-three" />
      <span className="css-signal-node node-one" />
      <span className="css-signal-node node-two" />
      <span className="css-signal-node node-three" />
      <span className="css-signal-line line-one" />
      <span className="css-signal-line line-two" />
    </div>
  );
}

function AnimatedOrbitField({ className = '' }) {
  return (
    <div className={`css-orbit-field ${className}`} aria-hidden="true">
      <span className="css-orbit-core" />
      <span className="css-orbit-ring orbit-one" />
      <span className="css-orbit-ring orbit-two" />
      <span className="css-orbit-dot dot-one" />
      <span className="css-orbit-dot dot-two" />
      <span className="css-orbit-dot dot-three" />
    </div>
  );
}

function RailIcon({ kind }) {
  switch (kind) {
    case 'spark':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3.5 14.5 9l5.5 2.5-5.5 2.5L12 19.5 9.5 14 4 11.5 9.5 9 12 3.5Z" fill="currentColor" opacity="0.2" />
          <path d="M12 3.5 14.5 9l5.5 2.5-5.5 2.5L12 19.5 9.5 14 4 11.5 9.5 9 12 3.5Z" />
          <path d="M18.5 4.5v3M20 6h-3" />
        </svg>
      );
    case 'pulse':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3.5 12h4l2.2-4.5 4.2 9 2.4-4.5h4.2" />
          <path d="M5 18.5h14" opacity="0.45" />
        </svg>
      );
    case 'chat':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5.5 6.5h13a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H11l-4.5 3v-3H5.5a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2Z" />
          <path d="M8 10h8M8 13h5" />
        </svg>
      );
    case 'database':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <ellipse cx="12" cy="6" rx="6.5" ry="2.5" />
          <path d="M5.5 6v6c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5V6" />
          <path d="M5.5 12v6c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5v-6" />
        </svg>
      );
    default:
      return null;
  }
}

function StreamingText({ text, active = false, reduceMotion = false, speed = 200, className = '' }) {
  const normalized = String(text || '');
  const [visibleText, setVisibleText] = useState(reduceMotion || !active ? normalized : '');

  useEffect(() => {
    if (reduceMotion || !active) {
      setVisibleText(normalized);
      return undefined;
    }

    let timerId;
    setVisibleText((previous) => (normalized.startsWith(previous) ? previous : ''));

    const step = () => {
      setVisibleText((previous) => {
        if (previous === normalized) {
          return previous;
        }
        const nextValue = normalized.slice(0, previous.length + 1);
        if (nextValue !== normalized) {
          timerId = window.setTimeout(step, speed);
        }
        return nextValue;
      });
    };

    timerId = window.setTimeout(step, speed);
    return () => window.clearTimeout(timerId);
  }, [active, normalized, reduceMotion, speed]);

  return <span className={className}>{visibleText}</span>;
}

function initialStatus() {
  return {
    results: [],
    news: [],
    issues: [],
    vector_count: 0,
    latest_log_briefing: '',
    latest_news_briefing: '',
    latest_regulation_analysis: '',
    regulation_files: [],
    regulation_file_stats: [],
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
    last_new_item_time: null,
    last_log_ingest_time: null,
    news_agent_ollama_enabled: false,
    log_agent_ollama_enabled: false,
    regulation_upload_summary_enabled: false,
    ollama_gpu_enabled: true,
    ontology_query_priority_enabled: true,
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

function hashText(value) {
  const text = String(value || '');
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = ((hash << 5) - hash) + text.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash);
}

function tokenizeVectorSearchQuery(value) {
  return String(value || '')
    .toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
}

function scoreVectorSimilarity(item, tokens) {
  if (!tokens.length) {
    return 0;
  }

  const haystacks = [
    String(item.name || ''),
    String(item.type || ''),
    String(item.product || ''),
    String(item.source || ''),
    String(item.snippet || ''),
    Array.isArray(item.reject_reason_codes) ? item.reject_reason_codes.join(' ') : '',
    Object.keys(item.features || {}).join(' '),
    Object.values(item.features || {}).map((value) => String(value || '')).join(' '),
  ].map((entry) => entry.toLowerCase());

  let score = 0;
  tokens.forEach((token) => {
    haystacks.forEach((entry, index) => {
      if (!entry) {
        return;
      }
      if (entry.includes(token)) {
        score += index <= 2 ? 5 : 2;
        if (entry.startsWith(token)) {
          score += 1;
        }
      }
    });
  });

  return score;
}

function findSimilarVectorItems(items, query, limit = 8) {
  const tokens = tokenizeVectorSearchQuery(query);
  if (!tokens.length) {
    return [];
  }

  return [...(items || [])]
    .map((item) => ({ item, score: scoreVectorSimilarity(item, tokens) }))
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
    .map((entry) => ({
      ...entry.item,
      similarityScore: entry.score,
    }));
}

function getVectorSceneStoreMeta(storeName) {
  return VECTOR_SCENE_STORE_META[String(storeName || '').trim().toLowerCase()] || VECTOR_SCENE_STORE_META.unknown;
}

function buildVectorSceneInsights(items, totalCount) {
  const rows = items || [];
  if (!rows.length) {
    return {
      totalCount: Number(totalCount || 0),
      loadedCount: 0,
      dominantStore: '-',
      dominantType: '-',
      richestVector: '-',
    };
  }

  const storeCounts = {};
  const typeCounts = {};
  let richestItem = rows[0];
  let richestScore = -1;

  rows.forEach((item) => {
    const storeKey = String(item.store || 'unknown').trim().toLowerCase() || 'unknown';
    const typeKey = String(item.type || 'unknown').trim().toLowerCase() || 'unknown';
    storeCounts[storeKey] = Number(storeCounts[storeKey] || 0) + 1;
    typeCounts[typeKey] = Number(typeCounts[typeKey] || 0) + 1;

    const featureCount = Object.keys(item.features || {}).length;
    const rejectCount = Array.isArray(item.reject_reason_codes) ? item.reject_reason_codes.length : 0;
    const snippetBoost = Math.min(String(item.snippet || '').length / 180, 2.2);
    const richnessScore = featureCount + rejectCount + snippetBoost;
    if (richnessScore > richestScore) {
      richestItem = item;
      richestScore = richnessScore;
    }
  });

  const dominantStore = Object.entries(storeCounts).sort((left, right) => right[1] - left[1])[0]?.[0] || '-';
  const dominantType = Object.entries(typeCounts).sort((left, right) => right[1] - left[1])[0]?.[0] || '-';

  return {
    totalCount: Number(totalCount || rows.length),
    loadedCount: rows.length,
    dominantStore,
    dominantType,
    richestVector: truncate(richestItem?.name || richestItem?.id || '-', 28),
  };
}

function buildClusterRelationModel(clusterPayload) {
  const clusters = Array.isArray(clusterPayload?.clusters) ? clusterPayload.clusters : [];
  const productGroups = Object.entries(clusters.reduce((acc, cluster) => {
    const product = cluster.product || 'UNKNOWN';
    if (!acc[product]) {
      acc[product] = {
        product,
        count: 0,
        records: 0,
        clusters: [],
        metrics: {},
        decisions: {},
      };
    }
    acc[product].count += 1;
    acc[product].records += Number(cluster.count || 0);
    acc[product].clusters.push(cluster);
    acc[product].decisions[cluster.decision || '미상'] = (acc[product].decisions[cluster.decision || '미상'] || 0) + 1;
    (cluster.metric_summary || []).forEach((metric) => {
      const key = metric.axis_key || metric.label || 'metric';
      if (!acc[product].metrics[key]) {
        acc[product].metrics[key] = [];
      }
      if (metric.display) {
        acc[product].metrics[key].push(metric.display);
      }
    });
    return acc;
  }, {}))
    .map(([, value]) => value)
    .sort((left, right) => right.records - left.records);

  const totalRecords = productGroups.reduce((total, item) => total + item.records, 0);
  const topClusters = [...clusters].sort((left, right) => Number(right.count || 0) - Number(left.count || 0)).slice(0, 8);
  const bandCounts = countBy(clusters, (cluster) => `${cluster.age_band || '연령 미상'} · ${cluster.income_band || '소득 미상'} · ${cluster.amount_band || '한도 미상'}`);
  const metricNames = Array.from(new Set(clusters.flatMap((cluster) => (cluster.metric_summary || []).map((metric) => metric.label || metric.axis_key).filter(Boolean))));

  return {
    clusters,
    productGroups,
    topClusters,
    bandCounts,
    totalRecords,
    metricNames,
    cacheMeta: clusterPayload?.meta || {},
  };
}

function buildVectorSceneFigure(items, totalCount, palette, animationTick, selectedStoreLabel, selectedVectorTypeLabel, highlightedIds = new Set(), searchLabel = '') {
  const rows = [...(items || [])].slice(-180);
  if (!rows.length) {
    return null;
  }

  const scenePulse = animationTick * 0.035;
  const buildClusterKey = (item) => {
    const typeKey = String(item.type || 'unknown').trim().toLowerCase() || 'unknown';
    const productKey = String(item.product || 'no-product').trim() || 'no-product';
    const rejectKey = Array.isArray(item.reject_reason_codes) && item.reject_reason_codes.length
      ? String(item.reject_reason_codes[0] || 'clean').trim()
      : 'clean';
    return `${typeKey}__${productKey}__${rejectKey}`;
  };

  const clusterMap = rows.reduce((bucket, item) => {
    const clusterKey = buildClusterKey(item);
    if (!bucket[clusterKey]) {
      bucket[clusterKey] = [];
    }
    bucket[clusterKey].push(item);
    return bucket;
  }, {});

  const sortedClusters = Object.entries(clusterMap)
    .sort((left, right) => right[1].length - left[1].length)
    .slice(0, 12);

  const traces = [];
  const annotations = [];
  const matchedPoints = [];

  sortedClusters.forEach(([clusterKey, clusterItems], clusterIndex) => {
    const sampleItem = clusterItems[0] || {};
    const storeMeta = getVectorSceneStoreMeta(sampleItem.store);
    const [typeKey, productKey, rejectKey] = clusterKey.split('__');
    const x = [];
    const y = [];
    const text = [];
    const customdata = [];
    const sizes = [];
    const markerSymbols = [];

    const columnCount = Math.max(1, Math.ceil(Math.sqrt(sortedClusters.length)));
    const rowIndex = Math.floor(clusterIndex / columnCount);
    const columnIndex = clusterIndex % columnCount;
    const centerOffsetX = ((columnCount - 1) * 1.7) / 2;
    const centerOffsetY = (Math.ceil(sortedClusters.length / columnCount) - 1) * 1.45 / 2;
    const clusterCenterX = (columnIndex * 1.7) - centerOffsetX;
    const clusterCenterY = centerOffsetY - (rowIndex * 1.45);
    const bubbleRadius = 0.34 + Math.min(0.22, clusterItems.length * 0.024);

    clusterItems.forEach((item, itemIndex) => {
      const hash = hashText(`${item.id}:${item.type}:${item.product}`);
      const featureCount = Object.keys(item.features || {}).length;
      const rejectCount = Array.isArray(item.reject_reason_codes) ? item.reject_reason_codes.length : 0;
      const snippetLength = String(item.snippet || '').length;
      const angle = scenePulse + (itemIndex * 0.62) + ((hash % 19) * 0.07);
      const radius = 0.08 + ((itemIndex % 5) * 0.035) + (((hash >> 3) % 100) / 100) * 0.06;
      const jitterX = ((((hash % 100) / 100) - 0.5) * 0.08);
      const jitterY = (((((hash >> 4) % 100) / 100) - 0.5) * 0.08);
      const pointX = clusterCenterX + (Math.cos(angle) * radius) + jitterX;
      const pointY = clusterCenterY + (Math.sin(angle) * radius) + jitterY;

      x.push(pointX);
      y.push(pointY);
      text.push([
        truncate(item.name || item.id || 'vector', 32),
        `store ${item.store || '-'}`,
        `type ${item.type || '-'}`,
        `product ${item.product || '-'}`,
        `reject ${rejectKey || 'clean'}`,
      ].join('<br>'));
      customdata.push(item.id);
      sizes.push(8 + Math.min(16, (featureCount * 1.8) + (rejectCount * 1.2) + Math.min(snippetLength / 90, 7)));
      markerSymbols.push(rejectCount ? 'diamond' : 'circle');

      if (highlightedIds.has(item.id)) {
        matchedPoints.push({
          x: pointX,
          y: pointY,
          id: item.id,
          text: `${truncate(item.name || item.id || 'vector', 28)}<br>유사도 검색 매치`,
        });
      }
    });

    traces.push({
      type: 'scatter',
      mode: 'markers',
      name: `${typeKey} · ${truncate(productKey, 16)}`,
      x,
      y,
      text,
      customdata,
      hovertemplate: '%{text}<extra></extra>',
      marker: {
        size: sizes,
        color: storeMeta.color,
        opacity: 0.84,
        line: { color: '#ffffff', width: 0.8 },
        symbol: markerSymbols,
      },
    });

    const ringSteps = 54;
    const ringX = [];
    const ringY = [];
    for (let stepIndex = 0; stepIndex <= ringSteps; stepIndex += 1) {
      const angle = (Math.PI * 2 * stepIndex) / ringSteps;
      ringX.push(clusterCenterX + (Math.cos(angle) * bubbleRadius * 1.18));
      ringY.push(clusterCenterY + (Math.sin(angle) * bubbleRadius));
    }
    traces.push({
      type: 'scatter',
      mode: 'lines',
      showlegend: false,
      hoverinfo: 'skip',
      x: ringX,
      y: ringY,
      line: { color: storeMeta.glow, width: 4 },
      opacity: 0.6,
    });

    annotations.push({
      x: clusterCenterX,
      y: clusterCenterY + bubbleRadius + 0.18,
      text: `${typeKey}<br>${truncate(productKey, 16)} · ${clusterItems.length}`,
      showarrow: false,
      font: { color: palette.text, size: 11, family: APP_CHART_FONT },
      bgcolor: 'rgba(0,0,0,0)',
    });
  });

  const recentRows = rows.slice(-14);
  traces.push({
    type: 'scatter',
    mode: 'markers',
    name: 'Recent Pulse',
    x: recentRows.map((item, index) => {
      const clusterIndex = sortedClusters.findIndex(([clusterKey]) => clusterKey === buildClusterKey(item));
      const normalizedIndex = clusterIndex >= 0 ? clusterIndex : 0;
      const columnCount = Math.max(1, Math.ceil(Math.sqrt(sortedClusters.length || 1)));
      const rowIndex = Math.floor(normalizedIndex / columnCount);
      const columnIndex = normalizedIndex % columnCount;
      const centerOffsetX = ((columnCount - 1) * 1.7) / 2;
      const centerOffsetY = (Math.ceil((sortedClusters.length || 1) / columnCount) - 1) * 1.45 / 2;
      const clusterCenterX = (columnIndex * 1.7) - centerOffsetX;
      return clusterCenterX + (Math.sin(scenePulse + index) * 0.12);
    }),
    y: recentRows.map((item, index) => {
      const clusterIndex = sortedClusters.findIndex(([clusterKey]) => clusterKey === buildClusterKey(item));
      const normalizedIndex = clusterIndex >= 0 ? clusterIndex : 0;
      const columnCount = Math.max(1, Math.ceil(Math.sqrt(sortedClusters.length || 1)));
      const rowIndex = Math.floor(normalizedIndex / columnCount);
      const centerOffsetY = (Math.ceil((sortedClusters.length || 1) / columnCount) - 1) * 1.45 / 2;
      const clusterCenterY = centerOffsetY - (rowIndex * 1.45);
      return clusterCenterY - 0.16 + (Math.cos(scenePulse + index) * 0.08);
    }),
    text: recentRows.map((item) => `${truncate(item.name || item.id || 'recent', 28)}<br>최근 적재 하이라이트`),
    customdata: recentRows.map((item) => item.id),
    hovertemplate: '%{text}<extra></extra>',
    marker: {
      size: recentRows.map((item) => 14 + Math.min(10, Object.keys(item.features || {}).length * 1.6)),
      color: '#ffffff',
      opacity: 0.92,
      symbol: 'diamond',
      line: { color: '#ffb774', width: 2.2 },
    },
  });

  if (matchedPoints.length) {
    traces.push({
      type: 'scatter',
      mode: 'markers+text',
      name: 'Search Match',
      x: matchedPoints.map((point) => point.x),
      y: matchedPoints.map((point) => point.y),
      text: matchedPoints.map(() => 'match'),
      textposition: 'top center',
      textfont: { color: palette.text, size: 10, family: APP_CHART_FONT },
      customdata: matchedPoints.map((point) => point.id),
      hovertext: matchedPoints.map((point) => point.text),
      hovertemplate: '%{hovertext}<extra></extra>',
      marker: {
        size: 20,
        color: '#ffffff',
        symbol: 'star-diamond',
        opacity: 0.96,
        line: { color: '#ff8ca8', width: 2.4 },
      },
    });
  }

  const xAxisRange = (() => {
    if (!matchedPoints.length) {
      return [-3.2, 3.2];
    }
    const values = matchedPoints.map((point) => point.x);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const padding = Math.max(0.52, ((maxValue - minValue) * 0.7) + 0.34);
    return [minValue - padding, maxValue + padding];
  })();

  const yAxisRange = (() => {
    if (!matchedPoints.length) {
      return [-3.1, 3.1];
    }
    const values = matchedPoints.map((point) => point.y);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const padding = Math.max(0.52, ((maxValue - minValue) * 0.7) + 0.34);
    return [minValue - padding, maxValue + padding];
  })();

  return {
    data: traces,
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 0, r: 0, t: 18, b: 0 },
      showlegend: true,
      legend: {
        orientation: 'h',
        y: 1.03,
        x: 0,
        font: { color: palette.text, size: 11 },
        bgcolor: 'rgba(0,0,0,0)',
      },
      annotations,
      hovermode: 'closest',
      xaxis: {
        title: '유사 로그 군집',
        color: palette.text,
        gridcolor: palette.grid,
        zerolinecolor: palette.gridStrong,
        showbackground: false,
        showticklabels: false,
        range: xAxisRange,
      },
      yaxis: {
        title: '로그 성격 레인',
        color: palette.text,
        gridcolor: palette.grid,
        zerolinecolor: palette.gridStrong,
        showbackground: false,
        showticklabels: false,
        range: yAxisRange,
        scaleanchor: 'x',
        scaleratio: 1,
      },
      title: {
        text: searchLabel
          ? `FAISS Log Cluster Map · ${selectedStoreLabel} · ${selectedVectorTypeLabel} · 검색: ${truncate(searchLabel, 24)}`
          : `FAISS Log Cluster Map · ${selectedStoreLabel} · ${selectedVectorTypeLabel}`,
        font: { color: palette.text, size: 15, family: APP_CHART_FONT },
        x: 0.02,
      },
    },
  };
}

function getOllamaAgentLabel(agent) {
  switch (String(agent || '').trim()) {
    case 'credit_planning_agent':
      return '신용기획부';
    case 'sales_strategy_agent':
      return '금융영업부';
    case 'solution_planning_agent':
      return '금융솔루션부';
    case 'log_agent':
      return '로그 에이전트';
    case 'news_agent':
      return '뉴스 에이전트';
    case 'regulation_agent':
      return '규제 에이전트';
    case 'orchestrator':
      return '오케스트레이터';
    default:
      return 'Ollama';
  }
}

function formatOllamaModel(model) {
  const text = String(model || '').trim();
  return text || '기본 모델';
}

function formatPromptLength(value) {
  const size = Number(value || 0);
  return size > 0 ? `${size.toLocaleString()} chars` : '길이 정보 없음';
}

function formatElapsedMs(value) {
  const amount = Number(value || 0);
  if (!Number.isFinite(amount) || amount <= 0) {
    return '-';
  }
  if (amount < 1000) {
    return `${amount}ms`;
  }
  return `${(amount / 1000).toFixed(1)}s`;
}

function buildReviewerPromptTemplate(personaId, promptText) {
  if (personaId === 'credit_planning_agent') {
    return `${promptText}\n\n[토론 주제]\n{{question}}\n\n[시장 신호]\n{{market_signals}}\n\n지시:\n1. 향후 리스크 2개 이내로 예측하라\n2. 현재 정책 취약점을 짧게 짚어라\n3. 보완 심사 기준을 제안하라\n4. 실행 룰을 간단히 작성하라\n\n{{output_rules}}`;
  }
  if (personaId === 'sales_strategy_agent') {
    return `${promptText}\n\n[토론 주제]\n{{question}}\n\n[현재 고객]\n{{current_customer}}\n\n[승인 사례]\n{{approved_cases}}\n\n[거절 사례]\n{{rejected_cases}}\n\n지시:\n1. 승인과 거절 사례 차이를 짧게 분석하라\n2. 현재 고객의 핵심 거절 원인을 찾아라\n3. 승인 전환 조건을 제시하라\n4. 실행 전략을 2개 이내로 정리하라\n\n{{output_rules}}`;
  }
  return `${promptText}\n\n[토론 주제]\n{{question}}\n\n[리스크 정책]\n{{credit_result}}\n\n[영업 전략]\n{{sales_result}}\n\n지시:\n1. 두 전략의 충돌 지점을 분석하라\n2. 리스크 통제형 상품 구조를 설계하라\n3. 신상품 1개를 제안하라\n4. 기존 상품 개선안을 짧게 제시하라\n\n{{output_rules}}`;
}

function getReviewerOutputRules(personaId) {
  if (personaId === 'credit_planning_agent') {
    return '출력 규칙:\n- 반드시 JSON만 출력하라\n{"risk_forecast": [], "policy_weakness": [], "new_rules": [], "expected_effect": []}';
  }
  if (personaId === 'sales_strategy_agent') {
    return '출력 규칙:\n- 반드시 JSON만 출력하라\n{"current_status": "", "rejection_reason": [], "conversion_conditions": [], "action_plan": []}';
  }
  return '출력 규칙:\n- 반드시 JSON만 출력하라\n{"conflict_analysis": "", "new_product": {"name": "", "target": "", "structure": "", "risk_control": [], "profit_model": ""}, "improvement": []}';
}

function getReviewerPlaceholderKeys(personaId) {
  if (personaId === 'credit_planning_agent') {
    return ['question', 'market_signals', 'output_rules'];
  }
  if (personaId === 'sales_strategy_agent') {
    return ['question', 'current_customer', 'approved_cases', 'rejected_cases', 'output_rules'];
  }
  return ['question', 'credit_result', 'sales_result', 'output_rules'];
}

const REVIEWER_CHAT_OPENERS = {
  credit_planning_agent: '안녕하세요, 신용기획부입니다. 저는 시장 신호와 심사 정책을 기준으로 이번 안건의 위험 구간을 먼저 판별하겠습니다. 시작은 차분하게 하되 기준선은 분명하게 잡겠습니다.',
  sales_strategy_agent: '안녕하세요, 금융영업부입니다. 저는 승인 사례와 거절 사례를 비교해서 지금 고객을 어디까지 승인 쪽으로 돌릴 수 있는지 바로 보겠습니다. 너무 무겁지 않게, 대신 전환 포인트는 빠르게 잡아보겠습니다.',
  solution_planning_agent: '안녕하세요, 금융솔루션부입니다. 저는 앞선 두 부서 판단을 한 화면에 올려두고 실제 상품 구조로 연결하는 역할을 맡고 있습니다. 충돌은 줄이고 실행감은 살리는 쪽으로 묶겠습니다.',
};

const REVIEWER_CHAT_THINKING = {
  credit_planning_agent: '잠시만요. 최신 뉴스 신호와 심사 취약 구간을 겹쳐 보면서 어떤 리스크를 먼저 차단해야 하는지 우선순위를 정리하고 있습니다. 이번에는 기준이 흔들리지 않게 보수적으로 보겠습니다.',
  sales_strategy_agent: '현재 고객, 승인 사례, 거절 사례를 한 줄로 맞춰 보고 있습니다. 어디를 바꾸면 승인 전환이 되는지 감이 잡히는 중입니다. 고객이 실제로 움직일 수 있는 조건만 남기겠습니다.',
  solution_planning_agent: '앞선 두 결과를 상품 구조에 녹이는 중입니다. 수익성은 살리고 리스크 통제 장치는 더 단단하게 맞춰보겠습니다. 구조가 복잡해지지 않게 한 번 더 정리하겠습니다.',
};

const REVIEWER_CHAT_HANDOFF = {
  credit_planning_agent: '제가 먼저 리스크 기준선을 세워두겠습니다. 뒤에서 영업과 상품 쪽이 이어받기 쉽게 핵심만 짧게 정리하겠습니다.',
  sales_strategy_agent: '신용기획부에서 짚은 리스크 선을 넘지 않는 범위에서, 승인으로 돌릴 수 있는 조건만 골라서 바로 이어보겠습니다.',
  solution_planning_agent: '앞선 두 부서 의견이 충돌하지 않게 제가 가운데서 정리하겠습니다. 실행 가능한 상품 구조로 마무리하겠습니다.',
};

function buildQuotedDebateSnippet(value, fallback) {
  const text = truncate(value || fallback || '-', 72);
  return text === '-' ? fallback || '-' : text;
}

function buildReviewerHandoffMessage(personaId, roundResults = []) {
  const creditResult = roundResults.find((item) => item.persona_id === 'credit_planning_agent');
  const salesResult = roundResults.find((item) => item.persona_id === 'sales_strategy_agent');

  if (personaId === 'sales_strategy_agent') {
    if (!creditResult) {
      return '신용기획부가 리스크 기준선을 잡는 동안 저는 승인 사례와 거절 사례를 나란히 보면서 승인 전환 조건을 동시에 추려보겠습니다.';
    }
    const quoted = buildQuotedDebateSnippet(creditResult?.preview || creditResult?.summary, '리스크 기준을 먼저 보겠습니다.');
    return `방금 신용기획부에서 ${quoted} 라고 정리했네요. 그 선은 넘지 않으면서 승인 전환이 가능한 조건만 빠르게 추려보겠습니다.`;
  }

  if (personaId === 'solution_planning_agent') {
    const creditQuoted = buildQuotedDebateSnippet(creditResult?.preview || creditResult?.summary, '리스크 기준을 확인했습니다.');
    const salesQuoted = buildQuotedDebateSnippet(salesResult?.preview || salesResult?.summary, '전환 조건도 이어서 보겠습니다.');
    return `지금까지는 신용기획부가 ${creditQuoted} 로 기준을 잡았고, 금융영업부는 ${salesQuoted} 쪽으로 이어가고 있습니다. 이 둘이 충돌하지 않게 상품 구조로 묶겠습니다.`;
  }

  return REVIEWER_CHAT_HANDOFF[personaId] || '앞선 논의를 이어받아 핵심만 정리하겠습니다.';
}

function buildReviewerResultFollowup(personaId, roundResults = []) {
  const creditResult = roundResults.find((item) => item.persona_id === 'credit_planning_agent');
  const salesResult = roundResults.find((item) => item.persona_id === 'sales_strategy_agent');
  const solutionResult = roundResults.find((item) => item.persona_id === 'solution_planning_agent');

  if (personaId === 'credit_planning_agent') {
    return '리스크 기준선은 여기까지 먼저 잠그겠습니다. 영업부는 이 선 안에서 전환 가능 조건을 붙여주세요.';
  }

  if (personaId === 'sales_strategy_agent') {
    const quoted = buildQuotedDebateSnippet(creditResult?.preview || creditResult?.summary, '리스크 기준은 확인했습니다.');
    return `좋습니다. 방금 정리된 ${quoted} 범위 안에서 고객 설득 포인트와 승인 전환 조건을 이어서 붙이겠습니다.`;
  }

  if (personaId === 'solution_planning_agent' && creditResult && salesResult) {
    const creditQuoted = buildQuotedDebateSnippet(creditResult?.preview || creditResult?.summary, '리스크 기준은 확보했습니다.');
    const salesQuoted = buildQuotedDebateSnippet(salesResult?.preview || salesResult?.summary, '전환 조건도 확보했습니다.');
    return `좋습니다. 지금 확보된 기준은 ${creditQuoted} 이고, 영업 쪽에서는 ${salesQuoted} 로 이어졌습니다. 이제 이 둘을 실제 상품 구조로 묶겠습니다.`;
  }

  if (personaId !== 'solution_planning_agent' && solutionResult) {
    return '이제 솔루션부가 최종 구조를 매만지는 단계입니다. 저는 필요한 근거가 더 없는지 짧게 확인하겠습니다.';
  }

  return '';
}

function buildReferencePreview(title, summary, tone = 'neutral') {
  return {
    title: truncate(title || '-', 42),
    summary: truncate(summary || '-', 92),
    tone,
  };
}

function buildReferenceTitleSnippet(previews = [], fallback) {
  const titles = previews
    .map((item) => String(item?.title || '').trim())
    .filter(Boolean)
    .slice(0, 2);
  if (!titles.length) {
    return fallback;
  }
  return titles.join(' / ');
}

function buildReviewerReferenceNarration(personaId, referenceContext) {
  const previews = referenceContext?.previews || [];

  if (personaId === 'credit_planning_agent') {
    const spotlight = buildReferenceTitleSnippet(previews, '시장 신호와 정책 취약 구간');
    return `지금은 ${spotlight} 쪽을 먼저 훑고 있습니다. 뉴스 신호와 현재 심사 정책을 겹쳐 보면서 어디서 리스크가 먼저 터질지 기준선을 세우겠습니다. 판단 기준은 조금 엄격하게 두겠습니다.`;
  }

  if (personaId === 'sales_strategy_agent') {
    const spotlight = buildReferenceTitleSnippet(previews, '현재 고객 정보와 승인/거절 사례');
    return `저는 ${spotlight} 를 같은 축에 놓고 보고 있습니다. 어떤 차이 때문에 승인과 거절이 갈렸는지 먼저 짚고, 그다음에 승인 전환 조건으로 좁혀보겠습니다. 고객이 체감할 수 있는 실행 포인트 위주로 보겠습니다.`;
  }

  const spotlight = buildReferenceTitleSnippet(previews, '앞선 두 부서 결과와 상품 설계 기준');
  return `지금은 ${spotlight} 를 한데 모아 보고 있습니다. 충돌나는 조건은 줄이고, 상품 구조와 통제 장치로 자연스럽게 묶어보겠습니다. 설명은 간결하게, 구조는 바로 실행 가능하게 가져가겠습니다.`;
}

function buildParallelSceneMessage(personaId, referenceContext) {
  const items = (referenceContext?.items || []).slice(0, 2).join(', ');

  if (personaId === 'credit_planning_agent') {
    return `저는 먼저 ${items || '시장 위험 신호'} 쪽부터 빠르게 체크하겠습니다. 위험 구간을 먼저 잠그고 공유드리겠습니다. 영업부는 고객 전환 조건을 같이 봐주세요.`;
  }

  if (personaId === 'sales_strategy_agent') {
    return `네, 저는 ${items || '승인 사례와 거절 사례'} 를 바로 비교하겠습니다. 신용기획부가 기준선을 잡아주면 그 안에서 전환 포인트를 고객 언어로 바꿔보겠습니다.`;
  }

  return '';
}

function getDebateReferenceContext(personaId, status, roundResults = []) {
  const signalCount = Math.min(3, (status.news || []).length);
  const newsPreviews = (status.news || []).slice(0, 2).map((item, index) => buildReferencePreview(item.title || `뉴스 신호 ${index + 1}`, item.summary || item.content || '요약 정보가 아직 없습니다.', 'news'));
  const casePreviews = (status.results || []).slice(0, 2).map((item, index) => buildReferencePreview(`${item.product || '고객 케이스'} · ${item.decision || '판단 미상'}`, item.snippet || item.summary || item.reason?.join(', ') || `사례 ${index + 1} 요약 정보가 아직 없습니다.`, 'case'));
  const debatePreviews = roundResults.slice(0, 2).map((item) => buildReferencePreview(item.name || item.persona_id || '이전 단계', item.preview || item.summary || item.raw_text || '아직 요약이 없습니다.', 'debate'));
  switch (personaId) {
    case 'credit_planning_agent':
      return {
        summary: `지금은 signal_news ${signalCount || 0}건과 현재 심사 정책 취약점을 함께 보고 있습니다. 금리, 경기, 연체 관련 신호가 먼저 반영됩니다.`,
        items: [`signal_news ${signalCount || 0}건`, '시장 위험 신호', '현재 심사 정책 취약점'],
        previews: newsPreviews.length ? newsPreviews : [buildReferencePreview('시장 신호 대기', '아직 불러온 뉴스 신호가 없어 기본 리스크 프레임으로 먼저 정리합니다.', 'news')],
      };
    case 'sales_strategy_agent':
      return {
        summary: '현재 고객 정보에 승인 사례와 거절 사례를 겹쳐서 보고 있습니다. 무엇이 승인과 거절을 갈랐는지 먼저 설명한 뒤 전환 조건을 찾습니다.',
        items: ['현재 고객 정보', '승인 사례 비교', '거절 사례 비교'],
        previews: casePreviews.length ? casePreviews : [buildReferencePreview('사례 비교 대기', '승인/거절 비교 사례가 아직 없어 현재 고객 정보 중심으로 먼저 전환 조건을 가정합니다.', 'case')],
      };
    case 'solution_planning_agent':
      return {
        summary: '앞선 두 부서 결과를 합쳐서 상품 구조, 수익 모델, 리스크 통제 장치를 같이 보고 있습니다. 충돌 나는 지점을 먼저 풀어내는 단계입니다.',
        items: [
          roundResults.find((item) => item.persona_id === 'credit_planning_agent') ? '리스크 정책 결과' : '리스크 정책 대기',
          roundResults.find((item) => item.persona_id === 'sales_strategy_agent') ? '영업 전략 결과' : '영업 전략 대기',
          '상품 구조 설계 기준',
        ],
        previews: debatePreviews.length ? debatePreviews : [buildReferencePreview('전 단계 결과 대기', '아직 앞선 부서 결과가 없어서 기본 상품 설계 원칙과 통제 장치부터 준비하고 있습니다.', 'debate')],
      };
    default:
      return {
        summary: '실시간 토론 컨텍스트를 불러오는 중입니다.',
        items: ['실시간 토론 컨텍스트'],
        previews: [],
      };
  }
}

function buildDebateChatMessages(status, sessionRunId = 0) {
  const debate = status.cardloan_debate || {};
  const runtime = status.ollama_runtime || {};
  const roundResults = debate.round_results || [];
  const agentStatuses = status.agent_statuses || {};
  const currentStage = String(debate.current_stage || '').trim();
  const currentStageIndex = REVIEWER_PERSONAS.findIndex((persona) => persona.name === currentStage || persona.display === currentStage);
  const started = debate.status === 'running' || debate.status === 'completed' || debate.status === 'failed' || roundResults.length > 0;

  if (!started) {
    return [
      {
        id: `debate-idle-${sessionRunId}`,
        type: 'system',
        stream: false,
        title: '토론실 대기 중',
        text: '토론시작을 누르면 세 부서가 차례대로 입장해서 리스크, 전환 전략, 상품 구조를 메신저처럼 주고받습니다.',
        meta: '준비 완료',
      },
    ];
  }

  const messages = [
    {
      id: `debate-open-${sessionRunId}`,
      type: 'system',
      stream: true,
      title: '토론실 연결됨',
      text: debate.question || '카드론 전략 토론을 시작합니다.',
      meta: debate.started_at ? formatTime(debate.started_at) : '방금 전',
    },
  ];

  REVIEWER_PERSONAS.forEach((persona, index) => {
    const result = roundResults.find((item) => item.persona_id === persona.id) || null;
    const info = agentStatuses[persona.id] || {};
    const isParallelFirstWave = index < 2 && started;
    const hasActiveParticipation = Boolean(
      result
      || isParallelFirstWave
      || runtime.agent === persona.id
      || ['running', 'completed', 'failed'].includes(String(info.status || ''))
      || (currentStageIndex >= 0 && index <= currentStageIndex),
    );

    const referenceContext = getDebateReferenceContext(persona.id, status, roundResults);
    const referenceNarration = buildReviewerReferenceNarration(persona.id, referenceContext);
    const parallelSceneText = index < 2 && !result ? buildParallelSceneMessage(persona.id, referenceContext) : '';
    const resultFollowup = result ? buildReviewerResultFollowup(persona.id, roundResults) : '';

    messages.push({
      id: `intro-${sessionRunId}-${persona.id}`,
      type: 'intro',
      priority: 10 + index,
      stream: true,
      personaId: persona.id,
      personaName: persona.name,
      emoji: persona.emoji,
      accent: persona.accent,
      text: REVIEWER_CHAT_OPENERS[persona.id] || `${persona.name}가 토론에 참여했습니다.`,
      meta: index < 2 ? '동시 입장' : '대기석에서 청취 중',
      referenceSummary: referenceContext.summary,
      references: referenceContext.items,
      referencePreviews: referenceContext.previews,
    });

    if (!hasActiveParticipation) {
      return;
    }

    if (!result) {
      messages.push({
        id: `reference-${sessionRunId}-${persona.id}`,
        type: 'thinking',
        priority: 20 + index,
        stream: true,
        personaId: persona.id,
        personaName: persona.name,
        emoji: persona.emoji,
        accent: persona.accent,
        text: referenceNarration,
        meta: '참조 자료 설명',
        referenceSummary: referenceContext.summary,
        references: referenceContext.items,
        referencePreviews: referenceContext.previews,
      });
    }

    if (parallelSceneText) {
      messages.push({
        id: `parallel-${sessionRunId}-${persona.id}`,
        type: 'bridge',
        priority: 30 + index,
        stream: true,
        personaId: persona.id,
        personaName: persona.name,
        emoji: persona.emoji,
        accent: persona.accent,
        text: parallelSceneText,
        meta: '병렬 분석 조율',
        referenceSummary: referenceContext.summary,
        references: referenceContext.items,
        referencePreviews: referenceContext.previews,
      });
    }

    messages.push({
      id: `handoff-${sessionRunId}-${persona.id}`,
      type: 'bridge',
      priority: 40 + index,
      stream: true,
      personaId: persona.id,
      personaName: persona.name,
      emoji: persona.emoji,
      accent: persona.accent,
      text: buildReviewerHandoffMessage(persona.id, roundResults),
      meta: index === 0 ? '토론 방향 설정' : '앞 단계 의견 반영',
      referenceSummary: referenceContext.summary,
      references: referenceContext.items,
      referencePreviews: referenceContext.previews,
    });

    if (runtime.agent === persona.id && runtime.status === 'running') {
      messages.push({
        id: `thinking-${sessionRunId}-${persona.id}`,
        type: 'thinking',
        priority: 50 + index,
        stream: true,
        personaId: persona.id,
        personaName: persona.name,
        emoji: persona.emoji,
        accent: persona.accent,
        text: REVIEWER_CHAT_THINKING[persona.id] || '답변을 정리하는 중입니다.',
        meta: formatOllamaModel(runtime.model),
        referenceSummary: referenceContext.summary,
        references: referenceContext.items,
        referencePreviews: referenceContext.previews,
      });

      if (String(runtime.response_text || '').trim()) {
        messages.push({
          id: `draft-${sessionRunId}-${persona.id}`,
          type: 'draft',
          priority: 80 + index,
          stream: true,
          personaId: persona.id,
          personaName: persona.name,
          emoji: persona.emoji,
          accent: persona.accent,
          text: truncate(runtime.response_text, 260),
          meta: '초안 작성 중',
          referenceSummary: referenceContext.summary,
          references: referenceContext.items,
          referencePreviews: referenceContext.previews,
        });
      }
    }

    if (info.status === 'failed') {
      messages.push({
        id: `failed-${sessionRunId}-${persona.id}`,
        type: 'failed',
        priority: 110 + index,
        stream: false,
        personaId: persona.id,
        personaName: persona.name,
        emoji: persona.emoji,
        accent: persona.accent,
        text: info.detail || '응답 생성 중 오류가 발생했습니다.',
        meta: info.updated_at ? formatTime(info.updated_at) : '실패',
        referenceSummary: referenceContext.summary,
        references: referenceContext.items,
        referencePreviews: referenceContext.previews,
      });
    }

    if (result) {
      messages.push({
        id: `result-${sessionRunId}-${persona.id}`,
        type: 'result',
        priority: 120 + index,
        stream: true,
        personaId: persona.id,
        personaName: persona.name,
        emoji: persona.emoji,
        accent: persona.accent,
        text: result.preview || result.summary || result.raw_text || `${persona.name} 결과가 도착했습니다.`,
        meta: result.completed_at ? formatTime(result.completed_at) : '결과 도착',
        referenceSummary: referenceContext.summary,
        references: referenceContext.items,
        referencePreviews: referenceContext.previews,
      });

      if (resultFollowup) {
        messages.push({
          id: `result-followup-${sessionRunId}-${persona.id}`,
          type: 'bridge',
          priority: 130 + index,
          stream: true,
          personaId: persona.id,
          personaName: persona.name,
          emoji: persona.emoji,
          accent: persona.accent,
          text: resultFollowup,
          meta: '후속 조율',
          referenceSummary: referenceContext.summary,
          references: referenceContext.items,
          referencePreviews: referenceContext.previews,
        });
      }
    }
  });

  if (debate.status === 'completed') {
    messages.push({
      id: `debate-complete-${sessionRunId}`,
      type: 'system',
      priority: 140,
      stream: true,
      title: '토론 정리 완료',
      text: debate.summary || '세 부서 의견이 모두 정리되었습니다.',
      meta: debate.completed_at ? formatTime(debate.completed_at) : '완료',
    });
  }

  if (debate.status === 'failed') {
    messages.push({
      id: `debate-failed-${sessionRunId}`,
      type: 'system-error',
      priority: 150,
      stream: true,
      title: '토론실 오류',
      text: debate.error || debate.summary || '토론 진행 중 문제가 발생했습니다.',
      meta: debate.updated_at ? formatTime(debate.updated_at) : '오류',
    });
  }

  return messages;
}

function getDebateMessageTypingSpeed(message) {
  switch (message?.type) {
    case 'intro':
    case 'bridge':
      return 65;
    case 'thinking':
      return 75;
    case 'system':
    case 'system-error':
      return 90;
    case 'result':
      return 135;
    case 'draft':
      return 110;
    default:
      return 90;
  }
}

function getDebateMessagePlaybackMs(message, reduceMotion = false) {
  if (!message) {
    return 0;
  }
  if (reduceMotion) {
    return 0;
  }

  const baseText = String(message.text || '');
  const referenceText = String(message.referenceSummary || '');
  const perChar = getDebateMessageTypingSpeed(message);
  const textLength = Math.max(baseText.length, Math.floor(referenceText.length * 0.9));
  const fixedDelay = message.type === 'system' || message.type === 'system-error' ? 300 : 450;
  return Math.min(12000, Math.max(1800, fixedDelay + (textLength * perChar)));
}

function getPlaceholderFieldLabel(key) {
  switch (key) {
    case 'question':
      return '{{question}}';
    case 'market_signals':
      return '{{market_signals}}';
    case 'current_customer':
      return '{{current_customer}}';
    case 'approved_cases':
      return '{{approved_cases}}';
    case 'rejected_cases':
      return '{{rejected_cases}}';
    case 'credit_result':
      return '{{credit_result}}';
    case 'sales_result':
      return '{{sales_result}}';
    case 'output_rules':
      return '{{output_rules}}';
    default:
      return key;
  }
}

function buildReviewerPlaceholderDefaults(personaId, question = DEFAULT_QUESTION) {
  const defaults = {
    question,
    market_signals: '...실시간 뉴스 신호 JSON...',
    current_customer: '...현재 고객 JSON...',
    approved_cases: '...승인 사례 JSON...',
    rejected_cases: '...거절 사례 JSON...',
    credit_result: '...신용기획부 결과 JSON...',
    sales_result: '...금융영업부 결과 JSON...',
    output_rules: getReviewerOutputRules(personaId),
  };
  return Object.fromEntries(getReviewerPlaceholderKeys(personaId).map((key) => [key, defaults[key] || '']));
}

function buildDefaultReviewerSetting(personaId) {
  return {
    temperature: 0.5,
    placeholders: buildReviewerPlaceholderDefaults(personaId),
    market_signal_feature_keys: personaId === 'credit_planning_agent' ? 'signal_summary, risk_signal, linked_decision' : '',
  };
}

function renderReviewerPromptPreview(personaId, template, question, overrides = {}) {
  const replacements = {
    ...buildReviewerPlaceholderDefaults(personaId, question),
    ...(overrides || {}),
  };
  return String(template || '').replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (_, key) => replacements[key] || '');
}

function getDefaultReviewerTemplate(persona) {
  return buildReviewerPromptTemplate(persona.id, persona.defaultPrompt);
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

function buildAgentFlowDetail(flowId, status) {
  const resolvedFlowId = AGENT_FLOW_EDGES.some((edge) => edge.id === flowId) ? flowId : AGENT_FLOW_EDGES[0].id;
  const latestVector = (status.vector_events || [])[0] || null;
  const roundResults = status.cardloan_debate?.round_results || [];

  switch (resolvedFlowId) {
    case 'source_news__news_collector':
      return {
        kicker: 'Clicked Flow',
        title: 'News -> NEWS_COLLECTOR',
        description: '뉴스 크롤링 레이어가 실제로 끌어오는 원천 기사 목록입니다.',
        preview: truncate(status.last_news_time ? `${formatTime(status.last_news_time)} 기준 최근 뉴스 수집 완료` : '아직 수집된 뉴스 타임스탬프가 없습니다.', 220),
        metrics: [
          `크롤링 상태 ${status.news_crawl_running ? '실행 중' : '대기/완료'}`,
          `수집 뉴스 ${(status.news || []).length}건`,
        ],
        samples: (status.news || []).slice(0, 3).map((item, index) => item?.title || item?.summary || item?.text || `뉴스 원문 ${index + 1}`),
      };
    case 'news_collector__news_agent':
      return {
        kicker: 'Clicked Flow',
        title: 'NEWS_COLLECTOR -> News Agent',
        description: '뉴스 에이전트가 실제로 읽는 최근 기사/요약 데이터입니다.',
        preview: truncate(status.latest_news_briefing || '아직 뉴스 브리핑이 없습니다.', 220),
        metrics: [`입력 뉴스 ${(status.news || []).length}건`, `이슈 ${(status.issues || []).length}건`],
        samples: (status.news || []).slice(0, 3).map((item, index) => item?.title || item?.summary || item?.text || `뉴스 샘플 ${index + 1}`),
      };
    case 'source_logs__log_analyzer':
      return {
        kicker: 'Clicked Flow',
        title: 'Logs -> LOG_ANALYZER',
        description: '원본 로그가 분석 가능한 사건/필드 단위로 바로 정규화될 때의 입력 데이터입니다.',
        preview: truncate(status.last_log_ingest_time ? `${formatTime(status.last_log_ingest_time)} 기준 로그 적재가 수행되었습니다.` : '최근 로그 적재 기록이 없습니다.', 220),
        metrics: [`로그 적재 ${(status.results || []).length}건`, `최근 적재 ${formatTime(status.last_log_ingest_time)}`],
        samples: (status.results || []).slice(0, 3).map((item, index) => item?.preview || item?.summary || `${item?.product || '미상 상품'} · 적재 샘플 ${index + 1}`),
      };
    case 'log_analyzer__log_agent':
      return {
        kicker: 'Clicked Flow',
        title: 'LOG_ANALYZER -> Log Agent',
        description: '로그 분석기가 정리한 결과가 로그 브리핑 에이전트로 전달될 때의 데이터입니다.',
        preview: truncate(status.latest_log_briefing || '아직 로그 브리핑이 없습니다.', 220),
        metrics: [`분석 결과 ${(status.results || []).length}건`, `실행 상태 ${getStatusPalette(status.agent_statuses?.log_agent?.status).label}`],
        samples: (status.results || []).slice(0, 3).map((item, index) => item?.preview || item?.summary || `${item?.product || '미상 상품'} · 분석 샘플 ${index + 1}`),
      };
    case 'log_agent__regulation_agent':
      return {
        kicker: 'Clicked Flow',
        title: 'Log Agent -> Regulation',
        description: '로그 기반 심사 정보가 규제 판단 단계로 넘어갈 때 쓰인 핵심 근거입니다.',
        preview: truncate(status.latest_regulation_analysis || status.latest_log_briefing || '규제 판단 입력이 아직 없습니다.', 220),
        metrics: [`심사 케이스 ${(status.results || []).length}건`, `로그 브리핑 1건`],
        samples: (status.results || []).slice(0, 3).map((item, index) => item?.preview || item?.summary || `${item?.product || '미상 상품'} · 로그 샘플 ${index + 1}`),
      };
    case 'news_agent__regulation_agent':
      return {
        kicker: 'Clicked Flow',
        title: 'News Agent -> Regulation',
        description: '시장 뉴스 신호가 규제/리스크 판단에 반영될 때 참고한 데이터입니다.',
        preview: truncate(status.latest_regulation_analysis || status.latest_news_briefing || '규제 판단 입력이 아직 없습니다.', 220),
        metrics: [`시장 신호 ${(status.news || []).length}건`, `탐지 이슈 ${(status.issues || []).length}건`],
        samples: (status.issues || []).slice(0, 3).map((item, index) => item?.title || item?.summary || item?.reason || `이슈 샘플 ${index + 1}`),
      };
    case 'regulation_agent__orchestrator':
      return {
        kicker: 'Clicked Flow',
        title: 'Regulation -> Orchestrator',
        description: '규제 판단 결과가 오케스트레이터 질문/전략 조립으로 이어질 때의 데이터입니다.',
        preview: truncate(status.latest_strategy_question || status.latest_regulation_analysis || '전략 질문이 아직 생성되지 않았습니다.', 220),
        metrics: [`규제 분석 1건`, `오케스트레이션 상태 ${getStatusPalette(status.agent_statuses?.orchestrator?.status).label}`],
        samples: [
          status.latest_regulation_analysis,
          status.latest_strategy_question,
          status.agent_statuses?.orchestrator?.detail,
        ].filter(Boolean).slice(0, 3).map((item) => truncate(item, 120)),
      };
    case 'orchestrator__credit_planning_agent': {
      const credit = roundResults.find((item) => item.persona_id === 'credit_planning_agent');
      return {
        kicker: 'Clicked Flow',
        title: 'Orchestrator -> 신용기획부',
        description: '오케스트레이터가 신용기획부에 전달한 질문과 정책 검토 단계입니다.',
        preview: truncate(credit?.request?.prompt || status.latest_strategy_question || '아직 신용기획부 실행 입력이 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(status.agent_statuses?.credit_planning_agent?.status).label}`],
        samples: [credit?.preview, credit?.summary, credit?.request?.prompt].filter(Boolean).slice(0, 3).map((item) => truncate(item, 120)),
      };
    }
    case 'orchestrator__sales_strategy_agent': {
      const sales = roundResults.find((item) => item.persona_id === 'sales_strategy_agent');
      return {
        kicker: 'Clicked Flow',
        title: 'Orchestrator -> 금융영업부',
        description: '오케스트레이터가 금융영업부에 전달한 승인 전환 전략 단계입니다.',
        preview: truncate(sales?.request?.prompt || status.latest_strategy_question || '아직 금융영업부 실행 입력이 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(status.agent_statuses?.sales_strategy_agent?.status).label}`],
        samples: [sales?.preview, sales?.summary, sales?.request?.prompt].filter(Boolean).slice(0, 3).map((item) => truncate(item, 120)),
      };
    }
    case 'orchestrator__solution_planning_agent': {
      const solution = roundResults.find((item) => item.persona_id === 'solution_planning_agent');
      return {
        kicker: 'Clicked Flow',
        title: 'Orchestrator -> 금융솔루션부',
        description: '오케스트레이터가 금융솔루션부에 전달한 최종 상품 구조 설계 단계입니다.',
        preview: truncate(solution?.request?.prompt || status.latest_strategy_question || '아직 금융솔루션부 실행 입력이 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(status.agent_statuses?.solution_planning_agent?.status).label}`],
        samples: [solution?.preview, solution?.summary, solution?.request?.prompt].filter(Boolean).slice(0, 3).map((item) => truncate(item, 120)),
      };
    }
    case 'orchestrator__vector_store':
      return {
        kicker: 'Clicked Flow',
        title: 'Orchestrator -> Vector DB',
        description: '오케스트레이터 산출물이 벡터 저장소에 적재될 때의 최근 이벤트입니다.',
        preview: latestVector
          ? `${formatTime(latestVector.timestamp)} 기준 ${latestVector.added_count || 0}건 적재, 누적 ${latestVector.after_count || 0}건`
          : '최근 벡터 적재 이벤트가 없습니다.',
        metrics: [`누적 벡터 ${status.vector_count || 0}건`, `최근 이벤트 ${(status.vector_events || []).length}건`],
        samples: (status.vector_events || []).slice(0, 3).map((item) => `${formatTime(item.timestamp)} · +${item.added_count || 0}건 · 누적 ${item.after_count || 0}건`),
      };
    default:
      return {
        kicker: 'Clicked Flow',
        title: 'Logs -> LOG_ANALYZER',
        description: '원본 로그가 LOG_ANALYZER 로 들어가기 직전의 입력 데이터입니다.',
        preview: truncate(status.last_log_ingest_time ? `${formatTime(status.last_log_ingest_time)} 기준 최근 로그 유입이 기록되었습니다.` : '아직 로그 유입 기록이 없습니다.', 220),
        metrics: [`입력 로그 ${(status.results || []).length}건`, `최근 적재 ${formatTime(status.last_log_ingest_time)}`],
        samples: (status.results || []).slice(0, 3).map((item, index) => item?.preview || item?.summary || `${item?.product || '미상 상품'} · 케이스 ${index + 1}`),
      };
  }
}

function buildAgentNodeDetail(nodeId, status) {
  const statuses = status.agent_statuses || {};
  const roundResults = status.cardloan_debate?.round_results || [];
  switch (nodeId) {
    case 'source_logs':
      return {
        kicker: 'Clicked Node',
        title: 'Logs',
        description: '원본 로그 입력 레이어입니다.',
        preview: truncate(status.last_log_ingest_time ? `${formatTime(status.last_log_ingest_time)} 기준 최근 로그가 유입되었습니다.` : '최근 로그 유입 기록이 없습니다.', 220),
        metrics: [`로그 입력 ${(status.results || []).length}건`],
        samples: (status.results || []).slice(0, 3).map((item, index) => item?.preview || item?.summary || `${item?.product || '미상 상품'} · 로그 ${index + 1}`),
      };
    case 'log_analyzer':
      return {
        kicker: 'Clicked Node',
        title: 'LOG_ANALYZER',
        description: '로그 분석 레이어가 현재 처리 중인 심사 데이터입니다.',
        preview: truncate(status.latest_log_briefing || '아직 로그 분석 결과가 없습니다.', 220),
        metrics: [`분석 대상 ${(status.results || []).length}건`],
        samples: (status.results || []).slice(0, 3).map((item, index) => item?.preview || item?.summary || `분석 샘플 ${index + 1}`),
      };
    case 'log_agent':
      return {
        kicker: 'Clicked Node',
        title: 'Log Agent',
        description: '로그 브리핑 에이전트가 생성한 최근 요약입니다.',
        preview: truncate(status.latest_log_briefing || statuses.log_agent?.detail || '아직 로그 브리핑이 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(statuses.log_agent?.status).label}`],
        samples: (status.results || []).slice(0, 3).map((item, index) => item?.preview || item?.summary || `브리핑 샘플 ${index + 1}`),
      };
    case 'source_news':
      return {
        kicker: 'Clicked Node',
        title: 'News',
        description: '원본 뉴스 입력 레이어입니다.',
        preview: truncate(status.last_news_time ? `${formatTime(status.last_news_time)} 기준 최근 뉴스가 수집되었습니다.` : '최근 뉴스 수집 기록이 없습니다.', 220),
        metrics: [`뉴스 입력 ${(status.news || []).length}건`],
        samples: (status.news || []).slice(0, 3).map((item, index) => item?.title || item?.summary || `뉴스 ${index + 1}`),
      };
    case 'news_collector':
      return {
        kicker: 'Clicked Node',
        title: 'NEWS_COLLECTOR',
        description: '뉴스 수집 레이어가 최근 모은 기사 목록입니다.',
        preview: truncate(status.news_crawl_running ? '현재 뉴스 크롤링이 실행 중입니다.' : '최근 수집된 뉴스 배치를 보유하고 있습니다.', 220),
        metrics: [`크롤링 상태 ${status.news_crawl_running ? '실행 중' : '대기/완료'}`],
        samples: (status.news || []).slice(0, 3).map((item, index) => item?.title || item?.summary || `수집 뉴스 ${index + 1}`),
      };
    case 'news_agent':
      return {
        kicker: 'Clicked Node',
        title: 'News Agent',
        description: '뉴스 브리핑 에이전트가 생성한 최근 요약입니다.',
        preview: truncate(status.latest_news_briefing || statuses.news_agent?.detail || '아직 뉴스 브리핑이 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(statuses.news_agent?.status).label}`],
        samples: (status.news || []).slice(0, 3).map((item, index) => item?.title || item?.summary || `뉴스 샘플 ${index + 1}`),
      };
    case 'regulation_agent':
      return {
        kicker: 'Clicked Node',
        title: 'Regulation',
        description: '규제 판단 레이어가 현재 참조하는 분석 결과입니다.',
        preview: truncate(status.latest_regulation_analysis || statuses.regulation_agent?.detail || '규제 판단 결과가 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(statuses.regulation_agent?.status).label}`],
        samples: (status.issues || []).slice(0, 3).map((item, index) => item?.title || item?.summary || item?.reason || `이슈 ${index + 1}`),
      };
    case 'orchestrator':
      return {
        kicker: 'Clicked Node',
        title: 'Orchestrator',
        description: '오케스트레이터가 최근 조립한 질문과 전략입니다.',
        preview: truncate(status.latest_strategy_question || statuses.orchestrator?.detail || '전략 질문이 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(statuses.orchestrator?.status).label}`],
        samples: [status.latest_strategy_question, statuses.orchestrator?.detail].filter(Boolean).map((item) => truncate(item, 120)),
      };
    case 'credit_planning_agent': {
      const credit = roundResults.find((item) => item.persona_id === 'credit_planning_agent');
      return {
        kicker: 'Clicked Node',
        title: '신용기획부',
        description: '리스크 정책과 심사 기준선을 정리하는 부서 에이전트입니다.',
        preview: truncate(credit?.preview || credit?.summary || statuses.credit_planning_agent?.detail || '아직 실행 결과가 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(statuses.credit_planning_agent?.status).label}`],
        samples: [credit?.preview, credit?.summary, credit?.request?.prompt].filter(Boolean).slice(0, 3).map((item) => truncate(item, 120)),
      };
    }
    case 'sales_strategy_agent': {
      const sales = roundResults.find((item) => item.persona_id === 'sales_strategy_agent');
      return {
        kicker: 'Clicked Node',
        title: '금융영업부',
        description: '승인 전환 전략과 실행 조건을 정리하는 부서 에이전트입니다.',
        preview: truncate(sales?.preview || sales?.summary || statuses.sales_strategy_agent?.detail || '아직 실행 결과가 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(statuses.sales_strategy_agent?.status).label}`],
        samples: [sales?.preview, sales?.summary, sales?.request?.prompt].filter(Boolean).slice(0, 3).map((item) => truncate(item, 120)),
      };
    }
    case 'solution_planning_agent': {
      const solution = roundResults.find((item) => item.persona_id === 'solution_planning_agent');
      return {
        kicker: 'Clicked Node',
        title: '금융솔루션부',
        description: '앞선 판단을 실제 상품 구조로 묶는 부서 에이전트입니다.',
        preview: truncate(solution?.preview || solution?.summary || statuses.solution_planning_agent?.detail || '아직 실행 결과가 없습니다.', 220),
        metrics: [`상태 ${getStatusPalette(statuses.solution_planning_agent?.status).label}`],
        samples: [solution?.preview, solution?.summary, solution?.request?.prompt].filter(Boolean).slice(0, 3).map((item) => truncate(item, 120)),
      };
    }
    case 'vector_store':
      return {
        kicker: 'Clicked Node',
        title: 'Vector DB',
        description: '벡터 저장소 최근 적재 이벤트입니다.',
        preview: (status.vector_events || [])[0]
          ? `${formatTime(status.vector_events[0].timestamp)} · +${status.vector_events[0].added_count || 0}건 · 누적 ${status.vector_events[0].after_count || 0}건`
          : '최근 적재 이벤트가 없습니다.',
        metrics: [`누적 벡터 ${status.vector_count || 0}건`],
        samples: (status.vector_events || []).slice(0, 3).map((item) => `${formatTime(item.timestamp)} · +${item.added_count || 0}건`),
      };
    default:
      return buildAgentFlowDetail(AGENT_FLOW_EDGES[0].id, status);
  }
}

function getAgentFlowKind(edgeId) {
  if (edgeId.includes('news')) {
    return 'news';
  }
  if (edgeId.includes('regulation')) {
    return 'regulation';
  }
  if (edgeId.includes('orchestrator') || edgeId.includes('vector_store') || edgeId.includes('credit_planning_agent') || edgeId.includes('sales_strategy_agent') || edgeId.includes('solution_planning_agent')) {
    return 'strategy';
  }
  return 'log';
}

function buildQuadraticCurvePoints(start, end, controlPoint, steps = 24) {
  return Array.from({ length: steps + 1 }, (_, index) => {
    const t = index / steps;
    const inverse = 1 - t;
    return {
      x: (inverse * inverse * start.x) + (2 * inverse * t * controlPoint.x) + (t * t * end.x),
      y: (inverse * inverse * start.y) + (2 * inverse * t * controlPoint.y) + (t * t * end.y),
    };
  });
}

function getEdgePathPoints(edge, start, end) {
  if (!edge.controlPoint) {
    return [start, end];
  }
  return buildQuadraticCurvePoints(start, end, edge.controlPoint);
}

function getPointAlongPath(pathPoints, progress) {
  if (!pathPoints.length) {
    return { x: 0, y: 0 };
  }
  if (pathPoints.length === 1) {
    return pathPoints[0];
  }

  const segments = [];
  let totalLength = 0;
  for (let index = 0; index < pathPoints.length - 1; index += 1) {
    const start = pathPoints[index];
    const end = pathPoints[index + 1];
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    segments.push({ start, end, length });
    totalLength += length;
  }

  if (!totalLength) {
    return pathPoints[0];
  }

  let targetLength = totalLength * progress;
  for (const segment of segments) {
    if (targetLength <= segment.length) {
      const ratio = segment.length ? targetLength / segment.length : 0;
      return {
        x: segment.start.x + ((segment.end.x - segment.start.x) * ratio),
        y: segment.start.y + ((segment.end.y - segment.start.y) * ratio),
      };
    }
    targetLength -= segment.length;
  }

  return pathPoints[pathPoints.length - 1];
}

function buildAgentFlowFigure(status) {
  const palette = arguments[1] || getPlotPalette('midnight');
  const animationTick = arguments[2] || 0;
  const selectedFlowId = arguments[3] || '';
  const flowActivityMap = arguments[4] || {};
  const selectedNodeId = arguments[5] || '';
  const statuses = status.agent_statuses || {};
  const latestVector = (status.vector_events || [])[0] || {};
  const colorMap = {
    running: palette.cyan,
    completed: palette.green,
    failed: palette.red,
    pending: palette.blue,
  };
  const logAnalyzerStatus = statuses.log_analyzer?.status || statuses.log_agent?.status || ((status.results || []).length ? 'completed' : 'pending');
  const newsCollectorStatus = statuses.news_collector?.status || (status.news_crawl_running ? 'running' : ((status.news || []).length ? 'completed' : 'pending'));
  const creditPlanningStatus = statuses.credit_planning_agent?.status || 'pending';
  const salesStrategyStatus = statuses.sales_strategy_agent?.status || 'pending';
  const solutionPlanningStatus = statuses.solution_planning_agent?.status || 'pending';
  const selectedEdge = AGENT_FLOW_EDGES.find((edge) => edge.id === selectedFlowId) || null;
  const selectedNodeIds = new Set(selectedEdge ? [selectedEdge.start, selectedEdge.end] : []);
  if (selectedNodeId) {
    selectedNodeIds.add(selectedNodeId);
  }
  const nodes = [
    { id: 'source_logs', label: 'Logs', x: 0.04, y: 0.8, status: 'completed', detail: `유입 로그 ${(status.results || []).length}건`, symbol: 'diamond', size: 34 },
    { id: 'log_analyzer', label: 'LOG_ANALYZER', x: 0.28, y: 0.8, status: logAnalyzerStatus, detail: `분석 대상 ${(status.results || []).length}건`, symbol: 'circle', size: 46 },
    { id: 'log_agent', label: 'Log Agent', x: 0.56, y: 0.68, status: statuses.log_agent?.status || 'pending', detail: truncate(status.latest_log_briefing || statuses.log_agent?.detail || '대기 중', 96), symbol: 'circle', size: 58, emphasis: true },
    { id: 'source_news', label: 'News', x: 0.04, y: 0.36, status: 'completed', detail: `수집 뉴스 ${(status.news || []).length}건`, symbol: 'diamond', size: 34 },
    { id: 'news_collector', label: 'NEWS_COLLECTOR', x: 0.22, y: 0.36, status: newsCollectorStatus, detail: `${formatTime(status.last_news_time)} · 크롤링`, symbol: 'circle', size: 42 },
    { id: 'news_agent', label: 'News Agent', x: 0.4, y: 0.36, status: statuses.news_agent?.status || 'pending', detail: truncate(status.latest_news_briefing || statuses.news_agent?.detail || '대기 중', 96), symbol: 'circle', size: 58, emphasis: true },
    { id: 'regulation_agent', label: 'Regulation', x: 0.62, y: 0.54, status: statuses.regulation_agent?.status || 'pending', detail: truncate(status.latest_regulation_analysis || statuses.regulation_agent?.detail || '대기 중', 96), symbol: 'hexagon', size: 66, emphasis: true },
    { id: 'orchestrator', label: 'Orchestrator', x: 0.68, y: 0.34, status: statuses.orchestrator?.status || 'pending', detail: truncate(status.latest_strategy_question || statuses.orchestrator?.detail || '질문 대기', 96), symbol: 'hexagon', size: 64, emphasis: true },
    { id: 'credit_planning_agent', label: '신용기획부', x: 0.14, y: 0.08, status: creditPlanningStatus, detail: truncate(statuses.credit_planning_agent?.detail || '대기 중', 96), symbol: 'circle', size: 50, emphasis: true },
    { id: 'sales_strategy_agent', label: '금융영업부', x: 0.3, y: 0.08, status: salesStrategyStatus, detail: truncate(statuses.sales_strategy_agent?.detail || '대기 중', 96), symbol: 'circle', size: 50, emphasis: true },
    { id: 'solution_planning_agent', label: '금융솔루션부', x: 0.46, y: 0.08, status: solutionPlanningStatus, detail: truncate(statuses.solution_planning_agent?.detail || '대기 중', 96), symbol: 'circle', size: 50, emphasis: true },
    { id: 'vector_store', label: 'Vector DB', x: 0.95, y: 0.18, status: statuses.vector_store?.status || 'pending', detail: `누적 ${status.vector_count || 0}건 · 최근 +${latestVector.added_count || 0}`, symbol: 'square', size: 48 },
  ];
  const lookup = Object.fromEntries(nodes.map((node) => [node.id, node]));
  const emphasizedNodes = nodes.filter((node) => node.emphasis);
  const traces = [
    {
      x: [0.02, 1.14],
      y: [0.22, 0.22],
      mode: 'lines',
      line: { width: 1.4, color: palette.line, dash: 'dot' },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
    },
    {
      x: [0.18, 0.66],
      y: [0.88, 0.88],
      text: ['MAIN PROCESS'],
      mode: 'text',
      textfont: { color: palette.text, size: 11, family: APP_CHART_FONT },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
    },
    {
      x: [0.3],
      y: [0.145],
      text: ['DEPARTMENT AGENTS'],
      mode: 'text',
      textfont: { color: palette.text, size: 11, family: APP_CHART_FONT },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
    },
    ...AGENT_FLOW_EDGES.flatMap((edge, index) => {
    const start = lookup[edge.start];
    const end = lookup[edge.end];
    const pathPoints = getEdgePathPoints(edge, start, end);
    const pathX = pathPoints.map((point) => point.x);
    const pathY = pathPoints.map((point) => point.y);
    const activityBoost = Number(flowActivityMap[edge.id] || 0);
    const isSelected = selectedFlowId === edge.id;
    const flowKind = getAgentFlowKind(edge.id);
    const packetColor = palette.flowPacketMap[flowKind] || palette.flowPacket;
    const travel = (animationTick * (0.045 + index * 0.004) + index * 0.17) % 1;
    const pulsePoint = getPointAlongPath(pathPoints, travel);
    const edgeColor = isSelected ? palette.flowActive : activityBoost > 0 ? palette.flowPulse : palette.line;
    const hitboxPoints = Array.from({ length: 7 }, (_, stepIndex) => getPointAlongPath(pathPoints, (stepIndex + 1) / 8));

    return [
      ...(isSelected ? [{
        x: pathX,
        y: pathY,
        mode: 'lines',
        line: { width: palette.flowWidth + 6, color: palette.flowHalo, dash: 'solid', shape: edge.controlPoint ? 'spline' : 'linear', smoothing: edge.controlPoint ? 1.15 : 0 },
        opacity: 0.22,
        hoverinfo: 'skip',
        showlegend: false,
        type: 'scatter',
      }] : []),
      {
        x: pathX,
        y: pathY,
        mode: 'lines',
        line: { width: palette.flowWidth + (isSelected ? 1.6 : 0) + Math.min(activityBoost * 0.4, 1.6), color: edgeColor, dash: palette.flowDash, shape: edge.controlPoint ? 'spline' : 'linear', smoothing: edge.controlPoint ? 1.15 : 0 },
        customdata: pathPoints.map(() => `flow:${edge.id}`),
        hovertemplate: `<b>${edge.label}</b><br>${edge.description}<br>클릭하면 처리 데이터 상세 표시<extra></extra>`,
        showlegend: false,
        type: 'scatter',
      },
      {
        x: hitboxPoints.map((point) => point.x),
        y: hitboxPoints.map((point) => point.y),
        mode: 'markers',
        marker: {
          size: 20,
          color: 'rgba(0,0,0,0.003)',
          line: { width: 0, color: 'rgba(0,0,0,0)' },
        },
        customdata: hitboxPoints.map(() => `flow:${edge.id}`),
        hoverinfo: 'skip',
        showlegend: false,
        type: 'scatter',
      },
      {
        x: [pulsePoint.x],
        y: [pulsePoint.y],
        mode: 'markers',
        marker: {
          size: palette.flowPacketSize + activityBoost * 1.4 + (isSelected ? 2 : 0),
          color: isSelected ? palette.flowActive : activityBoost > 0 ? palette.flowPulse : packetColor,
          line: { width: isSelected ? 2.2 : 1.4, color: palette.flowPacketOutline },
          opacity: 0.92,
        },
        customdata: [`flow:${edge.id}`],
        hovertemplate: `<b>${edge.label}</b><br>${edge.description}<br>클릭하면 처리 데이터 상세 표시<extra></extra>`,
        showlegend: false,
        type: 'scatter',
      },
    ];
  })];
  traces.push({
    x: emphasizedNodes.map((node) => node.x),
    y: emphasizedNodes.map((node) => node.y),
    mode: 'markers',
    marker: {
      size: emphasizedNodes.map((node) => node.size + 18 + (selectedNodeIds.has(node.id) ? 8 : 0)),
      color: emphasizedNodes.map((node) => selectedNodeIds.has(node.id) ? palette.flowHaloStrong : palette.nodeGlow),
      symbol: emphasizedNodes.map((node) => palette.nodeSymbolMap[node.symbol] || node.symbol),
      opacity: 0.28,
      line: { width: 0, color: 'rgba(0,0,0,0)' },
    },
    hoverinfo: 'skip',
    showlegend: false,
    type: 'scatter',
  });
  traces.push({
    x: nodes.map((node) => node.x),
    y: nodes.map((node) => node.y),
    text: nodes.map((node) => node.label),
    customdata: nodes.map((node) => `node:${node.id}`),
    hovertext: nodes.map((node) => node.detail),
    mode: 'markers+text',
    textposition: 'bottom center',
    textfont: { color: palette.text, size: 12, family: APP_CHART_FONT },
    marker: {
      size: nodes.map((node) => node.size + (selectedNodeIds.has(node.id) ? 12 : 0)),
      color: nodes.map((node) => selectedNodeIds.has(node.id) ? palette.flowActive : (colorMap[node.status] || colorMap.pending)),
      symbol: nodes.map((node) => palette.nodeSymbolMap[node.symbol] || node.symbol),
      opacity: palette.nodeOpacity,
      line: {
        width: nodes.map((node) => selectedNodeIds.has(node.id) ? palette.nodeBorderWidth + 2 : palette.nodeBorderWidth),
        color: nodes.map((node) => selectedNodeIds.has(node.id) ? palette.flowActiveOutline : palette.nodeBorder),
      },
    },
    hovertemplate: '<b>%{text}</b><br>%{hovertext}<extra></extra>',
    showlegend: false,
    type: 'scatter',
  });
  return {
    data: traces,
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 18, r: 18, t: 18, b: 8 },
      clickmode: 'event',
      hovermode: 'closest',
      uirevision: 'agent-flow',
      xaxis: { visible: false, range: [-0.04, 1.12] },
      yaxis: { visible: false, range: [0, 1.02] },
      height: 560,
    },
  };
}


function buildPulseFigure(status) {
  const palette = arguments[1] || getPlotPalette('midnight');
  const rows = [
    { bucket: 'Vectors', value: Number(status.vector_count || 0), color: palette.cyan },
    { bucket: 'News', value: (status.news || []).length, color: palette.blue },
    { bucket: 'Issues', value: (status.issues || []).length, color: palette.red },
    { bucket: 'Events', value: (status.agent_activity_log || []).length, color: palette.amber },
  ];
  return {
    data: [{ type: 'bar', x: rows.map((item) => item.bucket), y: rows.map((item) => item.value), marker: { color: rows.map((item) => item.color), line: { color: palette.barLine, width: palette.barLineWidth } }, opacity: palette.barOpacity }],
    layout: { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: palette.text }, height: 320, margin: { l: 16, r: 16, t: 8, b: 20 }, showlegend: false, xaxis: { title: '' }, yaxis: { gridcolor: palette.gridStrong, title: '건수' } },
  };
}

function buildVectorFigures(events) {
  const palette = arguments[1] || getPlotPalette('midnight');
  const rows = [...(events || [])].slice(0, 20).reverse();
  if (!rows.length) {
    return { line: null, bar: null };
  }
  return {
    line: {
      data: [{ type: 'scatter', x: rows.map((item) => formatTime(item.timestamp)), y: rows.map((item) => item.after_count || 0), mode: 'lines+markers', line: { color: palette.cyan, width: palette.lineWidth, shape: palette.lineShape, smoothing: palette.lineSmoothing }, marker: { color: palette.amber, size: palette.smallMarkerSize, symbol: palette.markerSymbol, line: { color: palette.markerBorder, width: palette.markerBorderWidth } }, fill: palette.vectorFillMode, fillcolor: palette.vectorFillColor }],
      layout: { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: palette.text }, height: 260, margin: { l: 16, r: 16, t: 16, b: 16 }, xaxis: { title: '시간', gridcolor: palette.grid }, yaxis: { title: '누적 벡터 수', gridcolor: palette.gridStrong } },
    },
    bar: {
      data: [{ type: 'bar', x: rows.map((item) => formatTime(item.timestamp)), y: rows.map((item) => item.added_count || 0), marker: { color: palette.blue, line: { color: palette.barLine, width: palette.barLineWidth } }, opacity: palette.barOpacity }],
      layout: { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: palette.text }, height: 180, margin: { l: 16, r: 16, t: 16, b: 16 }, xaxis: { title: '시간', gridcolor: palette.grid }, yaxis: { title: '추가량', gridcolor: palette.gridStrong } },
    },
  };
}

function getPlotPalette(theme) {
  if (theme === 'cute') {
    return {
      text: '#5d4c73',
      cyan: '#5fd6d3',
      blue: '#8ba8ff',
      amber: '#ffb774',
      red: '#ff8ca8',
      green: '#7fd9a8',
      fill: 'rgba(95,214,211,0.18)',
      grid: 'rgba(194, 171, 214, 0.18)',
      gridStrong: 'rgba(194, 171, 214, 0.28)',
      line: 'rgba(163, 146, 210, 0.28)',
      nodeBorder: 'rgba(255, 246, 251, 0.95)',
      lineWidth: 4,
      lineShape: 'spline',
      lineSmoothing: 1.1,
      markerSize: 11,
      smallMarkerSize: 9,
      markerSymbol: 'circle',
      markerBorder: 'rgba(255,255,255,0.92)',
      markerBorderWidth: 2,
      barLine: 'rgba(255,255,255,0.86)',
      barLineWidth: 1.5,
      barOpacity: 0.94,
      pieHole: 0.68,
      pieTextMode: 'percent+label',
      pieLine: 'rgba(255,255,255,0.96)',
      pieLineWidth: 3,
      piePull: 0.03,
      flowWidth: 3.2,
      flowDash: 'dot',
      flowActive: '#ff8ca8',
      flowPulse: '#5fd6d3',
      flowHalo: 'rgba(255, 140, 168, 0.55)',
      flowHaloStrong: 'rgba(255, 140, 168, 0.72)',
      flowActiveOutline: 'rgba(255,255,255,0.96)',
      nodeGlow: 'rgba(255, 183, 116, 0.22)',
      flowPacket: '#ffffff',
      flowPacketMap: {
        log: AGENT_FLOW_PACKET_COLORS.log.cute,
        news: AGENT_FLOW_PACKET_COLORS.news.cute,
        regulation: AGENT_FLOW_PACKET_COLORS.regulation.cute,
        strategy: AGENT_FLOW_PACKET_COLORS.strategy.cute,
      },
      flowPacketOutline: 'rgba(255,255,255,0.92)',
      flowPacketSize: 10,
      nodeOpacity: 0.96,
      nodeBorderWidth: 3,
      nodeSymbolMap: { diamond: 'diamond', circle: 'circle', hexagon: 'hexagon', square: 'square' },
      vectorFillMode: 'tozeroy',
      vectorFillColor: 'rgba(139,168,255,0.12)',
    };
  }

  return {
    text: '#e7f4ff',
    cyan: '#61f4de',
    blue: '#8fb9d6',
    amber: '#ffbf69',
    red: '#ff6b6b',
    green: '#6ee7b7',
    fill: 'rgba(97,244,222,0.10)',
    grid: 'rgba(151,196,225,0.10)',
    gridStrong: 'rgba(151,196,225,0.12)',
    line: 'rgba(151,196,225,0.35)',
    nodeBorder: 'rgba(7,19,30,0.95)',
    lineWidth: 3,
    lineShape: 'linear',
    lineSmoothing: 0,
    markerSize: 9,
    smallMarkerSize: 8,
    markerSymbol: 'diamond',
    markerBorder: 'rgba(7,19,30,0.95)',
    markerBorderWidth: 1.5,
    barLine: 'rgba(151,196,225,0.18)',
    barLineWidth: 1,
    barOpacity: 0.9,
    pieHole: 0.58,
    pieTextMode: 'percent',
    pieLine: 'rgba(7,19,30,0.72)',
    pieLineWidth: 1,
    piePull: 0,
    flowWidth: 2.5,
    flowDash: 'solid',
    flowActive: '#ffbf69',
    flowPulse: '#61f4de',
    flowHalo: 'rgba(255, 191, 105, 0.38)',
    flowHaloStrong: 'rgba(255, 191, 105, 0.55)',
    flowActiveOutline: 'rgba(7,19,30,0.95)',
    nodeGlow: 'rgba(97, 244, 222, 0.18)',
    flowPacket: '#e7f4ff',
    flowPacketMap: {
      log: AGENT_FLOW_PACKET_COLORS.log.midnight,
      news: AGENT_FLOW_PACKET_COLORS.news.midnight,
      regulation: AGENT_FLOW_PACKET_COLORS.regulation.midnight,
      strategy: AGENT_FLOW_PACKET_COLORS.strategy.midnight,
    },
    flowPacketOutline: 'rgba(7,19,30,0.95)',
    flowPacketSize: 8,
    nodeOpacity: 1,
    nodeBorderWidth: 2,
    nodeSymbolMap: { diamond: 'diamond', circle: 'circle', hexagon: 'hexagon', square: 'square' },
    vectorFillMode: 'none',
    vectorFillColor: 'rgba(97,244,222,0)',
  };
}

function getChartBoxMotion(theme, prefersReducedMotion, index = 0) {
  if (prefersReducedMotion) {
    return {
      initial: false,
      animate: undefined,
      transition: undefined,
    };
  }

  if (theme === 'cute') {
    return {
      initial: { opacity: 0, y: 24, scale: 0.94, rotate: index % 2 === 0 ? -1.2 : 1.2 },
      animate: { opacity: 1, y: 0, scale: 1, rotate: 0 },
      transition: { duration: 0.56, delay: 0.08 * index, ease: [0.22, 1, 0.36, 1] },
    };
  }

  return {
    initial: { opacity: 0, y: 18, scale: 0.98 },
    animate: { opacity: 1, y: 0, scale: 1 },
    transition: { duration: 0.42, delay: 0.06 * index, ease: 'easeOut' },
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
      return Object.fromEntries(REVIEWER_PERSONAS.map((persona) => [persona.id, getDefaultReviewerTemplate(persona)]));
    }
    const parsed = JSON.parse(raw);
    return Object.fromEntries(
      REVIEWER_PERSONAS.map((persona) => {
        const storedPrompt = parsed?.[persona.id];
        if (!storedPrompt || storedPrompt === LEGACY_DEFAULT_PROMPTS[persona.id]) {
          return [persona.id, getDefaultReviewerTemplate(persona)];
        }
        if (!String(storedPrompt).includes('{{')) {
          return [persona.id, buildReviewerPromptTemplate(persona.id, String(storedPrompt))];
        }
        return [persona.id, storedPrompt];
      }),
    );
  } catch {
    return Object.fromEntries(REVIEWER_PERSONAS.map((persona) => [persona.id, getDefaultReviewerTemplate(persona)]));
  }
}

function saveStoredPrompts(prompts) {
  window.localStorage.setItem('reviewer_prompts', JSON.stringify(prompts));
}

function loadStoredReviewerSettings() {
  try {
    const raw = window.localStorage.getItem('reviewer_settings');
    const parsed = raw ? JSON.parse(raw) : {};
    return Object.fromEntries(
      REVIEWER_PERSONAS.map((persona) => {
        const defaults = buildDefaultReviewerSetting(persona.id);
        const stored = parsed?.[persona.id] || {};
        return [
          persona.id,
          {
            temperature: Number.isFinite(Number(stored.temperature)) ? Number(stored.temperature) : defaults.temperature,
            market_signal_feature_keys: String(stored.market_signal_feature_keys || defaults.market_signal_feature_keys || ''),
            placeholders: {
              ...defaults.placeholders,
              ...Object.fromEntries(
                Object.entries(stored.placeholders || {}).map(([key, value]) => [key, String(value || '')]),
              ),
            },
          },
        ];
      }),
    );
  } catch {
    return Object.fromEntries(REVIEWER_PERSONAS.map((persona) => [persona.id, buildDefaultReviewerSetting(persona.id)]));
  }
}

function saveStoredReviewerSettings(settings) {
  window.localStorage.setItem('reviewer_settings', JSON.stringify(settings));
}

function loadStoredTheme() {
  try {
    const raw = window.localStorage.getItem('app_theme');
    if (raw === 'midnight') {
      return 'cute';
    }
    return THEME_OPTIONS.some((item) => item.value === raw) ? raw : 'cute';
  } catch {
    return 'cute';
  }
}

function saveStoredTheme(theme) {
  window.localStorage.setItem('app_theme', theme);
}

function RegulationLearningBunny() {
  const { RiveComponent } = useRive({
    src: '/gbunny.riv',
    stateMachines: 'State Machine 1',
    autoplay: true,
  });

  return <RiveComponent className="regulation-learning-bunny-rive" aria-label="학습 중인 Bunny" />;
}

function getRegulationLearningStage(progress, busy) {
  if (!busy) {
    if (progress >= 100) {
      return '학습 완료';
    }
    return '학습 대기';
  }
  if (progress < 28) return '문서 읽는 중';
  if (progress < 55) return '청크 생성 중';
  if (progress < 82) return '벡터 적재 중';
  return '근거 인덱싱 중';
}

function mapDepartmentToTheme(departmentId) {
  switch (String(departmentId || '').toLowerCase()) {
    case 'credit':
      return 'mint';
    case 'sales':
      return 'lemon';
    case 'it':
      return 'orange';
    case 'solution':
    default:
      return 'cute';
  }
}

function App() {
  const prefersReducedMotion = useReducedMotion();
  const [status, setStatus] = useState(initialStatus);
  const [charts, setCharts] = useState({});
  const [selectedSection, setSelectedSection] = useState('온톨로지');
  const [selectedStore, setSelectedStore] = useState('');
  const [selectedVectorType, setSelectedVectorType] = useState('');
  const [vectorSummary, setVectorSummary] = useState({ items: [], total_count: 0 });
  const [clusterRelationState, setClusterRelationState] = useState({ loading: true, error: '', payload: null });
  const [selectedVectorEntry, setSelectedVectorEntry] = useState(null);
  const [vectorDetailBusy, setVectorDetailBusy] = useState(false);
  const [vectorSearchQuery, setVectorSearchQuery] = useState('');
  const [vectorSearchMatches, setVectorSearchMatches] = useState([]);
  const [vectorSearchBusy, setVectorSearchBusy] = useState(false);
  const [regulationFiles, setRegulationFiles] = useState([]);
  const [regulationBusy, setRegulationBusy] = useState(false);
  const [regulationLearningProgress, setRegulationLearningProgress] = useState(0);
  const [showRegulationUploadModal, setShowRegulationUploadModal] = useState(false);
  const [debateBusy, setDebateBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [reviewerPrompts, setReviewerPrompts] = useState(loadStoredPrompts);
  const [reviewerSettings, setReviewerSettings] = useState(loadStoredReviewerSettings);
  const [selectedReviewerId, setSelectedReviewerId] = useState(null);
  const [activeToast, setActiveToast] = useState(null);
  const [debateSessionActive, setDebateSessionActive] = useState(false);
  const [debateRunId, setDebateRunId] = useState(0);
  const [visibleDebateMessageIds, setVisibleDebateMessageIds] = useState([]);
  const [activeDebateMessageId, setActiveDebateMessageId] = useState(null);
  const [theme, setTheme] = useState(loadStoredTheme);
  const [themeMenuOpen, setThemeMenuOpen] = useState(false);
  const [agentOllamaToggleBusy, setAgentOllamaToggleBusy] = useState({ news: false, log: false, gpu: false, ontology: false, regulationSummary: false });
  const [agentFlowTick, setAgentFlowTick] = useState(0);
  const [selectedGraphTarget, setSelectedGraphTarget] = useState({ kind: 'flow', id: AGENT_FLOW_EDGES[0].id });
  const [agentFlowPanelOpen, setAgentFlowPanelOpen] = useState(false);
  const [flowActivityMap, setFlowActivityMap] = useState({});
  const lastRuntimeSignatureRef = useRef('');
  const lastAgentFlowSnapshotRef = useRef(null);
  const regulationInputRef = useRef(null);
  const debateChatRoomRef = useRef(null);
  const debatePlaybackTimerRef = useRef(null);
  const themeMenuRef = useRef(null);

  useEffect(() => {
    saveStoredPrompts(reviewerPrompts);
  }, [reviewerPrompts]);

  useEffect(() => {
    saveStoredReviewerSettings(reviewerSettings);
  }, [reviewerSettings]);

  useEffect(() => {
    saveStoredTheme(theme);
    document.documentElement.dataset.theme = theme;
    return () => {
      delete document.documentElement.dataset.theme;
    };
  }, [theme]);

  useEffect(() => {
    if (!themeMenuOpen) {
      return undefined;
    }

    const handlePointerDown = (event) => {
      if (!themeMenuRef.current?.contains(event.target)) {
        setThemeMenuOpen(false);
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, [themeMenuOpen]);

  useEffect(() => () => {
    if (debatePlaybackTimerRef.current) {
      window.clearTimeout(debatePlaybackTimerRef.current);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    async function bootstrap() {
      try {
        const [health, chartPayload] = await Promise.all([fetchHealth(), fetchCharts()]);
        if (ignore) {
          return;
        }
        setStatus((previous) => ({
          ...previous,
          ...health,
          cardloan_debate: debateSessionActive ? (health.cardloan_debate || previous.cardloan_debate) : previous.cardloan_debate,
          ollama_runtime: debateSessionActive ? (health.ollama_runtime || previous.ollama_runtime) : previous.ollama_runtime,
        }));
        setCharts(chartPayload?.charts || {});
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
  }, [debateSessionActive]);

  useEffect(() => {
    let cancelled = false;
    async function loadVectorSummary() {
      try {
        const payload = await fetchFaissEntriesByStore(120, selectedStore, selectedVectorType);
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
  }, [selectedSection, selectedStore, selectedVectorType, status.last_faiss_time]);

  useEffect(() => {
    if (selectedSection !== 'Vector DB') {
      return undefined;
    }
    let cancelled = false;
    async function loadClusterRelations() {
      try {
        setClusterRelationState((previous) => ({ ...previous, loading: true, error: '' }));
        const payload = await fetchFeatureOntologyClusters({ product: '', limit: 48 });
        if (!cancelled) {
          setClusterRelationState({ loading: false, error: '', payload });
        }
      } catch (error) {
        if (!cancelled) {
          setClusterRelationState({ loading: false, error: String(error.message || error), payload: null });
        }
      }
    }
    loadClusterRelations();
    return () => {
      cancelled = true;
    };
  }, [selectedSection]);

  useEffect(() => {
    const socket = createFaissWebSocket((payload) => {
      if (payload?.snapshot) {
        setStatus((previous) => ({
          ...previous,
          ...payload.snapshot,
          cardloan_debate: debateSessionActive ? (payload.snapshot.cardloan_debate || previous.cardloan_debate) : previous.cardloan_debate,
          ollama_runtime: debateSessionActive ? (payload.snapshot.ollama_runtime || previous.ollama_runtime) : previous.ollama_runtime,
        }));
      }
      if (payload?.event) {
        setStatus((previous) => ({ ...previous, vector_events: [payload.event, ...(previous.vector_events || [])].slice(0, 40) }));
      }
    });
    socket.onopen = () => setErrorMessage('');
    socket.onerror = () => setErrorMessage('실시간 상태 연결이 잠시 끊겼습니다.');
    return () => socket.close();
  }, [debateSessionActive]);

  useEffect(() => {
    if (prefersReducedMotion) {
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      setAgentFlowTick((previous) => (previous + 1) % 1000);
    }, 140);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [prefersReducedMotion]);

  useEffect(() => {
    if (prefersReducedMotion) {
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      setFlowActivityMap((previous) => {
        const nextEntries = Object.entries(previous)
          .map(([key, value]) => [key, Math.max(0, Number(value || 0) - 1)])
          .filter(([, value]) => value > 0);
        return nextEntries.length ? Object.fromEntries(nextEntries) : {};
      });
    }, 520);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [prefersReducedMotion]);

  useEffect(() => {
    const nextSnapshot = {
      resultsCount: (status.results || []).length,
      newsCount: (status.news || []).length,
      issuesCount: (status.issues || []).length,
      vectorCount: Number(status.vector_count || 0),
      lastVectorTime: String(status.last_faiss_time || (status.vector_events || [])[0]?.timestamp || ''),
      regulation: String(status.latest_regulation_analysis || ''),
      strategy: String(status.latest_strategy_question || ''),
    };

    const previousSnapshot = lastAgentFlowSnapshotRef.current;
    lastAgentFlowSnapshotRef.current = nextSnapshot;
    if (!previousSnapshot) {
      return;
    }

    const boosts = {};
    if (nextSnapshot.resultsCount !== previousSnapshot.resultsCount) {
      boosts.source_logs__log_ingestor = 4;
      boosts.log_ingestor__log_analyzer = 4;
      boosts.log_analyzer__log_agent = 4;
      boosts.log_agent__regulation_agent = 3;
    }
    if (nextSnapshot.newsCount !== previousSnapshot.newsCount) {
      boosts.source_news__news_collector = 4;
      boosts.news_collector__news_agent = 4;
      boosts.news_agent__regulation_agent = 3;
    }
    if (
      nextSnapshot.issuesCount !== previousSnapshot.issuesCount
      || nextSnapshot.regulation !== previousSnapshot.regulation
    ) {
      boosts.regulation_agent__orchestrator = 4;
    }
    if (
      nextSnapshot.vectorCount !== previousSnapshot.vectorCount
      || nextSnapshot.lastVectorTime !== previousSnapshot.lastVectorTime
      || nextSnapshot.strategy !== previousSnapshot.strategy
    ) {
      boosts.orchestrator__vector_store = 4;
    }

    const boostIds = Object.keys(boosts);
    if (!boostIds.length) {
      return;
    }

    setFlowActivityMap((previous) => {
      const next = { ...previous };
      boostIds.forEach((id) => {
        next[id] = Math.max(Number(next[id] || 0), boosts[id]);
      });
      return next;
    });
  }, [status.results, status.news, status.issues, status.vector_count, status.last_faiss_time, status.latest_regulation_analysis, status.latest_strategy_question, status.vector_events]);

  useEffect(() => {
    const runtime = status.ollama_runtime || {};
    const runtimeStatus = String(runtime.status || '').trim();
    if (!runtimeStatus) {
      return;
    }

    let signature = '';
    let nextToast = null;

    if (runtimeStatus === 'running' && runtime.started_at) {
      signature = `running:${runtime.agent || '-'}:${runtime.started_at}`;
      nextToast = {
        id: signature,
        tone: 'running',
        kicker: 'OLLAMA START',
        title: `${getOllamaAgentLabel(runtime.agent)} 질의 시작`,
        meta: `${formatOllamaModel(runtime.model)} · 실행 중`,
        message: truncate(runtime.prompt || '프롬프트를 생성하고 응답을 요청했습니다.', 120),
      };
    } else if (runtimeStatus === 'completed' && runtime.completed_at) {
      signature = `completed:${runtime.agent || '-'}:${runtime.completed_at}`;
      nextToast = {
        id: signature,
        tone: 'completed',
        kicker: 'OLLAMA DONE',
        title: `${getOllamaAgentLabel(runtime.agent)} 질의 완료`,
        meta: `${formatOllamaModel(runtime.model)} · 응답 생성 완료`,
        message: truncate(runtime.response_text || '응답 생성이 완료되었습니다.', 120),
      };
    } else if (runtimeStatus === 'failed' && runtime.completed_at) {
      signature = `failed:${runtime.agent || '-'}:${runtime.completed_at}`;
      nextToast = {
        id: signature,
        tone: 'failed',
        kicker: 'OLLAMA FAILED',
        title: `${getOllamaAgentLabel(runtime.agent)} 질의 실패`,
        meta: `${formatOllamaModel(runtime.model)} · 오류 발생`,
        message: truncate(runtime.error || '응답 생성 중 오류가 발생했습니다.', 120),
      };
    }

    if (!signature || signature === lastRuntimeSignatureRef.current || !nextToast) {
      return;
    }

    lastRuntimeSignatureRef.current = signature;
    setActiveToast(nextToast);
  }, [status.ollama_runtime]);

  useEffect(() => {
    if (!regulationBusy) {
      setRegulationLearningProgress((previous) => {
        if (previous >= 100) return 100;
        return previous > 0 ? 100 : 0;
      });
      return undefined;
    }

    setRegulationLearningProgress((previous) => (previous <= 2 ? 8 : previous));
    const timerId = window.setInterval(() => {
      setRegulationLearningProgress((previous) => {
        if (previous >= 94) {
          return 92 + Math.floor(Math.random() * 3);
        }
        return Math.min(94, previous + 3 + Math.floor(Math.random() * 4));
      });
    }, 900);

    return () => window.clearInterval(timerId);
  }, [regulationBusy]);

  async function handleToggleAgentOllama(agentKey, enabled) {
    const isNewsAgent = agentKey === 'news';
    const updater = isNewsAgent ? setNewsAgentOllamaEnabled : setLogAgentOllamaEnabled;
    const statusKey = isNewsAgent ? 'news_agent_ollama_enabled' : 'log_agent_ollama_enabled';
    const busyKey = isNewsAgent ? 'news' : 'log';

    setAgentOllamaToggleBusy((previous) => ({ ...previous, [busyKey]: true }));
    setStatus((previous) => ({ ...previous, [statusKey]: Boolean(enabled) }));

    try {
      await updater(enabled);
      setActiveToast({
        id: `agent-ollama-${agentKey}-${Date.now()}`,
        tone: enabled ? 'completed' : 'running',
        kicker: enabled ? 'OLLAMA ENABLED' : 'OLLAMA DISABLED',
        title: `${isNewsAgent ? '뉴스 에이전트' : '로그 에이전트'} ${enabled ? '호출 허용' : '호출 차단'}`,
        meta: enabled ? 'server setting synced' : 'fallback briefing mode',
        message: enabled
          ? '다음 주기 실행부터 Ollama를 다시 호출합니다.'
          : '다음 주기 실행부터 Ollama 대신 fallback 브리핑을 생성합니다.',
      });
    } catch (error) {
      setStatus((previous) => ({ ...previous, [statusKey]: !Boolean(enabled) }));
      setErrorMessage(String(error.message || error));
    } finally {
      setAgentOllamaToggleBusy((previous) => ({ ...previous, [busyKey]: false }));
    }
  }

  async function handleToggleOllamaRuntime(settingKey, enabled) {
    const isGpuSetting = settingKey === 'gpu';
    const updater = isGpuSetting ? setOllamaGpuEnabled : setOntologyQueryPriorityEnabled;
    const statusKey = isGpuSetting ? 'ollama_gpu_enabled' : 'ontology_query_priority_enabled';
    const busyKey = isGpuSetting ? 'gpu' : 'ontology';

    setAgentOllamaToggleBusy((previous) => ({ ...previous, [busyKey]: true }));
    setStatus((previous) => ({ ...previous, [statusKey]: Boolean(enabled) }));

    try {
      await updater(enabled);
      setActiveToast({
        id: `ollama-runtime-${settingKey}-${Date.now()}`,
        tone: enabled ? 'completed' : 'running',
        kicker: isGpuSetting ? 'OLLAMA GPU' : 'ONTOLOGY PRIORITY',
        title: isGpuSetting
          ? `Ollama GPU ${enabled ? '사용' : '해제'}`
          : `온톨로지 질의 ${enabled ? '최우선 처리' : '일반 처리'}`,
        meta: isGpuSetting
          ? (enabled ? 'server runtime uses ollama default GPU path' : 'force CPU mode')
          : (enabled ? 'non-ontology ollama calls will yield first' : 'shared lock fairness'),
        message: isGpuSetting
          ? (enabled ? '다음 Ollama 호출부터 GPU 사용 가능 경로로 전환합니다.' : '다음 Ollama 호출부터 CPU 우선 모드로 되돌립니다.')
          : (enabled ? '온톨로지 질의가 들어오면 다른 Ollama 작업보다 먼저 처리되도록 조정합니다.' : '온톨로지 질의를 일반 Ollama 작업과 동일 우선순위로 처리합니다.'),
      });
    } catch (error) {
      setStatus((previous) => ({ ...previous, [statusKey]: !Boolean(enabled) }));
      setErrorMessage(String(error.message || error));
    } finally {
      setAgentOllamaToggleBusy((previous) => ({ ...previous, [busyKey]: false }));
    }
  }

  async function handleToggleRegulationUploadSummary(enabled) {
    setAgentOllamaToggleBusy((previous) => ({ ...previous, regulationSummary: true }));
    setStatus((previous) => ({ ...previous, regulation_upload_summary_enabled: Boolean(enabled) }));
    try {
      await setRegulationUploadSummaryEnabled(enabled);
      setActiveToast({
        id: `regulation-summary-toggle-${Date.now()}`,
        tone: enabled ? 'completed' : 'running',
        kicker: 'REGULATION UPLOAD',
        title: enabled ? '업로드 요약 생성 켜짐' : '업로드 요약 생성 꺼짐',
        meta: enabled ? '문서 업로드 후 요약 생성 포함' : '문서 업로드 시 요약 단계 생략',
        message: enabled ? '다음 업로드부터 규제 요약을 생성합니다.' : '다음 업로드부터 요약 생성 없이 벡터 적재만 수행합니다.',
      });
    } catch (error) {
      setStatus((previous) => ({ ...previous, regulation_upload_summary_enabled: !Boolean(enabled) }));
      setErrorMessage(String(error.message || error));
    } finally {
      setAgentOllamaToggleBusy((previous) => ({ ...previous, regulationSummary: false }));
    }
  }

  useEffect(() => {
    if (!activeToast) {
      return undefined;
    }
    const timeoutId = window.setTimeout(() => {
      setActiveToast((previous) => (previous?.id === activeToast.id ? null : previous));
    }, 4200);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [activeToast]);

  async function handleOpenVectorDetail(item) {
    const docId = item?.id;
    if (!docId) {
      return;
    }
    setSelectedVectorEntry({
      id: docId,
      metadata: {
        name: item?.name || '',
        store: item?.store || selectedStore || '',
        type: item?.type || '',
        product: item?.product || '',
        source: item?.source || '',
      },
      page_content: String(item?.snippet || ''),
      loading: true,
      error: '',
    });
    try {
      setVectorDetailBusy(true);
      const payload = await fetchFaissEntry(docId);
      setSelectedVectorEntry({
        id: payload?.id || docId,
        metadata: payload?.metadata || {},
        page_content: String(payload?.page_content || ''),
        loading: false,
        error: '',
      });
      setErrorMessage('');
    } catch (error) {
      const message = String(error.message || error);
      setSelectedVectorEntry((previous) => (previous && previous.id === docId ? {
        ...previous,
        loading: false,
        error: message,
      } : previous));
      setErrorMessage(message);
    } finally {
      setVectorDetailBusy(false);
    }
  }

  async function handleRegulationUpload(files = regulationFiles) {
    const filesToUpload = Array.from(files || []);
    if (!filesToUpload.length) {
      return;
    }
    try {
      setRegulationBusy(true);
      setShowRegulationUploadModal(true);
      setRegulationFiles(filesToUpload);
      const result = await uploadRegulationFiles(filesToUpload);
      setStatus((previous) => ({
        ...previous,
        vector_count: result.vector_count,
        latest_regulation_analysis: result.summary,
        regulation_files: Array.isArray(result.files) ? result.files : (previous.regulation_files || []),
        regulation_file_stats: Array.isArray(result.file_stats) ? result.file_stats : (previous.regulation_file_stats || []),
        agent_statuses: {
          ...(previous.agent_statuses || {}),
          regulation_agent: { status: 'completed', detail: result.detail, updated_at: result.updated_at },
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

  function handleRegulationFileSelection(event) {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length || regulationBusy) {
      return;
    }
    handleRegulationUpload(selectedFiles);
    event.target.value = '';
  }

  function openRegulationFilePicker() {
    if (regulationBusy) {
      return;
    }
    regulationInputRef.current?.click();
  }

  function openRegulationUploadModal() {
    setShowRegulationUploadModal(true);
  }

  function handleAddRegulationFiles() {
    openRegulationFilePicker();
  }

  async function handleStartDebate() {
    try {
      setDebateBusy(true);
      setDebateSessionActive(true);
      setDebateRunId((previous) => previous + 1);
      setStatus((previous) => ({
        ...previous,
        cardloan_debate: {
          status: 'running',
          current_stage: '신용기획부',
          question: DEFAULT_QUESTION,
          round_results: [],
          summary: '토론실에 입장해서 순서대로 의견을 정리하고 있습니다.',
          timings: {},
        },
        ollama_runtime: {},
      }));
      const result = await startCardloanDebate(DEFAULT_QUESTION, reviewerPrompts, reviewerSettings);
      setStatus((previous) => ({ ...previous, cardloan_debate: result }));
      setErrorMessage('');
    } catch (error) {
      setErrorMessage(String(error.message || error));
    } finally {
      setDebateBusy(false);
    }
  }

  const selectedReviewer = REVIEWER_PERSONAS.find((persona) => persona.id === selectedReviewerId) || null;
  const selectedReviewerSetting = selectedReviewer ? (reviewerSettings[selectedReviewer.id] || buildDefaultReviewerSetting(selectedReviewer.id)) : null;

  const plotPalette = getPlotPalette(theme);
  const currentThemeOption = THEME_OPTIONS.find((item) => item.value === theme) || THEME_OPTIONS[0];
  const selectedFlowId = selectedGraphTarget.kind === 'flow' ? selectedGraphTarget.id : '';
  const selectedNodeId = selectedGraphTarget.kind === 'node' ? selectedGraphTarget.id : '';
  const agentFlowFigure = buildAgentFlowFigure(status, plotPalette, agentFlowTick, selectedFlowId, flowActivityMap, selectedNodeId);
  const selectedFlowDetail = selectedGraphTarget.kind === 'node'
    ? buildAgentNodeDetail(selectedGraphTarget.id, status)
    : buildAgentFlowDetail(selectedGraphTarget.id, status);
  const vectorFigures = buildVectorFigures(status.vector_events || [], plotPalette);
  const isDebateSection = selectedSection === 'AI 카드론 토론실';
  const isOntologySection = selectedSection === '온톨로지';
  const isImmersiveSection = isDebateSection || isOntologySection;
  const selectedStoreLabel = STORE_OPTIONS.find((item) => item.value === selectedStore)?.label || '전체 DB';
  const summaryItems = vectorSummary.items || [];
  const selectedVectorTypeLabel = VECTOR_TYPE_OPTIONS.find((item) => item.value === selectedVectorType)?.label || '전체 type';
  const clusterRelationModel = buildClusterRelationModel(clusterRelationState.payload);
  const vectorSceneInsights = buildVectorSceneInsights(
    summaryItems,
    vectorSummary.total_count || status.vector_count || summaryItems.length,
  );
  const normalizedVectorSearchQuery = String(vectorSearchQuery || '').trim();
  const recentVectorItems = (
    normalizedVectorSearchQuery && vectorSearchMatches.length
      ? vectorSearchMatches
      : summaryItems
  ).slice(0, 24);
  const effectiveNewsItems = (status.news && status.news.length)
    ? status.news
    : (status.recent_news_fallback || []);
  const latestNewsItem = effectiveNewsItems.find((item) => String(item?.content || item?.summary || '').trim()) || effectiveNewsItems[0] || null;
  const latestNewsSignal = latestNewsItem
    ? {
      id: `${latestNewsItem.link || latestNewsItem.title || 'news'}::${status.last_new_item_time || ''}`,
      title: String(latestNewsItem.title || '').trim(),
      summary: truncate(latestNewsItem.summary || latestNewsItem.content || '카드론 심사 연관 뉴스를 탐지했습니다.', 180),
      time: status.last_new_item_time || status.last_news_time || latestNewsItem.published_at || latestNewsItem.collected_at || '',
    }
    : null;
  const latestNewsItems = (effectiveNewsItems || [])
    .filter((item) => String(item?.title || item?.summary || item?.content || '').trim())
    .slice(0, 6)
    .map((item, index) => ({
      id: `${item.link || item.title || 'news'}-${index}`,
      title: truncate(item.title || `뉴스 ${index + 1}`, 72),
      summary: truncate(item.summary || item.content || '요약 정보가 없습니다.', 120),
      time: item.published_at || item.collected_at || status.last_news_time || '',
      rawTitle: String(item.title || '').trim(),
      rawSummary: String(item.summary || item.content || '').trim(),
      link: String(item.link || '').trim(),
      publisher: String(item.publisher || '').trim(),
      publishedAt: String(item.published_at || item.published || item.collected_at || '').trim(),
    }));
  const visibleDebate = debateSessionActive
    ? (status.cardloan_debate || { status: 'idle', round_results: [] })
    : { status: 'idle', round_results: [], current_stage: null, summary: '', timings: {}, priority_active: false };
  const visibleOllamaRuntime = debateSessionActive ? (status.ollama_runtime || {}) : {};
  const debateTimings = visibleDebate?.timings || {};
  const ontologyOperationsSummary = {
    metrics: [
      { label: '로그', value: `${(status.results || []).length}건` },
      { label: '뉴스', value: `${(status.news || []).length}건` },
      { label: '이슈', value: `${(status.issues || []).length}건` },
      { label: '벡터', value: `${Number(status.vector_count || 0)}건` },
    ],
    briefs: [
      { title: '로그 브리핑', detail: truncate(status.latest_log_briefing || '아직 로그 브리핑이 없습니다.', 120) },
      { title: '뉴스 브리핑', detail: truncate(status.latest_news_briefing || '아직 뉴스 브리핑이 없습니다.', 120) },
      { title: '규제 요약', detail: truncate(status.latest_regulation_analysis || '아직 규제 요약이 없습니다.', 120) },
    ],
    timeline: (status.agent_activity_log || []).slice(0, 4).map((item, index) => ({
      id: `${item.timestamp || 'activity'}-${index}`,
      title: item.source || 'runtime',
      detail: truncate(item.detail || '최근 처리 이벤트가 없습니다.', 110),
      time: formatTime(item.timestamp),
    })),
  };
  ontologyOperationsSummary.newsHealth = {
    newsAgentOllamaEnabled: Boolean(status.news_agent_ollama_enabled),
    newsCount: Number((effectiveNewsItems || []).length || 0),
    lastNewsTime: String(status.last_news_time || ''),
    crawlRunning: Boolean(status.news_crawl_running),
    crawlSuccessCount: Number(status.news_crawl_success_count || 0),
    crawlFailureCount: Number(status.news_crawl_failure_count || 0),
    lastNewsBriefingTime: String(status.last_news_briefing_time || ''),
    latestNewsBriefing: String(status.latest_news_briefing || ''),
  };
  const ontologyDebateSummary = {
    status: visibleDebate?.status || 'idle',
    currentStage: visibleDebate?.current_stage || '대기',
    summary: truncate(visibleDebate?.summary || '토론을 시작하면 세 부서 의견이 이 패널에 요약됩니다.', 140),
    priorityActive: Boolean(visibleDebate?.priority_active),
    timings: debateTimings,
    roundResults: (visibleDebate?.round_results || []).slice(0, 4).map((item, index) => ({
      id: `${item.persona_id || 'round'}-${index}`,
      title: item.name || item.persona_id || `단계 ${index + 1}`,
      detail: truncate(item.preview || item.summary || item.raw_text || '아직 요약이 없습니다.', 100),
      time: item.completed_at ? formatTime(item.completed_at) : '진행 중',
    })),
  };
  const uploadedRegulationFiles = Array.isArray(status.regulation_files) ? status.regulation_files : [];
  const uploadedRegulationFileStats = Array.isArray(status.regulation_file_stats) ? status.regulation_file_stats : [];
  const regulationAgentStatus = String(status.agent_statuses?.regulation_agent?.status || (regulationBusy ? 'running' : 'pending'));
  const normalizedRegulationProgress = regulationBusy ? Math.min(regulationLearningProgress, 96) : (regulationLearningProgress > 0 ? 100 : 0);
  const regulationLearningStage = getRegulationLearningStage(normalizedRegulationProgress, regulationBusy);
  const debateChatMessages = buildDebateChatMessages({ ...status, cardloan_debate: visibleDebate, ollama_runtime: visibleOllamaRuntime }, debateRunId);
  const sortedDebateMessageIds = debateChatMessages
    .map((message, index) => ({ id: message.id, priority: Number(message.priority || 0), index }))
    .sort((left, right) => {
      const priorityGap = left.priority - right.priority;
      if (priorityGap !== 0) {
        return priorityGap;
      }
      return left.index - right.index;
    })
    .map((item) => item.id);
  const activeDebateMessage = activeDebateMessageId
    ? (debateChatMessages.find((message) => message.id === activeDebateMessageId) || null)
    : null;
  const activeDebateMessageSignature = activeDebateMessage
    ? [
      activeDebateMessage.id,
      activeDebateMessage.type,
      activeDebateMessage.text,
      activeDebateMessage.referenceSummary,
      activeDebateMessage.meta,
    ].join('::')
    : '';
  const activeDebatePlaybackMs = activeDebateMessage ? getDebateMessagePlaybackMs(activeDebateMessage, prefersReducedMotion) : 0;
  const visibleDebateMessageIdSet = new Set(visibleDebateMessageIds);
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

  const liftCard = (transform = { y: -6 }) => (prefersReducedMotion ? undefined : transform);

  function handleAgentFlowClick(event) {
    const rawPayload = event?.points?.[0]?.customdata;
    const payload = Array.isArray(rawPayload) ? rawPayload[0] : rawPayload;
    if (typeof payload !== 'string' || !payload.includes(':')) {
      return;
    }
    const [kind, nextId] = payload.split(':');
    if (kind === 'flow' && AGENT_FLOW_EDGES.some((edge) => edge.id === nextId)) {
      setSelectedGraphTarget({ kind: 'flow', id: nextId });
      setFlowActivityMap((previous) => ({
        ...previous,
        [nextId]: Math.max(Number(previous[nextId] || 0), 5),
      }));
    } else if (kind === 'node') {
      if (nextId === 'vector_store') {
        setSelectedGraphTarget({ kind: 'node', id: nextId });
        setAgentFlowPanelOpen(false);
        setSelectedSection('Vector DB');
        return;
      }
      setSelectedGraphTarget({ kind: 'node', id: nextId });
    } else {
      return;
    }
    setAgentFlowPanelOpen(true);
  }

  async function handleVectorSearchSubmit(event) {
    event?.preventDefault?.();
    const normalizedQuery = String(vectorSearchQuery || '').trim();
    if (!normalizedQuery) {
      setVectorSearchMatches([]);
      return;
    }

    try {
      setVectorSearchBusy(true);
      const payload = await fetchSimilarLogVectors(normalizedQuery, 8);
      const nextMatches = Array.isArray(payload?.items) ? payload.items : [];
      setVectorSearchMatches(nextMatches);
      setErrorMessage('');
    } catch (error) {
      setVectorSearchMatches([]);
      setErrorMessage(String(error.message || error));
    } finally {
      setVectorSearchBusy(false);
    }
  }

  function handleClearVectorSearch() {
    setVectorSearchQuery('');
    setVectorSearchMatches([]);
  }

  useEffect(() => {
    if (debatePlaybackTimerRef.current) {
      window.clearTimeout(debatePlaybackTimerRef.current);
    }

    setActiveDebateMessageId(null);

    if (!debateSessionActive) {
      setVisibleDebateMessageIds(debateChatMessages.map((message) => message.id));
      return;
    }

    if (prefersReducedMotion) {
      setVisibleDebateMessageIds(debateChatMessages.map((message) => message.id));
      return;
    }

    const immediateMessageIds = sortedDebateMessageIds.slice(0, 2);
    const [firstMessageId] = immediateMessageIds;
    setVisibleDebateMessageIds(immediateMessageIds);
    setActiveDebateMessageId(firstMessageId || null);
  }, [debateRunId, debateSessionActive, prefersReducedMotion]);

  useEffect(() => {
    if (!debateSessionActive || prefersReducedMotion) {
      return undefined;
    }

    if (activeDebateMessageId) {
      return undefined;
    }

    const nextId = sortedDebateMessageIds.find((messageId) => !visibleDebateMessageIds.includes(messageId));
    const nextMessage = debateChatMessages.find((message) => message.id === nextId);
    if (!nextMessage) {
      return undefined;
    }

    setVisibleDebateMessageIds((previous) => (previous.includes(nextId) ? previous : [...previous, nextId]));
    setActiveDebateMessageId(nextId);
    return undefined;
  }, [debateChatMessages, debateSessionActive, prefersReducedMotion, visibleDebateMessageIds, activeDebateMessageId, sortedDebateMessageIds]);

  useEffect(() => {
    if (!activeDebateMessageId || prefersReducedMotion) {
      return undefined;
    }

    if (!activeDebateMessage) {
      return undefined;
    }

    debatePlaybackTimerRef.current = window.setTimeout(() => {
      setActiveDebateMessageId((current) => (current === activeDebateMessageId ? null : current));
    }, activeDebatePlaybackMs);

    return () => {
      if (debatePlaybackTimerRef.current) {
        window.clearTimeout(debatePlaybackTimerRef.current);
      }
    };
  }, [activeDebateMessageId, activeDebateMessageSignature, activeDebatePlaybackMs, prefersReducedMotion]);

  useEffect(() => {
    const room = debateChatRoomRef.current;
    if (!room) {
      return;
    }
    room.scrollTo({ top: room.scrollHeight, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
  }, [visibleDebateMessageIds, activeDebateMessageId, prefersReducedMotion]);

  const renderContent = () => {
    if (selectedSection === '운영 현황') {
      return (
        <motion.section key="operations" className="content-stack" {...sectionTransition}>
          <MotionCard className="panel operations-flow-panel">
            <div className="panel-head">
              <div>
                <div className="panel-kicker">Graph View</div>
                <h2>Agent 간 데이터 흐름 시각화</h2>
              </div>
              <p>로그, 뉴스, 규제 판단 이후 오케스트레이터가 세 부서 에이전트와 벡터 적재까지 어떻게 제어하는지 한 화면에 묶었습니다.</p>
            </div>
            <div className={`agent-flow-stage ${agentFlowPanelOpen ? 'panel-open' : ''}`}>
              <div className="agent-flow-plot-shell">
                <LazyPlot data={agentFlowFigure.data} layout={agentFlowFigure.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} loadingHeight={560} onClick={handleAgentFlowClick} />
              </div>
              <AnimatePresence>
                {agentFlowPanelOpen ? (
                  <motion.aside
                    className="agent-flow-side-panel"
                    initial={{ opacity: 0, x: 28 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 28 }}
                    transition={{ duration: 0.26, ease: 'easeOut' }}
                  >
                    <div className="panel-head panel-head-spread">
                      <div>
                        <div className="panel-kicker">{selectedFlowDetail.kicker}</div>
                        <h2>{selectedFlowDetail.title}</h2>
                      </div>
                      <button className="agent-flow-close" type="button" onClick={() => setAgentFlowPanelOpen(false)}>닫기</button>
                    </div>
                    <p className="long-copy">{selectedFlowDetail.preview}</p>
                    <div className="detail-meta-row">
                      {selectedFlowDetail.metrics.map((item) => <span className="sample-pill" key={item}>{item}</span>)}
                    </div>
                    <div className="summary-box agent-flow-side-summary">
                      <div className="summary-box-title">처리 데이터 샘플</div>
                      <p>{selectedFlowDetail.description}</p>
                      {selectedFlowDetail.samples.length ? (
                        <div className="table-shell compact-table">
                          <table>
                            <tbody>
                              {selectedFlowDetail.samples.map((item, index) => (
                                <tr key={`${selectedFlowDetail.title}-${index}`}>
                                  <td>{index + 1}</td>
                                  <td>{truncate(item, 140)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="empty-box">표시할 최근 처리 데이터가 없습니다.</div>
                      )}
                    </div>
                  </motion.aside>
                ) : null}
              </AnimatePresence>
            </div>
          </MotionCard>
        </motion.section>
      );
    }

    if (selectedSection === 'AI 카드론 토론실') {
      return (
        <motion.section key="debate" className="debate-fullscreen-shell" {...sectionTransition}>
          <motion.div className="debate-fullscreen-stage" initial={{ opacity: 0, scale: 0.985 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.99 }} transition={{ duration: 0.28, ease: 'easeOut' }}>
          <MotionCard className="debate-hero">
            <div>
              <div className="panel-kicker">Cardloan Strategy Room</div>
              <h2>AI 카드론 토론실</h2>
              <p>신용기획부가 리스크 정책을 먼저 정리하고, 금융영업부가 승인 전환 전략을 만들고, 금융솔루션부가 상품 구조를 설계합니다.</p>
            </div>
            <AnimatedSignalWave className="debate-wave" />
            <button className="primary-button" type="button" disabled={debateBusy || visibleDebate?.status === 'running'} onClick={handleStartDebate}>{debateBusy || visibleDebate?.status === 'running' ? '토론 진행 중' : '토론시작'}</button>
          </MotionCard>
          <motion.div className="reviewer-grid" variants={containerVariants} initial="hidden" animate="show">
            {REVIEWER_PERSONAS.map((persona) => {
              const result = (visibleDebate?.round_results || []).find((item) => item.persona_id === persona.id) || {};
              const info = status.agent_statuses?.[persona.id] || {};
              const runtime = visibleOllamaRuntime || {};
              const isSpeaking = debateSessionActive && runtime.agent === persona.id && runtime.status === 'running';
              const palette = getStatusPalette(info.status || (result.name ? 'completed' : 'pending'));
              return (
                <MotionCard
                  className={`reviewer-card ${isSpeaking ? 'speaking' : ''} ${palette.tone}`}
                  as="article"
                  key={persona.id}
                  style={{ '--persona-accent': persona.accent }}
                  whileHover={liftCard({ y: -8, rotateX: -2, rotateY: 1 })}
                  onClick={() => setSelectedReviewerId(persona.id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setSelectedReviewerId(persona.id);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <div className="reviewer-card-glow" />
                  <div className="reviewer-topline">{persona.emoji} Reviewer</div>
                  <div className="reviewer-visual-row">
                    <div className="reviewer-avatar-shell">
                      <div className="reviewer-avatar-illustration" aria-hidden="true">
                        <span className="reviewer-avatar-halo" />
                        <span className="reviewer-avatar-badge">{persona.emoji}</span>
                        <span className="reviewer-avatar-monitor">
                          <span />
                          <span />
                          <span />
                        </span>
                      </div>
                      {isSpeaking ? <span className="reviewer-speaking-ring" /> : null}
                    </div>
                    <div className="reviewer-heading">
                      <h3>{persona.name}</h3>
                      <div className="reviewer-tone">{persona.tone}</div>
                      <p>{persona.description}</p>
                    </div>
                  </div>
                  <div className="status-pill-row">
                    <span className={`status-pill ${palette.tone}`}>{palette.label}</span>
                    <span className="status-pill neutral">{truncate(result.preview || info.detail || persona.tagline, 48)}</span>
                  </div>
                  <div className="reviewer-edit-hint">카드를 클릭하면 전체 프롬프트를 보고 수정할 수 있습니다.</div>
                </MotionCard>
              );
            })}
          </motion.div>
          <MotionCard className="panel debate-chat-panel">
            <div className="panel-head panel-head-spread">
              <div>
                <div className="panel-kicker">Live Messenger</div>
                <h2>토론 대화방</h2>
              </div>
              <div className="debate-chat-meta">
                <span className={`status-pill ${getStatusPalette(visibleDebate?.status).tone}`}>{getStatusPalette(visibleDebate?.status).label}</span>
                <span className="sample-pill stage">{visibleDebate?.current_stage || '대기'}</span>
              </div>
            </div>
            <div className="debate-chat-room" ref={debateChatRoomRef}>
              {debateChatMessages.filter((message) => visibleDebateMessageIdSet.has(message.id)).map((message, index) => {
                const shouldStream = message.stream && activeDebateMessageId === message.id;
                const typingSpeed = getDebateMessageTypingSpeed(message);
                return (
                <motion.div
                  className={`debate-chat-message ${message.type}`}
                  key={message.id}
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.28, delay: index * 0.04, ease: 'easeOut' }}
                  style={{ '--bubble-accent': message.accent || '#8fb9d6' }}
                >
                  {message.type === 'system' || message.type === 'system-error' ? (
                    <div className="debate-chat-system-pill">
                      <strong>{message.title}</strong>
                      <p>
                        <StreamingText text={message.text} active={shouldStream} reduceMotion={prefersReducedMotion} speed={typingSpeed} className="debate-chat-copy" />
                        {shouldStream ? <span className="debate-chat-cursor" aria-hidden="true" /> : null}
                      </p>
                      <span>{message.meta}</span>
                    </div>
                  ) : (
                    <>
                      <div className="debate-chat-avatar" aria-hidden="true">{message.emoji}</div>
                      <div className="debate-chat-bubble">
                        <div className="debate-chat-bubble-head">
                          <strong>{message.personaName}</strong>
                          <span>{message.meta}</span>
                        </div>
                        <p>
                          <StreamingText text={message.text} active={shouldStream} reduceMotion={prefersReducedMotion} speed={typingSpeed} className="debate-chat-copy" />
                          {shouldStream ? <span className="debate-chat-cursor" aria-hidden="true" /> : null}
                        </p>
                        {message.references?.length ? (
                          <div className="debate-chat-reference-box">
                            <strong>참조 자료</strong>
                            {message.referenceSummary ? <p className="debate-chat-reference-summary"><StreamingText text={message.referenceSummary} active={shouldStream} reduceMotion={prefersReducedMotion} speed={typingSpeed} className="debate-chat-copy" /></p> : null}
                            <div className="debate-chat-reference-list">
                              {message.references.map((item) => <span className="sample-pill debate-reference-pill" key={`${message.id}-${item}`}>{item}</span>)}
                            </div>
                            {message.referencePreviews?.length ? (
                              <div className="debate-chat-reference-previews">
                                {message.referencePreviews.map((item, previewIndex) => (
                                  <div className={`debate-reference-card ${item.tone || 'neutral'}`} key={`${message.id}-preview-${previewIndex}`}>
                                    <strong>{item.title}</strong>
                                    <p>{item.summary}</p>
                                  </div>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                        {message.type === 'thinking' ? <div className="debate-chat-typing"><span /><span /><span /></div> : null}
                      </div>
                    </>
                  )}
                </motion.div>
                );
              })}
            </div>
          </MotionCard>
          <div className="debate-collapsible-stack">
            <details className="debate-collapsible">
              <summary>실시간 토론 현황 펼쳐보기</summary>
              <div className="debate-collapsible-body">
                <motion.div className="debate-status-card" whileHover={liftCard({ scale: 1.01 })}>
                  <strong>{visibleDebate?.current_stage || '대기'}</strong>
                  <span className={`status-pill ${getStatusPalette(visibleDebate?.status).tone}`}>{getStatusPalette(visibleDebate?.status).label}</span>
                  <p>{truncate(visibleDebate?.summary || '시작 버튼을 누르면 카드론 토론이 순차적으로 실행됩니다.', 240)}</p>
                  <div className="detail-meta-row">
                    <span className="sample-pill">준비 {formatElapsedMs(debateTimings.prep_elapsed_ms)}</span>
                    <span className="sample-pill">첫 요청 시작 {formatElapsedMs(debateTimings.first_ollama_wait_ms)}</span>
                    <span className="sample-pill">락 대기 {formatElapsedMs(debateTimings.first_queue_wait_ms)}</span>
                    <span className="sample-pill">우선순위 {visibleDebate?.priority_active ? '최우선' : '해제'}</span>
                  </div>
                </motion.div>
                <motion.div className="transcript-stack" variants={containerVariants} initial="hidden" animate="show">
                  {(visibleDebate?.round_results || []).map((item, index) => (
                    <MotionCard className="transcript-card" as="article" key={`${item.persona_id || 'round'}-${index}`} whileHover={liftCard({ x: 6 })}>
                      <div className="event-card-head"><strong>{item.name || item.persona_id || `stage-${index + 1}`}</strong><span>{item.completed_at ? formatTime(item.completed_at) : '진행 중'}</span></div>
                      <p>{truncate(item.preview || item.summary || item.raw_text || '-', 240)}</p>
                    </MotionCard>
                  ))}
                  {!(visibleDebate?.round_results || []).length ? <div className="empty-box">아직 정리된 토론 라운드가 없습니다.</div> : null}
                </motion.div>
              </div>
            </details>
            <details className="debate-collapsible">
              <summary>단계별 결과 펼쳐보기</summary>
              <div className="debate-collapsible-body">
                <motion.div className="round-detail-stack" variants={containerVariants} initial="hidden" animate="show">
                  {(visibleDebate?.round_results || []).map((item, index) => (
                    <MotionCard className="detail-card" as="section" key={`${item.persona_id || 'detail'}-${index}`}>
                      <h3>{item.name || `단계 ${index + 1}`}</h3>
                      <p>{truncate(item.preview || '-', 180)}</p>
                      <div className="detail-meta-row">
                        <span className="sample-pill">{formatPromptLength(item.request?.prompt_length)}</span>
                        <span className="sample-pill">단계 {formatElapsedMs(item.request?.elapsed_ms)}</span>
                        <span className="sample-pill">락 대기 {formatElapsedMs(item.request?.queue_wait_ms)}</span>
                        <span className="sample-pill">실제 질의문 포함</span>
                      </div>
                      <div className="detail-subtitle">실제 Ollama 질의</div>
                      <pre>{item.request?.prompt || '기록된 질의문이 없습니다.'}</pre>
                      <div className="detail-subtitle">단계 결과 JSON</div>
                      <pre>{JSON.stringify(item.response?.parsed || item, null, 2)}</pre>
                    </MotionCard>
                  ))}
                  {!(visibleDebate?.round_results || []).length ? <div className="empty-box">아직 기록된 상세 결과가 없습니다.</div> : null}
                </motion.div>
              </div>
            </details>
          </div>
          </motion.div>
        </motion.section>
      );
    }

    if (selectedSection === '온톨로지') {
      return (
        <Suspense
          fallback={(
            <motion.section key="ontology-loading" className="content-stack" {...sectionTransition}>
              <MotionCard className="panel">
                <div className="panel-head">
                  <div>
                    <div className="panel-kicker">Semantic Runtime</div>
                    <h2>온톨로지 콘솔을 불러오는 중입니다</h2>
                  </div>
                </div>
                <div className="empty-box">React Flow, retrieval trace, runtime inspector 청크를 로딩하고 있습니다.</div>
              </MotionCard>
            </motion.section>
          )}
        >
          <OntologyWorkbench
            theme={theme}
            onDepartmentThemeChange={(departmentId) => {
              const mappedTheme = mapDepartmentToTheme(departmentId);
              setTheme(mappedTheme);
            }}
            reduceMotion={prefersReducedMotion}
            onError={(message) => setErrorMessage(message)}
            onToast={(toast) => setActiveToast(toast)}
            onRequestRegulationUpload={openRegulationUploadModal}
            regulationBusy={regulationBusy}
            regulationStatus={status.agent_statuses?.regulation_agent?.status || 'pending'}
            regulationUpdatedAt={status.agent_statuses?.regulation_agent?.updated_at || ''}
            regulationSummary={truncate(status.latest_regulation_analysis || status.agent_statuses?.regulation_agent?.detail || '업로드된 규제 요약이 없습니다.', 260)}
            latestNewsSignal={latestNewsSignal}
            latestNewsItems={latestNewsItems}
            operationsSummary={ontologyOperationsSummary}
            debateSummary={ontologyDebateSummary}
            debateBusy={debateBusy}
            onStartDebate={handleStartDebate}
          />
        </Suspense>
      );
    }

    return (
      <motion.section key="vector" className="content-stack" {...sectionTransition}>
        <MotionCard className="panel">
          <div className="panel-head panel-head-spread"><div><div className="panel-kicker">Vector Runtime</div><h2>Vector DB 실시간 적재 현황</h2></div><select className="store-select" value={selectedStore} onChange={(event) => setSelectedStore(event.target.value)}>{STORE_OPTIONS.map((item) => <option key={item.value || 'all'} value={item.value}>{item.label}</option>)}</select></div>
          <motion.div className="metric-grid compact" variants={containerVariants} initial="hidden" animate="show">
            <MotionCard className="metric-card blue" as="article"><span>{selectedStoreLabel} 벡터 수</span><strong>{vectorSummary.total_count || 0}</strong></MotionCard>
            <MotionCard className="metric-card cyan" as="article"><span>문서 type 수</span><strong>{Object.keys(typeCounts).length}</strong></MotionCard>
            <MotionCard className="metric-card amber" as="article"><span>상품 코드 수</span><strong>{Object.keys(productCounts).length}</strong></MotionCard>
            <MotionCard className="metric-card red" as="article"><span>마지막 증감</span><strong>{status.vector_events?.[0]?.added_count || 0}</strong></MotionCard>
          </motion.div>
          <div className="table-shell"><table><thead><tr><th>DB</th><th>store</th><th>현재 로드</th><th>주요 type</th><th>주요 product</th></tr></thead><tbody>{(selectedStore ? [{ label: selectedStoreLabel, store: selectedStore, loaded: summaryItems.length, topType: Object.keys(typeCounts)[0] || '-', topProduct: Object.keys(productCounts)[0] || '-' }] : groupedStoreCounts).map((row) => <tr key={row.store}><td>{row.label}</td><td>{row.store}</td><td>{row.loaded}</td><td>{row.topType}</td><td>{row.topProduct}</td></tr>)}</tbody></table></div>
        </MotionCard>
        <MotionCard className="panel vector-cluster-relation-panel">
          <div className="panel-head panel-head-spread">
            <div>
              <div className="panel-kicker">feature_customer_clusters.json</div>
              <h2>고객 군집 관계망</h2>
            </div>
            <div className="detail-meta-row">
              <span className="sample-pill">{clusterRelationModel.clusters.length} clusters</span>
              <span className="sample-pill">{clusterRelationModel.totalRecords.toLocaleString()} records</span>
              <span className="sample-pill">cache v{clusterRelationModel.cacheMeta.cache_version || '-'}</span>
            </div>
          </div>
          {clusterRelationState.error ? <div className="error-banner">군집 관계망 조회 실패: {clusterRelationState.error}</div> : null}
          {clusterRelationState.loading ? <div className="empty-box">feature_customer_clusters.json 관계를 불러오는 중입니다.</div> : null}
          {!clusterRelationState.loading && !clusterRelationModel.clusters.length ? <div className="empty-box">표시할 고객 군집이 없습니다.</div> : null}
          {!clusterRelationState.loading && clusterRelationModel.clusters.length ? (
            <div className="vector-cluster-relation-layout">
              <section className="vector-cluster-map" aria-label="상품별 군집 관계">
                <div className="vector-cluster-source-node">
                  <span>source</span>
                  <strong>feature_customer_clusters.json</strong>
                  <small>{clusterRelationModel.cacheMeta.built_at || 'cache timestamp 없음'}</small>
                </div>
                <div className="vector-cluster-product-rail">
                  {clusterRelationModel.productGroups.map((group) => {
                    const share = clusterRelationModel.totalRecords ? Math.round((group.records / clusterRelationModel.totalRecords) * 100) : 0;
                    const decisionTop = Object.entries(group.decisions).sort((left, right) => right[1] - left[1])[0]?.[0] || '-';
                    return (
                      <article className="vector-cluster-product-node" key={group.product}>
                        <div className="vector-cluster-node-head">
                          <span>{group.product}</span>
                          <strong>{group.records.toLocaleString()}건</strong>
                        </div>
                        <div className="vector-cluster-share-track"><span style={{ width: `${Math.max(6, share)}%` }} /></div>
                        <div className="detail-chip-row">
                          <span className="sample-pill">{group.count} clusters</span>
                          <span className="sample-pill">{decisionTop}</span>
                        </div>
                        <div className="vector-cluster-mini-list">
                          {group.clusters.slice(0, 3).map((cluster) => (
                            <span key={cluster.cluster_id}>{cluster.age_band || '-'} · {cluster.income_band || '-'} · {cluster.amount_band || '-'}</span>
                          ))}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </section>
              <aside className="vector-cluster-side">
                <div className="vector-cluster-metric-strip">
                  {clusterRelationModel.metricNames.length ? clusterRelationModel.metricNames.map((name) => <span key={name}>{name}</span>) : <span>metric summary 없음</span>}
                </div>
                <div className="vector-cluster-top-list">
                  {clusterRelationModel.topClusters.slice(0, 6).map((cluster) => (
                    <article key={cluster.cluster_id} className="vector-cluster-top-card">
                      <div>
                        <span>{cluster.product || '-'}</span>
                        <strong>{cluster.label || cluster.cluster_id}</strong>
                      </div>
                      <small>{Number(cluster.count || 0).toLocaleString()} records</small>
                      <div className="detail-chip-row">
                        {(cluster.metric_summary || []).map((metric) => <span className="sample-pill" key={`${cluster.cluster_id}-${metric.axis_key}`}>{metric.label}: {metric.display || '-'}</span>)}
                      </div>
                    </article>
                  ))}
                </div>
              </aside>
              <div className="vector-cluster-band-grid">
                {Object.entries(clusterRelationModel.bandCounts).sort((left, right) => right[1] - left[1]).slice(0, 8).map(([band, count]) => (
                  <div className="vector-cluster-band-card" key={band}>
                    <span>{band}</span>
                    <strong>{count}</strong>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </MotionCard>
        <div className="two-column">
          <MotionCard className="panel">
            <div className="panel-head"><div><div className="panel-kicker">Execution Timeline</div><h2>실행 타임라인</h2></div></div>
            <motion.div className="timeline-grid single-column" variants={containerVariants} initial="hidden" animate="show">
              {(status.agent_activity_log || []).slice(0, 8).map((event, index) => (
                <MotionCard className="event-card" as="article" key={`${event.timestamp || 'activity'}-${index}`} whileHover={liftCard({ x: 6 })}>
                  <div className="event-card-head"><strong>{event.source || '-'}</strong><span>{formatTime(event.timestamp)}</span></div>
                  <span className={`status-pill ${getStatusPalette(event.status).tone}`}>{getStatusPalette(event.status).label}</span>
                  <p>{truncate(event.detail || '-', 160)}</p>
                </MotionCard>
              ))}
            </motion.div>
          </MotionCard>
          <MotionCard className="panel">
            <div className="panel-head"><div><div className="panel-kicker">Vector Event</div><h2>적재 이벤트</h2></div></div>
            {vectorFigures.line ? <LazyPlot data={vectorFigures.line.data} layout={vectorFigures.line.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} loadingHeight={260} /> : <div className="empty-box">아직 기록된 벡터 적재 이벤트가 없습니다.</div>}
            {vectorFigures.bar ? <LazyPlot data={vectorFigures.bar.data} layout={vectorFigures.bar.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }} loadingHeight={180} /> : null}
          </MotionCard>
        </div>
        <div className="two-column">
          <MotionCard className="panel"><div className="panel-head"><div><div className="panel-kicker">Type Distribution</div><h2>선택 DB type 분포</h2></div></div><div className="table-shell compact-table"><table><thead><tr><th>type</th><th>count</th></tr></thead><tbody>{Object.entries(typeCounts).sort((left, right) => right[1] - left[1]).map(([key, value]) => <tr key={key}><td>{key}</td><td>{value}</td></tr>)}</tbody></table></div></MotionCard>
          <MotionCard className="panel"><div className="panel-head"><div><div className="panel-kicker">Product Distribution</div><h2>선택 DB product 분포</h2></div></div><div className="table-shell compact-table"><table><thead><tr><th>product</th><th>count</th></tr></thead><tbody>{Object.entries(productCounts).sort((left, right) => right[1] - left[1]).map(([key, value]) => <tr key={key}><td>{key}</td><td>{value}</td></tr>)}</tbody></table></div></MotionCard>
        </div>
        <MotionCard className="panel">
          <div className="panel-head panel-head-spread"><div><div className="panel-kicker">Recent Search</div><h2>FAISS 최근 적재 검색기</h2></div><div className="vector-search-controls"><select className="store-select" value={selectedStore} onChange={(event) => setSelectedStore(event.target.value)}>{STORE_OPTIONS.map((item) => <option key={`search-store-${item.value || 'all'}`} value={item.value}>{item.label}</option>)}</select><select className="store-select" value={selectedVectorType} onChange={(event) => setSelectedVectorType(event.target.value)}>{VECTOR_TYPE_OPTIONS.map((item) => <option key={item.value || 'all-type'} value={item.value}>{item.label}</option>)}</select></div></div>
          <div className="detail-meta-row">
            <span className="sample-pill">최근 적재 기준</span>
            <span className="sample-pill">store {selectedStoreLabel}</span>
            <span className="sample-pill">type {selectedVectorTypeLabel}</span>
            <span className="sample-pill">표시 {recentVectorItems.length}건</span>
          </div>
          {!recentVectorItems.length ? <div className="empty-box">선택한 store/type에 해당하는 최근 적재 항목이 없습니다.</div> : null}
          {recentVectorItems.length ? <div className="table-shell"><table><thead><tr><th>id</th><th>store</th><th>type</th><th>product</th><th>source</th><th>name</th><th>full text</th></tr></thead><tbody>{recentVectorItems.map((item) => <tr key={item.id}><td>{truncate(item.id, 18)}</td><td>{item.store || '-'}</td><td>{item.type || '-'}</td><td>{item.product || '-'}</td><td>{item.source || '-'}</td><td>{truncate(item.name || '-', 32)}</td><td><button className="secondary-button table-action-button" type="button" disabled={vectorDetailBusy} onClick={() => handleOpenVectorDetail(item)}>FULLTEXT</button></td></tr>)}</tbody></table></div> : null}
        </MotionCard>
      </motion.section>
    );
  };

  return (
    <div className={`app-shell ${isImmersiveSection ? 'loan-gpt-mode' : ''} theme-${theme}`}>
      <div className="theme-toolbar" ref={themeMenuRef}>
        <button
          className={`theme-toggle-button ${themeMenuOpen ? 'open' : ''}`}
          type="button"
          onClick={() => setThemeMenuOpen((previous) => !previous)}
          aria-haspopup="menu"
          aria-expanded={themeMenuOpen}
        >
          <span className="theme-toggle-emoji" aria-hidden="true">🎨</span>
          <span className="theme-toggle-copy">
            <strong>옵션</strong>
            <small>{currentThemeOption?.description || '부서 테마'}</small>
          </span>
        </button>
        <AnimatePresence>
          {themeMenuOpen ? (
            <motion.div
              className="theme-menu"
              initial={{ opacity: 0, y: -8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.98 }}
              transition={{ duration: 0.18, ease: 'easeOut' }}
            >
              <div className="theme-menu-kicker theme-menu-kicker-spaced">Agent Ollama</div>
              <div className="theme-agent-ollama-controls">
                <div className="theme-agent-ollama-card">
                  <div className="theme-agent-ollama-copy">
                    <strong>뉴스 에이전트</strong>
                    <span>{status.news_agent_ollama_enabled ? 'Ollama 호출 사용' : 'Fallback 브리핑 모드'}</span>
                  </div>
                  <button
                    type="button"
                    className={`theme-agent-toggle ${status.news_agent_ollama_enabled ? 'active' : ''}`}
                    onClick={() => handleToggleAgentOllama('news', !status.news_agent_ollama_enabled)}
                    disabled={agentOllamaToggleBusy.news}
                  >
                    {agentOllamaToggleBusy.news ? '적용 중...' : (status.news_agent_ollama_enabled ? '호출 끄기' : '호출 켜기')}
                  </button>
                </div>
                <div className="theme-agent-ollama-card">
                  <div className="theme-agent-ollama-copy">
                    <strong>로그 에이전트</strong>
                    <span>{status.log_agent_ollama_enabled ? 'Ollama 호출 사용' : 'Fallback 브리핑 모드'}</span>
                  </div>
                  <button
                    type="button"
                    className={`theme-agent-toggle ${status.log_agent_ollama_enabled ? 'active' : ''}`}
                    onClick={() => handleToggleAgentOllama('log', !status.log_agent_ollama_enabled)}
                    disabled={agentOllamaToggleBusy.log}
                  >
                    {agentOllamaToggleBusy.log ? '적용 중...' : (status.log_agent_ollama_enabled ? '호출 끄기' : '호출 켜기')}
                  </button>
                </div>
                <div className="theme-agent-ollama-card">
                  <div className="theme-agent-ollama-copy">
                    <strong>Ollama GPU 실행</strong>
                    <span>{status.ollama_gpu_enabled ? 'Ollama 기본 GPU 경로 사용' : 'CPU 우선 모드 유지'}</span>
                  </div>
                  <button
                    type="button"
                    className={`theme-agent-toggle ${status.ollama_gpu_enabled ? 'active' : ''}`}
                    onClick={() => handleToggleOllamaRuntime('gpu', !status.ollama_gpu_enabled)}
                    disabled={agentOllamaToggleBusy.gpu}
                  >
                    {agentOllamaToggleBusy.gpu ? '적용 중...' : (status.ollama_gpu_enabled ? 'GPU 끄기' : 'GPU 켜기')}
                  </button>
                </div>
                <div className="theme-agent-ollama-card">
                  <div className="theme-agent-ollama-copy">
                    <strong>규제 업로드 요약 생성</strong>
                    <span>{status.regulation_upload_summary_enabled ? '업로드 후 요약 생성 포함' : '업로드 시 요약 단계 생략(빠른 모드)'}</span>
                  </div>
                  <button
                    type="button"
                    className={`theme-agent-toggle ${status.regulation_upload_summary_enabled ? 'active' : ''}`}
                    onClick={() => handleToggleRegulationUploadSummary(!status.regulation_upload_summary_enabled)}
                    disabled={agentOllamaToggleBusy.regulationSummary}
                  >
                    {agentOllamaToggleBusy.regulationSummary ? '적용 중...' : (status.regulation_upload_summary_enabled ? '요약 끄기' : '요약 켜기')}
                  </button>
                </div>
                <div className="theme-agent-ollama-card">
                  <div className="theme-agent-ollama-copy">
                    <strong>온톨로지 질의 최우선</strong>
                    <span>{status.ontology_query_priority_enabled ? '온톨로지 요청이 다른 Ollama 작업보다 먼저 진입' : '공유 락을 일반 우선순위로 사용'}</span>
                  </div>
                  <button
                    type="button"
                    className={`theme-agent-toggle ${status.ontology_query_priority_enabled ? 'active' : ''}`}
                    onClick={() => handleToggleOllamaRuntime('ontology', !status.ontology_query_priority_enabled)}
                    disabled={agentOllamaToggleBusy.ontology}
                  >
                    {agentOllamaToggleBusy.ontology ? '적용 중...' : (status.ontology_query_priority_enabled ? '우선처리 끄기' : '우선처리 켜기')}
                  </button>
                </div>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
      <AnimatePresence>
        {selectedVectorEntry ? (
          <motion.div className="prompt-modal-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setSelectedVectorEntry(null)}>
            <motion.section className="prompt-modal vector-detail-modal" initial={{ opacity: 0, y: 20, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 14, scale: 0.98 }} transition={{ duration: 0.22, ease: 'easeOut' }} onClick={(event) => event.stopPropagation()}>
              <div className="prompt-modal-head">
                <div>
                  <div className="panel-kicker">Vector Full Text</div>
                  <h3>{truncate(selectedVectorEntry.metadata?.name || selectedVectorEntry.id || 'FAISS 문서', 72)}</h3>
                  <p>{`${selectedVectorEntry.metadata?.store || selectedStore || '-'} · ${selectedVectorEntry.metadata?.type || '-'}`}</p>
                </div>
                <button className="secondary-button" type="button" onClick={() => setSelectedVectorEntry(null)}>닫기</button>
              </div>
              <div className="detail-meta-row">
                <span className="sample-pill">ID {truncate(selectedVectorEntry.id, 28)}</span>
                <span className="sample-pill">store {selectedVectorEntry.metadata?.store || selectedStore || '-'}</span>
                <span className="sample-pill">type {selectedVectorEntry.metadata?.type || '-'}</span>
                <span className="sample-pill">product {selectedVectorEntry.metadata?.product || '-'}</span>
                <span className="sample-pill">{selectedVectorEntry.loading ? 'FULLTEXT 불러오는 중' : 'FULLTEXT 준비됨'}</span>
              </div>
              {selectedVectorEntry.error ? <div className="error-banner">FULLTEXT 조회 실패: {selectedVectorEntry.error}</div> : null}
              <div className="vector-fulltext-shell">
                <pre>{selectedVectorEntry.page_content || (selectedVectorEntry.loading ? '본문을 불러오는 중입니다.' : '표시할 full text가 없습니다.')}</pre>
              </div>
            </motion.section>
          </motion.div>
        ) : null}
        {showRegulationUploadModal ? (
          <motion.div
            className="prompt-modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => { if (!regulationBusy) setShowRegulationUploadModal(false); }}
          >
            <motion.section
              className="prompt-modal regulation-upload-modal"
              initial={{ opacity: 0, y: 20, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 14, scale: 0.98 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="prompt-modal-head">
                <div>
                  <div className="panel-kicker">Bunny Learning Lab</div>
                  <h3>버니 규제 문서 학습실</h3>
                  <p>{regulationBusy ? '버니가 문서를 읽으며 근거 지식을 업데이트하고 있어요.' : '문서를 추가하면 버니가 바로 학습을 시작해요.'}</p>
                </div>
                <button className="secondary-button" type="button" onClick={() => setShowRegulationUploadModal(false)} disabled={regulationBusy}>닫기</button>
              </div>
              <div className="detail-meta-row">
                <span className={`sample-pill ${regulationBusy ? 'is-running' : ''}`}>상태 {regulationAgentStatus}</span>
                <span className="sample-pill">문서 {uploadedRegulationFiles.length}건</span>
              </div>
              <div className="regulation-learning-stage" role="status" aria-live="polite">
                <div className="regulation-learning-bunny-shell">
                  <RegulationLearningBunny />
                </div>
                <div className="regulation-learning-copy">
                  <strong>{regulationBusy ? 'Bunny가 실시간으로 학습 중입니다' : 'Bunny가 학습 준비를 마쳤어요'}</strong>
                  <p>{regulationBusy ? '문서 파싱 → 청크 생성 → 벡터 적재 → 근거 인덱싱 순서로 처리하고 있어요.' : '문서를 추가하면 같은 순서로 학습을 진행하고 결과를 답변에 반영합니다.'}</p>
                  <small>{status.agent_statuses?.regulation_agent?.detail || '학습 파이프라인을 준비 중입니다.'}</small>
                </div>
              </div>
              <div className="regulation-learning-progress-wrap" aria-label="학습 진행률">
                <div className="regulation-learning-progress-head">
                  <strong>{regulationLearningStage}</strong>
                  <span>{normalizedRegulationProgress}%</span>
                </div>
                <div className={`regulation-learning-progress-track ${regulationBusy ? 'is-running' : ''}`}>
                  <span style={{ width: `${Math.max(4, normalizedRegulationProgress)}%` }} />
                </div>
              </div>
              <div className={`regulation-upload-live ${regulationBusy ? 'is-running' : ''}`}>
                <span />
                <span />
                <span />
              </div>
              <div className="regulation-upload-file-list">
                {(uploadedRegulationFileStats.length || uploadedRegulationFiles.length) ? (
                  (uploadedRegulationFileStats.length
                    ? uploadedRegulationFileStats
                    : uploadedRegulationFiles.map((name) => ({ name, chunk_count: 0, page_count: 0, status: 'unknown' })))
                    .map((item, index) => (
                      <div key={`${item.name || 'regulation-file'}-${index}`} className="regulation-upload-file-item">
                        <strong>{item.name || `문서 ${index + 1}`}</strong>
                        <small>{`페이지 ${Number(item.page_count || 0)} · 청크 ${Number(item.chunk_count || 0)} · ${String(item.status || 'unknown')}`}</small>
                      </div>
                    ))
                ) : (
                  <div className="empty-box compact">업로드된 규제 문서가 없습니다.</div>
                )}
              </div>
              <div className="prompt-modal-actions">
                <button className="primary-button" type="button" onClick={handleAddRegulationFiles} disabled={regulationBusy}>추가</button>
              </div>
            </motion.section>
          </motion.div>
        ) : null}
        {activeToast ? (
          <motion.aside
            key={activeToast.id}
            className={`toast-banner ${activeToast.tone}`}
            initial={{ opacity: 0, y: -18, x: 20 }}
            animate={{ opacity: 1, y: 0, x: 0 }}
            exit={{ opacity: 0, y: -12, x: 12 }}
            transition={{ duration: 0.28, ease: 'easeOut' }}
          >
            <div className="toast-kicker">{activeToast.kicker}</div>
            <div className="toast-title">{activeToast.title}</div>
            <div className="toast-meta">{activeToast.meta}</div>
            <div className="toast-message">{activeToast.message}</div>
          </motion.aside>
        ) : null}
      </AnimatePresence>
      <input ref={regulationInputRef} className="hidden-file-input" type="file" accept=".pdf,.txt,.md" multiple onChange={handleRegulationFileSelection} />
      <AnimatePresence>
        {selectedReviewer ? (
          <motion.div className="prompt-modal-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setSelectedReviewerId(null)}>
            <motion.section className="prompt-modal" initial={{ opacity: 0, y: 20, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 14, scale: 0.98 }} transition={{ duration: 0.22, ease: 'easeOut' }} onClick={(event) => event.stopPropagation()}>
              <div className="prompt-modal-head">
                <div>
                  <div className="panel-kicker">Reviewer Prompt</div>
                  <h3>{selectedReviewer.name}</h3>
                  <p>{selectedReviewer.description}</p>
                </div>
                <button className="secondary-button" type="button" onClick={() => setSelectedReviewerId(null)}>닫기</button>
              </div>
              <label className="prompt-editor prompt-editor-modal">
                <span>실제 백엔드 조립 템플릿</span>
                <textarea
                  value={reviewerPrompts[selectedReviewer.id] || getDefaultReviewerTemplate(selectedReviewer)}
                  onChange={(event) => setReviewerPrompts((previous) => ({ ...previous, [selectedReviewer.id]: event.target.value }))}
                />
              </label>
              <label className="prompt-editor">
                <span>Ollama Temperature</span>
                <input
                  className="store-select"
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={selectedReviewerSetting?.temperature ?? 0.5}
                  onChange={(event) => setReviewerSettings((previous) => ({
                    ...previous,
                    [selectedReviewer.id]: {
                      ...(previous[selectedReviewer.id] || buildDefaultReviewerSetting(selectedReviewer.id)),
                      temperature: Math.max(0, Math.min(1, Number(event.target.value) || 0.5)),
                    },
                  }))}
                />
              </label>
              {selectedReviewer.id === 'credit_planning_agent' ? (
                <label className="prompt-editor">
                  <span>signal_news feature 기준</span>
                  <input
                    className="store-select"
                    type="text"
                    value={selectedReviewerSetting?.market_signal_feature_keys || ''}
                    onChange={(event) => setReviewerSettings((previous) => ({
                      ...previous,
                      [selectedReviewer.id]: {
                        ...(previous[selectedReviewer.id] || buildDefaultReviewerSetting(selectedReviewer.id)),
                        market_signal_feature_keys: event.target.value,
                      },
                    }))}
                    placeholder="signal_summary, risk_signal, linked_decision"
                  />
                  <small className="prompt-help-text">사용 가능: signal_summary, risk_signal, opportunity_signal, linked_decision, tags</small>
                </label>
              ) : null}
              <div className="placeholder-editor-grid">
                {getReviewerPlaceholderKeys(selectedReviewer.id).map((key) => (
                  <label className="prompt-editor" key={key}>
                    <span>{getPlaceholderFieldLabel(key)}</span>
                    <textarea
                      className="placeholder-editor-textarea"
                      value={selectedReviewerSetting?.placeholders?.[key] || ''}
                      onChange={(event) => setReviewerSettings((previous) => ({
                        ...previous,
                        [selectedReviewer.id]: {
                          ...(previous[selectedReviewer.id] || buildDefaultReviewerSetting(selectedReviewer.id)),
                          placeholders: {
                            ...((previous[selectedReviewer.id] || buildDefaultReviewerSetting(selectedReviewer.id)).placeholders || {}),
                            [key]: event.target.value,
                          },
                        },
                      }))}
                    />
                  </label>
                ))}
              </div>
              <div className="prompt-preview-block">
                <div className="prompt-preview-title">렌더링 미리보기</div>
                <pre>{renderReviewerPromptPreview(selectedReviewer.id, reviewerPrompts[selectedReviewer.id] || getDefaultReviewerTemplate(selectedReviewer), DEFAULT_QUESTION, selectedReviewerSetting?.placeholders || {})}</pre>
              </div>
              <div className="prompt-token-guide">
                <strong>사용 가능한 placeholder</strong>
                <p>{getReviewerPlaceholderKeys(selectedReviewer.id).map((key) => getPlaceholderFieldLabel(key)).join(', ')}</p>
                <p>각 값을 직접 수정하면 실제 백엔드 조립 프롬프트에도 그대로 반영됩니다.</p>
                {selectedReviewer.id === 'credit_planning_agent' ? <p>시장 신호는 signal_news 문서 중 최근 3개만 사용하고, 위 feature 기준에 값이 있는 문서만 고릅니다.</p> : null}
              </div>
              <div className="prompt-modal-actions">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => {
                    setReviewerPrompts((previous) => ({ ...previous, [selectedReviewer.id]: getDefaultReviewerTemplate(selectedReviewer) }));
                    setReviewerSettings((previous) => ({ ...previous, [selectedReviewer.id]: buildDefaultReviewerSetting(selectedReviewer.id) }));
                  }}
                >
                  기본값 복원
                </button>
                <button className="primary-button" type="button" onClick={() => setSelectedReviewerId(null)}>저장하고 닫기</button>
              </div>
            </motion.section>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <div className="app-backdrop">
        <AmbientOrb className="ambient-orb orb-cyan" delay={0.2} duration={10} />
        <AmbientOrb className="ambient-orb orb-amber" delay={1.1} duration={12} />
        <AmbientOrb className="ambient-orb orb-violet" delay={0.7} duration={14} />
      </div>
      <AnimatePresence>
        {!isImmersiveSection ? (
          <motion.section className="hero-panel page-title-hero" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -14 }} transition={{ duration: 0.46, ease: 'easeOut' }}>
            <div className="hero-panel-copy hero-panel-copy-compact">
              <div className="hero-kicker">AI Review Control Tower</div>
              <h1>심사 로그, 뉴스, 규제 분석를 AI 오케스트라로 한번에!</h1>
              <p>실시간 상태, 요약 브리핑, 벡터 적재 현황을 통합해서 심사 운영 흐름을 빠르게 파악하도록 만든 시스템입니다.</p>
            </div>
            <div className="hero-visual hero-visual-layered">
              <motion.div className="hero-visual-shell hero-visual-shell-title" animate={prefersReducedMotion ? undefined : { y: [0, -6, 0] }} transition={{ duration: 4.6, repeat: Infinity, ease: 'easeInOut' }}>
                <div className="hero-control-grid" />
                <div className="hero-control-bars" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                </div>
                <div className="hero-control-cards" aria-hidden="true">
                  <span className="hero-control-card primary" />
                  <span className="hero-control-card secondary" />
                  <span className="hero-control-card accent" />
                </div>
                <div className="hero-control-scan" />
                <div className="hero-control-dots" aria-hidden="true">
                  <span />
                  <span />
                </div>
                <div className="hero-wave-layer">
                  <AnimatedSignalWave className="rail-lottie" />
                </div>
                <div className="hero-orbit-layer">
                  <AnimatedOrbitField className="hero-lottie" />
                </div>
                <motion.div className="hero-signal-tag" animate={prefersReducedMotion ? undefined : { scale: [1, 1.04, 1], opacity: [0.82, 1, 0.82] }} transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}>LIVE CONTROL</motion.div>
              </motion.div>
            </div>
          </motion.section>
        ) : null}
      </AnimatePresence>
      <div className={`page-grid ${isImmersiveSection ? 'loan-gpt-layout' : ''}`}>
        <motion.aside className={`left-rail app-section-rail ${isImmersiveSection ? 'immersive' : ''}`} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.35, ease: 'easeOut' }}>
          {SECTION_RAIL_ITEMS.map((item) => (
            <motion.button
              key={item.id}
              type="button"
              className={`rail-nav-button ${selectedSection === item.id ? 'active' : ''}`}
              onClick={() => setSelectedSection(item.id)}
              whileHover={liftCard({ y: -2, scale: 1.02 })}
              whileTap={prefersReducedMotion ? undefined : { scale: 0.98 }}
              aria-label={item.id}
              title={item.id}
            >
              <span className="rail-nav-icon" aria-hidden="true"><RailIcon kind={item.icon} /></span>
              <span className="rail-nav-label">{item.label}</span>
            </motion.button>
          ))}
        </motion.aside>
        <motion.main className={`main-rail ${isImmersiveSection ? 'expanded' : ''}`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55, ease: 'easeOut', delay: 0.08 }}>
          {errorMessage ? <motion.div className="error-banner" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>{errorMessage}</motion.div> : null}
          <AnimatePresence mode="wait">{renderContent()}</AnimatePresence>
        </motion.main>
      </div>
    </div>
  );
}

export default App;