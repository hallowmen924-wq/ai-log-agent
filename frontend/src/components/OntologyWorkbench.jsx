import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';
import { EventType, useRive, useStateMachineInput } from '@rive-app/react-canvas';
import ReactFlow, { Background, Controls, Handle, MarkerType, MiniMap, Position } from 'reactflow';
import 'reactflow/dist/style.css';
import './OntologyWorkbench.local.css';
import regulationDocumentAnimation from '../assets/regulation-document.json';
import promptSubmitAnimation from '../assets/prompt-submit.json';
import cardLoanNewsAnimation from '../assets/card-loan-news.json';
import { createProductDevelopmentAgendas, createProductDevelopmentDebate, fetchFeatureOntologyRuntimeJob, fetchFeatureOntologySegmentMetricCube, fetchFeatureOntologySemanticRefreshStatus, fetchOntologyState, startFeatureOntologyRuntimeJob } from '../api';
import { RUNTIME_STAGES, useOntologyRuntimeStore } from './ontologyRuntimeStore';

const ONTOLOGY_PROMPT_EXAMPLES = [
  {
    group: '승인 요인',
    accent: 'blue',
    prompts: [
      '40대 직장인의 카드론 승인에 중요한 요인은?',
      '이지신용대출 신청자의 승인 한도에 영향을 주는 feature는?',
      '저소득 카드론 승인 고객군에서 공통으로 강한 신호는?',
    ],
  },
  {
    group: '거절/탈락 사유',
    accent: 'rose',
    prompts: [
      '40대 카드론 신청자들의 평균 탈락 사유는?',
      '이지론 거절 고객군에서 가장 자주 연결되는 reject reason은?',
      '이지신용대출 탈락 사유와 관련 feature를 같이 보여줘',
    ],
  },
  {
    group: '금리/한도',
    accent: 'green',
    prompts: [
      '이지신용대출 평균 금리와 한도는?',
      '이지신용대출 승인 고객의 평균 금리와 승인 한도 분포를 알려줘',
      '이지론 승인군의 평균 한도와 소득 구간 관계를 비교해줘',
    ],
  },
  {
    group: '군집/벡터',
    accent: 'amber',
    prompts: [
      '이지신용대출 고객군집에서 승인과 거절 군집을 비교해줘',
      '40대 신청자 군집을 소득, 한도, 거절 사유 기준으로 나눠줘',
      'feature_customer_clusters.json 기준으로 가장 반복되는 관계를 설명해줘',
    ],
  },
];

function makeDotLayer({ ind, color, size = 11, x = 32, y = 32, delay = 0 }) {
  return {
    ddd: 0,
    ind,
    ty: 4,
    nm: `dot-${ind}`,
    sr: 1,
    ks: {
      o: { a: 0, k: 100 },
      r: { a: 0, k: 0 },
      p: {
        a: 1,
        k: [
          { t: 0 + delay, s: [x, y, 0], e: [x, y - 4, 0] },
          { t: 18 + delay, s: [x, y - 4, 0], e: [x, y, 0] },
          { t: 36 + delay, s: [x, y, 0], e: [x, y - 3, 0] },
          { t: 54 + delay, s: [x, y - 3, 0] },
        ],
      },
      a: { a: 0, k: [0, 0, 0] },
      s: {
        a: 1,
        k: [
          { t: 0 + delay, s: [85, 85, 100], e: [118, 118, 100] },
          { t: 18 + delay, s: [118, 118, 100], e: [85, 85, 100] },
          { t: 54 + delay, s: [85, 85, 100] },
        ],
      },
    },
    ao: 0,
    shapes: [
      {
        ty: 'gr',
        it: [
          { ty: 'el', p: { a: 0, k: [0, 0] }, s: { a: 0, k: [size, size] }, nm: 'dot' },
          { ty: 'fl', c: { a: 0, k: color }, o: { a: 0, k: 100 }, r: 1, bm: 0, nm: 'fill' },
          { ty: 'tr', p: { a: 0, k: [0, 0] }, a: { a: 0, k: [0, 0] }, s: { a: 0, k: [100, 100] }, r: { a: 0, k: 0 }, o: { a: 0, k: 100 } },
        ],
        nm: 'dot-group',
      },
    ],
  };
}

function makePromptExamplesLottieData() {
  return {
    v: '5.8.1',
    fr: 30,
    ip: 0,
    op: 60,
    w: 64,
    h: 64,
    nm: 'prompt-examples',
    ddd: 0,
    assets: [],
    layers: [
      makeDotLayer({ ind: 1, color: [0.25, 0.62, 1, 1], size: 12, x: 22, y: 32, delay: 0 }),
      makeDotLayer({ ind: 2, color: [0.24, 0.82, 0.92, 1], size: 12, x: 32, y: 28, delay: 6 }),
      makeDotLayer({ ind: 3, color: [0.98, 0.72, 0.29, 1], size: 12, x: 42, y: 32, delay: 12 }),
    ],
  };
}

function PromptLottieActionButton({ variant = 'examples', className = '', ariaLabel = '', title = '', onClick }) {
  const animationData = useMemo(
    () => (variant === 'send' ? promptSubmitAnimation : makePromptExamplesLottieData()),
    [variant],
  );
  return (
    <button type="button" className={`prompt-lottie-button is-${variant} ${className}`} aria-label={ariaLabel} title={title || ariaLabel} onClick={onClick}>
      <DotLottieReact className="prompt-lottie-icon" data={animationData} loop autoplay />
    </button>
  );
}

function GeneralAnswerModeIntro() {
  const { RiveComponent } = useRive({
    src: '/general-answer-mode.riv',
    autoplay: true,
  });

  return (
    <div className="general-answer-mode-intro-rive" aria-hidden="true">
      <RiveComponent />
    </div>
  );
}

function StreamingText({ text, active = false, reduceMotion = false, speed = 26, className = '' }) {
  const normalized = String(text || '');
  const [visibleText, setVisibleText] = useState(reduceMotion || !active ? normalized : '');

  useEffect(() => {
    if (reduceMotion || !active) {
      setVisibleText(normalized);
      return undefined;
    }

    let timerId;
    setVisibleText('');

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

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function deriveAnswerHighlightTerms(text, baseTerms = [], tooltipTerms = []) {
  const source = String(text || '');
  const regexTerms = [
    ...source.match(/\bK\d{3}\b/g) || [],
    ...source.match(/\d+(?:,\d{3})*건/g) || [],
    ...source.match(/\d+(?:\.\d+)?%/g) || [],
  ];
  const fixedTerms = ['거절사유코드', '이지론(C9)', '개인사업자대출(C11)', '이지대환대출(C12)', '이지신용대출(C6)', 'DSR 초과', '가계부채', '분석 완료'];
  return [...new Set([...baseTerms, ...tooltipTerms, ...regexTerms, ...fixedTerms]
    .map((item) => String(item || '').trim())
    .filter((item) => item && source.includes(item)))]
    .sort((left, right) => right.length - left.length)
    .slice(0, 24);
}

function HighlightedAnswerText({ text, terms = [], termMeta = {}, className = '' }) {
  const normalized = String(text || '');
  const highlightTerms = deriveAnswerHighlightTerms(normalized, terms, Object.keys(termMeta || {}));
  if (!normalized || !highlightTerms.length) {
    return <span className={className}>{normalized}</span>;
  }
  const pattern = new RegExp(`(${highlightTerms.map(escapeRegExp).join('|')})`, 'g');
  return (
    <span className={className}>
      {normalized.split(pattern).map((part, index) => (
        highlightTerms.includes(part)
          ? (
            <mark
              key={`${part}-${index}`}
              className="ontology-answer-keyword"
              data-tooltip={termMeta?.[part] || ''}
              title={termMeta?.[part] || undefined}
            >
              {part}
            </mark>
          )
          : <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>
      ))}
    </span>
  );
}

function RuntimeProgressPanel({
  complete = false,
  progress = 0,
  headline = '진행 중',
  subLabel = '진행 중',
  liveText = '진행 상태를 확인하는 중',
  activeStage,
  runtimeStages = [],
  slowestCompletedStage,
}) {
  return (
    <motion.div
      className={`ontology-progress-theatre ${complete ? 'is-complete' : ''}`}
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.22, ease: 'easeOut' }}
    >
      <div className="ontology-progress-orbit" aria-hidden="true"><span /><span /><span /></div>
      <div className="ontology-progress-main">
        <div className="ontology-progress-head">
          <div><span className="panel-kicker">진행 상황</span><strong>{headline}</strong></div>
          <div className="ontology-progress-number"><strong>{progress}%</strong><span>{subLabel}</span></div>
        </div>
        <div className="ontology-progress-track"><motion.span initial={false} animate={{ width: `${Math.max(4, progress)}%` }} transition={{ duration: 0.28, ease: 'easeOut' }} /></div>
        <div className="ontology-progress-stage-rail">
          {runtimeStages.map((stage, index) => {
            const duration = Number(stage.duration_ms || stage.meta?.duration_ms || 0);
            const isActive = stage.key === activeStage?.key;
            return (
              <div key={stage.key} className={`ontology-progress-stage-dot ${toneClass(stage.status)} ${isActive ? 'active' : ''}`}>
                <span>{index + 1}</span>
                <small>{stageShortLabel(stage.key)}</small>
                {duration > 0 ? <em>{formatRuntimeMs(duration)}</em> : null}
              </div>
            );
          })}
        </div>
        <div className="ontology-progress-live-row">
          <span>{liveText}</span>
          {slowestCompletedStage ? <span>가장 오래 걸림: {stageShortLabel(slowestCompletedStage.key)} {formatRuntimeMs(slowestCompletedStage.duration_ms || slowestCompletedStage.meta?.duration_ms)}</span> : <span>소요 시간 확인 중</span>}
        </div>
      </div>
    </motion.div>
  );
}

function FinancialToolCard({ tool }) {
  if (!tool) {
    return null;
  }
  // 영향 feature는 최대 5개, 군집은 최대 2개만 노출
  const shapValues = (tool.shap_values || []).slice(0, 5);
  const clusters = (tool.clusters || []).slice(0, 2);
  const conflicts = tool.conflicts || [];
  const metrics = tool.metrics || [];
  const personas = tool.personas || [];
  const visualization = tool.visualization || null;
  const visualizationPoints = visualization?.points || [];

  // 승인 vs 거절 비교 탭에서는 군집 시각화 숨김
  const isApproveVsRejectTab = (tool?.id === 'cluster' && tool?.title?.includes('승인 vs 거절'));
  const clusterBars = clusters.map((item) => ({
    label: String(item.decision || item.display_label || item.label || '군집').replace(/\s+/g, ' ').slice(0, 12),
    value: Number(item.records || item.count || item.size || 0),
    tone: String(item.decision || '').includes('거절') ? 'reject' : 'approve',
  }));
  const maxClusterBar = Math.max(1, ...clusterBars.map((item) => item.value));
  return (
    <motion.article
      className={`financial-tool-card tool-${tool.id || 'generic'}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: 'easeOut' }}
    >
      <div className="financial-tool-card-head">
        <div>
          <span className="panel-kicker">{tool.llm_call ? 'LLM Tool' : 'Tool Agent'}</span>
          <strong>{tool.title || 'Agent Tool'}</strong>
        </div>
        <span className="sample-pill">{tool.status || 'ready'}</span>
      </div>
      <p>{tool.summary || '대화 흐름에 필요한 분석 도구를 실행했습니다.'}</p>
      {metrics.length ? (
        <div className="tool-metric-row">
          {metrics.map((item) => <div key={item.label} className={`tool-metric is-${item.tone || 'neutral'}`}><span>{item.label}</span><strong>{item.value}</strong></div>)}
        </div>
      ) : null}
      {shapValues.length ? (
        <div className="tool-impact-list">
          {shapValues.map((item) => (
            <div key={`${tool.id}-${item.feature}`} className="tool-impact-row">
              <div className="tool-impact-meta">
                <strong>{item.feature}</strong>
                <span>{item.evidence || item.direction || ''}</span>
              </div>
              <div className="tool-impact-bar"><span style={{ width: `${Math.min(100, Number(item.impact || 0))}%` }} /></div>
              <b>{Number(item.impact || 0).toFixed(1)}%</b>
            </div>
          ))}
        </div>
      ) : null}
      {clusters.length ? (
        <div className="tool-mini-grid">
          {clusters.map((item) => (
            <div key={item.cluster_id || item.label} className="tool-mini-card">
              <span>{item.decision || '군집'} · {item.records || 0}건</span>
              <strong>{item.display_label || item.label}</strong>
              <small>금리 {item.avg_rate || '-'} · 한도 {item.avg_limit || '-'}</small>
              <small>연체/부실 {item.delinquency_rate || '-'} · 모델 {item.avg_model_score || '-'}</small>
            </div>
          ))}
        </div>
      ) : null}
      {clusterBars.length > 1 ? (
        <>
          <div className="tool-cluster-bar-chart-desc" style={{marginBottom: 4, fontWeight: 500, color: '#2a2a2a'}}>
            승인/거절 고객군별 주요 분포 비교 (승인률, 인원수 등)
          </div>
          <div className="tool-cluster-bar-chart" aria-label="군집 비교 막대 차트">
            {clusterBars.map((item, index) => (
              <span key={`${tool.id}-cluster-bar-${item.label}-${index}`} className={`is-${item.tone}`}>
                <i style={{ height: `${Math.max(18, Math.round((item.value / maxClusterBar) * 100))}%` }} />
                <b>{item.label}</b>
              </span>
            ))}
          </div>
        </>
      ) : null}
      {/* 승인 vs 거절 비교 탭에서는 군집 시각화 숨김 */}
      {visualizationPoints.length && !isApproveVsRejectTab ? (
        <div className="tool-cluster-visual">
          <div className="tool-cluster-visual-head">
            <span>{visualization?.x_label || 'x'}</span>
            <strong>군집 시각화</strong>
            <span>{visualization?.y_label || 'y'}</span>
          </div>
          <div className="tool-cluster-plot" aria-label="고객군집 분포">
            {visualizationPoints.map((point, index) => {
              const left = Math.min(88, Math.max(8, 12 + index * 18));
              const top = Math.min(82, Math.max(12, 78 - Number(point.y || 0) * 2.8));
              const size = Math.min(34, Math.max(14, Math.sqrt(Number(point.size || 1)) * 2.1));
              return (
                <span
                  key={point.id || point.label}
                  className="tool-cluster-dot"
                  style={{ left: `${left}%`, top: `${top}%`, width: size, height: size }}
                  title={`${point.label} · ${point.x_display} · ${point.y_display} · 연체/부실 ${point.risk}`}
                >
                  {index + 1}
                </span>
              );
            })}
          </div>
        </div>
      ) : null}
      {conflicts.length ? (
        <div className="tool-mini-list">
          {conflicts.map((item) => <span key={item.title} className={`tool-status-line is-${item.level || 'info'}`}>{item.title}</span>)}
        </div>
      ) : null}
      {personas.length ? (
        <div className="tool-mini-grid">
          {personas.map((item) => (
            <div key={item.name} className="tool-mini-card">
              <span>{item.focus}</span>
              <strong>{item.name}</strong>
              <small>{item.view}</small>
            </div>
          ))}
        </div>
      ) : null}
    </motion.article>
  );
}

function StrategyWorkspace({ panels = [], workflow = [], semanticLayer = {} }) {
  const panelMap = Object.fromEntries((panels || []).filter(Boolean).map((item) => [item.id, item]));
  return (
    <section className="strategy-workspace-panel">
      <div className="strategy-workspace-head">
        <div>
          <span className="panel-kicker">Version 2</span>
          <h3>전략/분석 Workspace</h3>
          <p>같은 Semantic Layer와 Agent Workflow를 전략부서, 신용기획, 운영 관점으로 재배치합니다.</p>
        </div>
        <span className="sample-pill">shared backend</span>
      </div>
      <div className="strategy-workspace-grid">
        <FinancialToolCard tool={panelMap.explainability} />
        <FinancialToolCard tool={panelMap.cluster} />
        <FinancialToolCard tool={panelMap.policy} />
        <FinancialToolCard tool={panelMap.strategy} />
      </div>
      <div className="agent-workflow-strip">
        {(workflow || []).map((item) => (
          <div key={`${item.step}-${item.agent}`} className="agent-workflow-step">
            <span>{item.step}</span>
            <strong>{item.agent}</strong>
            <small>{item.llm_call ? 'Ollama 최소 호출' : 'Tool 처리'} · {item.output}</small>
          </div>
        ))}
      </div>
      <div className="semantic-layer-strip">
        {Object.entries(semanticLayer || {}).slice(0, 5).map(([key, value]) => (
          <span key={key}>{key}: {String(value)}</span>
        ))}
      </div>
    </section>
  );
}

function statusLabel(status) {
  switch (status) {
    case 'running':
      return 'Running';
    case 'completed':
      return 'Completed';
    case 'failed':
      return 'Failed';
    case 'warning':
      return 'Warning';
    default:
      return 'Standby';
  }
}

function formatRuntimeMs(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return '0.0s';
  }
  if (numericValue < 1000) {
    return `${Math.round(numericValue)}ms`;
  }
  return `${(numericValue / 1000).toFixed(numericValue >= 10000 ? 0 : 1)}s`;
}

function stageWaitCopy(stageKey) {
  switch (stageKey) {
    case 'extraction':
      return 'JSON과 로그 파일을 읽는 중';
    case 'alias':
      return '상품 범위와 후보 feature를 좁히는 중';
    case 'mapping':
      return '질문과 가까운 feature를 정렬하는 중';
    case 'ontology':
      return '핵심 기준과 영향 feature를 묶는 중';
    case 'faiss':
      return '고객군집 캐시와 분포를 확인하는 중';
    case 'retrieval':
      return '답변 근거를 모으는 중';
    case 'ollama':
      return '답변을 짧게 정리하는 중';
    default:
      return '파이프라인을 준비하는 중';
  }
}

function statusLabelKo(status) {
  switch (status) {
    case 'running':
      return '진행 중';
    case 'completed':
      return '완료';
    case 'failed':
      return '오류';
    case 'warning':
      return '확인 필요';
    default:
      return '대기';
  }
}

function stageWaitCopyKo(stageKey) {
  switch (stageKey) {
    case 'extraction':
      return '필요한 자료를 확인하는 중';
    case 'alias':
      return '상품 조건에 맞게 범위를 좁히는 중';
    case 'mapping':
      return '질문과 가까운 기준을 찾는 중';
    case 'ontology':
      return '가장 중요한 판단 기준을 고르는 중';
    case 'faiss':
      return '비슷한 고객 그룹을 확인하는 중';
    case 'retrieval':
      return '답변에 쓸 근거를 모으는 중';
    case 'ollama':
      return '답변을 쉽게 정리하는 중';
    default:
      return '준비하는 중';
  }
}

function stageShortLabel(stageKey) {
  const labels = {
    extraction: '자료 확인',
    alias: '상품 필터',
    mapping: '기준 찾기',
    ontology: '핵심 선택',
    faiss: '고객군 확인',
    retrieval: '근거 수집',
    ollama: '답변 정리',
  };
  return labels[stageKey] || '진행';
}

function bunnySpeech({ agentState, activeStageKey, showSuccess, isHovered, hasSubmittedRuntimeQuery }) {
  if (showSuccess) {
    return { message: '잘 배웠어요! 이제 그 내용을 반영해서 답할게요.', detail: '학습 완료', mood: 'success' };
  }
  if (!hasSubmittedRuntimeQuery && !isHovered) {
    return { message: '궁금한 걸 물어봐 주세요. 제가 같이 찾아볼게요!', detail: '대기 중', mood: 'idle' };
  }
  if (isHovered && agentState === 'idle') {
    return { message: '어떤 상품이나 고객 기준이 궁금한지 물어봐 주세요.', detail: '질문 기다리는 중', mood: 'listening' };
  }
  if (agentState === 'idle') {
    return { message: '지금은 쉬고 있어요. 궁금한 걸 던져 주세요.', detail: '대기 중', mood: 'idle' };
  }
  return { message: stageWaitCopyKo(activeStageKey), detail: '작업 중', mood: 'talking' };
}
function toneClass(status) {
  switch (status) {
    case 'running':
      return 'is-running';
    case 'completed':
      return 'is-completed';
    case 'failed':
      return 'is-failed';
    case 'warning':
      return 'is-warning';
    default:
      return 'is-idle';
  }
}

function SemanticNode({ data }) {
  return (
    <div className={`semantic-node semantic-node-${data.kind || 'runtime'} ${data.active ? 'active' : ''} ${data.selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} className="semantic-handle" />
      <div className="semantic-node-head">
        <span className={`semantic-node-indicator ${toneClass(data.status)}`} />
        <span className="semantic-node-title">{data.title}</span>
      </div>
      <strong className="semantic-node-key">{data.keyLabel}</strong>
      <p className="semantic-node-description">{data.description}</p>
      {data.meta?.length ? (
        <div className="semantic-node-meta">
          {data.meta.slice(0, 2).map((item) => (
            <span key={`${data.keyLabel}-${item}`} className="semantic-node-chip">{item}</span>
          ))}
        </div>
      ) : null}
      <Handle type="source" position={Position.Right} className="semantic-handle" />
    </div>
  );
}

const MemoSemanticNode = React.memo(SemanticNode);

function RoniEffectCanvas({ stageKey, reduceMotion = false }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || reduceMotion) {
      return undefined;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return undefined;
    }

    let frameId = 0;
    let startTime = performance.now();

    const paletteByStage = {
      extraction: ['rgba(255, 191, 105, 0.95)', 'rgba(255, 226, 176, 0.42)'],
      alias: ['rgba(255, 191, 105, 0.92)', 'rgba(143, 185, 214, 0.36)'],
      mapping: ['rgba(97, 244, 222, 0.92)', 'rgba(143, 185, 214, 0.34)'],
      ontology: ['rgba(110, 231, 183, 0.92)', 'rgba(97, 244, 222, 0.34)'],
      faiss: ['rgba(139, 168, 255, 0.92)', 'rgba(97, 244, 222, 0.32)'],
      retrieval: ['rgba(97, 244, 222, 0.92)', 'rgba(255, 191, 105, 0.32)'],
      ollama: ['rgba(255, 191, 105, 0.92)', 'rgba(255, 140, 168, 0.3)'],
    };

    const particles = Array.from({ length: 18 }, (_, index) => ({
      orbit: 52 + ((index % 6) * 14),
      radius: 1.8 + (index % 3),
      speed: 0.45 + ((index % 5) * 0.08),
      angle: (index / 18) * Math.PI * 2,
      drift: ((index % 4) - 1.5) * 0.09,
    }));

    function resizeCanvas() {
      const bounds = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function draw(now) {
      const elapsed = (now - startTime) / 1000;
      const { width, height } = canvas.getBoundingClientRect();
      const centerX = width / 2;
      const centerY = height / 2;
      const [primary, secondary] = paletteByStage[stageKey] || paletteByStage.mapping;

      context.clearRect(0, 0, width, height);

      for (let index = 0; index < 3; index += 1) {
        const radius = 58 + (index * 20) + (Math.sin((elapsed * 1.6) + index) * 6);
        context.beginPath();
        context.lineWidth = 1.2;
        context.strokeStyle = index % 2 === 0 ? secondary : primary;
        context.globalAlpha = 0.18 - (index * 0.04);
        context.arc(centerX, centerY, radius, 0, Math.PI * 2);
        context.stroke();
      }

      context.globalAlpha = 1;
      particles.forEach((particle, index) => {
        const angle = particle.angle + (elapsed * particle.speed) + (Math.sin(elapsed + index) * particle.drift);
        const x = centerX + (Math.cos(angle) * particle.orbit);
        const y = centerY + (Math.sin(angle) * (particle.orbit * 0.58));
        context.beginPath();
        context.fillStyle = index % 2 === 0 ? primary : secondary;
        context.globalAlpha = 0.34 + ((Math.sin(elapsed * 2.1 + index) + 1) * 0.14);
        context.arc(x, y, particle.radius + (Math.sin(elapsed * 1.8 + index) * 0.4), 0, Math.PI * 2);
        context.fill();
      });

      if (stageKey === 'mapping' || stageKey === 'retrieval' || stageKey === 'ollama') {
        context.globalAlpha = 0.32;
        context.strokeStyle = primary;
        context.lineWidth = 2;
        for (let index = 0; index < 2; index += 1) {
          const waveOffset = elapsed * 90 + (index * 46);
          context.beginPath();
          for (let x = 14; x <= width - 14; x += 8) {
            const y = centerY + Math.sin((x + waveOffset) / 28) * (12 + (index * 5));
            if (x === 14) {
              context.moveTo(x, y);
            } else {
              context.lineTo(x, y);
            }
          }
          context.stroke();
        }
      }

      context.globalAlpha = 1;
      frameId = window.requestAnimationFrame(draw);
    }

    resizeCanvas();
    frameId = window.requestAnimationFrame(draw);
    window.addEventListener('resize', resizeCanvas);

    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [reduceMotion, stageKey]);

  return <canvas ref={canvasRef} className="roni-effect-canvas" aria-hidden="true" />;
}

function RoniCoreIllustration({ stageKey, surprised = false, success = false, reduceMotion = false }) {
  const palette = {
    extraction: { primary: '#8fc8f5', secondary: '#dff4ff', accent: '#f2c57c' },
    alias: { primary: '#7bb8f0', secondary: '#e1f4ff', accent: '#8be0d4' },
    mapping: { primary: '#66c5eb', secondary: '#d7f4ff', accent: '#9fd6ff' },
    ontology: { primary: '#6fb8e8', secondary: '#e5f7ff', accent: '#90dfc8' },
    faiss: { primary: '#86b2f6', secondary: '#e5f0ff', accent: '#9fd6ff' },
    retrieval: { primary: '#59c8df', secondary: '#def7ff', accent: '#c3e8ff' },
    ollama: { primary: '#78b8ec', secondary: '#eef8ff', accent: '#ffd59e' },
  }[stageKey] || { primary: '#66c5eb', secondary: '#d7f4ff', accent: '#9fd6ff' };

  return (
    <motion.svg
      viewBox="0 0 220 220"
      className="roni-core-svg"
      aria-hidden="true"
      animate={reduceMotion ? undefined : { rotate: surprised ? [0, -2, 2, 0] : 0, scale: success ? [1, 1.03, 1] : 1 }}
      transition={{ duration: 3.4, repeat: Infinity, ease: 'easeInOut' }}
    >
      <circle className="roni-core-halo" cx="110" cy="110" r="88" style={{ fill: `${palette.primary}20`, stroke: `${palette.primary}55` }} />
      <circle className="roni-core-orbit" cx="110" cy="110" r="73" style={{ stroke: `${palette.accent}66` }} />
      <g className="roni-core-shell">
        <path d="M110 46c31 0 57 24 60 55l3 25c2 24-17 45-42 45H89c-25 0-44-21-42-45l3-25c3-31 29-55 60-55Z" style={{ fill: palette.primary }} />
        <path d="M110 57c23 0 42 17 45 40l2 18c2 17-12 32-30 32H93c-18 0-32-15-30-32l2-18c3-23 22-40 45-40Z" style={{ fill: '#10283a' }} />
        <path d="M79 78h62c10 0 18 8 18 18v15H61V96c0-10 8-18 18-18Z" style={{ fill: palette.secondary, opacity: 0.18 }} />
        <rect x="83" y="91" width="20" height={surprised ? '16' : '12'} rx="6" style={{ fill: palette.secondary }} />
        <rect x="117" y="91" width="20" height={surprised ? '16' : '12'} rx="6" style={{ fill: palette.secondary }} />
        <path d={success ? 'M84 126c9 11 18 16 26 16s17-5 26-16' : surprised ? 'M96 126c4-7 10-11 14-11s10 4 14 11' : 'M90 129c7 7 14 10 20 10s13-3 20-10'} style={{ stroke: success ? '#90dfc8' : palette.secondary, fill: 'none', strokeWidth: '5', strokeLinecap: 'round' }} />
        <path d="M74 66c8-11 20-18 36-20M146 66c-8-11-20-18-36-20" style={{ stroke: `${palette.accent}aa`, fill: 'none', strokeWidth: '4', strokeLinecap: 'round' }} />
        <circle cx="110" cy="75" r="6" style={{ fill: palette.accent, opacity: 0.9 }} />
      </g>
      <g className="roni-core-satellites">
        <circle cx="52" cy="94" r="6" style={{ fill: palette.accent }} />
        <circle cx="169" cy="83" r="5" style={{ fill: palette.secondary }} />
        <circle cx="161" cy="149" r="7" style={{ fill: palette.accent }} />
      </g>
    </motion.svg>
  );
}

/** @typedef {'idle' | 'retrieving' | 'graph_expanding' | 'thinking' | 'clarifying' | 'explaining' | 'learning'} AgentState */

const AGENT_STATES = Object.freeze(['idle', 'retrieving', 'graph_expanding', 'thinking', 'clarifying', 'explaining', 'learning']);
const THINKING_PHASES = Object.freeze(['intake', 'refine', 'commit']);

function deriveThinkingPhase(activeStageKey) {
  if (activeStageKey === 'extraction') {
    return 'intake';
  }
  if (activeStageKey === 'alias' || activeStageKey === 'mapping' || activeStageKey === 'ontology') {
    return 'refine';
  }
  return 'commit';
}

function deriveAgentState({
  activeStageKey,
  runtimeJobStatus,
  hasWorkbench,
  isRoniHovered,
  regulationBusy,
  showRoniSuccess,
  hasError,
}) {
  if (regulationBusy || showRoniSuccess) {
    return 'learning';
  }
  if (isRoniHovered) {
    return 'clarifying';
  }
  if (hasError) {
    return 'clarifying';
  }
  if (hasWorkbench && runtimeJobStatus === 'completed') {
    return 'explaining';
  }
  if (activeStageKey === 'retrieval' || activeStageKey === 'faiss') {
    return 'retrieving';
  }
  if (activeStageKey === 'ontology' || activeStageKey === 'mapping') {
    return 'graph_expanding';
  }
  if (runtimeJobStatus === 'running' || runtimeJobStatus === 'queued') {
    return 'thinking';
  }
  return 'idle';
}

function agentStateLabel(agentState, thinkingPhase = 'intake') {
  if (agentState === 'thinking') {
    if (thinkingPhase === 'intake') {
      return '질문 확인';
    }
    if (thinkingPhase === 'refine') {
      return '기준 정리';
    }
    return '답변 준비';
  }
  const labels = {
    idle: '대기 중',
    retrieving: '근거 수집',
    graph_expanding: '관계 확인',
    clarifying: '질문 확인',
    explaining: '답변 중',
    learning: '문서 학습',
  };
  return labels[agentState] || '진행 중';
}

function getRiveEventText(eventData) {
  const properties = eventData?.properties && typeof eventData.properties === 'object'
    ? Object.entries(eventData.properties).flatMap(([key, value]) => [key, value])
    : [];
  return [
    eventData?.name,
    eventData?.url,
    eventData?.target,
    ...properties,
  ].filter((value) => value !== undefined && value !== null).join(' ').toLowerCase();
}

function resolveRiveEventAction(eventData) {
  const text = getRiveEventText(eventData);
  if (!text) {
    return null;
  }
  if (/document|regulation|upload|pencil|edit|pen|file|규제|문서/.test(text)) {
    return { key: 'document', state: 'learning', phase: 'commit' };
  }
  if (/listen|clarif|question|intent|mic|voice/.test(text)) {
    return { key: 'listen', state: 'clarifying', phase: 'intake' };
  }
  if (/retriev|search|hand|fetch|scan/.test(text)) {
    return { key: 'retrieve', state: 'retrieving', phase: 'refine' };
  }
  if (/explain|talk|answer|star|summary/.test(text)) {
    return { key: 'explain', state: 'explaining', phase: 'commit' };
  }
  return { key: 'event', state: 'thinking', phase: 'intake' };
}

function RoniAvatar({
  stageKey,
  agentState = 'idle',
  thinkingPhase = 'intake',
  liveSpeech,
  reduceMotion = false,
  surprised = false,
  success = false,
  onDocumentRequest,
}) {
  const BUNNY_STATE_MACHINE = 'State Machine 1';
  const [manualAction, setManualAction] = useState(null);
  const safeAgentState = AGENT_STATES.includes(agentState) ? agentState : 'idle';
  const safeThinkingPhase = THINKING_PHASES.includes(thinkingPhase) ? thinkingPhase : 'intake';
  const effectiveAgentState = manualAction?.state || safeAgentState;
  const effectiveThinkingPhase = manualAction?.phase || safeThinkingPhase;
  const stageTheme = {
    idle: { aura: 'sky' },
    retrieving: { aura: 'cyan' },
    graph_expanding: { aura: 'mint' },
    thinking: { aura: 'amber' },
    clarifying: { aura: 'sky' },
    explaining: { aura: 'blue' },
    learning: { aura: 'mint' },
  }[effectiveAgentState] || { aura: 'sky' };

  const { RiveComponent, rive } = useRive({
    src: '/interactive-bunny-character.riv',
    stateMachines: BUNNY_STATE_MACHINE,
    autoplay: true,
    automaticallyHandleEvents: false,
    shouldDisableRiveListeners: false,
  });

  const inputListeningA = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'isListening');
  const inputListeningB = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'listening');
  const inputListeningC = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'Listening');
  const inputThinkingA = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'isThinking');
  const inputThinkingB = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'thinking');
  const inputThinkingC = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'Thinking');
  const inputTalkingA = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'isTalking');
  const inputTalkingB = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'talking');
  const inputTalkingC = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'Talking');
  const poseTriggerInput = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'Trigger 1');
  const poseInput = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'Pose');
  const clickInput = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'Click');
  const buttonInput = useStateMachineInput(rive, BUNNY_STATE_MACHINE, 'tombol');

  useEffect(() => {
    const listeningInputs = [inputListeningA, inputListeningB, inputListeningC].filter(Boolean);
    const thinkingInputs = [inputThinkingA, inputThinkingB, inputThinkingC].filter(Boolean);
    const talkingInputs = [inputTalkingA, inputTalkingB, inputTalkingC].filter(Boolean);

    const isListening = effectiveAgentState === 'clarifying' || (effectiveAgentState === 'thinking' && (stageKey === 'extraction' || stageKey === 'alias'));
    const isThinking = effectiveAgentState === 'thinking' || effectiveAgentState === 'retrieving' || effectiveAgentState === 'graph_expanding' || effectiveAgentState === 'learning';
    const isTalking = effectiveAgentState === 'explaining' || Boolean(success);

    listeningInputs.forEach((input) => {
      input.value = isListening;
    });
    thinkingInputs.forEach((input) => {
      input.value = isThinking;
    });
    talkingInputs.forEach((input) => {
      input.value = isTalking;
    });
  }, [
    inputListeningA,
    inputListeningB,
    inputListeningC,
    inputThinkingA,
    inputThinkingB,
    inputThinkingC,
    inputTalkingA,
    inputTalkingB,
    inputTalkingC,
    effectiveAgentState,
    stageKey,
    success,
  ]);

  useEffect(() => {
    if (!manualAction) {
      return undefined;
    }
    const timerId = window.setTimeout(() => setManualAction(null), 1200);
    return () => window.clearTimeout(timerId);
  }, [manualAction]);

  useEffect(() => {
    if (!rive) {
      return undefined;
    }

    const activatePoseAnimation = () => {
      const triggerCandidates = [poseTriggerInput, clickInput, buttonInput].filter(Boolean);
      triggerCandidates.forEach((input) => {
        input.fire?.();
      });

      if (poseInput) {
        if (typeof poseInput.value === 'number') {
          poseInput.value = 1;
        } else if (typeof poseInput.value === 'boolean') {
          poseInput.value = true;
          window.setTimeout(() => {
            if (poseInput) {
              poseInput.value = false;
            }
          }, 180);
        } else {
          poseInput.fire?.();
        }
      }
    };

    const handleRiveEvent = (event) => {
      const action = resolveRiveEventAction(event?.data);
      if (!action) {
        return;
      }
      activatePoseAnimation();
      setManualAction(action);
      if (action.key === 'document') {
        onDocumentRequest?.();
      }
    };

    rive.on(EventType.RiveEvent, handleRiveEvent);
    return () => rive.off(EventType.RiveEvent, handleRiveEvent);
  }, [buttonInput, clickInput, onDocumentRequest, poseInput, poseTriggerInput, rive]);

  const avatarMotion = reduceMotion
    ? undefined
    : {
      y: effectiveAgentState === 'retrieving' ? [0, -9, 0] : effectiveAgentState === 'learning' ? [0, -11, 0] : [0, -6, 0],
      rotate: surprised ? [0, -2, 2, 0] : effectiveAgentState === 'clarifying' ? [0, -1.4, 1.4, 0] : [0, -0.8, 0.8, 0],
      scale: success || effectiveAgentState === 'learning' ? [1, 1.05, 1] : effectiveAgentState === 'explaining' ? [1, 1.02, 1] : [1, 1.015, 1],
    };
  return (
    <div className="roni-avatar-shell" aria-label="Bunny semantic runtime indicator">
      <motion.div
        className={`roni-cloud-scene roni-cloud-${stageTheme.aura} state-${effectiveAgentState} thinking-${effectiveThinkingPhase} ${manualAction ? 'is-action-triggered' : ''} ${surprised ? 'is-surprised' : ''} ${success ? 'is-success' : ''}`}
        initial={{ opacity: 0, y: 10, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.32, ease: 'easeOut' }}
      >
        <motion.div
          className="roni-cloud-avatar"
          animate={avatarMotion}
          transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut' }}
        >
          <RiveComponent className="roni-bunny-rive" aria-label="Bunny assistant avatar" />
        </motion.div>
        <motion.div
          key={`${liveSpeech?.message || ''}-${liveSpeech?.detail || ''}`}
          className={`roni-live-speech mood-${liveSpeech?.mood || 'idle'}`}
          initial={{ opacity: 0, y: 8, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.24, ease: 'easeOut' }}
          aria-live="polite"
        >
          <div className="roni-live-speech-head">
            <span>Bunny</span>
            <i aria-hidden="true"><b /><b /><b /></i>
          </div>
          <p>
            <StreamingText
              text={liveSpeech?.message || agentStateLabel(effectiveAgentState, effectiveThinkingPhase)}
              active={!reduceMotion}
              reduceMotion={reduceMotion}
              speed={18}
            />
          </p>
        </motion.div>
        <span className="roni-cloud-stage-tag">{liveSpeech?.detail || agentStateLabel(effectiveAgentState, effectiveThinkingPhase)}</span>
      </motion.div>
    </div>
  );
}

const NODE_DETAILS = {
  log_source: {
    title: '심사 로그 입력',
    subtitle: 'Runtime log fragments',
    relation: ['IN_AGE', 'IN_INCOME', 'OVERDUE_HISTORY'],
    products: ['C11', 'C12'],
    clusters: ['Raw Event Stream'],
  },
  canonical_applicant_age: {
    title: 'Canonical Feature',
    subtitle: 'applicant.age',
    relation: ['IN_연령', '연령', 'AGE', 'applicant.age'],
    products: ['C6', 'C9', 'C11', 'C12'],
    clusters: ['cluster_40_office_worker', 'cluster_prime_income'],
  },
  cluster_40_office_worker: {
    title: 'Customer Cluster',
    subtitle: '40s Office Worker',
    relation: ['applicant.age', 'income.monthly', 'overdue.history'],
    products: ['C11', 'C12'],
    clusters: ['income.monthly', 'dsr.level'],
  },
  feature_income: {
    title: 'Related Feature',
    subtitle: 'income.monthly',
    relation: ['cluster_40_office_worker', 'dsr.level'],
    products: ['C9', 'C11', 'C12'],
    clusters: ['Prime Income Growth'],
  },
  query_node: {
    title: 'User Query',
    subtitle: 'Semantic intent',
    relation: ['Ontology Retrieval', 'Cluster Retrieval'],
    products: ['C12'],
    clusters: ['Question Context'],
  },
};

function buildGraph({ currentStageIndex, selectedNodeId, question, runtimeStages, workbenchData }) {
  const activeStageKey = RUNTIME_STAGES[currentStageIndex]?.key || 'extraction';
  const stageStatus = Object.fromEntries(runtimeStages.map((item) => [item.key, item.status]));
  const retrievalTop = (workbenchData?.retrieval_results || []).slice(0, 3).map((item) => item.record_id || item.product || 'candidate');
  const primarySelection = workbenchData?.primary_feature_selection || {};
  const selectedFeature = workbenchData?.selected_feature || {};
  const representativeFeatures = (primarySelection?.representative_features || workbenchData?.representative_features || (selectedFeature?.feature_id ? [selectedFeature] : [])).slice(0, 3);
  const relatedFeatures = workbenchData?.related_features || [];
  const tokenMappings = (workbenchData?.question_token_mappings || []).slice(0, 6);
  const topCandidates = (primarySelection?.reference_features || primarySelection?.top_k || []).slice(0, 3);
  const selectedCluster = (workbenchData?.customer_clusters || [])[0] || {};
  const questionTokens = tokenMappings.length ? tokenMappings.map((item) => item.token) : String(question || '').split(/\s+/).filter(Boolean).slice(0, 6);
  const representativeFeatureNames = representativeFeatures.map((item) => item?.feature_name || item?.feature_id).filter(Boolean);
  const selectedFeatureName = representativeFeatureNames.join(' / ') || selectedFeature?.feature_name || selectedFeature?.feature_id || '대표 축';
  const supportLabels = (relatedFeatures || []).slice(0, 3).map((item) => item.feature_name || item.feature_id).filter(Boolean);
  const topCandidateLabels = topCandidates.map((item) => item.feature_name || item.feature_id).filter(Boolean);

  const tokenNodes = tokenMappings.map((item, index) => ({
    id: `token-node-${index}`,
    type: 'semantic',
    position: { x: 420 + ((index % 3) * 170), y: 20 + (Math.floor(index / 3) * 82) },
    data: {
      title: item?.direct_feature_match ? '직접 토큰' : '문맥 토큰',
      keyLabel: item?.token || `token-${index + 1}`,
      description: item?.reason || '질문 해석에 사용한 토큰입니다.',
      kind: item?.direct_feature_match ? 'entity' : 'question',
      status: stageStatus.alias,
      active: activeStageKey === 'alias' || activeStageKey === 'mapping',
      selected: selectedNodeId === `token-node-${index}`,
      meta: item?.feature_links?.length ? item.feature_links.slice(0, 1).map((link) => link.feature_name || link.feature_id) : [item?.concept_label || '연결 feature 없음'],
    },
  }));

  const candidateNodes = topCandidates.map((item, index) => ({
    id: `candidate-node-${item.feature_id || index}`,
    type: 'semantic',
    position: { x: 960 + (index * 230), y: 20 },
    data: {
      title: `Top-K Feature #${index + 1}`,
      keyLabel: item.feature_name || item.feature_id || `후보 ${index + 1}`,
      description: `base ${Number(item.base_score || 0).toFixed(2)} / intent ${Number(item.intent_score || 0).toFixed(2)} / graph ${Number(item.graph_score || 0).toFixed(2)}`,
      kind: 'entity',
      status: stageStatus.mapping,
      active: activeStageKey === 'mapping' || activeStageKey === 'ontology',
      selected: selectedNodeId === `candidate-node-${item.feature_id || index}`,
      meta: [`hybrid ${Number(item.hybrid_score || 0).toFixed(2)}`, ...(item.support_labels || []).slice(0, 1)],
    },
  }));

  const nodes = [
    { id: 'log_source', type: 'semantic', position: { x: 10, y: 180 }, data: { title: 'User Query', keyLabel: question || '질문', description: '사용자 질문을 파이프라인에 넣습니다.', kind: 'source', status: stageStatus.extraction, active: activeStageKey === 'extraction', selected: selectedNodeId === 'log_source', meta: ['query'] } },
    { id: 'feature_extraction', type: 'semantic', position: { x: 270, y: 180 }, data: { title: 'Intent Router', keyLabel: '질문 의도 분기', description: '나이, 상품, 승인, 거절 같은 토큰을 해석합니다.', kind: 'pipeline', status: stageStatus.extraction, active: activeStageKey === 'extraction', selected: selectedNodeId === 'feature_extraction', meta: questionTokens.slice(0, 2) } },
    { id: 'alias_resolution', type: 'semantic', position: { x: 560, y: 180 }, data: { title: 'Ontology Domain Select', keyLabel: '도메인 선택', description: `질문 토큰 ${questionTokens.join(', ') || '없음'}을 ontology feature 검색에 연결합니다.`, kind: 'pipeline', status: stageStatus.alias, active: activeStageKey === 'alias', selected: selectedNodeId === 'alias_resolution', meta: questionTokens.length ? questionTokens.slice(0, 3) : ['질문 토큰'] } },
    { id: 'canonical_mapping', type: 'semantic', position: { x: 860, y: 180 }, data: { title: 'Scoped FAISS Search', keyLabel: '범위 축소 검색', description: '상품과 도메인 범위로 feature 검색 공간을 줄입니다.', kind: 'pipeline', status: stageStatus.mapping, active: activeStageKey === 'mapping', selected: selectedNodeId === 'canonical_mapping', meta: topCandidateLabels.length ? topCandidateLabels : ['후보 계산 중'] } },
    { id: 'ontology_update', type: 'semantic', position: { x: 1160, y: 180 }, data: { title: 'Top-K Features', keyLabel: '후보 3개 압축', description: `상위 feature 후보 ${topCandidateLabels.length || 0}개를 묶었습니다.`, kind: 'pipeline', status: stageStatus.ontology, active: activeStageKey === 'ontology', selected: selectedNodeId === 'ontology_update', meta: topCandidateLabels.length ? topCandidateLabels : ['후보 계산 중'] } },
    { id: 'faiss_indexing', type: 'semantic', position: { x: 1460, y: 180 }, data: { title: 'Graph Expansion', keyLabel: '주변 연결 확장', description: `${selectedFeatureName} 주변의 관련 feature를 확장합니다.`, kind: 'pipeline', status: stageStatus.faiss, active: activeStageKey === 'faiss', selected: selectedNodeId === 'faiss_indexing', meta: supportLabels.length ? supportLabels : ['연결 근거 계산 중'] } },
    { id: 'semantic_retrieval', type: 'semantic', position: { x: 1760, y: 180 }, data: { title: 'LLM Context', keyLabel: '근거 컨텍스트', description: '대표 축, cluster, retrieval 결과를 최종 설명용 컨텍스트로 묶습니다.', kind: 'pipeline', status: stageStatus.retrieval, active: activeStageKey === 'retrieval', selected: selectedNodeId === 'semantic_retrieval', meta: retrievalTop.length ? retrievalTop : ['context'] } },
    { id: 'ollama_node', type: 'semantic', position: { x: 2010, y: 180 }, data: { title: 'Answer Summary', keyLabel: 'final narrative', description: 'LLM 또는 fallback summary가 최종 답변을 만듭니다.', kind: 'pipeline', status: stageStatus.ollama, active: activeStageKey === 'ollama', selected: selectedNodeId === 'ollama_node', meta: ['summary'] } },
    ...tokenNodes,
    ...candidateNodes,
    { id: 'canonical_applicant_age', type: 'semantic', position: { x: 1660, y: 40 }, data: { title: '대표 Axes', keyLabel: selectedFeatureName, description: 'top-3 후보를 함께 보고 최종 묶은 중심 축입니다.', kind: 'canonical', status: stageStatus.ontology, active: activeStageKey === 'ontology' || activeStageKey === 'retrieval', selected: selectedNodeId === 'canonical_applicant_age', meta: representativeFeatureNames.length ? [`axes ${representativeFeatureNames.length}`, representativeFeatureNames[0]] : ['대표 선택 중'] } },
    { id: 'query_node', type: 'semantic', position: { x: 1335, y: 360 }, data: { title: 'User Question', keyLabel: question, description: '운영자가 보낸 질의의 실행 상태입니다.', kind: 'question', status: stageStatus.retrieval, active: activeStageKey === 'retrieval' || activeStageKey === 'ollama', selected: selectedNodeId === 'query_node', meta: ['semantic intent'] } },
    { id: 'cluster_40_office_worker', type: 'semantic', position: { x: 1595, y: 360 }, data: { title: 'Cluster Retrieval', keyLabel: selectedCluster?.label || 'cluster', description: `${selectedFeatureName}를 바탕으로 가까운 고객군을 찾습니다.`, kind: 'cluster', status: stageStatus.retrieval, active: activeStageKey === 'retrieval', selected: selectedNodeId === 'cluster_40_office_worker', meta: [selectedCluster?.cluster_id || 'cluster', `records ${(selectedCluster?.count || 0)}`] } },
    { id: 'feature_income', type: 'semantic', position: { x: 1835, y: 320 }, data: { title: 'Graph Support #1', keyLabel: relatedFeatures[0]?.feature_name || relatedFeatures[0]?.feature_id || 'support 1', description: relatedFeatures[0]?.description || '대표 축을 지지하는 첫 번째 연결입니다.', kind: 'feature', status: stageStatus.retrieval, active: activeStageKey === 'retrieval', selected: selectedNodeId === 'feature_income', meta: ['graph support'] } },
    { id: 'feature_dsr', type: 'semantic', position: { x: 1835, y: 420 }, data: { title: 'Graph Support #2', keyLabel: relatedFeatures[1]?.feature_name || relatedFeatures[1]?.feature_id || 'support 2', description: relatedFeatures[1]?.description || '대표 축을 지지하는 두 번째 연결입니다.', kind: 'feature', status: stageStatus.retrieval, active: activeStageKey === 'retrieval', selected: selectedNodeId === 'feature_dsr', meta: ['graph support'] } },
    { id: 'feature_overdue', type: 'semantic', position: { x: 2045, y: 370 }, data: { title: 'Related Feature', keyLabel: relatedFeatures[2]?.feature_name || relatedFeatures[2]?.feature_id || 'related', description: relatedFeatures[2]?.description || '대표 축과 함께 설명되는 연관 feature입니다.', kind: 'feature', status: relatedFeatures[2] ? stageStatus.retrieval : 'warning', active: activeStageKey === 'retrieval', selected: selectedNodeId === 'feature_overdue', meta: ['연관 feature'] } },
  ];

  const pipelineEdges = [
    ['log_source', 'feature_extraction', 'extraction'],
    ['feature_extraction', 'alias_resolution', 'alias'],
    ['alias_resolution', 'canonical_mapping', 'mapping'],
    ['canonical_mapping', 'ontology_update', 'ontology'],
    ['ontology_update', 'faiss_indexing', 'faiss'],
    ['faiss_indexing', 'semantic_retrieval', 'retrieval'],
    ['semantic_retrieval', 'ollama_node', 'ollama'],
  ].map(([source, target, stage]) => ({
    id: `${source}-${target}`,
    source,
    target,
    animated: activeStageKey === stage,
    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
    style: { stroke: activeStageKey === stage ? '#61f4de' : 'rgba(143,185,214,0.28)', strokeWidth: activeStageKey === stage ? 2.4 : 1.4 },
  }));

  const detailEdges = [
    ['query_node', 'canonical_applicant_age', activeStageKey === 'retrieval'],
    ['canonical_applicant_age', 'cluster_40_office_worker', activeStageKey === 'retrieval'],
    ['cluster_40_office_worker', 'feature_income', activeStageKey === 'retrieval'],
    ['cluster_40_office_worker', 'feature_dsr', activeStageKey === 'retrieval'],
    ['feature_dsr', 'feature_overdue', activeStageKey === 'retrieval'],
    ['semantic_retrieval', 'query_node', activeStageKey === 'retrieval' || activeStageKey === 'ollama'],
    ['ollama_node', 'query_node', activeStageKey === 'ollama'],
  ].map(([source, target, animated], index) => ({
    id: `detail-${source}-${target}-${index}`,
    source,
    target,
    animated,
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
    style: { stroke: animated ? '#ffbf69' : 'rgba(255,255,255,0.16)', strokeWidth: animated ? 2.2 : 1.2 },
  }));

  const tokenFeedEdges = tokenNodes.map((node, index) => ({
    id: `token-feed-${index}`,
    source: 'alias_resolution',
    target: node.id,
    animated: activeStageKey === 'alias',
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
    style: { stroke: activeStageKey === 'alias' ? '#ffbf69' : 'rgba(255,255,255,0.14)', strokeWidth: activeStageKey === 'alias' ? 2 : 1.1 },
  }));

  const candidateFeedEdges = candidateNodes.map((node, index) => ({
    id: `candidate-feed-${index}`,
    source: 'canonical_mapping',
    target: node.id,
    animated: activeStageKey === 'mapping' || activeStageKey === 'ontology',
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
    style: { stroke: activeStageKey === 'mapping' || activeStageKey === 'ontology' ? '#61f4de' : 'rgba(255,255,255,0.14)', strokeWidth: activeStageKey === 'mapping' || activeStageKey === 'ontology' ? 2 : 1.1 },
  }));

  const tokenToCandidateEdges = tokenMappings.flatMap((item, tokenIndex) => (item?.feature_links || []).slice(0, 3).map((link, linkIndex) => ({
    id: `token-link-${tokenIndex}-${link.feature_id || linkIndex}`,
    source: `token-node-${tokenIndex}`,
    target: `candidate-node-${link.feature_id || linkIndex}`,
    animated: activeStageKey === 'mapping',
    markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12 },
    style: { stroke: activeStageKey === 'mapping' ? '#8bb6ff' : 'rgba(139,182,255,0.24)', strokeWidth: 1.8 },
  })));

  const candidateToRepresentativeEdges = topCandidates.map((item, index) => ({
    id: `candidate-rep-${item.feature_id || index}`,
    source: `candidate-node-${item.feature_id || index}`,
    target: 'canonical_applicant_age',
    animated: activeStageKey === 'ontology',
    markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12 },
    style: { stroke: activeStageKey === 'ontology' ? '#7ef0b0' : 'rgba(126,240,176,0.24)', strokeWidth: activeStageKey === 'ontology' ? 2.1 : 1.1 },
  }));

  return { nodes, edges: [...pipelineEdges, ...detailEdges, ...tokenFeedEdges, ...candidateFeedEdges, ...tokenToCandidateEdges, ...candidateToRepresentativeEdges] };
}
const NODE_TYPES = Object.freeze({ semantic: MemoSemanticNode });
const REACT_FLOW_PRO_OPTIONS = Object.freeze({ hideAttribution: true });
const REACT_FLOW_DEFAULT_VIEWPORT = Object.freeze({ x: 0, y: 0, zoom: 0.73 });
const ONTOLOGY_SUBTABS = Object.freeze([
  { id: 'question', label: '1. 질문 해석', kicker: 'input parsing' },
  { id: 'feature', label: '2. 축 선택', kicker: 'semantic rank' },
  { id: 'cluster', label: '3. 군집 검색', kicker: 'retrieval build' },
  { id: 'answer', label: '4. 답변 생성', kicker: 'summary build' },
]);

const PRODUCT_KEYWORD_MAP = Object.freeze({
  C6: ['이지신용대출', '신용대출', '신용'],
  C9: ['이지론', '카드론', 'card loan'],
  C11: ['개인사업자대출', '개인사업자', '사업자대출', '사업자'],
  C12: ['이지대환대출', '대환대출', '대환'],
});

const PRODUCT_DISPLAY_NAMES = Object.freeze({
  C6: '이지신용대출(C6)',
  C9: '이지론(C9)',
  C11: '개인사업자대출(C11)',
  C12: '이지대환대출(C12)',
});

function getProductDisplayName(codeOrName = '') {
  const value = String(codeOrName || '').trim();
  if (!value) {
    return '-';
  }
  const upperValue = value.toUpperCase();
  if (PRODUCT_DISPLAY_NAMES[upperValue]) {
    return PRODUCT_DISPLAY_NAMES[upperValue];
  }
  const matchedCode = Object.entries(PRODUCT_DISPLAY_NAMES).find(([, label]) => label === value)?.[0];
  return matchedCode ? PRODUCT_DISPLAY_NAMES[matchedCode] : value;
}

function getAnswerSourceTag(answerSummary = {}, ollamaStatus = '') {
  const source = String(answerSummary?.source || '').toLowerCase();
  if (source.includes('regulation')) {
    return '규제문서';
  }
  if (source.includes('ollama-general')) {
    return '일반답변';
  }
  if (source.includes('log-cluster') || source.includes('customer-cluster')) {
    return '심사로그';
  }
  if (String(ollamaStatus || '') === 'completed' || source.includes('ollama')) {
    return 'ollama-linked';
  }
  return 'server-linked';
}

function shouldShowProductChip(product = '', sourceTag = '') {
  const normalizedProduct = String(product || '').trim().toUpperCase();
  const normalizedSource = String(sourceTag || '').trim();
  if (!normalizedProduct || normalizedProduct === 'ALL') {
    return false;
  }
  return !['규제문서', '일반답변'].includes(normalizedSource);
}

function buildCitationHighlights(citation = {}, query = '') {
  const explicitHighlights = Array.isArray(citation?.highlights) ? citation.highlights : [];
  if (explicitHighlights.length) {
    return explicitHighlights.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 2);
  }
  const snippet = String(citation?.snippet || '').replace(/\s+/g, ' ').trim();
  if (!snippet) {
    return [];
  }
  const queryTokens = String(query || '')
    .toLowerCase()
    .split(/[^0-9a-z\u3131-\u318e\uac00-\ud7a3.]+/g)
    .map((item) => item.replace(/[은는이가을를]$/g, ''))
    .filter((item) => item.length >= 2 && !['오늘', '최근', '현재', '무엇', '뭐야', '알려줘'].includes(item));
  const priorityTokens = Array.from(new Set([...queryTokens, 'dsr', '3단계', '스트레스', '시행', '2025', '25.7.1']));
  const sentences = snippet
    .split(/(?<=[.!?。])\s+|(?<=다)\s+/g)
    .map((item) => item.trim())
    .filter(Boolean);
  const scored = sentences
    .map((sentence) => {
      const compact = sentence.toLowerCase().replace(/\s+/g, '');
      const score = priorityTokens.reduce((total, token) => total + (compact.includes(token.replace(/\s+/g, '')) ? 1 : 0), 0);
      return { sentence, score };
    })
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score);
  return (scored.length ? scored : sentences.map((sentence) => ({ sentence, score: 0 })))
    .map((item) => item.sentence)
    .slice(0, 2);
}

const ANSWER_MODES = Object.freeze([
  { id: 'general', label: '일반답변모드', icon: '💬', hint: '현재 대화형 답변' },
  { id: 'memo', label: '메모모드', icon: '📝', hint: '부서 메모 우선 반영' },
  { id: 'product', label: '상품개발모드', icon: '✨', hint: '전략·부서 관점 중심' },
]);

const MEMO_DEPARTMENTS = Object.freeze([
  { id: 'solution', label: '금융솔루션부', icon: '🧩', defaultConcept: '신상품 총괄과 계획을 세우며 규제, 뉴스, 시장 변화를 함께 봅니다.' },
  { id: 'credit', label: '신용기획부', icon: '📊', defaultConcept: '심사솔루션 기준으로 상품별 금리, 한도, 거절코드와 리스크 통제를 봅니다.' },
  { id: 'sales', label: '금융영업부', icon: '🚀', defaultConcept: '취급량과 취급률 목표를 중시하며 승인 가능 고객을 적극적으로 찾습니다.' },
  { id: 'it', label: 'IT개발자', icon: '💻', defaultConcept: 'KCB, NICE, 신정원, 공공마이데이터 연계와 개발공수, 정합성 검증을 중시합니다.' },
]);

const PRODUCT_DEBATE_PEOPLE = Object.freeze([
  { id: 'solution', department: '금융솔루션부', name: '금프로', short: '금', tone: 'solution', role: '상품 구조 총괄' },
  { id: 'credit', department: '신용기획부', name: '신프로', short: '신', tone: 'risk', role: '리스크 기준선' },
  { id: 'sales', department: '금융영업부', name: '영프로', short: '영', tone: 'sales', role: '승인 전환 전략' },
  { id: 'it', department: 'IT개발자', name: '아프로', short: '아', tone: 'tech', role: '데이터/개발 검증' },
]);

const PRODUCT_DEBATE_WARMUP_LINES = Object.freeze([
  { id: 'hello-solution', speaker: '금프로', department: '금융솔루션부', tone: 'solution', message: '회의실 들어왔습니다. 오늘 안건은 작게 실험하되, 기존 심사 룰은 크게 흔들지 않는 방향으로 열어볼게요.' },
  { id: 'hello-credit', speaker: '신프로', department: '신용기획부', tone: 'risk', message: '좋습니다. 저는 먼저 리스크 상한선을 잡겠습니다. 승인 전환은 하되, 연체 가능성 높은 구간은 분명히 분리해야 합니다.' },
  { id: 'hello-sales', speaker: '영프로', department: '금융영업부', tone: 'sales', message: '저는 거절 고객 중 어디까지 되살릴 수 있는지 보겠습니다. 현장에서는 조건이 명확해야 바로 움직일 수 있어요.' },
  { id: 'hello-it', speaker: '아프로', department: 'IT개발자', tone: 'tech', message: '저는 필요한 데이터가 실제로 붙는지 볼게요. KCB, NICE, 신정원, 공공마이데이터 연계 난이도도 같이 체크합니다.' },
  { id: 'smalltalk-1', speaker: '금프로', department: '금융솔루션부', tone: 'solution', message: '커피는 각자 챙기셨죠? OLLAMA가 생각하는 동안 우리는 안건의 뼈대를 먼저 맞춰두겠습니다.' },
  { id: 'smalltalk-2', speaker: '신프로', department: '신용기획부', tone: 'risk', message: '핵심은 씬파일 고객을 무작정 승인하지 않는 겁니다. 소액, 단기, 관찰 가능 조건이면 논의할 수 있습니다.' },
  { id: 'smalltalk-3', speaker: '영프로', department: '금융영업부', tone: 'sales', message: '그럼 고객 안내 문구도 중요하겠네요. “조건부 승인”처럼 이해하기 쉬운 언어가 필요합니다.' },
  { id: 'smalltalk-4', speaker: '아프로', department: 'IT개발자', tone: 'tech', message: '실시간 소득/재직 확인이 들어가면 개발 공수가 늘어납니다. 대신 룰이 명확하면 배치 검증부터 작게 시작할 수 있어요.' },
  { id: 'debate-1', speaker: '금프로', department: '금융솔루션부', tone: 'solution', message: '좋아요. 신상품 후보와 기존 상품 보완안을 분리해서, 최종 산출물은 실험 상품 1개와 룰 개선안으로 묶겠습니다.' },
]);

const MEMO_STORAGE_KEY = 'ontology-workbench-memo-preferences';
function normalizeSearchText(value) {
  return String(value || '').toLowerCase().replace(/\s+/g, ' ').trim();
}

function resolveProductForQuestion(question, products) {
  const normalizedQuestion = normalizeSearchText(question);
  if (!normalizedQuestion) {
    return '';
  }

  const compactQuestion = normalizedQuestion.replace(/[^0-9a-z\u3131-\u318e\uac00-\ud7a3]+/g, '');
  const scoredProducts = (products || []).map((product) => {
    const terms = [product?.code, product?.label, ...(PRODUCT_KEYWORD_MAP[product?.code] || [])]
      .filter(Boolean)
      .map((item) => normalizeSearchText(item));
    let score = 0;
    for (const term of terms) {
      if (!term) {
        continue;
      }
      if (normalizedQuestion.includes(term)) {
        score += Math.max(4, term.length);
      }
      const compactTerm = term.replace(/[^0-9a-z\u3131-\u318e\uac00-\ud7a3]+/g, '');
      if (compactTerm && compactQuestion.includes(compactTerm)) {
        score += Math.max(3, compactTerm.length - 1);
      }
    }
    return { code: product?.code || '', score };
  });

  scoredProducts.sort((left, right) => right.score - left.score);
  return scoredProducts[0]?.score > 0 ? scoredProducts[0].code : '';
}

function summarizeList(items, fallback = '-') {
  const values = (items || []).map((item) => String(item || '').trim()).filter(Boolean);
  return values.length ? values.join(', ') : fallback;
}

function addTooltipTerm(map, label, parts = []) {
  const key = String(label || '').trim();
  if (!key || map[key]) {
    return;
  }
  const detail = parts.map((item) => String(item || '').trim()).filter(Boolean).join(' 쨌 ');
  if (detail) {
    map[key] = detail;
  }
}

function buildAnswerTermMeta(workbench, productCode = '') {
  const meta = {};
  Object.entries(PRODUCT_DISPLAY_NAMES).forEach(([code, label]) => {
    addTooltipTerm(meta, code, ['상품 코드', label]);
    addTooltipTerm(meta, label, [`상품 코드 ${code}`, `화면 표시: ${label}`]);
  });
  const features = [
    ...(workbench?.representative_features || []),
    ...(workbench?.related_features || []),
    ...(workbench?.search_results || []),
  ];
  features.forEach((feature) => {
    const featureName = feature?.feature_name || feature?.feature_id;
    addTooltipTerm(meta, featureName, [
      feature?.feature_id ? `FEATURE ID ${feature.feature_id}` : '',
      feature?.category ? `분류 ${feature.category}` : '',
      Array.isArray(feature?.products) && feature.products.length ? `상품 ${feature.products.map(getProductDisplayName).join(', ')}` : '',
    ]);
  });
  (workbench?.summary?.top_reject_codes || []).forEach((item) => {
    addTooltipTerm(meta, item?.code, [
      item?.description || '거절사유코드',
      item?.count ? `로그 ${Number(item.count).toLocaleString('ko-KR')}건` : '',
      item?.share ? `비중 ${(Number(item.share) * 100).toFixed(1)}%` : '',
    ]);
    if (item?.description) {
      addTooltipTerm(meta, item.description, [
        item?.code ? `실제 출력 코드 ${item.code}` : '',
        item?.count ? `실제 로그 ${Number(item.count).toLocaleString('ko-KR')}건` : '',
      ]);
    }
  });
  (workbench?.answer_summary?.highlights || []).forEach((item) => {
    addTooltipTerm(meta, item?.value, [item?.label ? `답변 핵심: ${item.label}` : '답변 핵심 키워드']);
  });
  if (productCode) {
    addTooltipTerm(meta, productCode, ['선택 상품', getProductDisplayName(productCode)]);
  }
  return meta;
}

function isRejectDrivenFeature(feature) {
  const haystack = [
    feature?.feature_id,
    feature?.feature_name,
    feature?.category,
    feature?.description,
    ...(feature?.aliases || []),
  ].join(' ').toLowerCase();
  return ['reject', '거절', 'knock', 'k코드'].some((token) => haystack.includes(token));
}

function getSemanticRankFormulaConfig(summary) {
  const formula = summary?.semantic_rank_formula || {};
  return {
    semanticWeight: Number(formula.semantic_weight ?? 20),
    haystackHitWeight: Number(formula.haystack_hit_weight ?? 4),
    featureIdHitWeight: Number(formula.feature_id_hit_weight ?? 5),
    featureNameHitWeight: Number(formula.feature_name_hit_weight ?? 6),
    productMatchBonus: Number(formula.product_match_bonus ?? 3),
    coverageBonusCap: Number(formula.coverage_bonus_cap ?? 6),
  };
}

function computeSemanticRankPreview(item, weights) {
  const breakdown = item?.score_breakdown || {};
  const coverageBonus = Math.min(Number(weights.coverageBonusCap || 0), Number(breakdown.coverage_bonus || 0));
  const previewScore =
    (Number(breakdown.semantic_score || 0) * Number(weights.semanticWeight || 0))
    + (Number(breakdown.token_haystack_hits || 0) * Number(weights.haystackHitWeight || 0))
    + (Number(breakdown.feature_id_hits || 0) * Number(weights.featureIdHitWeight || 0))
    + (Number(breakdown.feature_name_hits || 0) * Number(weights.featureNameHitWeight || 0))
    + (Number(breakdown.product_bonus || 0) > 0 ? Number(weights.productMatchBonus || 0) : 0)
    + coverageBonus
    + Number(breakdown.reject_boost || 0);
  return {
    ...item,
    preview_score: Number(previewScore.toFixed(4)),
    preview_breakdown: {
      ...breakdown,
      coverage_bonus_applied: coverageBonus,
    },
  };
}

function formatKrwCompact(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return '-';
  }
  if (Math.abs(numericValue) >= 100000000) {
    return `${(numericValue / 100000000).toFixed(1)}억원`;
  }
  if (Math.abs(numericValue) >= 10000) {
    return `${Math.round(numericValue / 10000).toLocaleString('ko-KR')}만원`;
  }
  return `${Math.round(numericValue).toLocaleString('ko-KR')}원`;
}

function formatThresholdLabel(item) {
  if (!item || item.max_value === undefined || item.max_value === null) {
    return '-';
  }
  if (Number(item.max_value) >= 999999999999) {
    return `${item.label}`;
  }
  return `${item.label} <= ${formatKrwCompact(item.max_value)}`;
}

function formatSemanticRefreshTime(value) {
  const text = String(value || '').trim();
  if (!text) {
    return '';
  }
  const parsed = new Date(text);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  }
  return text.slice(11, 16) || text;
}

function MetricCubeModal({ open, data, loading, error, activeTab, onTabChange, onClose, onRefresh }) {
  if (!open) {
    return null;
  }

  const meta = data?.meta || {};
  const productSummaries = data?.product_summaries || [];
  const grainSummary = data?.grain_summary || [];
  const reliabilitySummary = data?.reliability_summary || [];
  const availableMetrics = data?.available_metrics || [];
  const queryExamples = data?.query_examples || [];
  const clusterSummary = data?.cluster_summary || {};
  const clusterProducts = clusterSummary?.products || [];
  const tabs = [
    { id: 'cube', label: '통계 큐브', detail: '평균/승인률/연체위험' },
    { id: 'cluster', label: '군집 분석', detail: '고객군 캐시' },
  ];

  return (
    <motion.div className="prompt-modal-backdrop metric-cube-modal-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}>
      <motion.section
        className="prompt-modal metric-cube-modal"
        initial={{ opacity: 0, y: 18, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 12, scale: 0.98 }}
        transition={{ duration: 0.22, ease: 'easeOut' }}
        onClick={(event) => event.stopPropagation()}
        style={{
          maxWidth: '1080px',
          minWidth: '720px',
          width: '80vw',
        }}
      >
        <div className="metric-cube-modal-head">
          <div>
            <span className="panel-kicker">Semantic Metric Layer</span>
            <h2>통계 큐브와 군집분석 맵</h2>
            <p>평균, 승인률, 연체위험 질문은 통계 큐브에서 바로 찾고, 고객군 해석은 군집분석 캐시에서 봅니다.</p>
          </div>
          <div className="metric-cube-modal-actions">
            <button type="button" className="secondary-button" onClick={onRefresh} disabled={loading}>새로고침</button>
            <button type="button" className="secondary-button" onClick={onClose}>닫기</button>
          </div>
        </div>

        <div className="metric-cube-tabs" role="tablist" aria-label="통계 큐브 보기">
          {tabs.map((tab) => (
            <button key={tab.id} type="button" className={activeTab === tab.id ? 'active' : ''} onClick={() => onTabChange(tab.id)}>
              <strong>{tab.label}</strong>
              <span>{tab.detail}</span>
            </button>
          ))}
        </div>

        {loading ? <div className="empty-box metric-cube-loading">통계 큐브와 군집 캐시를 읽는 중입니다.</div> : null}
        {error ? <div className="error-banner metric-cube-error">통계 큐브를 불러오지 못했습니다: {error}</div> : null}

        {!loading && !error && activeTab === 'cube' ? (
          <div className="metric-cube-modal-body">
            <div className="metric-cube-summary-grid">
              <article>
                <span>심사 로그</span>
                <strong>{Number(meta.record_count || 0).toLocaleString('ko-KR')}건</strong>
                <p>{meta.source_path || 'data/full_text_records.json'} 기준</p>
              </article>
              <article>
                <span>세그먼트</span>
                <strong>{Number(meta.segment_count || 0).toLocaleString('ko-KR')}개</strong>
                <p>상품 × 승인/거절 × 연령 × 소득 × 한도 × K코드</p>
              </article>
              <article>
                <span>저장 파일</span>
                <strong>{meta.path || 'data/segment_metric_cube.json'}</strong>
                <p>rag/segment_metric_cube.py 로 조회</p>
              </article>
            </div>

            <div className="metric-cube-two-column">
              <section className="metric-cube-panel">
                <div className="metric-cube-section-head">
                  <span className="panel-kicker">Product KPI</span>
                  <strong>상품별 바로 조회 가능한 숫자</strong>
                </div>
                <div className="metric-product-list">
                  {productSummaries.map((item) => (
                    <article key={item.product} className="metric-product-card">
                      <div>
                        <span>{item.product}</span>
                        <strong>{item.product_label}</strong>
                      </div>
                      <div className="metric-product-card-grid">
                        <span>표본 <b>{Number(item.count || 0).toLocaleString('ko-KR')}건</b></span>
                        <span>승인률 <b>{item.approval_rate_percent}%</b></span>
                        <span>금리 <b>{item.avg_rate_display || '-'}</b></span>
                        <span>한도 <b>{item.avg_amount_display || '-'}</b></span>
                      </div>
                      {item.top_reject_codes?.length ? (
                        <div className="metric-reject-code-row">
                          {item.top_reject_codes.slice(0, 3).map((code) => <span key={`${item.product}-${code.code}`}>{code.code}</span>)}
                        </div>
                      ) : null}
                    </article>
                  ))}
                </div>
              </section>

              <section className="metric-cube-panel">
                <div className="metric-cube-section-head">
                  <span className="panel-kicker">Lookup Guide</span>
                  <strong>이런 질문을 바로 답할 수 있어요</strong>
                </div>
                <div className="metric-query-list">
                  {queryExamples.map((item) => <span key={item}>{item}</span>)}
                </div>
                <div className="metric-available-list">
                  {availableMetrics.map((item) => (
                    <article key={item.label}>
                      <strong>{item.label}</strong>
                      <p>{item.detail}</p>
                    </article>
                  ))}
                </div>
              </section>
            </div>

            <div className="metric-cube-two-column">
              <section className="metric-cube-panel">
                <div className="metric-cube-section-head">
                  <span className="panel-kicker">Cube Grain</span>
                  <strong>많이 만들어진 큐브 조합</strong>
                </div>
                <div className="metric-grain-list">
                  {grainSummary.slice(0, 8).map((item) => (
                    <span key={item.grain}><b>{item.count}</b> {item.grain}</span>
                  ))}
                </div>
              </section>
              <section className="metric-cube-panel">
                <div className="metric-cube-section-head">
                  <span className="panel-kicker">Reliability</span>
                  <strong>표본 안정성</strong>
                </div>
                <div className="metric-reliability-bars">
                  {reliabilitySummary.map((item) => (
                    <span key={item.label} className={`is-${item.label}`}>
                      <b>{item.label}</b>
                      <i style={{ width: `${Math.min(100, (Number(item.count || 0) / Math.max(1, Number(meta.segment_count || 1))) * 100)}%` }} />
                      <em>{Number(item.count || 0).toLocaleString('ko-KR')}개</em>
                    </span>
                  ))}
                </div>
              </section>
            </div>
          </div>
        ) : null}

        {!loading && !error && activeTab === 'cluster' ? (
          <div className="metric-cube-modal-body">
            <div className="metric-cube-summary-grid">
              <article>
                <span>군집 캐시</span>
                <strong>{Number(clusterSummary.total_clusters || 0).toLocaleString('ko-KR')}개</strong>
                <p>data/feature_customer_clusters.json 기준</p>
              </article>
              <article>
                <span>군집 표본</span>
                <strong>{Number(clusterSummary.record_count || 0).toLocaleString('ko-KR')}건</strong>
                <p>승인/거절 고객군 해석용</p>
              </article>
              <article>
                <span>역할</span>
                <strong>해석/비교</strong>
                <p>통계 큐브는 숫자, 군집은 고객군 특징 설명</p>
              </article>
            </div>
            <div className="metric-cluster-product-grid">
              {clusterProducts.map((item) => (
                <article key={item.product} className="metric-cluster-card">
                  <div className="metric-cluster-card-head">
                    <span>{item.product}</span>
                    <strong>{item.product_label}</strong>
                    <b>{item.cluster_count}개 군집</b>
                  </div>
                  <div className="metric-cluster-mini-list">
                    {(item.top_clusters || []).map((cluster) => (
                      <div key={`${item.product}-${cluster.label}`} className="metric-cluster-mini-item">
                        <span>{cluster.decision || '군집'} · {Number(cluster.count || 0).toLocaleString('ko-KR')}건</span>
                        <strong>{cluster.label}</strong>
                        <small>금리 {cluster.avg_rate_display || '-'} · 한도 {cluster.avg_amount_display || '-'} · 위험 {cluster.delinquency_proxy_rate_display || '-'}</small>
                      </div>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </div>
        ) : null}
      </motion.section>
    </motion.div>
  );
}

function DepartmentConceptModal({ open, concepts = [], onClose }) {
  if (!open) {
    return null;
  }
  return (
    <motion.div className="prompt-modal-backdrop department-concept-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}>
      <motion.section className="prompt-modal department-concept-modal" initial={{ opacity: 0, y: 18, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 12, scale: 0.98 }} transition={{ duration: 0.22, ease: 'easeOut' }} onClick={(event) => event.stopPropagation()}>
        <div className="metric-cube-modal-head">
          <div>
            <span className="panel-kicker">Department Concept</span>
            <h2>부서별 상품개발 관점</h2>
            <p>메모모드에 적은 기준은 상품개발모드의 안건 생성과 4인 토론에 같이 반영됩니다.</p>
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>닫기</button>
        </div>
        <div className="department-concept-grid">
          {concepts.map((item) => (
            <article key={item.id} className="department-concept-card">
              <div>
                <span aria-hidden="true">{item.icon}</span>
                <strong>{item.label}</strong>
              </div>
              <p>{item.concept}</p>
              {item.note ? <small>담당자 메모: {item.note}</small> : <small>메모가 없으면 기본 컨셉으로 참여합니다.</small>}
            </article>
          ))}
        </div>
      </motion.section>
    </motion.div>
  );
}

function resolveDebatePerson(message = {}) {
  const speakerText = String(message.speaker || message.department || '').toLowerCase();
  if (speakerText.includes('금융솔루션') || speakerText.includes('금프로') || speakerText.includes('solution')) {
    return PRODUCT_DEBATE_PEOPLE[0];
  }
  if (speakerText.includes('신용기획') || speakerText.includes('신프로') || speakerText.includes('risk') || speakerText.includes('credit')) {
    return PRODUCT_DEBATE_PEOPLE[1];
  }
  if (speakerText.includes('금융영업') || speakerText.includes('영프로') || speakerText.includes('sales')) {
    return PRODUCT_DEBATE_PEOPLE[2];
  }
  if (speakerText.includes('it') || speakerText.includes('개발') || speakerText.includes('아프로') || speakerText.includes('tech')) {
    return PRODUCT_DEBATE_PEOPLE[3];
  }
  return PRODUCT_DEBATE_PEOPLE.find((person) => person.tone === message.tone) || PRODUCT_DEBATE_PEOPLE[0];
}

function ProductDebateMeetingRoom({ loading = false, messages = [], selectedAgenda = null }) {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!loading) {
      return undefined;
    }
    const timerId = window.setInterval(() => {
      setTick((value) => value + 1);
    }, 1150);
    return () => window.clearInterval(timerId);
  }, [loading]);

  useEffect(() => {
    if (loading) {
      setTick(0);
    }
  }, [loading, selectedAgenda?.id, selectedAgenda?.title]);

  const stagedMessages = loading
    ? PRODUCT_DEBATE_WARMUP_LINES.slice(0, Math.min(PRODUCT_DEBATE_WARMUP_LINES.length, Math.max(4, tick + 1)))
    : messages.map((message, index) => {
      const person = resolveDebatePerson(message);
      return {
        id: `${person.id}-${index}`,
        speaker: message.speaker || person.name,
        department: person.department,
        tone: message.tone || person.tone,
        message: message.message || message.content || '',
      };
    });
  const activePerson = resolveDebatePerson(stagedMessages[stagedMessages.length - 1] || {});
  const agendaTitle = selectedAgenda?.title || '선택된 안건';

  return (
    <div className={`product-debate-room ${loading ? 'is-live' : 'is-complete'}`}>
      <div className="product-debate-room-head">
        <div>
          <span className="panel-kicker">Live Meeting Room</span>
          <strong>{loading ? '4명이 회의실에서 안건을 맞추는 중' : '4개 부서 토론 기록'}</strong>
        </div>
        <span className="sample-pill">{loading ? '실시간 구성 중' : `${messages.length}개 발언`}</span>
      </div>

      <div className="product-debate-table">
        <div className="product-debate-table-surface">
          <span>{agendaTitle}</span>
          <strong>{loading ? 'OLLAMA 결과를 기다리는 동안 사전 토론 중' : '토론 정리 완료'}</strong>
          <div className="product-debate-equalizer" aria-hidden="true">
            {Array.from({ length: 12 }).map((_, index) => <i key={index} style={{ animationDelay: `${index * 0.08}s` }} />)}
          </div>
        </div>
        {PRODUCT_DEBATE_PEOPLE.map((person, index) => {
          const isSpeaking = person.id === activePerson.id;
          return (
            <motion.div
              key={person.id}
              className={`product-debate-person person-${index + 1} is-${person.tone} ${isSpeaking ? 'is-speaking' : ''}`}
              initial={{ opacity: 0, y: 18, scale: 0.94 }}
              animate={{ opacity: 1, y: 0, scale: isSpeaking ? 1.04 : 1 }}
              transition={{ duration: 0.32, delay: index * 0.08, ease: 'easeOut' }}
            >
              <div className="product-debate-avatar">
                <span>{person.short}</span>
                <i aria-hidden="true" />
              </div>
              <div>
                <strong>{person.name}</strong>
                <small>{person.department} · {person.role}</small>
              </div>
            </motion.div>
          );
        })}
      </div>

      <div className="product-debate-live-feed">
        <AnimatePresence initial={false}>
          {stagedMessages.slice(-6).map((message, index) => {
            const person = resolveDebatePerson(message);
            return (
              <motion.article
                key={message.id || `${message.speaker}-${index}-${message.message}`}
                className={`product-dev-message is-${message.tone || person.tone}`}
                initial={{ opacity: 0, x: -12, y: 6 }}
                animate={{ opacity: 1, x: 0, y: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.22, ease: 'easeOut' }}
              >
                <div className="product-dev-message-speaker">
                  <span>{person.short}</span>
                  <strong>{message.speaker || person.name}</strong>
                  <small>{message.department || person.department}</small>
                </div>
                <p>{message.message}</p>
              </motion.article>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

function ProductDevelopmentWorkspace({
  state,
  concepts = [],
  semanticRefresh = {},
  onRefreshAgendas,
  onSelectAgenda,
  onOpenConcepts,
}) {
  const agendas = state.agendas || [];
  const debate = state.debate?.result || state.debate || null;
  const context = state.agendaPayload?.context || state.debate?.context || {};
  const productSummaries = context.product_summaries || debate?.product_cards || [];
  const messages = debate?.messages || [];
  const final = debate?.final || {};
  const newProduct = final.new_product || {};
  const improvements = final.product_logic_improvements || [];
  const sourceLabel = state.source === 'ollama' || state.debateSource === 'ollama' ? 'Ollama 1회 생성' : 'Fallback 초안';

  return (
    <motion.article className="product-development-workspace" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24, ease: 'easeOut' }}>
      <div className="product-dev-hero">
        <div>
          <span className="panel-kicker">상품개발모드</span>
          <h2>통계 큐브와 군집분석으로 4개 부서가 빠르게 상품을 설계합니다.</h2>
          <p>먼저 금융솔루션부가 토론 안건 2개를 제안하고, 안건을 고르면 4명이 서로 질문하며 신상품과 기존 상품 보완 로직을 같이 만듭니다.</p>
        </div>
        <div className="product-dev-actions">
          <button type="button" className="secondary-button" onClick={onOpenConcepts}>부서 컨셉 보기</button>
          <button type="button" className="primary-button product-dev-refresh-button" onClick={onRefreshAgendas} disabled={state.loadingAgendas || state.debateLoading}>
            {state.loadingAgendas ? '안건 생성 중' : '안건 다시 만들기'}
          </button>
        </div>
      </div>

      <div className="product-dev-stat-strip">
        {(productSummaries.length ? productSummaries : []).slice(0, 4).map((item) => (
          <article key={item.product || item.product_label}>
            <span>{item.product_label || item.product}</span>
            <strong>승인률 {item.approval_rate_percent ?? '-'}%</strong>
            <small>금리 {item.avg_rate_display || '-'} · 한도 {item.avg_amount_display || '-'} · 연체 {item.delinquency_proxy_rate_display || '-'}</small>
          </article>
        ))}
        {!productSummaries.length ? <div className="empty-box compact">통계 큐브를 읽어 상품별 숫자를 준비하는 중입니다.</div> : null}
      </div>

      <div className="product-dev-meta-row">
        <span className="sample-pill">{sourceLabel}</span>
        <span className="sample-pill">큐브 {Number(context?.cube_meta?.segment_count || semanticRefresh?.segment_count || 0).toLocaleString('ko-KR')}개 세그먼트</span>
        <span className="sample-pill">군집 {Number(semanticRefresh?.cluster_count || 0).toLocaleString('ko-KR')}개</span>
      </div>

      {state.agendaError ? <div className="error-banner">상품개발 안건 생성 실패: {state.agendaError}</div> : null}
      {state.debateError ? <div className="error-banner">상품개발 토론 생성 실패: {state.debateError}</div> : null}

      <section className="product-dev-section">
        <div className="product-dev-section-head">
          <span className="panel-kicker">Step 1</span>
          <strong>금융솔루션부가 던지는 토론 안건 2개</strong>
        </div>
        {state.loadingAgendas ? <div className="empty-box compact">통계자료, 한도, 금리, 거절코드, 연체가능성을 읽고 안건을 만드는 중입니다.</div> : null}
        <div className="product-dev-agenda-grid">
          {agendas.map((agenda, index) => {
            const isSelected = state.selectedAgenda?.id === agenda.id || (!agenda.id && state.selectedAgenda === agenda);
            return (
              <button key={agenda.id || `${agenda.title}-${index}`} type="button" className={`product-dev-agenda-card ${isSelected ? 'active' : ''}`} onClick={() => onSelectAgenda(agenda)} disabled={state.debateLoading}>
                <span>{agenda.type === 'new_product' ? '신상품 후보' : '로직 보완 후보'}</span>
                <strong>{agenda.title}</strong>
                <p>{agenda.summary}</p>
                <small>{agenda.target}</small>
                <div>
                  {(agenda.data_points || agenda.expected_effect || []).slice(0, 3).map((item) => <em key={item}>{item}</em>)}
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {(state.debateLoading || messages.length) ? (
        <section className="product-dev-section">
          <div className="product-dev-section-head">
            <span className="panel-kicker">Step 2</span>
            <strong>{state.debateLoading ? '4개 부서 실시간 회의' : '4개 부서 토론'}</strong>
          </div>
          <ProductDebateMeetingRoom loading={state.debateLoading} messages={messages} selectedAgenda={state.selectedAgenda} />
        </section>
      ) : null}

      {final.new_product || improvements.length ? (
        <section className="product-dev-section product-dev-final">
          <div className="product-dev-section-head">
            <span className="panel-kicker">Final</span>
            <strong>최종 산출물: 신상품 + 기존 상품별 보완 로직</strong>
          </div>
          <div className="product-dev-final-grid">
            <article className="product-dev-final-card is-new">
              <span>신상품</span>
              <strong>{newProduct.name || '신상품 초안'}</strong>
              <p>{newProduct.target || '-'}</p>
              <div className="product-dev-list">
                {(newProduct.core_logic || []).map((item) => <em key={item}>{item}</em>)}
                {newProduct.limit_rate_policy ? <em>{newProduct.limit_rate_policy}</em> : null}
              </div>
            </article>
            <article className="product-dev-final-card">
              <span>기존 상품 보완</span>
              <strong>상품별 심사 로직 개선안</strong>
              <div className="product-dev-improvement-list">
                {improvements.map((item) => (
                  <div key={`${item.product}-${item.change}`}>
                    <b>{item.product}</b>
                    <p>{item.change}</p>
                    <small>{item.expected_effect} · {item.dev_impact}</small>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>
      ) : null}

      <div className="product-dev-concept-row">
        {concepts.map((item) => <span key={item.id}>{item.icon} {item.label}</span>)}
      </div>
    </motion.article>
  );
}

export default function OntologyWorkbench({
  reduceMotion = false,
  onToast,
  onError,
  onRequestRegulationUpload,
  regulationBusy = false,
  regulationStatus = 'pending',
  regulationUpdatedAt = '',
  regulationSummary = '',
  latestNewsSignal = null,
  latestNewsItems = [],
  operationsSummary = {},
  debateSummary = {},
  debateBusy = false,
  onStartDebate,
}) {
  const formatConversationTime = () => new Date().toLocaleTimeString('ko-KR', { hour12: false });

  const {
    currentQuestion,
    currentStageIndex,
    queryRunNonce,
    products,
    clusters,
    semanticStats,
    runtimeStages,
    runtimeJobStatus,
    runtimeElapsedMs,
    activityFeed,
    liveLogs,
    retrievalTrace,
    answerSummary,
    selectedNodeId,
    submitQuestion,
    selectNode,
    hydrateRuntime,
    applyRuntimeSnapshot,
    ingestWorkbench,
    lastRunAt,
  } = useOntologyRuntimeStore();

  const [queryInput, setQueryInput] = useState(currentQuestion);
  const [backendState, setBackendState] = useState({ loading: true, error: '', workbench: null, ontology: null });
  const [activeSubtab, setActiveSubtab] = useState('question');
  const [answerTraceTab, setAnswerTraceTab] = useState('input');
  const [activeWorkspaceResultTab, setActiveWorkspaceResultTab] = useState('summary');
  const [workspaceMode, setWorkspaceMode] = useState('conversation');
  const [answerMode, setAnswerMode] = useState('general');
  const [memoDepartment, setMemoDepartment] = useState('solution');
  const [memoNotes, setMemoNotes] = useState('');
  const [memoDepartmentNotes, setMemoDepartmentNotes] = useState({});
  const [showConceptModal, setShowConceptModal] = useState(false);
  const [productDevelopmentState, setProductDevelopmentState] = useState({
    loadingAgendas: false,
    agendaError: '',
    agendaPayload: null,
    agendas: [],
    source: '',
    selectedAgenda: null,
    debateLoading: false,
    debateError: '',
    debate: null,
    debateSource: '',
  });
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [isRoniHovered, setIsRoniHovered] = useState(false);
  const [isNewsBadgeHovered, setIsNewsBadgeHovered] = useState(false);
  const [showRoniSuccess, setShowRoniSuccess] = useState(false);
  const [activeNewsBubble, setActiveNewsBubble] = useState(null);
  const [showPromptExamples, setShowPromptExamples] = useState(false);
  const [showMetricCubeModal, setShowMetricCubeModal] = useState(false);
  const [metricCubeTab, setMetricCubeTab] = useState('cube');
  const [metricCubeState, setMetricCubeState] = useState({ loading: false, error: '', data: null });
  const [semanticRefreshState, setSemanticRefreshState] = useState({ loading: true, error: '', refresh: null });
  const [conversationTurns, setConversationTurns] = useState([]);
  const [avoidedFeatureIds, setAvoidedFeatureIds] = useState([]);
  const [clarificationAnswered, setClarificationAnswered] = useState(false);
  const avoidedFeatureIdsRef = useRef([]);
  const onErrorRef = useRef(onError);
  const runtimeRequestRef = useRef('');
  const conversationSessionIdRef = useRef(`session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`);
  const conversationTurnsRef = useRef([]);
  const memoPreferencesRef = useRef({ answerMode: 'general', department: 'solution', notes: '' });
  const regulationStatusRef = useRef(regulationStatus);
  const regulationIntentRef = useRef(false);
  const latestNewsSignalRef = useRef('');
  const newsSignalInitializedRef = useRef(false);
  const semanticRefreshInitializedRef = useRef(false);
  const semanticRefreshCompletionRef = useRef('');
  const departmentConceptsRef = useRef({});

  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  useEffect(() => {
    try {
      const saved = JSON.parse(window.localStorage.getItem(MEMO_STORAGE_KEY) || '{}');
      if (saved.answerMode && ANSWER_MODES.some((item) => item.id === saved.answerMode)) {
        setAnswerMode(saved.answerMode);
      }
      if (saved.department && MEMO_DEPARTMENTS.some((item) => item.id === saved.department)) {
        setMemoDepartment(saved.department);
      }
      if (saved.department === 'developer') {
        setMemoDepartment('it');
      }
      if (saved.department_notes && typeof saved.department_notes === 'object') {
        const normalizedNotes = { ...saved.department_notes };
        if (normalizedNotes.developer && !normalizedNotes.it) {
          normalizedNotes.it = normalizedNotes.developer;
          delete normalizedNotes.developer;
        }
        setMemoDepartmentNotes(normalizedNotes);
      }
      if (typeof saved.notes === 'string') {
        setMemoNotes(saved.notes);
        if (!saved.department_notes) {
          setMemoDepartmentNotes({ [saved.department === 'developer' ? 'it' : (saved.department || 'solution')]: saved.notes });
        }
      }
    } catch {
      return;
    }
  }, []);

  useEffect(() => {
    setMemoNotes(memoDepartmentNotes[memoDepartment] || '');
  }, [memoDepartment, memoDepartmentNotes]);

  const departmentConcepts = useMemo(() => {
    const concepts = {};
    MEMO_DEPARTMENTS.forEach((department) => {
      const note = String(memoDepartmentNotes[department.id] || '').trim();
      concepts[department.id] = {
        id: department.id,
        label: department.label,
        icon: department.icon,
        note,
        concept: note
          ? `${department.defaultConcept} 담당자 메모: ${note}`
          : department.defaultConcept,
      };
    });
    return concepts;
  }, [memoDepartmentNotes]);

  const departmentConceptList = useMemo(
    () => MEMO_DEPARTMENTS.map((department) => departmentConcepts[department.id]),
    [departmentConcepts],
  );

  useEffect(() => {
    departmentConceptsRef.current = departmentConcepts;
  }, [departmentConcepts]);

  useEffect(() => {
    const allMemoNotes = MEMO_DEPARTMENTS
      .map((department) => {
        const note = String(memoDepartmentNotes[department.id] || '').trim();
        return note ? `${department.label}: ${note}` : '';
      })
      .filter(Boolean)
      .join('\n');
    const nextPreferences = {
      answerMode,
      department: memoDepartment,
      notes: allMemoNotes || memoNotes,
      department_notes: memoDepartmentNotes,
      department_concepts: departmentConcepts,
    };
    memoPreferencesRef.current = nextPreferences;
    try {
      window.localStorage.setItem(MEMO_STORAGE_KEY, JSON.stringify(nextPreferences));
    } catch {
      return;
    }
  }, [answerMode, departmentConcepts, memoDepartment, memoDepartmentNotes, memoNotes]);

  useEffect(() => {
    if (answerMode === 'product') {
      setWorkspaceMode('strategy');
    } else if (workspaceMode === 'strategy') {
      setWorkspaceMode('conversation');
    }
  }, [answerMode]);

  useEffect(() => {
    conversationTurnsRef.current = conversationTurns;
  }, [conversationTurns]);

  useEffect(() => {
    avoidedFeatureIdsRef.current = avoidedFeatureIds;
  }, [avoidedFeatureIds]);

  useEffect(() => {
    setQueryInput(currentQuestion);
  }, [currentQuestion]);

  const nodeTypes = useMemo(() => NODE_TYPES, []);
  const reactFlowProOptions = useMemo(() => REACT_FLOW_PRO_OPTIONS, []);
  const reactFlowDefaultViewport = useMemo(() => REACT_FLOW_DEFAULT_VIEWPORT, []);
  const selectedProductCode = useMemo(() => resolveProductForQuestion(currentQuestion, products), [currentQuestion, products]);
  const hasSubmittedRuntimeQuery = Number(queryRunNonce || 0) > 0;
  const selectedProductLabel = getProductDisplayName(selectedProductCode);
  const newsBadgeCount = latestNewsItems.length;
  const operationMetrics = operationsSummary?.metrics || [];
  const operationBriefs = operationsSummary?.briefs || [];
  const operationTimeline = operationsSummary?.timeline || [];
  const debateRounds = debateSummary?.roundResults || [];
  const debateStatus = debateSummary?.status || 'idle';
  const conversationStatus = backendState.error ? '오류' : backendState.loading ? '분석 중' : '답변 준비됨';

  useEffect(() => {
    let ignore = false;

    async function loadOntology() {
      try {
        setBackendState((previous) => ({ ...previous, loading: true, error: '' }));
        const payload = await fetchOntologyState();
        if (ignore) {
          return;
        }
        const commonFeatureCount = payload?.commonfeature?.statistics?.common_feature_count || payload?.commonfeature?.common_features?.length || 0;
        const productItems = Object.entries(payload?.ontology?.products || {}).map(([code, item]) => ({
          code,
          label: item?.product_name || code,
          count: Object.keys(item?.input_fields || {}).length + Object.keys(item?.output_fields || {}).length,
        }));
        hydrateRuntime({
          nodeCount: commonFeatureCount || undefined,
          relationCount: commonFeatureCount ? commonFeatureCount * 3 : undefined,
          liveAliasCount: commonFeatureCount ? Math.max(12, Math.floor(commonFeatureCount * 0.02)) : undefined,
          products: productItems,
          ontologyChanges: commonFeatureCount
            ? [
                { id: 'live-ontology-1', label: 'commonfeature.json refreshed', detail: `공통 feature ${commonFeatureCount}건 동기화` },
                { id: 'live-ontology-2', label: 'ontology product coverage updated', detail: `${productItems.length}개 상품 field mapping 반영` },
              ]
            : undefined,
        });
        setBackendState((previous) => ({ ...previous, loading: false, ontology: payload }));
      } catch (error) {
        if (ignore) {
          return;
        }
        const message = String(error.message || error);
        setBackendState((previous) => ({ ...previous, loading: false, error: message }));
      }
    }

    loadOntology();
    return () => {
      ignore = true;
    };
  }, [hydrateRuntime]);

  useEffect(() => {
    let ignore = false;

    async function pollSemanticRefreshStatus() {
      try {
        const payload = await fetchFeatureOntologySemanticRefreshStatus();
        if (ignore) {
          return;
        }
        const refresh = payload?.refresh || null;
        setSemanticRefreshState({ loading: false, error: '', refresh });

        const completionKey = refresh?.last_completed_at
          ? `completed:${refresh.run_count || 0}:${refresh.last_completed_at}`
          : '';
        const failureKey = refresh?.last_failed_at
          ? `failed:${refresh.last_failed_at}`
          : '';

        if (!semanticRefreshInitializedRef.current) {
          semanticRefreshInitializedRef.current = true;
          semanticRefreshCompletionRef.current = completionKey || failureKey;
          return;
        }

        if (completionKey && completionKey !== semanticRefreshCompletionRef.current) {
          semanticRefreshCompletionRef.current = completionKey;
          setShowRoniSuccess(true);
          onToast?.({
            id: `semantic-refresh-${Date.now()}`,
            kicker: 'Semantic Layer Sync',
            title: '통계 큐브와 군집분석 갱신 완료',
            meta: `${Number(refresh?.record_count || 0).toLocaleString('ko-KR')}건 · ${Number(refresh?.segment_count || 0).toLocaleString('ko-KR')}개 세그먼트`,
            message: '2분 자동 갱신이 끝났어요. 최신 심사 로그 기준으로 평균, 승인률, 군집분석 결과를 다시 반영했습니다.',
            tone: 'completed',
          });
          try {
            const cubePayload = await fetchFeatureOntologySegmentMetricCube({ forceRebuild: false });
            if (!ignore) {
              setMetricCubeState({ loading: false, error: '', data: cubePayload });
            }
          } catch {
            // The toast is still useful even if the optional UI cache refresh fails.
          }
          return;
        }

        if (failureKey && failureKey !== semanticRefreshCompletionRef.current) {
          semanticRefreshCompletionRef.current = failureKey;
          onToast?.({
            id: `semantic-refresh-failed-${Date.now()}`,
            kicker: 'Semantic Layer Sync',
            title: '통계 큐브 갱신 실패',
            meta: refresh?.last_failed_at || 'refresh failed',
            message: refresh?.last_error || '자동 갱신 중 오류가 발생했습니다.',
            tone: 'failed',
          });
        }
      } catch (error) {
        if (!ignore) {
          setSemanticRefreshState((previous) => ({ ...previous, loading: false, error: String(error.message || error) }));
        }
      }
    }

    pollSemanticRefreshStatus();
    const intervalId = window.setInterval(pollSemanticRefreshStatus, 10000);
    return () => {
      ignore = true;
      window.clearInterval(intervalId);
    };
  }, [onToast]);

  useEffect(() => {
    let ignore = false;
    let pollTimer = null;
    const requestKey = `runtime:${currentQuestion}:${queryRunNonce}`;

    if (!hasSubmittedRuntimeQuery || !String(currentQuestion || '').trim()) {
      runtimeRequestRef.current = '';
      setBackendState((previous) => ({ ...previous, loading: false, error: '', workbench: null }));
      return undefined;
    }

    if (runtimeRequestRef.current === requestKey) {
      return undefined;
    }
    runtimeRequestRef.current = requestKey;

    function applyWorkbenchPayload(payload) {
      const searchResults = payload?.search_results || [];
      const clusterResults = payload?.customer_clusters || [];
      ingestWorkbench({
        retrievalTrace: [
          { id: 'wb-1', label: 'Ontology Retrieval', value: searchResults[0]?.feature_id || 'applicant.age', score: searchResults[0]?.score || 0.97 },
          { id: 'wb-2', label: 'Cluster Retrieval', value: clusterResults[0]?.cluster_id || 'cluster_40_office_worker', score: clusterResults[0]?.score || 0.94 },
          ...((payload?.related_features || []).slice(0, 3).map((item, index) => ({ id: `wb-rf-${index}`, label: 'Related Feature', value: item.feature_id || item.feature_name, score: item.score || 0.8 }))),
          { id: 'wb-llm', label: 'Need LLM', value: 'YES', score: 1 },
        ],
        clusters: clusterResults.map((item) => ({ id: item.cluster_id, label: item.label || item.cluster_id, score: item.score || item.count || 0 })),
        activity: payload?.summary ? `Feature workbench 결과 ${payload.summary.record_count || 0}건을 retrieval trace에 반영했습니다.` : '',
        answerSummary: payload?.answer_summary,
      });
      setBackendState((previous) => ({ ...previous, loading: false, error: '', workbench: payload }));
    }

    async function pollJob(jobId) {
      try {
        const snapshot = await fetchFeatureOntologyRuntimeJob(jobId);
        if (ignore) {
          return;
        }
        applyRuntimeSnapshot(snapshot);
        if (snapshot?.status === 'completed') {
          applyWorkbenchPayload(snapshot?.result || {});
          return;
        }
        if (snapshot?.status === 'failed') {
          const message = String(snapshot?.error || 'runtime job failed');
          setBackendState((previous) => ({ ...previous, loading: false, error: message }));
          onErrorRef.current?.(message);
          return;
        }
        pollTimer = window.setTimeout(() => {
          void pollJob(jobId);
        }, 280);
      } catch (error) {
        if (ignore) {
          return;
        }
        const message = String(error.message || error);
        setBackendState((previous) => ({ ...previous, loading: false, error: previous.error || message }));
        onErrorRef.current?.(message);
      }
    }

    async function loadRuntimeJob() {
      try {
        setBackendState((previous) => ({ ...previous, loading: true, error: '', workbench: null }));
        const recentTurns = conversationTurnsRef.current.slice(-6);
        const history = recentTurns.map((turn) => ({
          question: String(turn?.question || ''),
          answer_headline: String(turn?.answerHeadline || ''),
          answer_body: String(turn?.answerBody || ''),
          selected_feature_ids: Array.isArray(turn?.selectedFeatureIds) ? turn.selectedFeatureIds.slice(0, 4) : [],
        }));
        const memoryKeywords = Array.from(new Set([
          ...String(currentQuestion || '').toLowerCase().split(/[^0-9a-zA-Z\u3131-\u318e\uac00-\ud7a3]+/).filter(Boolean),
          ...recentTurns.flatMap((turn) => String(turn?.axis || '').toLowerCase().split(/[^0-9a-zA-Z\u3131-\u318e\uac00-\ud7a3]+/).filter(Boolean)),
          ...String(memoPreferencesRef.current?.notes || '').toLowerCase().split(/\s+/).filter(Boolean),
          String(memoPreferencesRef.current?.department || '').toLowerCase(),
        ])).slice(0, 24);
        const preferredFeatureIds = recentTurns
          .slice()
          .reverse()
          .find((turn) => Array.isArray(turn?.selectedFeatureIds) && turn.selectedFeatureIds.length > 0)
          ?.selectedFeatureIds?.slice(0, 4) || [];
        const snapshot = await startFeatureOntologyRuntimeJob({
          product: selectedProductCode,
          query: currentQuestion,
          answer_mode: memoPreferencesRef.current?.answerMode || answerMode,
          department: memoPreferencesRef.current?.department || memoDepartment,
          memo_notes: memoPreferencesRef.current?.notes || memoNotes,
          session_id: conversationSessionIdRef.current,
          turn_id: `turn-${Date.now()}`,
          memory_keywords: memoryKeywords,
          history,
          feedback: {
            preferred_feature_ids: preferredFeatureIds,
            avoided_feature_ids: avoidedFeatureIdsRef.current.slice(0, 6),
          },
          allow_clarification: true,
          clarification_budget: 1,
        });
        if (ignore) {
          return;
        }
        applyRuntimeSnapshot(snapshot);
        if (snapshot?.status === 'completed') {
          applyWorkbenchPayload(snapshot?.result || {});
          return;
        }
        await pollJob(snapshot?.job_id);
      } catch (error) {
        if (ignore) {
          return;
        }
        const message = String(error.message || error);
        setBackendState((previous) => ({ ...previous, error: previous.error || message }));
        onErrorRef.current?.(message);
      }
    }

    loadRuntimeJob();
    return () => {
      ignore = true;
      runtimeRequestRef.current = '';
      if (pollTimer) {
        window.clearTimeout(pollTimer);
      }
    };
  }, [applyRuntimeSnapshot, currentQuestion, hasSubmittedRuntimeQuery, ingestWorkbench, queryRunNonce, selectedProductCode]);

  const activeStage = runtimeStages[currentStageIndex] || runtimeStages[0];
  const completedStageCount = runtimeStages.filter((item) => item.status === 'completed').length;
  const jobElapsedMs = Number(runtimeElapsedMs || 0);
  const overallRuntimeProgress = runtimeStages.length
    ? Math.min(100, Math.round(((completedStageCount + (activeStage?.status === 'running' ? 0.66 : 0)) / runtimeStages.length) * 100))
    : 0;
  const isRuntimeComplete = runtimeJobStatus === 'completed' || (runtimeStages.length > 0 && completedStageCount === runtimeStages.length);
  const progressHeadline = isRuntimeComplete ? '분석 완료' : stageWaitCopyKo(activeStage?.key);
  const progressSubLabel = isRuntimeComplete ? '답변 준비됨' : (jobElapsedMs ? formatRuntimeMs(jobElapsedMs) : statusLabelKo(runtimeJobStatus));
  const progressLiveText = isRuntimeComplete
    ? '답변이 준비됐어요'
    : (activeStage?.meta?.duration_ms ? `방금 완료: ${formatRuntimeMs(activeStage.meta.duration_ms)}` : '진행 상태를 확인하는 중');
  const slowestCompletedStage = runtimeStages
    .filter((item) => Number(item.duration_ms || item.meta?.duration_ms || 0) > 0)
    .sort((left, right) => Number(right.duration_ms || right.meta?.duration_ms || 0) - Number(left.duration_ms || left.meta?.duration_ms || 0))[0] || null;
  const agentState = useMemo(() => deriveAgentState({
    activeStageKey: activeStage?.key || 'extraction',
    runtimeJobStatus,
    hasWorkbench: Boolean(backendState.workbench),
    isRoniHovered,
    regulationBusy,
    showRoniSuccess,
    hasError: Boolean(backendState.error),
  }), [activeStage?.key, backendState.error, backendState.workbench, isRoniHovered, regulationBusy, runtimeJobStatus, showRoniSuccess]);
  const thinkingPhase = useMemo(() => deriveThinkingPhase(activeStage?.key || 'extraction'), [activeStage?.key]);
  const roniCaption = useMemo(() => bunnySpeech({
    agentState,
    activeStageKey: activeStage?.key || 'extraction',
    showSuccess: showRoniSuccess,
    isHovered: isRoniHovered,
    hasSubmittedRuntimeQuery,
  }), [activeStage?.key, agentState, hasSubmittedRuntimeQuery, isRoniHovered, showRoniSuccess]);
  const resolvedAnswerSummary = useMemo(() => {
    const runtimeSummary = backendState.workbench?.answer_summary;
    if (runtimeSummary) {
      return {
        ...runtimeSummary,
        highlights: (runtimeSummary?.highlights || []).map((item) => (
          item.label === 'Product' || item.label === '상품'
            ? { ...item, value: getProductDisplayName(selectedProductCode || backendState.workbench?.input?.product) }
            : item
        )),
      };
    }
    return {
      ...answerSummary,
      highlights: (answerSummary?.highlights || []).map((item) => (
        item.label === 'Product'
          ? { ...item, value: getProductDisplayName(selectedProductCode) }
          : item
      )),
    };
  }, [answerSummary, backendState.workbench?.answer_summary, backendState.workbench?.input?.product, selectedProductCode]);
  const resultSummary = backendState.workbench?.summary || {};
  const hasResolvedRuntimeAnswer = Boolean(backendState.workbench?.answer_summary);
  const agenticWorkspace = backendState.workbench?.agentic_workspace || {};
  const agentWorkflow = backendState.workbench?.agent_workflow || agenticWorkspace?.agent_workflow || [];
  const activeToolCards = (agenticWorkspace?.version_1?.active_tools || []).filter(Boolean);
  const strategyPanels = (agenticWorkspace?.version_2?.panels || []).filter(Boolean);
  const productModeToolCards = strategyPanels.filter((tool) => ['strategy', 'persona'].includes(tool?.id));
  const semanticFinancialLayer = backendState.workbench?.semantic_financial_layer || agenticWorkspace?.semantic_layer || {};
  const ollamaRuntime = backendState.workbench?.ollama_runtime || {};
  const ollamaInput = ollamaRuntime?.input || {};
  const ollamaOutput = ollamaRuntime?.output || {};
  const ollamaStatus = ollamaRuntime?.status || resultSummary.ollama_status || 'skipped';
  const currentAnswerSourceTag = getAnswerSourceTag(resolvedAnswerSummary, ollamaStatus);
  const isDocumentOrGeneralAnswer = ['규제문서', '일반답변'].includes(currentAnswerSourceTag);
  const finalAnswerBody = hasResolvedRuntimeAnswer ? (resolvedAnswerSummary?.explanation || ollamaOutput?.response_text || '') : '';
  const ollamaFullInput = ollamaInput?.prompt || [ollamaInput?.system_prompt, ollamaInput?.user_prompt].filter(Boolean).join('\n\n') || '아직 Ollama 입력이 없습니다.';
  const ollamaFullOutput = ollamaOutput?.response_text || ollamaOutput?.response_preview || ollamaRuntime?.error || '아직 Ollama 출력이 없습니다.';
  const resultTopCluster = backendState.workbench?.customer_clusters?.[0] || null;
  const searchResults = backendState.workbench?.search_results || [];
  const primaryFeatureSelection = backendState.workbench?.primary_feature_selection || {};
  const selectedFeature = backendState.workbench?.selected_feature || searchResults[0] || null;
  const representativeFeatures = (primaryFeatureSelection?.representative_features || backendState.workbench?.representative_features || (selectedFeature ? [selectedFeature] : [])).slice(0, 3);
  const representativeAxisDetails = (backendState.workbench?.representative_axis_details || []).slice(0, 3);
  const representativeFeatureNames = representativeFeatures.map((item) => item?.feature_name || item?.feature_id).filter(Boolean);
  const answerTermMeta = useMemo(
    () => buildAnswerTermMeta(backendState.workbench, selectedProductCode || backendState.workbench?.input?.product || ''),
    [backendState.workbench, selectedProductCode],
  );
  const representativeFeatureIds = representativeFeatures.map((item) => item?.feature_id).filter(Boolean);
  const relatedFeatures = backendState.workbench?.related_features || [];
  const customerClusters = backendState.workbench?.customer_clusters || [];
  const retrievalResults = backendState.workbench?.retrieval_results || [];
  const clusterStorage = backendState.workbench?.cluster_storage || {};
  const roadmap = backendState.workbench?.roadmap || [];
  const graph = useMemo(
    () => buildGraph({ currentStageIndex, selectedNodeId, question: currentQuestion, runtimeStages, workbenchData: backendState.workbench }),
    [backendState.workbench, currentQuestion, currentStageIndex, runtimeStages, selectedNodeId],
  );

  const selectedNodeDetail = NODE_DETAILS[selectedNodeId] || {
    title: graph.nodes.find((node) => node.id === selectedNodeId)?.data?.title || 'Runtime Detail',
    subtitle: graph.nodes.find((node) => node.id === selectedNodeId)?.data?.keyLabel || 'ontology.runtime',
    relation: graph.nodes.find((node) => node.id === selectedNodeId)?.data?.meta || [],
    products: products.slice(0, 3).map((item) => item.code),
    clusters: clusters.slice(0, 3).map((item) => item.label),
  };
  const retrievalChartItems = [
    ...retrievalTrace.map((item) => ({ id: item.id, label: item.label, value: item.value, score: Number(item.score) || 0 })),
    ...((backendState.workbench?.related_features || []).slice(0, 4).map((item, index) => ({
      id: `related-${index}`,
      label: item.feature_name || item.feature_id || `feature-${index + 1}`,
      value: item.feature_id || item.feature_name || '-',
      score: Number(item.score) || 0,
    }))),
  ].slice(0, 8);
  const stagePeak = Math.max(100, ...runtimeStages.map((item) => item.progress || 0));
  const questionTokens = currentQuestion.replace(/[?]/g, '').split(/\s+/).filter(Boolean).slice(0, 8);
  const topFeatureName = representativeFeatureNames.join(' / ') || selectedFeature?.feature_name || selectedFeature?.feature_id || resolvedAnswerSummary?.highlights?.find((item) => item.label === 'Top Axes')?.value || resolvedAnswerSummary?.highlights?.find((item) => item.label === 'Primary Axis')?.value || '-';
  const topFeatureAliases = (selectedFeature?.aliases || []).slice(0, 6);
  const topFeatureDirections = (selectedFeature?.directions || []).slice(0, 4);
  const topFeatureScore = searchResults.find((item) => item.feature_id === selectedFeature?.feature_id)?.score || 0;
  const primarySelectionMode = primaryFeatureSelection?.mode || resultSummary.primary_feature_select_mode || 'topk-intent-graph-hybrid';
  const selectedFeatureIdsForTurn = representativeFeatureIds.length
    ? representativeFeatureIds
    : (selectedFeature?.feature_id ? [selectedFeature.feature_id] : []);
  const primaryTopCandidates = primaryFeatureSelection?.top_k || [];
  const primaryReferenceFeatures = (primaryFeatureSelection?.reference_features || primaryTopCandidates).slice(0, 3);
  const primaryRepresentativeFeature = representativeFeatures[0] || selectedFeature;
  const selectedPrimaryCandidate = primaryTopCandidates.find((item) => item.feature_id === primaryRepresentativeFeature?.feature_id) || primaryTopCandidates[0] || null;
  const primarySelectionCards = primaryFeatureSelection?.graph_result_explanation || [];
  const primaryGraphSupports = selectedPrimaryCandidate?.graph_edges || [];
  const questionTokenMappings = backendState.workbench?.question_token_mappings || [];
  const topClusterRejectDescriptions = (resultTopCluster?.top_reject_descriptions || []).slice(0, 3);
  const topClusterRejectCodes = ((resultTopCluster?.top_reject_codes || []).slice(0, 3)).map((item) => item.code).filter(Boolean);
  const selectedFeatureIsRejectDriven = isRejectDrivenFeature(selectedFeature);
  const semanticRankFormulaConfig = useMemo(() => getSemanticRankFormulaConfig(resultSummary), [
    resultSummary?.semantic_rank_formula?.semantic_weight,
    resultSummary?.semantic_rank_formula?.haystack_hit_weight,
    resultSummary?.semantic_rank_formula?.feature_id_hit_weight,
    resultSummary?.semantic_rank_formula?.feature_name_hit_weight,
    resultSummary?.semantic_rank_formula?.product_match_bonus,
    resultSummary?.semantic_rank_formula?.coverage_bonus_cap,
  ]);
  const [semanticRankEditor, setSemanticRankEditor] = useState(() => semanticRankFormulaConfig);
  useEffect(() => {
    setSemanticRankEditor(semanticRankFormulaConfig);
  }, [currentQuestion, semanticRankFormulaConfig]);
  const previewSearchResults = useMemo(() => searchResults
    .map((item) => computeSemanticRankPreview(item, semanticRankEditor))
    .sort((left, right) => right.preview_score - left.preview_score), [searchResults, semanticRankEditor]);
  const previewTopFeature = previewSearchResults[0] || null;
  const selectedFeatureBreakdown = searchResults.find((item) => item.feature_id === selectedFeature?.feature_id)?.score_breakdown || null;
  const backendRankMap = useMemo(() => new Map(searchResults.map((item, index) => [item.feature_id || item.feature_name || `${index}`, index + 1])), [searchResults]);
  const representativeRetrievalGroups = useMemo(() => representativeAxisDetails.map((axis) => {
    const evidenceTerms = [
      axis.feature_name || axis.feature_id || '',
      ...(axis.matched_tokens || []),
      ...((axis.graph_supports || []).map((item) => item.target_feature_name || item.target_feature_id || '')),
      ...((axis.related_features || []).map((item) => item.feature_name || item.feature_id || '')),
    ].filter(Boolean);
    const scoredRecords = retrievalResults
      .map((item) => {
        const snippet = [
          item.snippet || '',
          ...(item.reject_descriptions || []),
          ...(item.reject_codes || []),
          item.product || '',
        ].join(' ').toLowerCase();
        const score = evidenceTerms.reduce((total, term, index) => {
          const normalized = String(term || '').toLowerCase().trim();
          if (!normalized) {
            return total;
          }
          if (snippet.includes(normalized)) {
            return total + (index === 0 ? 4 : 2);
          }
          return total;
        }, 0);
        return { ...item, axis_score: score };
      })
      .sort((left, right) => (right.axis_score - left.axis_score) || ((right.score || 0) - (left.score || 0)))
      .slice(0, 3);
    return {
      ...axis,
      evidence_terms: evidenceTerms.slice(0, 6),
      top_records: (scoredRecords.some((item) => Number(item.axis_score || 0) > 0) ? scoredRecords : retrievalResults.slice(0, 3).map((item) => ({ ...item, axis_score: 0 }))),
    };
  }), [representativeAxisDetails, retrievalResults]);
  const stageHoverDetails = useMemo(() => {
    const retrievalIds = retrievalResults.slice(0, 4).map((item) => item.record_id || item.product || item.cluster_id || 'candidate');
    const clusterNames = customerClusters.slice(0, 3).map((item) => item.label || item.cluster_id || 'cluster');

    return {
      extraction: {
        badge: `${semanticStats.nodeCount.toLocaleString()} nodes`,
        title: '자료를 읽고 질문을 준비',
        work: [
          'commonfeature.json과 full_text_records.json을 읽어 분석 기준 데이터를 준비합니다.',
          `질문에서 ${questionTokens.length || 0}개의 의미 단서를 뽑았습니다.`,
        ],
        output: [
          `온톨로지 노드 ${semanticStats.nodeCount.toLocaleString()}건`,
          `질문 단서: ${summarizeList(questionTokens, '아직 단서 없음')}`,
        ],
      },
      alias: {
        badge: selectedProductCode || 'ALL',
        title: '상품 범위 필터링',
        work: [
          '질문에 나온 상품명을 기준으로 분석할 feature 범위를 좁힙니다.',
          '상품명이 없으면 전체 상품 후보를 보고 다음 단계로 넘깁니다.',
        ],
        output: [
          `선택 상품: ${backendState.workbench?.input?.product || selectedProductCode || 'ALL'}${selectedProductCode ? ` (${selectedProductLabel})` : ''}`,
          `필터 후 feature 후보: ${resultSummary.feature_count || 0}건`,
        ],
      },
      mapping: {
        badge: primarySelectionMode || resultSummary.semantic_search_mode || 'rank',
        title: '질문과 가까운 후보 추리기',
        work: [
          '질문, feature 이름, 별칭, 방향성을 비교해 상위 후보를 고릅니다.',
          '한 개만 단정하지 않고 관련 후보를 함께 묶어 비교합니다.',
        ],
        output: [
          `주요 후보: ${summarizeList(primaryReferenceFeatures.map((item) => item.feature_name || item.feature_id), '아직 후보 없음')}`,
          `기본 랭킹 방식: ${resultSummary.semantic_search_mode || 'pending'}`,
        ],
      },
      ontology: {
        badge: `${relatedFeatures.length + representativeFeatures.length} linked`,
        title: '핵심 기준과 연결 근거 확인',
        work: [
          '상위 후보가 질문 의도와 직접 맞는지 다시 계산합니다.',
          '그래프 연결 근거를 확인해 핵심 축을 정합니다.',
        ],
        output: [
          `핵심 축: ${topFeatureName}`,
          `그래프 근거: ${summarizeList(primaryGraphSupports.slice(0, 3).map((item) => item.target_feature_name || item.target_feature_id), '그래프 근거 없음')}`,
        ],
      },
      faiss: {
        badge: `${customerClusters.length} clusters`,
        title: '고객군집 후보 확인',
        work: [
          '사전 계산된 고객군집 캐시를 불러와 질문과 맞는 후보를 찾습니다.',
          '상품과 feature 범위에 맞는 상위 군집을 계산합니다.',
        ],
        output: [
          `군집 저장 방식: ${resultSummary.cluster_storage_mode || 'file-cache'}`,
          `상위 군집: ${summarizeList(clusterNames, '아직 군집 없음')}`,
        ],
      },
      retrieval: {
        badge: `${retrievalResults.length} records`,
        title: '관련 로그와 근거 수집',
        work: [
          '핵심 축과 군집을 기준으로 관련 로그 근거를 모읍니다.',
          '답변에 쓸 수 있는 실제 코드와 기록을 정리합니다.',
        ],
        output: [
          `검색 결과: ${retrievalResults.length}건`,
          `상위 record: ${summarizeList(retrievalIds, '아직 record 없음')}`,
        ],
      },
      ollama: {
        badge: `${ollamaStatus}`,
        title: '최종 답변 생성',
        work: [
          '앞 단계에서 계산한 feature, 군집, 검색 결과를 짧은 컨텍스트로 묶습니다.',
          'Ollama는 마지막 자연어 답변 생성에만 사용합니다.',
        ],
        output: [
          `요약: ${resolvedAnswerSummary?.headline || '아직 요약 없음'}`,
          `입력 근거: ${summarizeList((ollamaInput?.context_preview || []).slice(0, 2), '아직 입력 없음')}`,
          `출력 미리보기: ${String(ollamaOutput?.response_preview || ollamaOutput?.response_text || ollamaRuntime?.error || '아직 답변 없음').slice(0, 120)}`,
        ],
      },
    };
  }, [
    backendState.workbench?.input?.product,
    customerClusters,
    ollamaInput?.context_preview,
    ollamaOutput?.response_preview,
    ollamaOutput?.response_text,
    ollamaRuntime?.error,
    ollamaStatus,
    questionTokens,
    relatedFeatures,
    primaryGraphSupports,
    primarySelectionMode,
    primaryReferenceFeatures,
    resolvedAnswerSummary,
    resultSummary.cluster_storage_mode,
    resultSummary.feature_count,
    resultSummary.semantic_search_mode,
    retrievalResults,
    representativeFeatures,
    selectedProductCode,
    selectedProductLabel,
    semanticStats.nodeCount,
    topFeatureName,
  ]);
  const querySignalCards = (questionTokenMappings.length ? questionTokenMappings : questionTokens.map((token, index) => ({ id: `token-${index}`, token, primary_label: '', reason: '토큰 근거를 계산 중입니다.', feature_links: [] }))).map((item, index) => {
    if (questionTokenMappings.length) {
      return {
        id: item.id || `token-${index}`,
        token: item.token,
        label: (item.feature_links || []).length
          ? (item.feature_links || []).slice(0, 3).map((link) => link.feature_name).join(' / ')
          : (item.primary_label || item.concept_label || '직접 연결된 feature 없음'),
        reason: item.reason || '질문과 연결되는 기준을 확인했습니다.',
        featureLinks: item.feature_links || [],
      };
    }
    const token = item.token;
    const normalizedToken = String(token || '').toLowerCase();
    if (normalizedToken.includes('카드론') || normalizedToken.includes('대출')) {
      return {
        id: `token-${index}`,
        token,
        label: selectedProductCode || 'ALL',
        reason: selectedProductCode ? `${selectedProductLabel} 상품 맥락을 지정하는 단어입니다.` : '상품명이 없어 전체 상품 범위에서 해석합니다.',
      };
    }
    if (/\d/.test(normalizedToken) || normalizedToken.includes('대')) {
      return { id: `token-${index}`, token, label: 'applicant.age', reason: '연령 조건으로 볼 수 있어 고객 연령 관련 feature 후보를 강화합니다.' };
    }
    if (normalizedToken.includes('승인') || normalizedToken.includes('거절')) {
      return { id: `token-${index}`, token, label: resultTopCluster?.decision || 'decision', reason: '승인/거절 같은 심사 결과 신호로 해석합니다.', featureLinks: [] };
    }
    const candidates = [...representativeFeatures, ...relatedFeatures]
      .filter(Boolean)
      .map((candidate) => {
        const haystack = [
          candidate?.feature_name,
          candidate?.feature_id,
          ...(candidate?.aliases || []),
          ...(candidate?.directions || []),
          candidate?.category,
        ].join(' ').toLowerCase();
        const matched = normalizedToken && haystack.includes(normalizedToken);
        return {
          label: candidate?.feature_name || candidate?.feature_id || '-',
          reason: candidate?.description || `${candidate?.category || 'feature'} 영역에서 해석합니다.`,
          matched,
        };
      });
    const directMatch = candidates.find((candidate) => candidate.matched);
    if (directMatch) {
      return { id: `token-${index}`, token, label: directMatch.label, reason: directMatch.reason, featureLinks: [] };
    }
    return { id: `token-${index}`, token, label: '직접 연결된 feature 없음', reason: '상위 feature와 바로 매칭되지는 않아 보조 단서로만 봅니다.', featureLinks: [] };
  });
  const answerNarrativeSteps = [
    { id: 'narrative-question', title: '질문 수신', detail: `사용자 질문 '${currentQuestion}'을 확인했습니다.` },
    { id: 'narrative-feature', title: '핵심 축 선택', detail: primaryFeatureSelection?.headline || `${topFeatureName}을 핵심 축으로 봅니다.` },
    { id: 'narrative-cluster', title: '관련 군집 결합', detail: `${resultTopCluster?.label || '상위 군집 계산 결과'}와 검색 결과를 붙입니다.` },
    { id: 'narrative-ollama-input', title: 'Ollama 입력 구성', detail: summarizeList((ollamaInput?.context_preview || []).slice(0, 3), '질문, 핵심 축, 군집 정보를 prompt로 묶습니다.') },
    { id: 'narrative-answer', title: '최종 답변', detail: String(ollamaOutput?.response_preview || finalAnswerBody || ollamaRuntime?.error || '최종 답변을 생성합니다.').slice(0, 180) },
  ];
  const conversationStatusValue = backendState.error
    ? 'failed'
    : backendState.loading
      ? 'running'
      : (finalAnswerBody || resolvedAnswerSummary?.headline)
        ? 'completed'
        : 'idle';

  useEffect(() => {
    if (!hasSubmittedRuntimeQuery) {
      return;
    }
    if (!currentQuestion) {
      return;
    }

    const nextHeadline = hasResolvedRuntimeAnswer ? String(resolvedAnswerSummary?.headline || '').trim() : '';
    const nextBody = String(finalAnswerBody || '').trim();

    setConversationTurns((previous) => {
      const nextTurns = [...previous];
      let targetIndex = -1;
      for (let index = nextTurns.length - 1; index >= 0; index -= 1) {
        if (nextTurns[index].question === currentQuestion) {
          targetIndex = index;
          break;
        }
      }

      if (targetIndex < 0) {
        nextTurns.push({
          id: `turn-${Date.now()}`,
          question: currentQuestion,
          answerHeadline: nextHeadline,
          answerBody: nextBody,
          status: conversationStatusValue,
          product: selectedProductCode || 'ALL',
          axis: topFeatureName,
          selectedFeatureIds: selectedFeatureIdsForTurn,
          timestamp: lastRunAt || formatConversationTime(),
          sourceTag: currentAnswerSourceTag,
        });
        return nextTurns.slice(-12);
      }

      const existing = nextTurns[targetIndex];
      const merged = {
        ...existing,
        answerHeadline: nextHeadline || existing.answerHeadline,
        answerBody: nextBody || existing.answerBody,
        status: conversationStatusValue,
        product: selectedProductCode || existing.product || 'ALL',
        axis: topFeatureName || existing.axis,
        selectedFeatureIds: selectedFeatureIdsForTurn.length ? selectedFeatureIdsForTurn : (existing.selectedFeatureIds || []),
        timestamp: lastRunAt || existing.timestamp || formatConversationTime(),
        sourceTag: currentAnswerSourceTag,
      };

      const unchanged = existing.answerHeadline === merged.answerHeadline
        && existing.answerBody === merged.answerBody
        && existing.status === merged.status
        && existing.product === merged.product
        && existing.axis === merged.axis
        && JSON.stringify(existing.selectedFeatureIds || []) === JSON.stringify(merged.selectedFeatureIds || [])
        && existing.timestamp === merged.timestamp
        && existing.sourceTag === merged.sourceTag;

      if (unchanged) {
        return previous;
      }

      nextTurns[targetIndex] = merged;
      return nextTurns.slice(-12);
    });
  }, [
    conversationStatusValue,
    currentQuestion,
    finalAnswerBody,
    hasSubmittedRuntimeQuery,
    hasResolvedRuntimeAnswer,
    lastRunAt,
    ollamaStatus,
    currentAnswerSourceTag,
    resolvedAnswerSummary?.headline,
    selectedProductCode,
    selectedFeatureIdsForTurn,
    topFeatureName,
  ]);

  const displayedConversationTurns = conversationTurns.length
    ? conversationTurns
    : hasSubmittedRuntimeQuery ? [{
      id: 'turn-initial',
      question: currentQuestion,
      answerHeadline: hasResolvedRuntimeAnswer ? (resolvedAnswerSummary?.headline || '') : '',
      answerBody: finalAnswerBody || '',
      status: conversationStatusValue,
      product: selectedProductCode || 'ALL',
      axis: topFeatureName,
      selectedFeatureIds: selectedFeatureIdsForTurn,
      timestamp: lastRunAt || formatConversationTime(),
      sourceTag: currentAnswerSourceTag,
    }] : [];

  const latestConversationTurnId = displayedConversationTurns[displayedConversationTurns.length - 1]?.id;
  const orderedConversationTurns = latestConversationTurnId
    ? [
        ...displayedConversationTurns.filter((turn) => turn.id === latestConversationTurnId),
        ...displayedConversationTurns.filter((turn) => turn.id !== latestConversationTurnId).reverse(),
      ]
    : displayedConversationTurns;

  function handleRunQuery() {
    const nextQuestion = queryInput.trim();
    if (!nextQuestion) {
      return;
    }
    const nextProductCode = resolveProductForQuestion(nextQuestion, products);
    setConversationTurns((previous) => ([
      ...previous,
      {
        id: `turn-${Date.now()}`,
        question: nextQuestion,
        answerHeadline: '',
        answerBody: '',
        status: 'running',
        product: nextProductCode || 'ALL',
        axis: '-',
        selectedFeatureIds: [],
        timestamp: formatConversationTime(),
        sourceTag: 'server-linked',
      },
    ].slice(-12)));
    setBackendState((previous) => ({ ...previous, loading: true, error: '', workbench: null }));
    setClarificationAnswered(false);
    submitQuestion(nextQuestion);
    onToast?.({
      id: `ontology-runtime-${Date.now()}`,
      kicker: 'Semantic Runtime',
      title: 'Ontology runtime query 실행',
      meta: 'JSON + FAISS pipeline',
      message: nextQuestion,
      tone: 'cyan',
    });
  }

  function handleUsePromptExample(prompt) {
    setQueryInput(prompt);
    setShowPromptExamples(false);
  }

  function handleOpenDetailModal(nextTab = activeSubtab) {
    setActiveSubtab(nextTab);
    setShowDetailModal(true);
  }

  async function handleOpenMetricCubeModal(nextTab = 'cube', forceRebuild = false) {
    setMetricCubeTab(nextTab);
    setShowMetricCubeModal(true);
    if (metricCubeState.data && !forceRebuild) {
      return;
    }
    setMetricCubeState((previous) => ({ ...previous, loading: true, error: '' }));
    try {
      const payload = await fetchFeatureOntologySegmentMetricCube({ forceRebuild });
      setMetricCubeState({ loading: false, error: '', data: payload });
    } catch (error) {
      setMetricCubeState((previous) => ({ ...previous, loading: false, error: String(error.message || error) }));
    }
  }

  async function handleLoadProductDevelopmentAgendas(force = false) {
    if (!force && productDevelopmentState.agendas?.length) {
      return;
    }
    setProductDevelopmentState((previous) => ({
      ...previous,
      loadingAgendas: true,
      agendaError: '',
      selectedAgenda: force ? null : previous.selectedAgenda,
      debate: force ? null : previous.debate,
      debateError: force ? '' : previous.debateError,
    }));
    try {
      const payload = await createProductDevelopmentAgendas({
        department_concepts: departmentConceptsRef.current,
      });
      setProductDevelopmentState((previous) => ({
        ...previous,
        loadingAgendas: false,
        agendaPayload: payload,
        agendas: payload?.agendas || [],
        source: payload?.source || '',
      }));
      onToast?.({
        id: `product-dev-agendas-${Date.now()}`,
        kicker: 'Product Workshop',
        title: '상품개발 안건 2개 준비',
        meta: payload?.source === 'ollama' ? 'Ollama 1회 호출' : '캐시 기반 초안',
        message: '금융솔루션부가 통계 큐브와 군집분석을 보고 토론 안건을 뽑았습니다.',
        tone: 'completed',
      });
    } catch (error) {
      setProductDevelopmentState((previous) => ({
        ...previous,
        loadingAgendas: false,
        agendaError: String(error.message || error),
      }));
    }
  }

  async function handleSelectProductDevelopmentAgenda(agenda) {
    setProductDevelopmentState((previous) => ({
      ...previous,
      selectedAgenda: agenda,
      debateLoading: true,
      debateError: '',
      debate: null,
    }));
    try {
      const payload = await createProductDevelopmentDebate({
        selected_agenda: agenda,
        department_concepts: departmentConceptsRef.current,
      });
      setProductDevelopmentState((previous) => ({
        ...previous,
        debateLoading: false,
        debate: payload,
        debateSource: payload?.source || '',
      }));
      onToast?.({
        id: `product-dev-debate-${Date.now()}`,
        kicker: '4-Department Debate',
        title: '상품개발 토론 완료',
        meta: payload?.source === 'ollama' ? 'Ollama 1회 호출' : '캐시 기반 초안',
        message: '신상품 제안과 기존 상품별 보완 로직을 함께 정리했습니다.',
        tone: 'completed',
      });
    } catch (error) {
      setProductDevelopmentState((previous) => ({
        ...previous,
        debateLoading: false,
        debateError: String(error.message || error),
      }));
    }
  }

  useEffect(() => {
    if (answerMode !== 'product') {
      return;
    }
    void handleLoadProductDevelopmentAgendas(false);
  }, [answerMode]);

  function handleClarificationSelect(option) {
    const clarificationQuery = `${currentQuestion} - ${option.feature_name || option.feature_id}`;
    setClarificationAnswered(true);
    setConversationTurns((previous) => ([
      ...previous,
      {
        id: `turn-${Date.now()}`,
        question: clarificationQuery,
        answerHeadline: '',
        answerBody: '',
        status: 'running',
        product: selectedProductCode || 'ALL',
        axis: option.feature_name || option.feature_id || '-',
        selectedFeatureIds: [option.feature_id].filter(Boolean),
        timestamp: formatConversationTime(),
        sourceTag: 'server-linked',
      },
    ].slice(-12)));
    setBackendState((previous) => ({ ...previous, loading: true, error: '', workbench: null }));
    submitQuestion(clarificationQuery);
  }

  function handleToggleAvoidedFeature(featureId) {
    setAvoidedFeatureIds((previous) => {
      if (previous.includes(featureId)) {
        return previous.filter((id) => id !== featureId);
      }
      return [...previous, featureId].slice(0, 8);
    });
  }

  function handleLearnClick() {
    regulationIntentRef.current = true;
    onRequestRegulationUpload?.();
    onToast?.({
      id: `ontology-regulation-${Date.now()}`,
      kicker: 'Roni Navigator',
      title: regulationBusy ? '규제 학습 진행 중' : '규제 문서 선택기 열기',
      meta: 'Ontology regulation intake',
      message: regulationBusy ? '이미 업로드된 문서를 학습하는 중입니다.' : '학습할 규제 문서를 선택하세요.',
      tone: 'amber',
    });
  }

  useEffect(() => {
    const previousStatus = regulationStatusRef.current;
    const completedNow = regulationStatus === 'completed' && previousStatus !== 'completed';
    const shouldCelebrate = completedNow && (regulationIntentRef.current || previousStatus === 'running');

    if (shouldCelebrate) {
      setShowRoniSuccess(true);
      onToast?.({
        id: `ontology-regulation-complete-${Date.now()}`,
        kicker: 'Roni Navigator',
        title: '규제 문서 학습 완료',
        meta: regulationUpdatedAt || 'regulation synced',
        message: regulationSummary || '규제 문서 학습이 완료되어 최신 요약을 반영했습니다.',
        tone: 'cyan',
      });
      regulationIntentRef.current = false;
    }

    regulationStatusRef.current = regulationStatus;
  }, [onToast, regulationStatus, regulationSummary, regulationUpdatedAt]);

  useEffect(() => {
    if (!showRoniSuccess) {
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setShowRoniSuccess(false);
    }, reduceMotion ? 900 : 2400);

    return () => window.clearTimeout(timerId);
  }, [reduceMotion, showRoniSuccess]);

  useEffect(() => {
    const nextNewsId = String(latestNewsSignal?.id || '');
    if (!nextNewsId) {
      return undefined;
    }

    if (!newsSignalInitializedRef.current) {
      newsSignalInitializedRef.current = true;
      latestNewsSignalRef.current = nextNewsId;
      return undefined;
    }

    if (latestNewsSignalRef.current === nextNewsId) {
      return undefined;
    }

    latestNewsSignalRef.current = nextNewsId;
    setActiveNewsBubble({
      title: String(latestNewsSignal?.title || '').trim() || '카드론 심사 영향 뉴스',
      summary: String(latestNewsSignal?.summary || '').trim() || '카드론 심사에 영향을 줄 수 있는 뉴스 신호를 찾았습니다.',
      time: String(latestNewsSignal?.time || '').trim(),
    });
    onToast?.({
      id: `ontology-news-${nextNewsId}`,
      kicker: 'Roni News Signal',
      title: '카드론 심사에 영향을 주는 뉴스 발견',
      meta: latestNewsSignal?.time || 'realtime news',
      message: String(latestNewsSignal?.title || '').trim() || '새 뉴스 신호가 들어왔습니다.',
      tone: 'cyan',
    });

    const timerId = window.setTimeout(() => {
      setActiveNewsBubble((current) => (current?.title === (latestNewsSignal?.title || '') ? null : current));
    }, 5000);

    return () => window.clearTimeout(timerId);
  }, [latestNewsSignal, onToast]);

  const isAgentIdle = !hasSubmittedRuntimeQuery && !backendState.loading && runtimeJobStatus !== 'running';
  const visibleAgentStages = (runtimeStages.length ? runtimeStages : RUNTIME_STAGES.map((stage) => ({ ...stage, status: 'pending' })))
    .map((stage) => ({ ...stage, status: isAgentIdle ? 'pending' : stage.status }))
    .slice(0, 5);
  const agentPanelHeadline = isAgentIdle ? '질문 대기 중' : (isRuntimeComplete ? '분석 완료' : progressHeadline);
  const activeToolSummary = (answerMode === 'product' ? productModeToolCards : activeToolCards)
    .map((tool) => tool.title || tool.id)
    .filter(Boolean)
    .slice(0, 3);
  const runtimeToolCards = answerMode === 'product' ? productModeToolCards : activeToolCards;
  const semanticRefresh = semanticRefreshState.refresh || {};
  const semanticRefreshStatus = String(semanticRefresh.status || '');
  const semanticRefreshLabel = semanticRefreshState.error
    ? '자동 갱신 상태 확인 실패'
    : semanticRefreshStatus === 'running'
      ? '2분 자동 갱신 중'
      : semanticRefresh.last_completed_at
        ? `마지막 갱신 ${formatSemanticRefreshTime(semanticRefresh.last_completed_at)}`
        : '2분마다 자동 갱신';
  const semanticRefreshDetail = semanticRefreshStatus === 'completed'
    ? `${Number(semanticRefresh.record_count || 0).toLocaleString('ko-KR')}건 · 세그먼트 ${Number(semanticRefresh.segment_count || 0).toLocaleString('ko-KR')}개`
    : semanticRefreshStatus === 'running'
      ? '최신 로그 반영 중'
      : '큐브와 군집을 함께 갱신';
  const runtimeToolMap = Object.fromEntries(runtimeToolCards.map((tool) => [String(tool?.id || '').toLowerCase(), tool]));
  const activeToolIds = new Set(runtimeToolCards.map((tool) => String(tool?.id || '').toLowerCase()).filter(Boolean));
  const workspaceToolTabs = hasResolvedRuntimeAnswer
    ? [
        { id: 'summary', label: '요약' },
        (activeToolIds.has('explainability') || runtimeToolCards.length > 0)
          ? { id: 'insight', label: '상세 인사이트' }
          : null,
        activeToolIds.has('policy') ? { id: 'policy', label: '정책/규제 영향' } : null,
        activeToolIds.has('strategy') ? { id: 'strategy', label: '상품 시뮬레이션' } : null,
      ].filter(Boolean)
    : [];
  const selectedWorkspaceResultTab = workspaceToolTabs.some((tab) => tab.id === activeWorkspaceResultTab)
    ? activeWorkspaceResultTab
    : (workspaceToolTabs[0]?.id || 'summary');
  const summaryHighlightItems = (resolvedAnswerSummary?.highlights || []).slice(0, 4);
  const clusterToolCard = runtimeToolMap.cluster;
  const insightToolCard = runtimeToolMap.explainability;
  const policyToolCard = runtimeToolMap.policy;
  const strategyToolCard = runtimeToolMap.strategy;
  const clusterInsightItems = customerClusters.slice(0, 3);
  const metricInsightItems = [
    ...(clusterToolCard?.metrics || []).slice(0, 3),
    ...(insightToolCard?.metrics || []).slice(0, 2),
  ].slice(0, 4);
  const promptDock = (
    <div className="roni-prompt-dock workspace-prompt-dock" style={{ maxWidth: '900px', width: '100%' }}>
      <div className="ontology-chat-input-shell copilot-composer-shell roni-prompt-dock-input-shell">
        <div className="workspace-prompt-textarea-shell">
          <textarea
            className="ontology-chat-textarea roni-prompt-textarea"
            value={queryInput}
            onChange={(event) => setQueryInput(event.target.value)}
            placeholder="추가로 궁금한 점을 입력해 주세요. 예: 승인군과 거절군의 평균 금리 차이는?"
            rows={2}
          />
          <PromptLottieActionButton
            variant="send"
            className="primary-button ontology-runtime-run workspace-inline-send"
            ariaLabel="보내기"
            title="보내기"
            onClick={handleRunQuery}
          />
        </div>
        <div className="ontology-chat-action-row">
          <div className="workspace-prompt-state workspace-mode-switcher" aria-label="답변 모드 선택">
            <div className="workspace-mode-choice-row" role="group" aria-label="답변 모드 선택">
              {ANSWER_MODES.map((mode) => {
                const isActiveMode = answerMode === mode.id;
                return (
                  <motion.button
                    key={mode.id}
                    type="button"
                    className={`workspace-mode-choice ${isActiveMode ? 'active' : ''}`}
                    onClick={() => setAnswerMode(mode.id)}
                    title={mode.hint}
                    aria-pressed={isActiveMode}
                    whileHover={{ y: -2, scale: 1.025 }}
                    whileTap={{ scale: 0.96 }}
                    transition={{ type: 'spring', stiffness: 420, damping: 28 }}
                  >
                    {isActiveMode ? <motion.span className="workspace-mode-choice-glow" layoutId="workspace-mode-choice-glow" transition={{ type: 'spring', stiffness: 420, damping: 34 }} /> : null}
                    <span className="workspace-mode-icon" aria-hidden="true">{mode.icon}</span>
                    <strong>{mode.label}</strong>
                  </motion.button>
                );
              })}
            </div>
            <span className="sample-pill">상품 {getProductDisplayName(selectedProductCode)}</span>
          </div>
          <div className="ontology-chat-action-buttons">
            <PromptLottieActionButton
              className={`secondary-button ontology-example-toggle ${showPromptExamples ? 'is-open' : ''}`}
              ariaLabel={showPromptExamples ? '예시 닫기' : '예시 보기'}
              title={showPromptExamples ? '예시 닫기' : '예시 보기'}
              onClick={() => setShowPromptExamples((value) => !value)}
            />
          </div>
        </div>
        <AnimatePresence>
          {showPromptExamples ? (
            <motion.div
              className="ontology-example-drawer"
              initial={{ opacity: 0, y: -8, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -8, height: 0 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              {ONTOLOGY_PROMPT_EXAMPLES.map((section) => (
                <div key={section.group} className={`ontology-example-group is-${section.accent}`}>
                  <span>{section.group}</span>
                  <div className="ontology-example-chip-grid">
                    {section.prompts.map((prompt) => (
                      <button key={prompt} type="button" className="ontology-example-chip" onClick={() => handleUsePromptExample(prompt)}>
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </motion.div>
          ) : null}
        </AnimatePresence>
        <AnimatePresence>
          {answerMode === 'memo' ? (
            <motion.div
              className="memo-mode-panel"
              initial={{ opacity: 0, y: -8, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -8, height: 0 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              <div className="memo-department-row">
                {MEMO_DEPARTMENTS.map((department) => (
                  <button
                    key={department.id}
                    type="button"
                    className={`memo-department-chip ${memoDepartment === department.id ? 'active' : ''}`}
                    onClick={() => setMemoDepartment(department.id)}
                  >
                    <span aria-hidden="true">{department.icon}</span>
                    <strong>{department.label}</strong>
                  </button>
                ))}
                <button type="button" className="memo-concept-button" onClick={() => setShowConceptModal(true)}>
                  컨셉 보기
                </button>
              </div>
              <textarea
                className="memo-mode-textarea"
                value={memoDepartmentNotes[memoDepartment] || ''}
                onChange={(event) => {
                  const nextValue = event.target.value;
                  setMemoNotes(nextValue);
                  setMemoDepartmentNotes((previous) => ({ ...previous, [memoDepartment]: nextValue }));
                }}
                placeholder={`${MEMO_DEPARTMENTS.find((item) => item.id === memoDepartment)?.label || '부서'}가 중요하게 보는 기준을 적어주세요.`}
                rows={2}
              />
              <p>저장된 메모는 부서 컨셉에 붙어서 상품개발 안건과 4인 토론에 반영됩니다.</p>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );

  return (
    <motion.section key="ontology-runtime-console" className="content-stack ontology-runtime-shell" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -14 }} transition={{ duration: 0.28, ease: 'easeOut' }}>
      <div className="ontology-copilot-layout">
        <div className="conversation-workspace-ribbon">
          <strong>Bunny 금융 워크스페이스</strong>
          <span>질문하면 답변, 군집, 정책 분석이 함께 열립니다</span>
        </div>
        <div className="ontology-copilot-main" style={{ maxWidth: '1680px', width: '100%', margin: '0 auto' }}>
          <section className="panel ontology-runtime-hero ontology-runtime-hero-compact roni-floating-stage" style={{ maxWidth: '100%', minWidth: 0 }}>
            <div
              className={`ontology-hero-roni-panel ontology-hero-roni-panel-top ${isRoniHovered ? 'is-hovered' : ''}`}
              onMouseEnter={() => setIsRoniHovered(true)}
              onMouseLeave={() => {
                setIsRoniHovered(false);
                setIsNewsBadgeHovered(false);
              }}
              onFocus={() => setIsRoniHovered(true)}
              onBlur={() => {
                setIsRoniHovered(false);
                setIsNewsBadgeHovered(false);
              }}
            >
              <div
                className={`roni-news-counter ${isNewsBadgeHovered ? 'active' : ''}`}
                onMouseEnter={() => setIsNewsBadgeHovered(true)}
                onMouseLeave={() => setIsNewsBadgeHovered(false)}
                onFocus={() => setIsNewsBadgeHovered(true)}
                onBlur={() => setIsNewsBadgeHovered(false)}
              >
                <span className="roni-news-lottie-frame" aria-hidden="true">
                  <DotLottieReact className="roni-news-lottie" data={cardLoanNewsAnimation} loop autoplay />
                </span>
                <strong className="roni-news-counter-count" aria-label={`카드론 뉴스 ${newsBadgeCount}건`}>{newsBadgeCount}</strong>
                <AnimatePresence>
                  {isNewsBadgeHovered ? (
                    <motion.aside className="roni-news-counter-popover" initial={{ opacity: 0, y: -8, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8, scale: 0.98 }} transition={{ duration: 0.18, ease: 'easeOut' }}>
                      <div className="roni-news-counter-head">
                        <span className="panel-kicker">Collected News</span>
                        <span className="sample-pill">{newsBadgeCount}건</span>
                      </div>
                      <div className="roni-news-counter-list">
                        {latestNewsItems.length ? latestNewsItems.map((item) => (
                          <article key={item.id} className="roni-news-counter-item">
                            <strong>{item.title}</strong>
                            <p>{item.summary}</p>
                          </article>
                        )) : <div className="empty-box compact">표시할 수집 뉴스가 없습니다.</div>}
                      </div>
                    </motion.aside>
                  ) : null}
                </AnimatePresence>
              </div>
              <RoniAvatar stageKey={activeStage.key} agentState={agentState} thinkingPhase={thinkingPhase} liveSpeech={roniCaption} reduceMotion={reduceMotion} surprised={isRoniHovered && !showRoniSuccess} success={showRoniSuccess} onDocumentRequest={handleLearnClick} />
              <div className={`roni-caption-box mood-${roniCaption.mood || 'idle'}`}>
                <strong>Bunny</strong>
                <p>
                  <StreamingText text={roniCaption.message} active={!reduceMotion} reduceMotion={reduceMotion} speed={20} />
                </p>
                <span><i aria-hidden="true"><b /><b /><b /></i>{roniCaption.detail}</span>
              </div>
              <div className="agent-panel-status-card">
                <div className="agent-panel-card-head">
                  <div>
                    <span className="panel-kicker">진행 상태</span>
                    <strong>{agentPanelHeadline}</strong>
                  </div>
                  <span className={`agent-live-dot ${!isAgentIdle && (backendState.loading || runtimeJobStatus === 'running') ? 'is-running' : ''}`} />
                </div>
                <div className="agent-progress-checklist">
                  {visibleAgentStages.map((stage) => (
                    <div key={stage.key} className={`agent-progress-step ${toneClass(stage.status)}`}>
                      <span aria-hidden="true">{stage.status === 'completed' ? '✓' : stage.status === 'running' ? '○' : '•'}</span>
                      <strong>{stageShortLabel(stage.key)}</strong>
                      <small>{statusLabelKo(stage.status)}</small>
                    </div>
                  ))}
                </div>
                <div className="agent-active-tool-row">
                  {(activeToolSummary.length ? activeToolSummary : ['질문 대기']).map((toolName) => (
                    <span key={toolName}>{toolName}</span>
                  ))}
                </div>
              </div>
              <motion.button
                type="button"
                className="agent-metric-cube-launcher"
                whileHover={reduceMotion ? undefined : { y: -3, scale: 1.015 }}
                whileTap={reduceMotion ? undefined : { scale: 0.985 }}
                onClick={() => handleOpenMetricCubeModal('cube')}
                aria-label="통계 큐브와 군집분석 보기"
              >
                <span className="agent-metric-cube-orb" aria-hidden="true"><i /><i /><i /></span>
                <span className="agent-metric-cube-copy">
                  <small>통계 큐브</small>
                  <strong>평균 · 승인률 · 연체위험 보기</strong>
                  <em>
                    {metricCubeState.data?.meta?.segment_count
                      ? `${Number(metricCubeState.data.meta.segment_count).toLocaleString('ko-KR')}개 세그먼트`
                      : 'data/segment_metric_cube.json'}
                    {' · '}
                    {metricCubeState.data?.cluster_summary?.total_clusters
                      ? `군집 ${Number(metricCubeState.data.cluster_summary.total_clusters).toLocaleString('ko-KR')}개`
                      : '군집분석 함께 보기'}
                  </em>
                  <small className={`agent-metric-cube-refresh ${semanticRefreshStatus === 'running' ? 'is-running' : ''}`}>
                    {semanticRefreshLabel} · {semanticRefreshDetail}
                  </small>
                </span>
                <span className="agent-metric-cube-arrow" aria-hidden="true">↗</span>
              </motion.button>
              <AnimatePresence>
                {activeNewsBubble ? (
                  <motion.aside className="roni-news-alert-bubble" initial={{ opacity: 0, y: -8, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -10, scale: 0.98 }} transition={{ duration: 0.22, ease: 'easeOut' }}>
                    <div className="roni-news-alert-head">
                      <span className="panel-kicker">Realtime News</span>
                      <span className="sample-pill">5s live</span>
                    </div>
                    <strong>카드론 심사에 영향을 주는 뉴스 발견</strong>
                    <p>{activeNewsBubble.title}</p>
                    <small>{activeNewsBubble.summary}</small>
                  </motion.aside>
                ) : null}
              </AnimatePresence>
              <AnimatePresence>
                <motion.button
                  type="button"
                  className={`roni-doc-bubble ${regulationBusy ? 'is-busy' : ''}`}
                  initial={{ opacity: 0, y: -6, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -6, scale: 0.95 }}
                  transition={{ duration: 0.18, ease: 'easeOut' }}
                  onClick={handleLearnClick}
                  disabled={regulationBusy}
                  aria-label={regulationBusy ? '규제 문서 학습 진행 중' : '규제 문서 업로드'}
                  title={regulationBusy ? `규제 문서 학습 진행 중 (${regulationUpdatedAt || '진행 중'})` : '규제 문서 업로드'}
                >
                  <span className="roni-doc-lottie-frame" aria-hidden="true">
                    <DotLottieReact className="roni-doc-lottie" data={regulationDocumentAnimation} loop autoplay />
                  </span>
                  {regulationBusy ? <span className="roni-doc-bubble-busy-dot" aria-hidden="true" /> : null}
                  <span className="roni-doc-bubble-tail" aria-hidden="true" />
                </motion.button>
              </AnimatePresence>

              <div className="roni-prompt-dock">
                <div className="ontology-chat-input-shell copilot-composer-shell roni-prompt-dock-input-shell">
                  <textarea
                    className="ontology-chat-textarea roni-prompt-textarea"
                    value={queryInput}
                    onChange={(event) => setQueryInput(event.target.value)}
                    placeholder="로니에게 이어서 질문하세요. 예: 40대 중 직위험군 기준으로 한도는?"
                    rows={2}
                  />
                  <div className="ontology-chat-action-row">
                    <div className="answer-mode-nav" aria-label="답변 모드 선택">
                      {ANSWER_MODES.map((mode) => (
                        <button
                          key={mode.id}
                          type="button"
                          className={`answer-mode-pill ${answerMode === mode.id ? 'active' : ''}`}
                          onClick={() => setAnswerMode(mode.id)}
                          title={mode.hint}
                        >
                          <span aria-hidden="true">{mode.icon}</span>
                          <strong>{mode.label}</strong>
                        </button>
                      ))}
                    </div>
                    <div className="ontology-chat-action-buttons">
                      <PromptLottieActionButton
                        className={`secondary-button ontology-example-toggle ${showPromptExamples ? 'is-open' : ''}`}
                        ariaLabel={showPromptExamples ? '예시 닫기' : '예시 보기'}
                        title={showPromptExamples ? '예시 닫기' : '예시 보기'}
                        onClick={() => setShowPromptExamples((value) => !value)}
                      />
                      <PromptLottieActionButton
                        variant="send"
                        className="primary-button ontology-runtime-run"
                        ariaLabel="보내기"
                        title="보내기"
                        onClick={handleRunQuery}
                      />
                    </div>
                  </div>
                  <AnimatePresence>
                    {showPromptExamples ? (
                      <motion.div
                        className="ontology-example-drawer"
                        initial={{ opacity: 0, y: -8, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: 'auto' }}
                        exit={{ opacity: 0, y: -8, height: 0 }}
                        transition={{ duration: 0.2, ease: 'easeOut' }}
                      >
                        {ONTOLOGY_PROMPT_EXAMPLES.map((section) => (
                          <div key={section.group} className={`ontology-example-group is-${section.accent}`}>
                            <span>{section.group}</span>
                            <div className="ontology-example-chip-grid">
                              {section.prompts.map((prompt) => (
                                <button key={prompt} type="button" className="ontology-example-chip" onClick={() => handleUsePromptExample(prompt)}>
                                  {prompt}
                                </button>
                              ))}
                            </div>
                          </div>
                        ))}
                      </motion.div>
                    ) : null}
                  </AnimatePresence>
                  <AnimatePresence>
                    {answerMode === 'memo' ? (
                      <motion.div
                        className="memo-mode-panel"
                        initial={{ opacity: 0, y: -8, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: 'auto' }}
                        exit={{ opacity: 0, y: -8, height: 0 }}
                        transition={{ duration: 0.2, ease: 'easeOut' }}
                      >
                        <div className="memo-department-row">
                          {MEMO_DEPARTMENTS.map((department) => (
                            <button
                              key={department.id}
                              type="button"
                              className={`memo-department-chip ${memoDepartment === department.id ? 'active' : ''}`}
                              onClick={() => setMemoDepartment(department.id)}
                            >
                              <span aria-hidden="true">{department.icon}</span>
                              <strong>{department.label}</strong>
                            </button>
                          ))}
                          <button type="button" className="memo-concept-button" onClick={() => setShowConceptModal(true)}>
                            컨셉 보기
                          </button>
                        </div>
                        <textarea
                          className="memo-mode-textarea"
                          value={memoDepartmentNotes[memoDepartment] || ''}
                          onChange={(event) => {
                            const nextValue = event.target.value;
                            setMemoNotes(nextValue);
                            setMemoDepartmentNotes((previous) => ({ ...previous, [memoDepartment]: nextValue }));
                          }}
                          placeholder={`${MEMO_DEPARTMENTS.find((item) => item.id === memoDepartment)?.label || '부서'}가 중요하게 보는 기준을 적어주세요.`}
                          rows={2}
                        />
                        <p>저장된 메모는 부서 컨셉에 붙어서 상품개발 안건과 4인 토론에 반영됩니다.</p>
                      </motion.div>
                    ) : null}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </section>

          <section className="panel ontology-chat-shell copilot-thread-shell" style={{ maxWidth: '100%', minWidth: 0 }}>
            <div className="ontology-conversation-stack ontology-conversation-thread">
              {answerMode === 'product' ? (
                <ProductDevelopmentWorkspace
                  state={productDevelopmentState}
                  concepts={departmentConceptList}
                  semanticRefresh={semanticRefresh}
                  onRefreshAgendas={() => handleLoadProductDevelopmentAgendas(true)}
                  onSelectAgenda={handleSelectProductDevelopmentAgenda}
                  onOpenConcepts={() => setShowConceptModal(true)}
                />
              ) : null}
              {!displayedConversationTurns.length && answerMode !== 'product' ? (
                <article className="ontology-empty-thread">
                  <GeneralAnswerModeIntro />
                  <span className="panel-kicker">일반답변모드</span>
                  <strong>궁금한 점을 편하게 물어보세요.</strong>
                  <p>버니가 질문에 맞는 심사 로그, 고객군 정보, 규제 근거를 필요한 만큼만 찾아보고 쉬운 말로 정리해드려요. 복잡한 분석 화면을 먼저 볼 필요 없이, 질문을 쓰고 보내기만 누르면 됩니다.</p>
                </article>
              ) : null}
              {orderedConversationTurns.map((turn) => {
                const isLatest = turn.id === latestConversationTurnId;
                const isRuntimePending = backendState.loading || runtimeJobStatus === 'queued' || runtimeJobStatus === 'running';
                const showAnswerProgress = isLatest && isRuntimePending && !hasResolvedRuntimeAnswer;
                const hasAnswer = Boolean(turn.answerBody || turn.answerHeadline);
                const answerStreamToken = `${turn.id}:${turn.answerHeadline || ''}:${turn.answerBody || ''}`;
                const answerHighlightTerms = isLatest
                  ? (resolvedAnswerSummary?.highlights || []).flatMap((item) => [item.label, item.value]).filter(Boolean)
                  : [turn.product, turn.axis].filter(Boolean);
                if (!isLatest) {
                  return (
                    <details key={turn.id} className="ontology-history-accordion">
                      <summary>
                        <div className="ontology-history-summary-main">
                          <span className="panel-kicker">이전 대화</span>
                          <strong>{turn.question}</strong>
                        </div>
                        <div className="ontology-history-summary-meta">
                          <span className="sample-pill">{turn.timestamp}</span>
                          {shouldShowProductChip(turn.product, turn.sourceTag) ? <span className="sample-pill">상품 {getProductDisplayName(turn.product)}</span> : null}
                        </div>
                      </summary>
                      <div className="ontology-history-accordion-body">
                        <article className="ontology-chat-bubble ontology-chat-bubble-user ontology-chat-bubble-archived">
                          <div className="ontology-chat-meta-row">
                            <span className="ontology-chat-role">User</span>
                            <span className="sample-pill">{turn.timestamp}</span>
                          </div>
                          <strong>{turn.question}</strong>
                        </article>
                        <article className="ontology-chat-bubble ontology-chat-bubble-assistant ontology-chat-bubble-archived">
                          <div className="ontology-runtime-answer-head">
                            <span className="panel-kicker">Answer</span>
                            <div className="detail-chip-row">
                              <span className="sample-pill">{turn.sourceTag}</span>
                              {shouldShowProductChip(turn.product, turn.sourceTag) ? <span className="sample-pill">상품 {getProductDisplayName(turn.product)}</span> : null}
                            </div>
                          </div>
                          <div className="ontology-answer-stream-shell">
                            <strong className="ontology-answer-stream-headline">
                              {turn.answerHeadline || '답변 기록'}
                            </strong>
                            <p className="ontology-answer-stream-body">
                              {turn.answerBody || '이전 답변을 불러오는 중입니다.'}
                            </p>
                          </div>
                          <div className="ontology-runtime-answer-grid ontology-runtime-answer-grid-compact">
                            <div className="ontology-runtime-answer-chip"><span>핵심 기준</span><strong>{turn.axis || '-'}</strong></div>
                          </div>
                        </article>
                      </div>
                    </details>
                  );
                }
                return (
                  <React.Fragment key={turn.id}>
                    <article className="ontology-chat-bubble ontology-chat-bubble-user">
                      <div className="ontology-chat-meta-row">
                        <span className="ontology-chat-role">User</span>
                        <span className="sample-pill">{turn.timestamp}</span>
                      </div>
                      <strong>{turn.question}</strong>
                    </article>

                    {isLatest && workspaceToolTabs.length ? (
                      <section className="workspace-result-panel" aria-label="질문 결과 분석 탭">
                        <div className="workspace-tool-tabs" role="tablist" aria-label="대화형 도구 뷰">
                          {workspaceToolTabs.map((tab) => (
                            <button
                              key={tab.id}
                              type="button"
                              role="tab"
                              aria-selected={selectedWorkspaceResultTab === tab.id}
                              className={selectedWorkspaceResultTab === tab.id ? 'active' : ''}
                              onClick={() => setActiveWorkspaceResultTab(tab.id)}
                            >
                              {tab.label}
                            </button>
                          ))}
                        </div>

                        <div className="workspace-result-tab-body">
                          {selectedWorkspaceResultTab === 'summary' ? (
                            <div className="workspace-summary-view">
                              <div className="workspace-summary-main">
                                <article className="workspace-summary-callout">
                                  <span aria-hidden="true">🐰</span>
                                  <div>
                                    <strong>
                                      <HighlightedAnswerText text={resolvedAnswerSummary?.headline || '분석 요약을 준비했습니다.'} terms={answerHighlightTerms} termMeta={answerTermMeta} />
                                    </strong>
                                    <p>
                                      <HighlightedAnswerText text={finalAnswerBody || '핵심 결과를 실제 로그와 고객군집 기준으로 정리했습니다.'} terms={answerHighlightTerms} termMeta={answerTermMeta} />
                                    </p>
                                  </div>
                                </article>
                                {summaryHighlightItems.length ? (
                                  <div className="workspace-summary-metric-grid">
                                    {summaryHighlightItems.map((item) => (
                                      <div key={`summary-${item.label}-${item.value}`} className="workspace-summary-metric">
                                        <span>{item.label}</span>
                                        <strong><HighlightedAnswerText text={String(item.value || '-')} terms={answerHighlightTerms} termMeta={answerTermMeta} /></strong>
                                      </div>
                                    ))}
                                  </div>
                                ) : null}
                                {/* K코드별 통계 표 노출 (summary 영역) */}
                                {resolvedAnswerSummary?.top_reject_codes?.length ? (
                                  <div className="summary-reject-code-table" style={{ margin: '16px 0', overflowX: 'auto' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                                      <thead>
                                        <tr style={{ background: '#f7f7fa' }}>
                                          <th style={{ padding: '4px 8px', border: '1px solid #eee' }}>K코드</th>
                                          <th style={{ padding: '4px 8px', border: '1px solid #eee' }}>설명</th>
                                          <th style={{ padding: '4px 8px', border: '1px solid #eee' }}>건수</th>
                                          <th style={{ padding: '4px 8px', border: '1px solid #eee' }}>비중</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {resolvedAnswerSummary.top_reject_codes.slice(0, 3).map((code) => (
                                          <tr key={code.code}>
                                            <td style={{ padding: '4px 8px', border: '1px solid #eee', fontWeight: 600 }}>{code.code}</td>
                                            <td style={{ padding: '4px 8px', border: '1px solid #eee' }}>{code.description || '-'}</td>
                                            <td style={{ padding: '4px 8px', border: '1px solid #eee', textAlign: 'right' }}>{Number(code.count || 0).toLocaleString('ko-KR')}</td>
                                            <td style={{ padding: '4px 8px', border: '1px solid #eee', textAlign: 'right' }}>{(Number(code.share || 0) * 100).toFixed(1)}%</td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                ) : null}
                              </div>
                              <div className="workspace-summary-tools">
                                {/* Customer Cluster Intelligence Card (군집 카드) */}
                                {clusterToolCard ? <FinancialToolCard tool={clusterToolCard} /> : null}
                                {insightToolCard ? <FinancialToolCard tool={insightToolCard} /> : null}
                                {/* 군집/로그/지표가 없을 때 Explainability Agent 결과로 대체 */}
                                {!clusterToolCard && !clusterInsightItems.length && !resolvedAnswerSummary?.top_reject_codes?.length ? (
                                  !insightToolCard
                                    ? <div className="empty-box compact">조건에 맞는 고객군, 로그, 지표가 없습니다.<br />상품, 연령, 기간 등 필터를 넓혀보세요.</div>
                                    : null
                                ) : null}
                                {!clusterToolCard && clusterInsightItems.length ? (
                                  <div className="tool-mini-grid">
                                    {clusterInsightItems.map((cluster) => (
                                      <div key={cluster.cluster_id || cluster.label} className="tool-mini-card">
                                        <span>{cluster.decision || '군집'} · {cluster.count || cluster.records || 0}건</span>
                                        <strong>{cluster.label || cluster.display_label || cluster.cluster_id}</strong>
                                        <small>금리 {cluster.avg_rate || '-'} · 한도 {cluster.avg_limit || '-'}</small>
                                      </div>
                                    ))}
                                  </div>
                                ) : null}
                              </div>
                            </div>
                          ) : null}

                          {selectedWorkspaceResultTab === 'cluster' ? (
                            <div className="workspace-cluster-view">
                              {clusterToolCard ? <FinancialToolCard tool={clusterToolCard} /> : null}
                              {!clusterToolCard && clusterInsightItems.length ? (
                                <div className="tool-mini-grid">
                                  {clusterInsightItems.map((cluster) => (
                                    <div key={cluster.cluster_id || cluster.label} className="tool-mini-card">
                                      <span>{cluster.decision || '군집'} · {cluster.count || cluster.records || 0}건</span>
                                      <strong>{cluster.label || cluster.display_label || cluster.cluster_id}</strong>
                                      <small>금리 {cluster.avg_rate || '-'} · 한도 {cluster.avg_limit || '-'}</small>
                                    </div>
                                  ))}
                                </div>
                              ) : null}
                              {!clusterToolCard && !clusterInsightItems.length ? <div className="empty-box compact">이 질문에 연결된 군집 결과가 아직 없습니다.</div> : null}
                            </div>
                          ) : null}

                          {selectedWorkspaceResultTab === 'insight' ? (
                            <div className="workspace-insight-view">
                              {insightToolCard ? <FinancialToolCard tool={insightToolCard} /> : null}
                              {metricInsightItems.length ? (
                                <div className="workspace-summary-metric-grid">
                                  {metricInsightItems.map((item) => (
                                    <div key={`insight-${item.label}-${item.value}`} className={`workspace-summary-metric is-${item.tone || 'neutral'}`}>
                                      <span>{item.label}</span>
                                      <strong>{item.value}</strong>
                                    </div>
                                  ))}
                                </div>
                              ) : null}
                              <div className="workspace-insight-grid">
                                <article>
                                  <span>핵심 기준</span>
                                  <strong>{topFeatureName}</strong>
                                  <p>질문과 직접 연결된 feature, 실제 로그, 고객군집 결과를 우선해서 봅니다.</p>
                                </article>
                                <article>
                                  <span>상위 고객군</span>
                                  <strong>{resultTopCluster?.label || '군집 계산 결과'}</strong>
                                  <p>{resultTopCluster ? `${resultTopCluster.decision || '-'} / ${resultTopCluster.age_band || '-'} / ${resultTopCluster.income_band || '-'}` : '답변과 연결된 고객군을 찾는 중입니다.'}</p>
                                </article>
                                <article>
                                  <span>근거 건수</span>
                                  <strong>{retrievalResults.length || resultSummary.record_count || '-'}건</strong>
                                  <p>retrieval evidence와 고객군집 캐시에서 확인한 근거입니다.</p>
                                </article>
                              </div>
                            </div>
                          ) : null}

                          {selectedWorkspaceResultTab === 'policy' ? (
                            <div className="workspace-policy-view">
                              {policyToolCard ? <FinancialToolCard tool={policyToolCard} /> : <div className="empty-box compact">이 질문에 직접 연결된 정책/규제 근거가 있을 때 표시됩니다.</div>}
                            </div>
                          ) : null}

                          {selectedWorkspaceResultTab === 'strategy' ? (
                            <div className="workspace-strategy-view">
                              {strategyToolCard ? <FinancialToolCard tool={strategyToolCard} /> : <div className="empty-box compact">상품개발모드에서 전략 시뮬레이션 결과가 표시됩니다.</div>}
                            </div>
                          ) : null}
                        </div>
                      </section>
                    ) : null}

                    <article className="ontology-chat-bubble ontology-chat-bubble-assistant">
                      <div className="ontology-runtime-answer-head">
                          <span className="panel-kicker">Answer</span>
                          <div className="detail-chip-row">
                            <span className="sample-pill">{turn.sourceTag}</span>
                            {shouldShowProductChip(turn.product, turn.sourceTag) ? <span className="sample-pill">상품 {getProductDisplayName(turn.product)}</span> : null}
                            {!showAnswerProgress ? <button type="button" className="secondary-button ontology-inline-detail-button" onClick={() => handleOpenDetailModal('answer')}>자세히</button> : null}
                          </div>
                      </div>
                      {showAnswerProgress ? (
                        <div className="ontology-answer-progress-shell">
                          <RuntimeProgressPanel
                            complete={isRuntimeComplete}
                            progress={overallRuntimeProgress}
                            headline={progressHeadline}
                            subLabel={progressSubLabel}
                            liveText={progressLiveText}
                            activeStage={activeStage}
                            runtimeStages={runtimeStages}
                            slowestCompletedStage={slowestCompletedStage}
                          />
                        </div>
                      ) : (
                        <>
                          <div className="ontology-answer-stream-shell">
                            <strong className="ontology-answer-stream-headline">
                              {turn.answerHeadline
                                ? <HighlightedAnswerText key={`headline-${answerStreamToken}`} text={turn.answerHeadline} terms={answerHighlightTerms} termMeta={answerTermMeta} />
                                : '답변 준비 중'}
                            </strong>
                            <p className="ontology-answer-stream-body">
                              {turn.answerBody
                                ? <HighlightedAnswerText key={`body-${answerStreamToken}`} text={turn.answerBody} terms={answerHighlightTerms} termMeta={answerTermMeta} className="ontology-answer-stream-copy" />
                                : '로니가 현재 질문을 해석하고 답변을 준비하고 있습니다.'}
                              {isLatest && hasAnswer ? <span className="ontology-chat-cursor" aria-hidden="true" /> : null}
                            </p>
                          </div>
                          <div className="ontology-runtime-answer-grid ontology-runtime-answer-grid-compact">
                              {isLatest
                                ? (resolvedAnswerSummary?.highlights || []).slice(0, 4).map((item) => <div key={`${turn.id}-${item.label}-${item.value}`} className="ontology-runtime-answer-chip"><span>{item.label}</span><strong><HighlightedAnswerText text={String(item.value || '')} terms={answerHighlightTerms} termMeta={answerTermMeta} /></strong></div>)
                                : <div className="ontology-runtime-answer-chip"><span>핵심 축</span><strong>{turn.axis || '-'}</strong></div>}
                          </div>
                          {isLatest && !workspaceToolTabs.length && (answerMode === 'product' ? productModeToolCards.length : activeToolCards.length) ? (
                            <div className={`financial-tool-card-stack ${answerMode === 'product' ? 'is-product-mode' : ''}`}>
                              {(answerMode === 'product' ? productModeToolCards : activeToolCards).map((tool) => <FinancialToolCard key={tool.id || tool.title} tool={tool} />)}
                            </div>
                          ) : null}
                        </>
                      )}

                      {isLatest && !backendState.loading && !isDocumentOrGeneralAnswer && representativeFeatures.length > 0 && (
                        <div className="ontology-avoided-feature-row">
                          <span className="ontology-avoided-feature-label">관련 없는 축 제외</span>
                          <div className="ontology-avoided-feature-chips">
                            {representativeFeatures.map((feature) => {
                              const featureId = feature.feature_id || feature.feature_name;
                              const isAvoided = avoidedFeatureIds.includes(featureId);
                              return (
                                <button
                                  key={featureId}
                                  type="button"
                                  className={`ontology-avoided-chip ${isAvoided ? 'is-avoided' : ''}`}
                                  onClick={() => handleToggleAvoidedFeature(featureId)}
                                  title={isAvoided ? '제외 취소' : '다음 질문에서 이 축을 제외'}
                                >
                                  {feature.feature_name || featureId}
                                  <span className="ontology-avoided-chip-icon">{isAvoided ? '-' : '+'}</span>
                                </button>
                              );
                            })}
                            {avoidedFeatureIds.length > 0 && (
                              <button type="button" className="ontology-avoided-reset-btn" onClick={() => setAvoidedFeatureIds([])}>초기화</button>
                            )}
                          </div>
                        </div>
                      )}

                      {isLatest && !backendState.loading && !isDocumentOrGeneralAnswer && !clarificationAnswered && backendState.workbench?.clarification?.needed && (
                        <div className="ontology-clarification-block">
                          <div className="ontology-clarification-question">
                            <span className="ontology-clarification-icon" aria-hidden="true">?</span>
                            <strong>{backendState.workbench.clarification.question || '어떤 축을 우선해서 볼까요?'}</strong>
                          </div>
                          <div className="ontology-clarification-options">
                            {(backendState.workbench.clarification.options || []).map((option) => (
                              <button
                                key={option.feature_id}
                                type="button"
                                className="ontology-clarification-option-btn"
                                onClick={() => handleClarificationSelect(option)}
                              >
                                <span className="ontology-clarification-option-name">{option.feature_name || option.feature_id}</span>
                                {option.axis_key ? <span className="ontology-clarification-option-axis">{option.axis_key}</span> : null}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {isLatest
                        && !backendState.loading
                        && Number(backendState.workbench?.regulation_evidence_meta?.shown_count ?? ((backendState.workbench?.semantic_context?.regulation_citations || []).length)) > 0
                        && (resolvedAnswerSummary?.citations || []).length > 0 && (
                        <div className="ontology-citation-block">
                          <span className="ontology-citation-label">Regulation Citations</span>
                          <div className="ontology-citation-list">
                            {(resolvedAnswerSummary?.citations || []).slice(0, 3).map((item, index) => {
                              const highlightedSentences = buildCitationHighlights(item, currentQuestion);
                              return (
                                <article key={`${turn.id}-citation-${index}`} className="ontology-citation-item">
                                  <strong>{item.name || 'regulation'}</strong>
                                  <span className="sample-pill">chunk #{item.chunk_index || 0}</span>
                                  {highlightedSentences.length ? (
                                    <div className="ontology-citation-highlight-box">
                                      <span>참조 문장</span>
                                      {highlightedSentences.map((sentence, sentenceIndex) => (
                                        <mark key={`${turn.id}-citation-${index}-sentence-${sentenceIndex}`}>{sentence}</mark>
                                      ))}
                                    </div>
                                  ) : null}
                                  <small>{item.snippet || '인용 스니펫 없음'}</small>
                                </article>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </article>
                  </React.Fragment>
                );
              })}
            </div>
            {answerMode === 'product' && hasResolvedRuntimeAnswer ? (
              <StrategyWorkspace panels={strategyPanels} workflow={agentWorkflow} semanticLayer={semanticFinancialLayer} />
            ) : null}
            {promptDock}

          </section>
        </div>
      </div>

      <AnimatePresence>
        {showDetailModal ? (
          <motion.div className="prompt-modal-backdrop ontology-detail-modal-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setShowDetailModal(false)}>
            <motion.section className="prompt-modal ontology-detail-modal" style={{ maxWidth: '900px', width: '100%' }} initial={{ opacity: 0, y: 20, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 14, scale: 0.98 }} transition={{ duration: 0.22, ease: 'easeOut' }} onClick={(event) => event.stopPropagation()}>
              <div className="ontology-detail-modal-head">
                <div>
                  <div className="panel-kicker">Detail View</div>
                  <h2>질문 해석과 답변 생성 과정</h2>
                  <p>Pipeline State 카드를 근거로, 필요한 과정만 전체 화면 팝업에서 봅니다.</p>
                </div>
                <button type="button" className="secondary-button" onClick={() => setShowDetailModal(false)}>닫기</button>
              </div>
              <div className="ontology-detail-modal-body">
                <div className="ontology-subtab-row ontology-subtab-row-modal">
                  {ONTOLOGY_SUBTABS.map((item) => (
                    <button key={item.id} type="button" className={`ontology-subtab ${activeSubtab === item.id ? 'active' : ''}`} onClick={() => setActiveSubtab(item.id)}>
                      <span>{item.kicker}</span>
                      <strong>{item.label}</strong>
                    </button>
                  ))}
                </div>

                {activeSubtab === 'question' ? (
                  <div className="ontology-detail-card-grid ontology-detail-card-grid-two">
                    <article className="ontology-detail-card">
                      <span>입력 질문</span>
                      <strong>{currentQuestion}</strong>
                      <p>질문 토큰과 상품 범위를 먼저 읽고, 어떤 의미 축으로 해석할지 준비합니다.</p>
                    </article>
                    <article className="ontology-detail-card">
                      <span>질문 토큰</span>
                      <div className="detail-chip-row">{questionTokens.map((item) => <span key={item} className="reason-chip">{item}</span>)}</div>
                    </article>
                    {querySignalCards.map((item) => <article key={item.id} className="ontology-detail-card"><span>{item.token}</span><strong>{item.label}</strong><p>{item.reason}</p></article>)}
                    <article className="ontology-detail-card ontology-ollama-box">
                      <span>진행 단계</span>
                      <div className="ontology-stage-timeline">{runtimeStages.map((item) => <div key={item.key} className="ontology-stage-tick"><span className={`ontology-stage-tick-dot ${toneClass(item.status)}`} /><strong>{item.label}</strong><small>{RUNTIME_STAGES.find((stage) => stage.key === item.key)?.detail}</small></div>)}</div>
                    </article>
                  </div>
                ) : null}

                {activeSubtab === 'feature' ? (
                  <div className="ontology-detail-card-grid ontology-detail-card-grid-two">
                    <article className="ontology-detail-card ontology-priority-card"><span>핵심 축</span><strong>{topFeatureName}</strong><p>{primaryFeatureSelection?.headline || '질문 의도와 그래프 연결을 함께 보고 핵심 축을 정했습니다.'}</p></article>
                    {representativeAxisDetails.map((axis) => <article key={axis.feature_id} className="ontology-detail-card ontology-rank-breakdown-card"><span>{axis.axis_key || 'axis'}</span><strong>{axis.feature_name || axis.feature_id}</strong><p>{axis.description || '핵심 축 설명이 아직 없습니다.'}</p><div className="detail-chip-row">{(axis.matched_tokens || []).map((item) => <span key={`${axis.feature_id}-${item}`} className="reason-chip">{item}</span>)}</div></article>)}
                    <article className="ontology-detail-card ontology-ollama-box"><span>관련 feature</span><div className="detail-chip-row">{relatedFeatures.slice(0, 6).map((item) => <span key={item.feature_id || item.feature_name} className="sample-pill">{item.feature_name || item.feature_id}</span>)}</div></article>
                  </div>
                ) : null}

                {activeSubtab === 'cluster' ? (
                  <div className="ontology-detail-card-grid ontology-detail-card-grid-two">
                    <article className="ontology-detail-card"><span>상위 군집</span><strong>{resultTopCluster?.label || '-'}</strong><p>{resultTopCluster ? `${resultTopCluster.decision} / ${resultTopCluster.age_band} / ${resultTopCluster.income_band}` : 'cluster 계산 중'}</p></article>
                    <article className="ontology-detail-card"><span>retrieval 결과</span><strong>{retrievalResults.length}건</strong><p>핵심 축과 가까운 기록을 모아 최종 답변 근거로 사용합니다.</p></article>
                    {representativeRetrievalGroups.map((axis) => <article key={axis.feature_id} className="ontology-detail-card"><span>{axis.axis_key || 'axis'}</span><strong>{axis.feature_name || axis.feature_id}</strong><div className="detail-chip-row">{(axis.top_records || []).slice(0, 3).map((item) => <span key={`${axis.feature_id}-${item.record_id || item.product}`} className="reason-chip">{item.record_id || item.product || '-'}</span>)}</div></article>)}
                  </div>
                ) : null}

                {activeSubtab === 'answer' ? (
                  <div className="ontology-answer-detail-layout">
                    <div className="ontology-detail-card-grid ontology-detail-card-grid-two">
                      {answerNarrativeSteps.map((item) => <article key={item.id} className="ontology-detail-card"><span>{item.title}</span><strong>{item.detail}</strong><p>최종 답변은 이전 단계 결과를 순서대로 결합해 만듭니다.</p></article>)}
                      <article className="ontology-detail-card ontology-ollama-trace-card"><span>최종 답변</span><strong>{resolvedAnswerSummary?.headline}</strong><p>{finalAnswerBody || '아직 답변 없음'}</p></article>
                    </div>
                    <aside className="ontology-ollama-debug-panel">
                      <div className="ontology-ollama-debug-head">
                        <div>
                          <span className="panel-kicker">Ollama Trace</span>
                          <strong>입력 / 출력 전체 텍스트</strong>
                        </div>
                        <span className="sample-pill">{ollamaStatus}</span>
                      </div>
                      <div className="ontology-ollama-debug-tabs" role="tablist" aria-label="Ollama trace tabs">
                        <button type="button" className={answerTraceTab === 'input' ? 'active' : ''} onClick={() => setAnswerTraceTab('input')}>Input</button>
                        <button type="button" className={answerTraceTab === 'output' ? 'active' : ''} onClick={() => setAnswerTraceTab('output')}>Output</button>
                      </div>
                      <pre className="ontology-ollama-debug-pre">{answerTraceTab === 'input' ? ollamaFullInput : ollamaFullOutput}</pre>
                    </aside>
                  </div>
                ) : null}
              </div>
            </motion.section>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <AnimatePresence>
        {showConceptModal ? (
          <DepartmentConceptModal
            open={showConceptModal}
            concepts={departmentConceptList}
            onClose={() => setShowConceptModal(false)}
          />
        ) : null}
      </AnimatePresence>

      <AnimatePresence>
        {showMetricCubeModal ? (
          <MetricCubeModal
            open={showMetricCubeModal}
            data={metricCubeState.data}
            loading={metricCubeState.loading}
            error={metricCubeState.error}
            activeTab={metricCubeTab}
            onTabChange={setMetricCubeTab}
            onClose={() => setShowMetricCubeModal(false)}
            onRefresh={() => handleOpenMetricCubeModal(metricCubeTab, true)}
          />
        ) : null}
      </AnimatePresence>
    </motion.section>
  );
}




