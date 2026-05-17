import { create } from 'zustand';

export const RUNTIME_STAGES = [
  { key: 'extraction', label: 'Load Runtime Data', heroLabel: 'JSON Load', tone: 'cyan', detail: 'commonfeature.json 과 full_text_records.json 을 읽습니다.' },
  { key: 'alias', label: 'Product Scope Filter', heroLabel: 'Product Filter', tone: 'amber', detail: '선택 상품 기준으로 feature 후보를 좁힙니다.' },
  { key: 'mapping', label: 'Semantic Feature Rank', heroLabel: 'Semantic Rank', tone: 'blue', detail: '질문과 가장 가까운 feature 를 점수화합니다.' },
  { key: 'ontology', label: 'Primary Axis Select', heroLabel: 'Axis Select', tone: 'green', detail: '대표 축과 related feature 를 확정합니다.' },
  { key: 'faiss', label: 'Cluster Cache Build', heroLabel: 'Cluster Build', tone: 'violet', detail: '고객군집 캐시와 cluster 후보를 계산합니다.' },
  { key: 'retrieval', label: 'Retrieval Result Build', heroLabel: 'Retrieval Build', tone: 'cyan', detail: '관련 레코드와 retrieval trace 후보를 만듭니다.' },
  { key: 'ollama', label: 'Answer Summary Build', heroLabel: 'Answer Summary', tone: 'amber', detail: '화면 상단에 보여줄 요약 답변을 만듭니다.' },
];

function stamp(offsetSeconds = 0) {
  const date = new Date(Date.now() + offsetSeconds * 1000);
  return date.toLocaleTimeString('ko-KR', { hour12: false });
}

function formatClock(value) {
  if (!value) {
    return stamp();
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return stamp();
  }
  return date.toLocaleTimeString('ko-KR', { hour12: false });
}

function createStageStatuses(currentIndex) {
  return RUNTIME_STAGES.map((stage, index) => {
    if (index < currentIndex) {
      return { ...stage, status: 'completed', progress: 100, started_at: null, completed_at: null, meta: {} };
    }
    if (index === currentIndex) {
      return { ...stage, status: 'running', progress: 66, started_at: null, completed_at: null, meta: {} };
    }
    return { ...stage, status: 'idle', progress: 0, started_at: null, completed_at: null, meta: {} };
  });
}

function deriveCurrentStageIndex(stages) {
  const runningIndex = stages.findIndex((item) => item.status === 'running');
  if (runningIndex >= 0) {
    return runningIndex;
  }
  const completedIndexes = stages.map((item, index) => (item.status === 'completed' ? index : -1)).filter((index) => index >= 0);
  return completedIndexes.length ? completedIndexes[completedIndexes.length - 1] : 0;
}

function createInitialAnswerSummary(question) {
  return {
    headline: '실제 서버 질의를 시작하면 answer summary 가 여기에 표시됩니다.',
    explanation: `현재 질문: ${question}`,
    highlights: [
      { label: 'Runtime', value: 'waiting' },
      { label: 'Product', value: 'C12' },
      { label: 'Primary Axis', value: '-' },
      { label: 'Top Axes', value: '-' },
      { label: 'Income Band', value: '-' },
    ],
  };
}

function initialState() {
  const defaultQuestion = '';
  const currentStageIndex = 0;
  return {
    currentStageIndex,
    queryRunNonce: 0,
    currentQuestion: defaultQuestion,
    selectedNodeId: 'canonical_applicant_age',
    selectedNav: 'ontology',
    lastRunAt: stamp(),
    runtimeJobId: '',
    runtimeJobStatus: 'idle',
    runtimeElapsedMs: 0,
    searchHistory: [
      { id: 'q-1', title: defaultQuestion, time: '방금 전' },
      { id: 'q-2', title: '연체 이력이 카드론 승인에 미치는 영향', time: '12분 전' },
      { id: 'q-3', title: 'C12 상품에서 DSR 경고 임계치', time: '28분 전' },
    ],
    featureCategories: [
      { id: 'fc-age', label: 'Applicant Identity', count: 32 },
      { id: 'fc-income', label: 'Income Signal', count: 48 },
      { id: 'fc-risk', label: 'Risk Signal', count: 19 },
      { id: 'fc-product', label: 'Product Constraints', count: 14 },
    ],
    products: [
      { code: 'C6', label: 'Revolving Lite', count: 88 },
      { code: 'C9', label: 'Prime Cash', count: 121 },
      { code: 'C11', label: 'Office Bridge', count: 96 },
      { code: 'C12', label: 'Card Loan Core', count: 164 },
    ],
    clusters: [
      { id: 'cluster_40_office_worker', label: '40s Office Worker', score: 0.94 },
      { id: 'cluster_prime_income', label: 'Prime Income Growth', score: 0.87 },
      { id: 'cluster_recovering_credit', label: 'Recovering Credit', score: 0.72 },
    ],
    alerts: [
      { id: 'alert-1', level: 'info', title: 'Server Runtime', detail: '실행 전에는 대기 상태입니다.' },
      { id: 'alert-2', level: 'info', title: 'Ontology merge ready', detail: '새 alias 후보 3건이 approval queue 에 있습니다.' },
    ],
    ontologyChanges: [
      { id: 'chg-1', label: 'applicant.age relation expanded', detail: 'cluster_40_office_worker 연결 추가' },
      { id: 'chg-2', label: 'income.monthly sample range updated', detail: 'C11, C12 최신 로그 반영' },
    ],
    semanticStats: {
      nodeCount: 1284,
      relationCount: 3912,
      liveAliasCount: 23,
      retrievalHitRate: 94,
      activeQuestion: defaultQuestion,
    },
    runtimeStages: createStageStatuses(currentStageIndex),
    activityFeed: [
      { id: 'feed-1', stage: 'extraction', title: 'Runtime Ready', detail: '질문을 실행하면 실제 서버 단계가 여기 누적됩니다.', time: stamp(-8) },
      { id: 'feed-2', stage: 'retrieval', title: 'Backend Linked', detail: '진행률은 runtime job snapshot 에서 읽습니다.', time: stamp(-5) },
      { id: 'feed-3', stage: 'ollama', title: 'Answer Summary', detail: '최종 결과는 서버가 만든 summary 카드로 표시됩니다.', time: stamp(-2) },
    ],
    liveLogs: [
      { id: 'log-1', tone: 'info', time: stamp(-9), text: '[runtime] 서버 질의 대기 중' },
      { id: 'log-2', tone: 'info', time: stamp(-6), text: '[backend] feature runtime job snapshot 연결 완료' },
      { id: 'log-3', tone: 'info', time: stamp(-2), text: '[answer] 최종 summary 는 서버 결과를 그대로 반영합니다.' },
    ],
    retrievalTrace: [
      { id: 'trace-1', label: 'Ontology Retrieval', value: 'applicant.age', score: 0.97 },
      { id: 'trace-2', label: 'Cluster Retrieval', value: 'cluster_40_office_worker', score: 0.94 },
      { id: 'trace-3', label: 'Related Feature', value: 'income.monthly', score: 0.91 },
      { id: 'trace-4', label: 'Related Feature', value: 'dsr.level', score: 0.88 },
      { id: 'trace-5', label: 'Related Feature', value: 'overdue.history', score: 0.84 },
      { id: 'trace-6', label: 'Need LLM', value: 'YES', score: 1 },
    ],
    answerSummary: createInitialAnswerSummary(defaultQuestion),
  };
}

export const useOntologyRuntimeStore = create((set) => ({
  ...initialState(),
  submitQuestion: (question) => {
    const nextQuestion = String(question || '').trim();
    if (!nextQuestion) {
      return;
    }
    set((state) => ({
      queryRunNonce: state.queryRunNonce + 1,
      currentQuestion: nextQuestion,
      selectedNav: 'ontology',
      currentStageIndex: 0,
      lastRunAt: stamp(),
      runtimeJobId: '',
      runtimeJobStatus: 'queued',
      runtimeElapsedMs: 0,
      runtimeStages: createStageStatuses(0),
      semanticStats: {
        ...state.semanticStats,
        activeQuestion: nextQuestion,
        retrievalHitRate: 0,
      },
      clusters: state.clusters.map((item, index) => ({
        ...item,
        id: `pending-cluster-${index + 1}`,
        label: index === 0 ? '새 군집 계산 중' : '대기 중',
        score: 0,
      })).slice(0, 3),
      retrievalTrace: [
        { id: 'trace-pending-1', label: 'Ontology Retrieval', value: 'pending', score: 0 },
        { id: 'trace-pending-2', label: 'Cluster Retrieval', value: 'pending', score: 0 },
        { id: 'trace-pending-3', label: 'Representative Axes', value: 'pending', score: 0 },
      ],
      searchHistory: [
        { id: `q-${Date.now()}`, title: nextQuestion, time: '방금 전' },
        ...state.searchHistory,
      ].slice(0, 6),
      activityFeed: [
        { id: `feed-${Date.now()}`, stage: 'extraction', title: 'User Query', detail: `질문 수신: ${nextQuestion}`, time: stamp() },
        ...state.activityFeed,
      ].slice(0, 7),
      liveLogs: [
        { id: `log-${Date.now()}`, tone: 'info', time: stamp(), text: `Semantic query received: ${nextQuestion}` },
        ...state.liveLogs,
      ].slice(0, 14),
      answerSummary: createInitialAnswerSummary(nextQuestion),
    }));
  },
  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),
  selectNav: (selectedNav) => set({ selectedNav }),
  hydrateRuntime: (payload) => set((state) => ({
    semanticStats: {
      ...state.semanticStats,
      nodeCount: payload?.nodeCount || state.semanticStats.nodeCount,
      relationCount: payload?.relationCount || state.semanticStats.relationCount,
      liveAliasCount: payload?.liveAliasCount || state.semanticStats.liveAliasCount,
    },
    ontologyChanges: payload?.ontologyChanges?.length ? payload.ontologyChanges : state.ontologyChanges,
    products: payload?.products?.length ? payload.products : state.products,
  })),
  applyRuntimeSnapshot: (snapshot) => set((state) => {
    const runtimeStages = (snapshot?.stages?.length ? snapshot.stages : state.runtimeStages).map((stage, index) => ({
      ...RUNTIME_STAGES[index],
      ...stage,
    }));
    const currentStageIndex = deriveCurrentStageIndex(runtimeStages);
    const liveLogs = (snapshot?.logs?.length ? snapshot.logs : []).map((item) => ({
      id: item.id || `log-${Math.random()}`,
      tone: item.tone === 'error' ? 'warning' : item.tone === 'success' ? 'success' : 'info',
      time: formatClock(item.time),
      text: item.text || '',
    }));
    const activityFeed = (snapshot?.logs?.length ? snapshot.logs : []).slice(0, 7).map((item) => ({
      id: item.id || `feed-${Math.random()}`,
      stage: item.stage || 'runtime',
      title: runtimeStages.find((stage) => stage.key === item.stage)?.label || 'Runtime Event',
      detail: item.text || '',
      time: formatClock(item.time),
    }));
    const activeStage = runtimeStages[currentStageIndex] || runtimeStages[0] || {};
    return {
      runtimeJobId: snapshot?.job_id || state.runtimeJobId,
      runtimeJobStatus: snapshot?.status || state.runtimeJobStatus,
      runtimeElapsedMs: Number(snapshot?.elapsed_ms || state.runtimeElapsedMs || 0),
      runtimeStages,
      currentStageIndex,
      lastRunAt: formatClock(snapshot?.updated_at),
      liveLogs: liveLogs.length ? liveLogs : state.liveLogs,
      activityFeed: activityFeed.length ? activityFeed : state.activityFeed,
      alerts: state.alerts.map((alert, index) => {
        if (index !== 0) {
          return alert;
        }
        return {
          ...alert,
          level: snapshot?.status === 'failed' ? 'warning' : 'info',
          detail: snapshot?.status === 'failed'
            ? `runtime job 실패: ${snapshot?.error || 'unknown error'}`
            : `${activeStage.label || 'Runtime'} 단계를 서버에서 실행 중입니다.`,
        };
      }),
    };
  }),
  ingestWorkbench: (payload) => set((state) => ({
    retrievalTrace: payload?.retrievalTrace?.length ? payload.retrievalTrace : state.retrievalTrace,
    clusters: payload?.clusters?.length ? payload.clusters : state.clusters,
    activityFeed: payload?.activity ? [{ id: `feed-${Date.now()}`, stage: 'retrieval', title: 'Workbench', detail: payload.activity, time: stamp() }, ...state.activityFeed].slice(0, 7) : state.activityFeed,
    answerSummary: payload?.answerSummary || state.answerSummary,
    semanticStats: {
      ...state.semanticStats,
      retrievalHitRate: payload?.retrievalTrace?.length ? Math.min(99, 82 + payload.retrievalTrace.length * 3) : state.semanticStats.retrievalHitRate,
    },
  })),
}));
