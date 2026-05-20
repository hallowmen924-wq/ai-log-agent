from __future__ import annotations

import collections
import concurrent.futures
import datetime
import hashlib
import math
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
import sys
import threading
import contextlib
import time
import uuid
import warnings
from typing import Callable

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Python 3.14 + LangChain의 pydantic v1 shim 경고를 런타임에서 필터링합니다.
# (동작에는 영향 없고, 경고 로그만 과도하게 출력되는 현상 완화)
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
    module=r"langchain_core\._api\.deprecation",
)

# 이 파일은 FastAPI의 실제 진입점입니다.
# Streamlit은 이 서버에 HTTP 요청을 보내고, 서버는 분석/벡터/챗 작업을 수행합니다.

from backend.schemas import (
    AgentOllamaToggleRequest,
    CardloanDebateRequest,
    CardloanDebateResponse,
    FaissBuildRequest,
    FeatureOntologyRuntimeRequest,
    FullAnalysisResponse,
    GenericMessage,
    LogPromptTemplateRequest,
    LogAnalyzeRequest,
    LogAnalyzeResponse,
    NewsPromptTemplateRequest,
    NewsCollectResponse,
    OntologySaveRequest,
    OllamaRuntimeToggleRequest,
    ProductSummaryResponse,
    RegulationUploadResponse,
    SearchRequest,
    StrategyChatRequest,
    StrategyChatResponse,
    WorkerConfigRequest,
)
from backend.services import (
    analyze_logs_bundle,
    ask_cardloan_debate,
    ask_strategy,
    build_backend_diagnostics,
    build_faiss_bundle,
    collect_news_bundle,
    enrich_results,
    get_chart_payloads,
    hydrate_state_from_existing_artifacts,
    record_activity_event,
    record_vector_event,
    run_full_analysis,
    search_faiss,
    state,
)
from backend.explainability_profiles import build_explainability_payload
from backend.evidence_guard import (
    apply_grounded_prompt_rules,
    build_blocked_runtime_answer,
    decorate_grounded_answer_summary,
    evaluate_runtime_evidence,
)
from backend.worker import worker
from backend.product_debate_orchestrator import run_product_debate_orchestration
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
import asyncio
import json
from mapper.reject_code_mapper import load_reject_code_mapping
from rag.decision_resolver import resolve_product_decisions
from rag.product_pattern_summary import DEFAULT_SUMMARY_PATH, load_product_pattern_summary
from rag.segment_metric_cube import DEFAULT_SEGMENT_CUBE_PATH, build_metric_answer_summary_from_cube, load_segment_metric_cube, query_has_segment_metric_intent, write_segment_metric_cube


def _strategy_chat_module():
    from agent import strategy_chat as strategy_chat_module

    return strategy_chat_module


def _vector_db_module():
    from rag import vector_db as vector_db_module

    return vector_db_module


def regulation_agent(*args, **kwargs):
    return _strategy_chat_module().regulation_agent(*args, **kwargs)


def get_ollama_runtime_preferences(*args, **kwargs):
    return _strategy_chat_module().get_ollama_runtime_preferences(*args, **kwargs)


def lightweight_ollama_generate(*args, **kwargs):
    return _strategy_chat_module().lightweight_ollama_generate(*args, **kwargs)


def set_ollama_gpu_enabled(*args, **kwargs):
    return _strategy_chat_module().set_ollama_gpu_enabled(*args, **kwargs)


def set_ontology_query_priority_enabled(*args, **kwargs):
    return _strategy_chat_module().set_ontology_query_priority_enabled(*args, **kwargs)


def get_embeddings(*args, **kwargs):
    return _vector_db_module().get_embeddings(*args, **kwargs)


def ingest_files(*args, **kwargs):
    return _vector_db_module().ingest_files(*args, **kwargs)


def ingest_files_with_report(*args, **kwargs):
    return _vector_db_module().ingest_files_with_report(*args, **kwargs)


def search_context(*args, **kwargs):
    return _vector_db_module().search_context(*args, **kwargs)


def search_regulation_evidence(*args, **kwargs):
    return _vector_db_module().search_regulation_evidence(*args, **kwargs)


def get_vector_count(*args, **kwargs):
    return _vector_db_module().get_vector_count(*args, **kwargs)


def _is_ollama_unavailable_error(error: Exception) -> bool:
    return error.__class__.__name__ == "OllamaUnavailableError"


def _probe_ollama_health(timeout_seconds: int = 2) -> dict[str, object]:
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    version_url = f"{base_url}/api/version"
    request = urllib.request.Request(version_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            version_payload = json.loads(raw_body) if raw_body.strip() else {}
            return {
                "status": "ok",
                "base_url": base_url,
                "version_url": version_url,
                "http_status": getattr(response, "status", 200),
                "version": version_payload.get("version") if isinstance(version_payload, dict) else None,
            }
    except urllib.error.HTTPError as error:
        return {
            "status": "error",
            "base_url": base_url,
            "version_url": version_url,
            "http_status": int(getattr(error, "code", 0) or 0),
            "error": str(error),
        }
    except Exception as error:
        return {
            "status": "error",
            "base_url": base_url,
            "version_url": version_url,
            "http_status": 0,
            "error": str(error),
        }


def _openai_chat_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> dict[str, object]:
    api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured on backend")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            body = json.loads(raw_body) if raw_body.strip() else {}
            if not isinstance(body, dict):
                body = {}
            return body
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace") if hasattr(error, "read") else ""
        detail = raw or str(error)
        raise HTTPException(status_code=int(getattr(error, "code", 502) or 502), detail=detail) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


OLLAMA_LIGHTWEIGHT_MODEL = os.environ.get("OLLAMA_LIGHTWEIGHT_MODEL", "mistral")
PRODUCT_DEBATE_MEMORY_PATH = pathlib.Path(__file__).resolve().parent / "data" / "product_debate_memory.json"
PRODUCT_DEBATE_MAX_CONCURRENCY = max(1, int(os.environ.get("PRODUCT_DEBATE_MAX_CONCURRENCY", "1") or 1))
PRODUCT_DEBATE_CALL_SEMAPHORE = threading.BoundedSemaphore(PRODUCT_DEBATE_MAX_CONCURRENCY)
PRODUCT_DEBATE_JOB_STATUS_LOCK = threading.Lock()
PRODUCT_DEBATE_JOB_STATUS: dict[str, dict[str, object]] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except Exception:
        value = default
    if minimum is not None:
        return max(minimum, value)
    return value


PRODUCT_DEBATE_FORCE_AUTOGEN = _env_flag("PRODUCT_DEBATE_FORCE_AUTOGEN", False)

try:
    from websockets.exceptions import ConnectionClosed
except Exception:
    ConnectionClosed = tuple()  # type: ignore[assignment]

app = FastAPI(title="AI Log Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_faiss_stats_cache: dict[str, object] = {
    "version": None,
    "data": None,
    "cached_at": 0.0,
}

ONTOLOGY_PATH = ROOT / "data" / "ontology.json"
COMMONFEATURE_PATH = ROOT / "data" / "commonfeature.json"
FULL_TEXT_RECORDS_PATH = ROOT / "data" / "full_text_records.json"
ONTOLOGY_RELATIONS_PATH = ROOT / "data" / "ontology_relations.json"
FEATURE_CLUSTER_CACHE_PATH = ROOT / "data" / "feature_customer_clusters.json"
FEATURE_EMBEDDING_CACHE_PATH = ROOT / "data" / "commonfeature_embeddings.npz"
FEATURE_CLUSTER_CACHE_VERSION = 9
FEATURE_EMBEDDING_CACHE_LIMIT = 8
SEMANTIC_REFRESH_INTERVAL_SECONDS = _env_int("SEMANTIC_REFRESH_INTERVAL_SECONDS", 120, minimum=30)
AUTO_START_WORKER = _env_flag("AUTO_START_WORKER", False)
AUTO_START_SEMANTIC_REFRESH = _env_flag("AUTO_START_SEMANTIC_REFRESH", False)
WORKER_START_INTERVAL_SECONDS = _env_int("WORKER_START_INTERVAL_SECONDS", 30, minimum=1)
HEALTH_CHECK_OLLAMA = _env_flag("HEALTH_CHECK_OLLAMA", False)
WORKBENCH_OLLAMA_TIMEOUT_SECONDS = 8
REGULATION_STATE_PATH = ROOT / "data" / "regulation_state.json"
REGULATION_UPLOAD_DIR = ROOT / "data" / "regulation_uploads"
PRODUCT_QUERY_ALIASES = {
    "C6": ["이지신용대출", "이지신용 대출", "신용대출"],
    "C9": ["카드론", "이지론", "card loan"],
    "C11": ["개인사업자대출", "사업자대출", "개인사업자"],
    "C12": ["이지대환대출", "이지대환", "대환대출"],
}
PRODUCT_DISPLAY_NAMES = {
    "C6": "이지신용대출(C6)",
    "C9": "이지론(C9)",
    "C11": "개인사업자대출(C11)",
    "C12": "이지대환대출(C12)",
}
PRODUCT_CROSS_AXIS_EXCLUSIONS = {
    "C6": ["카드론", "개인사업자", "이지대환"],
    "C9": ["이지신용대출", "개인사업자", "이지대환"],
    "C11": ["이지신용대출", "카드론", "이지대환"],
    "C12": ["이지신용대출", "카드론", "개인사업자"],
}


def _load_persisted_regulation_state() -> dict[str, object]:
    if not REGULATION_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(REGULATION_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        return payload
    except Exception:
        return {}


def _persist_regulation_state(payload: dict[str, object]) -> None:
    try:
        REGULATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGULATION_STATE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


def _persist_regulation_summary_toggle(enabled: bool) -> None:
    current = _load_persisted_regulation_state()
    if not isinstance(current, dict):
        current = {}
    current["regulation_upload_summary_enabled"] = bool(enabled)
    _persist_regulation_state(current)


def _resolve_regulation_uploaded_file(file_name: str) -> pathlib.Path:
    safe_name = pathlib.Path(str(file_name or "")).name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="invalid file name")
    base_dir = REGULATION_UPLOAD_DIR.resolve()
    target_path = (base_dir / safe_name).resolve()
    if base_dir not in target_path.parents and target_path != base_dir:
        raise HTTPException(status_code=400, detail="invalid file path")
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="regulation file not found")
    return target_path

CATEGORY_ENTITY_TYPE_MAP = {
    "applicant": "ApplicantAttribute",
    "application": "ApplicationAttribute",
    "auto_cluster": "DerivedFeature",
    "channel": "ChannelAttribute",
    "credit": "CreditAttribute",
    "credit_bureau": "CreditAttribute",
    "credit_model": "CreditModelAttribute",
    "customer_relationship": "CustomerRelationshipAttribute",
    "decision": "DecisionAttribute",
    "income": "IncomeAttribute",
    "loan": "LoanAttribute",
}

DEFAULT_SEMANTIC_GROUNDING_RULES = [
    "ontology expansion에 포함된 relation만 사용할 것",
    "retrieval result에 없는 정책, 모델, feature를 생성하지 말 것",
    "점수는 검색/랭킹 근거로만 사용하고 중요도 수치로 단정하지 말 것",
    "확실하지 않으면 추정 표현을 사용할 것",
    "금융 심사 explainability 관점으로만 설명할 것",
]

AGE_BAND_THRESHOLDS = [(29, "20대"), (39, "30대"), (49, "40대"), (59, "50대"), (999, "60대+")]
DEFAULT_INCOME_BAND_THRESHOLDS = [(26_000_000, "저소득"), (33_000_000, "중소득"), (44_000_000, "고소득"), (999_999_999_999, "초고소득")]
DEFAULT_AMOUNT_BAND_THRESHOLDS = [(5_000_000, "소액"), (15_000_000, "중액"), (36_000_000, "고액"), (999_999_999_999, "초대형")]
INCOME_MEASURE_SPECS = [
    ("최종연소득금액", 100_000, 10_000),
    ("최종연소득금액(4.0)", 100_000, 10_000),
    ("KCB추정소득", 100_000, 1_000),
    ("NICE추정소득", 100_000, 1_000),
    ("소득", 100_000, 1_000),
]
AMOUNT_MEASURE_SPECS = [
    ("최종대출가능금액_실시간", None, 1),
    ("최종대출가능금액", None, 1),
    ("초기한도", None, 1),
    ("대출금액", 100_000, 1_000),
]
RATE_MEASURE_SPECS = [
    ("산출금리", None, 1),
    ("대출이율", None, 1),
    ("적용금리", None, 1),
    ("applied_rate", None, 1),
]

_feature_embedding_cache: dict[str, object] = {
    "full_signature": None,
    "full_ids": [],
    "full_index_by_id": {},
    "full_matrix": None,
    "subset_matrices": collections.OrderedDict(),
}
_feature_embedding_cache_lock = threading.Lock()

INTENT_CLASSIFIER_PROTOTYPES: dict[str, dict[str, object]] = {
    "approval_factor": {
        "label": "Approval factor",
        "description": "Approval drivers, important features, influence factors, and explainability questions.",
        "output_categories": ["answer_summary", "feature_explainability", "cluster_analysis"],
        "examples": [
            "40대 직장인의 카드론 승인에 중요한 요인은?",
            "승인 가능성에 영향을 주는 feature를 알려줘",
            "이지신용대출 신청자의 승인 요인을 설명해줘",
            "승인 고객군에서 공통으로 강한 신호는?",
            "어떤 특성이 심사 결과에 가장 큰 영향을 줘?",
            "approval factor important feature explainability",
        ],
    },
    "reject_reason": {
        "label": "Reject reason",
        "description": "Rejected cases, reject reason codes, knock-out causes, and decline explanations.",
        "output_categories": ["answer_summary", "reject_code_analysis", "feature_explainability"],
        "examples": [
            "40대 카드론 신청자들의 평균 탈락 사유는?",
            "이지론 거절 고객군에서 가장 자주 연결되는 reject reason은?",
            "거절 사유 코드 K코드를 보여줘",
            "부결 원인과 관련 feature를 같이 설명해줘",
            "knock-out reject code decline reason",
            "탈락 사유와 거절 고객 패턴을 알려줘",
        ],
    },
    "rate_limit": {
        "label": "Rate and limit",
        "description": "Average rate, approved limit, amount, distribution, delinquency, and metric questions.",
        "output_categories": ["answer_summary", "metric_summary", "cluster_analysis"],
        "examples": [
            "이지신용대출 평균 금리와 한도는?",
            "승인 고객의 평균 금리와 승인 한도 분포를 알려줘",
            "평균 한도와 소득 구간 관계를 비교해줘",
            "연체율과 금리 분포를 보여줘",
            "average rate approved limit distribution metric",
            "한도 금리 부실률 지표를 요약해줘",
        ],
    },
    "cluster_vector": {
        "label": "Cluster and vector",
        "description": "Customer clusters, vector similarity, FAISS, embeddings, and segment comparisons.",
        "output_categories": ["answer_summary", "cluster_analysis", "feature_explainability"],
        "examples": [
            "이지신용대출 고객군집에서 승인과 거절 군집을 비교해줘",
            "40대 신청자 군집을 소득, 한도, 거절 사유 기준으로 나눠줘",
            "feature_customer_clusters 기준 반복 관계를 설명해줘",
            "FAISS 벡터에서 유사한 고객 패턴을 찾아줘",
            "embedding similarity cluster vector search",
            "군집별 공통 신호를 비교해줘",
        ],
    },
    "regulation_policy": {
        "label": "Regulation and policy",
        "description": "Regulation documents, policy, DSR, stress rules, citations, and compliance evidence.",
        "output_categories": ["answer_summary", "policy_regulation", "citation_answer"],
        "examples": [
            "규제 문서 기준으로 카드론 심사 정책을 설명해줘",
            "DSR 규제와 스트레스 금리 시행 내용을 알려줘",
            "첨부 문서 근거로 정책 영향을 요약해줘",
            "규제 강화 가능성과 심사 기준 변경점을 알려줘",
            "policy regulation compliance citation evidence",
            "금융 규제 문서에서 관련 근거를 찾아줘",
        ],
    },
    "strategy_simulation": {
        "label": "Strategy simulation",
        "description": "What-if simulations, pricing changes, product strategy, conversion, and scenario analysis.",
        "output_categories": ["answer_summary", "strategy_simulation", "product_strategy"],
        "examples": [
            "이지신용대출 금리를 올리면 승인률과 수익은 어떻게 될까?",
            "이지론 금리를 1%p 낮추면 고객군별 리스크가 어떻게 바뀔까?",
            "한도를 늘리면 승인 전환과 부실률은 어떻게 변할까?",
            "상품 전략 시뮬레이션을 해줘",
            "what if simulation strategy product conversion",
            "거절 고객을 승인 전환하려면 어떤 전략이 필요해?",
        ],
    },
    "general_fallback": {
        "label": "General fallback",
        "description": "General questions that do not clearly map to financial ontology evidence.",
        "output_categories": ["answer_summary", "general_answer"],
        "examples": [
            "전체 내용을 간단히 설명해줘",
            "무엇을 할 수 있는지 알려줘",
            "요약해줘",
            "일반적인 답변을 해줘",
            "general answer summarize",
        ],
    },
}

_intent_embedding_cache: dict[str, object] = {
    "signature": "",
    "intent_ids": [],
    "example_counts": {},
    "matrix": None,
}
_intent_embedding_cache_lock = threading.Lock()

WORKBENCH_RUNTIME_STAGES = [
    {
        "key": "extraction",
        "label": "Load Runtime Data",
        "heroLabel": "JSON Load",
        "detail": "commonfeature.json 과 full_text_records.json 을 읽습니다.",
    },
    {
        "key": "alias",
        "label": "Product Scope Filter",
        "heroLabel": "Product Filter",
        "detail": "선택한 상품 기준으로 feature 후보를 좁힙니다.",
    },
    {
        "key": "mapping",
        "label": "Semantic Feature Rank",
        "heroLabel": "Semantic Rank",
        "detail": "질문과 가장 가까운 feature 를 점수화합니다.",
    },
    {
        "key": "ontology",
        "label": "Primary Feature Select",
        "heroLabel": "Feature Select",
        "detail": "대표 축과 연관 feature 를 확정합니다.",
    },
    {
        "key": "faiss",
        "label": "Cluster Cache Build",
        "heroLabel": "Cluster Build",
        "detail": "고객군집 캐시와 cluster 후보를 계산합니다.",
    },
    {
        "key": "retrieval",
        "label": "Retrieval Result Build",
        "heroLabel": "Retrieval Build",
        "detail": "관련 레코드와 retrieval trace 후보를 만듭니다.",
    },
    {
        "key": "ollama",
        "label": "Answer Summary Build",
        "heroLabel": "Answer Summary",
        "detail": "화면 상단에 보여줄 요약 답변을 만듭니다.",
    },
]

_workbench_jobs: dict[str, dict[str, object]] = {}
_workbench_jobs_lock = threading.Lock()
_conversation_sessions: dict[str, dict[str, object]] = {}
_conversation_sessions_lock = threading.Lock()


def _iso_now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


_semantic_refresh_status_lock = threading.Lock()
_semantic_refresh_run_lock = threading.Lock()
_semantic_refresh_stop_event = threading.Event()
_semantic_refresh_thread: threading.Thread | None = None
_semantic_refresh_status: dict[str, object] = {
    "enabled": AUTO_START_SEMANTIC_REFRESH,
    "interval_seconds": SEMANTIC_REFRESH_INTERVAL_SECONDS,
    "status": "idle",
    "message": "자동 갱신은 기본 비활성화 상태입니다. 필요하면 수동 갱신하거나 AUTO_START_SEMANTIC_REFRESH=1로 켜세요.",
    "last_started_at": "",
    "last_completed_at": "",
    "last_failed_at": "",
    "last_error": "",
    "next_run_at": "",
    "run_count": 0,
    "record_count": 0,
    "segment_count": 0,
    "cluster_count": 0,
    "elapsed_ms": 0,
    "trigger": "",
}


def _semantic_refresh_snapshot() -> dict[str, object]:
    with _semantic_refresh_status_lock:
        return dict(_semantic_refresh_status)


def _update_semantic_refresh_status(**updates: object) -> dict[str, object]:
    with _semantic_refresh_status_lock:
        _semantic_refresh_status.update(updates)
        _semantic_refresh_status["interval_seconds"] = SEMANTIC_REFRESH_INTERVAL_SECONDS
        return dict(_semantic_refresh_status)


def _run_semantic_refresh_once(trigger: str = "timer") -> dict[str, object]:
    if not _semantic_refresh_run_lock.acquire(blocking=False):
        return _update_semantic_refresh_status(
            status="running",
            message="이전 통계 큐브/군집 갱신이 아직 진행 중입니다.",
            trigger=trigger,
        )

    started_at = _iso_now()
    started_perf = time.perf_counter()
    try:
        _update_semantic_refresh_status(
            status="running",
            message="심사 로그를 다시 읽고 통계 큐브와 군집분석을 최신화하는 중입니다.",
            last_started_at=started_at,
            last_error="",
            trigger=trigger,
        )
        record_activity_event(
            "semantic_refresh",
            "running",
            "2분 자동 갱신: 심사 로그 → full_text_records → 통계 큐브 → 군집분석 순서로 최신화합니다.",
            update_status=True,
        )

        # Keep full_text_records in sync with the generated live log before
        # rebuilding the precomputed semantic layers. Avoid the huge
        # log_analyzer_results.json intermediate during the recurring refresh.
        from tools.export_full_text_records import main as export_full_text_records

        export_full_text_records()

        write_segment_metric_cube(FULL_TEXT_RECORDS_PATH, DEFAULT_SEGMENT_CUBE_PATH)
        cube_payload = load_segment_metric_cube(DEFAULT_SEGMENT_CUBE_PATH)
        records = _read_record_list(FULL_TEXT_RECORDS_PATH)
        cluster_payload = _load_or_build_customer_cluster_cache(records, force_rebuild=True)

        segments = list(cube_payload.get("segments") or [])
        clusters_all = list(cluster_payload.get("all") or [])
        completed_at = _iso_now()
        elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
        with _semantic_refresh_status_lock:
            run_count = int(_semantic_refresh_status.get("run_count") or 0) + 1
        snapshot = _update_semantic_refresh_status(
            status="completed",
            message="통계 큐브와 군집분석이 최신 로그 기준으로 갱신됐습니다.",
            last_completed_at=completed_at,
            run_count=run_count,
            record_count=len(records),
            segment_count=int((cube_payload.get("meta") or {}).get("segment_count") or len(segments)),
            cluster_count=len(clusters_all),
            elapsed_ms=elapsed_ms,
            trigger=trigger,
        )
        record_activity_event(
            "semantic_refresh",
            "completed",
            f"통계 큐브/군집 자동 갱신 완료: 로그 {len(records)}건, 세그먼트 {snapshot['segment_count']}개, 군집 {len(clusters_all)}개",
            update_status=True,
        )
        return snapshot
    except Exception as error:
        failed_at = _iso_now()
        elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
        snapshot = _update_semantic_refresh_status(
            status="failed",
            message="통계 큐브/군집 자동 갱신에 실패했습니다.",
            last_failed_at=failed_at,
            last_error=str(error),
            elapsed_ms=elapsed_ms,
            trigger=trigger,
        )
        record_activity_event(
            "semantic_refresh",
            "failed",
            f"통계 큐브/군집 자동 갱신 실패: {error}",
            update_status=True,
        )
        return snapshot
    finally:
        _semantic_refresh_run_lock.release()


def _semantic_refresh_loop() -> None:
    while not _semantic_refresh_stop_event.is_set():
        if _semantic_refresh_stop_event.wait(SEMANTIC_REFRESH_INTERVAL_SECONDS):
            break
        _run_semantic_refresh_once("timer")
        next_run = datetime.datetime.now() + datetime.timedelta(seconds=SEMANTIC_REFRESH_INTERVAL_SECONDS)
        _update_semantic_refresh_status(next_run_at=next_run.isoformat(timespec="seconds"))


def _start_semantic_refresh_scheduler() -> None:
    global _semantic_refresh_thread
    if _semantic_refresh_thread is not None and _semantic_refresh_thread.is_alive():
        return
    _semantic_refresh_stop_event.clear()
    next_run = datetime.datetime.now() + datetime.timedelta(seconds=SEMANTIC_REFRESH_INTERVAL_SECONDS)
    _update_semantic_refresh_status(
        enabled=True,
        status="idle",
        message="2분 자동 갱신 스케줄러가 시작됐습니다.",
        next_run_at=next_run.isoformat(timespec="seconds"),
    )
    _semantic_refresh_thread = threading.Thread(
        target=_semantic_refresh_loop,
        name="semantic-refresh-scheduler",
        daemon=True,
    )
    _semantic_refresh_thread.start()


def _stop_semantic_refresh_scheduler() -> None:
    _semantic_refresh_stop_event.set()


def _create_workbench_stage_statuses() -> list[dict[str, object]]:
    return [
        {
            "key": stage["key"],
            "label": stage["label"],
            "heroLabel": stage["heroLabel"],
            "detail": stage["detail"],
            "status": "idle",
            "progress": 0,
            "started_at": None,
            "completed_at": None,
            "meta": {},
        }
        for stage in WORKBENCH_RUNTIME_STAGES
    ]


def _make_job_log(stage_key: str, text: str, tone: str = "info", meta: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "id": f"{stage_key}-{uuid.uuid4().hex[:10]}",
        "stage": stage_key,
        "text": text,
        "tone": tone,
        "time": _iso_now(),
        "meta": meta or {},
    }


def _snapshot_workbench_job(job_id: str) -> dict[str, object]:
    with _workbench_jobs_lock:
        job = _workbench_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown workbench job: {job_id}")
        snapshot = dict(job)
        started_monotonic = float(snapshot.get("started_monotonic") or 0.0)
        if started_monotonic:
            snapshot["elapsed_ms"] = int((time.perf_counter() - started_monotonic) * 1000)
        stage_timers = snapshot.pop("stage_timers", None)
        if stage_timers is not None:
            snapshot["stage_timer_count"] = len(stage_timers)
        return snapshot


def _store_workbench_job(job_id: str, job: dict[str, object]) -> None:
    with _workbench_jobs_lock:
        _workbench_jobs[job_id] = job


def _format_reject_code_summary(reject_code_summary: list[dict[str, object]], limit: int = 3) -> str:
    parts: list[str] = []
    for item in reject_code_summary[:limit]:
        code = str(item.get("code") or "").strip()
        count = int(item.get("count") or 0)
        description = str(item.get("description") or "").strip()
        if not code:
            continue
        label = f"{code} {count}건"
        if description:
            label = f"{label}({description})"
        parts.append(label)
    return ", ".join(parts)


def _format_reject_reason_stat_line(reject_code_summary: list[dict[str, object]], limit: int = 3) -> str:
    parts: list[str] = []
    for item in reject_code_summary[:limit]:
        code = str(item.get("code") or "").strip()
        description = str(item.get("description") or "").strip()
        count = int(item.get("count") or 0)
        share = item.get("share")
        if not code:
            continue
        label = f"{code}"
        if description:
            label = f"{label}({description})"
        try:
            share_text = f", {float(share) * 100:.1f}%"
        except (TypeError, ValueError):
            share_text = ""
        parts.append(f"{label} {count:,}건{share_text}")
    return " / ".join(parts)


def _build_reject_reason_highlights(
    reject_code_summary: list[dict[str, object]],
    selected_product: str,
    customer_clusters: list[dict],
) -> list[dict[str, object]]:
    product_name = _product_display_name(selected_product) or selected_product or "전체"
    top_item = reject_code_summary[0] if reject_code_summary else {}
    top_code = str(top_item.get("code") or "-")
    top_description = str(top_item.get("description") or "").strip()
    top_label = f"{top_code} {top_description}".strip()
    base_count = int(top_item.get("base_rejected_records") or 0)
    top_cluster = customer_clusters[0] if customer_clusters else {}
    cluster_label = " / ".join(
        part for part in [
            str(top_cluster.get("decision") or "").strip(),
            str(top_cluster.get("age_band") or "").strip(),
            str(top_cluster.get("income_band") or "").strip(),
            str(top_cluster.get("amount_band") or "").strip(),
        ]
        if part and part != "미상"
    )
    return [
        {"label": "상품", "value": product_name},
        {"label": "최빈 거절사유", "value": top_label},
        {"label": "거절 표본", "value": f"{base_count:,}건" if base_count else "-"},
        {"label": "상위 거절사유", "value": _format_reject_reason_stat_line(reject_code_summary)},
        {"label": "대표 고객군", "value": cluster_label or "-"},
    ]


def _format_reject_code_scope_note(reject_code_summary: list[dict[str, object]], selected_product: str) -> str:
    if not reject_code_summary:
        return ""
    first = reject_code_summary[0]
    age_filter = str(first.get("age_filter") or "").strip()
    if age_filter and first.get("age_filter_used") is False:
        product_name = _product_display_name(selected_product) or str(selected_product or "").strip() or "상품"
        return f"{age_filter} 기준 K코드가 부족해 {product_name} 전체 기준으로 봤습니다."
    return ""


def _build_answer_summary(query: str, selected_product: str, selected_feature: dict | None, representative_features: list[dict[str, object]], customer_clusters: list[dict], retrieval_results: list[dict], records: list[dict], related_features: list[dict[str, object]] | None = None, reject_code_summary: list[dict[str, object]] | None = None) -> dict[str, object]:
    top_cluster = customer_clusters[0] if customer_clusters else {}
    top_feature_name = str((selected_feature or {}).get("feature_name") or (selected_feature or {}).get("feature_id") or "핵심 기준")
    representative_names = _dedupe_text_items([
        item.get("feature_name") or item.get("feature_id") or ""
        for item in representative_features[:3]
    ], limit=3)
    reject_intent = _query_has_reject_intent(query, selected_feature)
    if reject_intent:
        representative_names = [
            name for name in representative_names
            if not _is_cross_product_feature_label(name, selected_product)
        ]
        representative_names = _dedupe_text_items([
            _product_display_name(selected_product),
            "거절사유코드",
            *representative_names,
        ], limit=3)
        if _is_cross_product_feature_label(top_feature_name, selected_product):
            top_feature_name = "거절사유코드"
    representative_label = " / ".join(representative_names) or top_feature_name
    income_band = str(top_cluster.get("income_band") or "미상")
    age_band = str(top_cluster.get("age_band") or "미상")
    amount_band = str(top_cluster.get("amount_band") or "미상")
    decision = str(top_cluster.get("decision") or "미상")
    count = int(top_cluster.get("count") or 0)
    reject_summary = _format_cluster_reject_reason_summary(top_cluster)
    # records가 이미 profiles라면 그대로 사용
    sample_records = records[:1000] if len(records) > 1000 else records
    if (not reject_code_summary or not len(reject_code_summary)) and records is not None:
        reject_code_summary = _build_reject_code_distribution(sample_records, selected_product, query, limit=3)
    reject_code_line = _format_reject_code_summary(reject_code_summary or [])
    reject_code_scope_note = _format_reject_code_scope_note(reject_code_summary or [], selected_product)
    if reject_intent and reject_code_summary:
        product_name = _product_display_name(selected_product) or selected_product or "전체"
        stat_line = _format_reject_reason_stat_line(reject_code_summary)
        top_item = reject_code_summary[0]
        top_code = str(top_item.get("code") or "").strip()
        top_description = str(top_item.get("description") or "").strip()
        top_label = f"{top_code}({top_description})" if top_description else top_code
        top_count = int(top_item.get("count") or 0)
        try:
            top_share = f"{float(top_item.get('share')) * 100:.1f}%"
        except (TypeError, ValueError):
            top_share = ""
        scope_prefix = f"{reject_code_scope_note} " if reject_code_scope_note else ""
        cluster_context = ""
        if top_cluster:
            cluster_parts = [
                str(top_cluster.get("decision") or "").strip(),
                str(top_cluster.get("age_band") or "").strip(),
                str(top_cluster.get("income_band") or "").strip(),
                str(top_cluster.get("amount_band") or "").strip(),
            ]
            cluster_label = " / ".join(part for part in cluster_parts if part and part != "미상")
            if cluster_label:
                cluster_context = f" 참고로 가장 가까운 고객군은 {cluster_label}이며 {count:,}건입니다."
        explanation = (
            f"{scope_prefix}{product_name} 거절 고객군에서 가장 자주 연결되는 reject reason은 "
            f"{top_label}입니다. 이 사유는 {top_count:,}건"
            f"{f'({top_share})' if top_share else ''}으로 가장 많이 나타났습니다. "
            f"상위 거절사유 분포는 {stat_line}입니다."
            f"{cluster_context}"
        )
        return {
            "headline": f"{product_name} 거절 고객군의 최빈 reject reason은 {top_label}입니다.",
            "explanation": explanation,
            "highlights": _build_reject_reason_highlights(reject_code_summary, selected_product, customer_clusters),
            "reject_code_summary": list(reject_code_summary or []),
            "top_reject_codes": list(reject_code_summary or []),
            "metric_summary": [],
            "source": "reject-reason-distribution",
        }
    avg_income = _format_krw_compact(top_cluster.get("avg_income"))
    avg_amount = _format_krw_compact(top_cluster.get("avg_amount"))
    avg_rate = str(top_cluster.get("avg_rate_display") or "")
    income_source = str(top_cluster.get("top_income_source") or "")
    amount_source = str(top_cluster.get("top_amount_source") or "")
    rate_source = str(top_cluster.get("top_rate_source") or "")
    metric_summary = _build_cluster_metric_summary(top_cluster, representative_features)
    # 대표축 해석이 headline에 먼저 오도록 개선
    headline = f"질문 '{query or '기본 질의'}'는 {selected_product or '전체'} / {representative_label} 기준으로 해석했습니다."
    # 군집 요약은 explanation에 포함
    if _query_asks_influence_features(query):
        # 영향 feature 최대 5개, 군집 최대 2개만 반환
        amount_axes = [
            item.get("feature_name") or item.get("feature_id") or ""
            for item in representative_features
            if str(item.get("feature_id") or "") in {"decision.approved_amount", "loan.requested_limit"}
            or str(item.get("axis_key") or "") in {"limit", "requested_limit"}
        ]
        influence_names = _dedupe_text_items([
            item.get("feature_name") or item.get("feature_id") or ""
            for item in (related_features or [])[:5]
        ], limit=5)
        product_label = _product_display_name(selected_product) or selected_product or "해당 상품"
        influence_buckets = _dedupe_text_items([
            _business_influence_bucket(item)[0]
            for item in (related_features or [])[:5]
            if isinstance(item, dict)
        ], limit=3)
        return {
            "headline": f"{product_label} 승인 한도는 {', '.join(influence_buckets) or '핵심 심사 기준'}이 좌우합니다.",
            "explanation": _build_business_friendly_influence_answer({"influence_features": (related_features or [])[:5]}, selected_product),
            "highlights": [
                {"label": "핵심 기준", "value": " / ".join(influence_buckets) or "-"},
                {"label": "주요 지표", "value": ", ".join(influence_names[:3]) or "-"},
                {"label": "상품", "value": product_label},
            ],
        }
        # axis_label, headline, explanation 등은 기존과 동일하게 유지
    if not customer_clusters:
        headline = f"{selected_product or '전체'} 기준으로 질문과 직접 연결된 군집을 찾지 못했습니다."
        if reject_code_line and reject_intent:
            product_name = _product_display_name(selected_product) or selected_product or "전체"
            headline = f"{product_name} 기준 거절사유코드 분포를 확인했습니다."
    explanation = (
        f"상위 고객군집은 {decision} {age_band} · {income_band} · {amount_band} 조합으로 {count}건입니다."
    )
    if avg_income or avg_amount:
        detail_parts = []
        if avg_income:
            detail_parts.append(f"평균 소득은 {avg_income}")
        if avg_rate:
            detail_parts.append(f"평균 금리는 {avg_rate}")
        if avg_amount:
            detail_parts.append(f"평균 가능금액은 {avg_amount}")
        explanation = f"{explanation} {' / '.join(detail_parts)} 수준입니다."
    if reject_summary and decision == "거절":
        explanation = f"{explanation} 대표 거절 사유는 {reject_summary} 입니다."
    if reject_code_line and reject_intent:
        scope_prefix = f"{reject_code_scope_note} " if reject_code_scope_note else ""
        explanation = f"{explanation} {scope_prefix}거절사유코드 상위 코드는 {reject_code_line} 입니다."
    if not customer_clusters:
        explanation = f"질문 '{query or '기본 질의'}' 에 대해 feature 랭킹과 retrieval 결과를 만들었지만, 노출할 군집 요약은 부족했습니다."
        if reject_code_line and reject_intent:
            product_name = _product_display_name(selected_product) or selected_product or "전체"
            scope_prefix = f"{reject_code_scope_note} " if reject_code_scope_note else ""
            explanation = f"질문 '{query or '기본 질의'}' 는 {product_name}의 거절사유코드 기준으로 해석했습니다. {scope_prefix}거절사유코드 상위 코드는 {reject_code_line} 입니다."
    return {
        "headline": headline,
        "explanation": explanation,
        "highlights": [
            {"label": "핵심 기준", "value": top_feature_name},
            {"label": "관련 기준", "value": representative_label},
            {"label": "Income Band", "value": income_band},
            {"label": "Avg Income", "value": avg_income or "없음"},
            {"label": "Income Source", "value": income_source or "없음"},
            {"label": "Age Band", "value": age_band},
            {"label": "Avg Rate", "value": avg_rate or "없음"},
            {"label": "Rate Source", "value": rate_source or "없음"},
            {"label": "Avg Amount", "value": avg_amount or "없음"},
            {"label": "Amount Source", "value": amount_source or "없음"},
            {"label": "Reject Reason", "value": reject_summary or "없음"},
            {"label": "Top Reject Codes", "value": reject_code_line or "없음"},
            {"label": "Matched Records", "value": str(len(retrieval_results))},
        ],
        "reject_code_summary": list(reject_code_summary or []),
        "metric_summary": metric_summary,
    }


def _query_asks_average_metrics(query: str) -> bool:
    query_text = str(query or "").lower()
    compact = re.sub(r"\s+", "", query_text)
    asks_average = any(marker in compact for marker in ["평균", "avg", "average", "수준", "얼마", "어느정도"])
    asks_metric = _query_has_rate_intent(query_text) or _query_has_limit_intent(query_text) or _query_has_delinquency_intent(query_text)
    asks_feature_reason = any(marker in compact for marker in ["영향", "feature", "피처", "변수", "요인", "왜"])
    return asks_average and asks_metric and not asks_feature_reason


def _query_has_delinquency_intent(query: str) -> bool:
    compact_query = _compact_search_text(query)
    return any(marker in compact_query for marker in ["연체", "연체율", "부실", "부실률", "delinquency", "default"])


def _mean_number(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _build_average_metric_answer_summary(query: str, selected_product: str, records: list[dict], customer_clusters: list[dict]) -> dict[str, object]:
    cube_answer = build_metric_answer_summary_from_cube(query, selected_product, DEFAULT_SEGMENT_CUBE_PATH)
    if cube_answer:
        return cube_answer

    reject_code_mapping = _get_reject_code_mapping()
    product_label = _product_display_name(selected_product) or selected_product or "전체 상품"
    profiles = [
        _build_record_profile(record, reject_code_mapping=reject_code_mapping)
        for record in records
    ]
    if selected_product:
        profiles = [profile for profile in profiles if str(profile.get("product") or "") == selected_product]
    metric_profiles = [
        profile for profile in profiles
        if profile.get("rate") is not None or profile.get("amount") is not None
    ] or profiles

    rate_values = [
        float(profile.get("rate"))
        for profile in metric_profiles
        if profile.get("rate") is not None
    ]
    amount_values = [
        float(profile.get("amount"))
        for profile in metric_profiles
        if profile.get("amount") is not None
    ]
    delinquency_rate_values = [
        float(profile.get("delinquency_rate"))
        for profile in metric_profiles
        if profile.get("delinquency_rate") is not None
    ]
    delinquency_proxy_values = [
        1.0 if profile.get("delinquency_signal") else 0.0
        for profile in metric_profiles
    ]
    avg_rate = _mean_number(rate_values)
    avg_amount = _mean_number(amount_values)
    avg_delinquency_rate = _mean_number(delinquency_rate_values)
    delinquency_proxy_rate = (_mean_number(delinquency_proxy_values) or 0.0) * 100 if delinquency_proxy_values else None
    avg_rate_display = f"{avg_rate:.2f}%" if avg_rate is not None else ""
    avg_amount_display = _format_krw_compact(avg_amount)
    delinquency_display = f"{avg_delinquency_rate:.2f}%" if avg_delinquency_rate is not None else (f"{delinquency_proxy_rate:.1f}%" if delinquency_proxy_rate is not None and _query_has_delinquency_intent(query) else "")

    top_cluster = next(
        (
            cluster
            for cluster in customer_clusters
            if cluster.get("avg_rate") is not None or cluster.get("avg_amount") is not None
        ),
        customer_clusters[0] if customer_clusters else {},
    )
    cluster_rate_display = str(top_cluster.get("avg_rate_display") or "")
    cluster_amount_display = str(top_cluster.get("avg_amount_display") or "")
    cluster_label_parts = [
        str(top_cluster.get("decision") or "").strip(),
        str(top_cluster.get("age_band") or "").strip(),
        str(top_cluster.get("income_band") or "").strip(),
        str(top_cluster.get("amount_band") or "").strip(),
    ]
    cluster_label = " / ".join(part for part in cluster_label_parts if part and part != "미상")
    total_count = len(profiles)
    matched_count = len(metric_profiles)

    rate_requested = _query_has_rate_intent(query)
    limit_requested = _query_has_limit_intent(query)
    delinquency_requested = _query_has_delinquency_intent(query)
    if not any([rate_requested, limit_requested, delinquency_requested]):
        rate_requested = True
        limit_requested = True
    headline_parts: list[str] = []
    if avg_rate_display and rate_requested:
        headline_parts.append(f"평균 금리는 {avg_rate_display}")
    if avg_amount_display and limit_requested:
        headline_parts.append(f"평균 한도는 {avg_amount_display}")
    if delinquency_display and delinquency_requested:
        headline_parts.append(f"연체/부실 proxy는 {delinquency_display}")
    headline = f"{product_label}의 " + ", ".join(headline_parts) + "입니다." if headline_parts else f"{product_label}의 평균 금리/한도 값을 찾지 못했습니다."

    explanation_lines = [
        f"{product_label} 전체 로그 {total_count}건 중 금리/한도/연체 신호가 있는 로그 {matched_count}건에서 금리 {len(rate_values)}건, 한도 {len(amount_values)}건을 집계했습니다."
    ]
    cluster_metric_parts: list[str] = []
    if cluster_rate_display:
        cluster_metric_parts.append(f"군집 평균 금리 {cluster_rate_display}")
    if cluster_amount_display:
        cluster_metric_parts.append(f"군집 평균 한도 {cluster_amount_display}")
    if cluster_label and cluster_metric_parts:
        explanation_lines.append(f"가장 가까운 고객군은 {cluster_label}이며, {' / '.join(cluster_metric_parts)}입니다.")
    elif cluster_metric_parts:
        explanation_lines.append(f"가장 가까운 고객군 기준 {' / '.join(cluster_metric_parts)}입니다.")
    explanation_lines.append("개념 설명은 빼고, 실제 로그와 군집 평균만 기준으로 정리했습니다.")

    highlights = [
        {"label": "상품", "value": product_label},
        {"label": "평균 금리", "value": avg_rate_display or "없음"},
        {"label": "평균 한도", "value": avg_amount_display or "없음"},
        {"label": "전체 로그", "value": f"{total_count}건"},
        {"label": "집계 로그", "value": f"{matched_count}건"},
        {"label": "금리 집계", "value": f"{len(rate_values)}건"},
        {"label": "한도 집계", "value": f"{len(amount_values)}건"},
    ]
    if delinquency_display:
        highlights.append({"label": "연체/부실 proxy", "value": delinquency_display})
    if cluster_label:
        highlights.append({"label": "가까운 군집", "value": cluster_label})

    metric_summary: list[dict[str, object]] = []
    if avg_rate is not None:
        metric_summary.append({
            "axis_key": "rate",
            "label": "평균 금리",
            "feature_id": "decision.applied_rate",
            "value": round(avg_rate, 4),
            "display": avg_rate_display,
            "source": "심사 로그",
        })
    if avg_amount is not None:
        metric_summary.append({
            "axis_key": "limit",
            "label": "평균 한도",
            "feature_id": "decision.approved_amount",
            "value": round(avg_amount, 2),
            "display": avg_amount_display,
            "source": "심사 로그",
        })
    if delinquency_display:
        metric_summary.append({
            "axis_key": "delinquency",
            "label": "연체/부실 proxy",
            "feature_id": "risk.delinquency_proxy",
            "value": round(avg_delinquency_rate if avg_delinquency_rate is not None else delinquency_proxy_rate, 4),
            "display": delinquency_display,
            "source": "심사 로그",
        })

    return {
        "headline": headline,
        "explanation": " ".join(explanation_lines),
        "highlights": highlights,
        "metric_summary": metric_summary,
        "source": "log-cluster-metric",
        "source_model": "rule-based",
        "citations": [],
    }


def _build_cluster_signal_items(cluster: dict[str, object]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    decision = str(cluster.get("decision") or "")
    if cluster.get("avg_model_score") is not None or cluster.get("avg_model_score_display"):
        items.append({
            "label": "모델 점수",
            "value": str(cluster.get("avg_model_score_display") or _format_metric_value(cluster.get("avg_model_score"), "점") or ""),
            "source": str(cluster.get("top_model_score_source") or "심사 로그"),
            "tone": "positive" if decision == "승인" else "warning",
        })
    if cluster.get("delinquency_proxy_rate") is not None or cluster.get("avg_delinquency_rate") is not None:
        value = str(cluster.get("avg_delinquency_rate_display") or cluster.get("delinquency_proxy_rate_display") or "")
        items.append({
            "label": "연체/부실 신호",
            "value": value,
            "source": str(cluster.get("top_delinquency_rate_source") or "로그 proxy"),
            "tone": "positive" if decision == "승인" and str(value).startswith("0") else "warning",
        })
    if cluster.get("avg_rate") is not None or cluster.get("avg_rate_display"):
        items.append({
            "label": "금리 수준",
            "value": str(cluster.get("avg_rate_display") or ""),
            "source": str(cluster.get("top_rate_source") or "심사 결과"),
            "tone": "neutral",
        })
    if cluster.get("avg_amount") is not None or cluster.get("avg_amount_display"):
        items.append({
            "label": "승인/대출 한도",
            "value": str(cluster.get("avg_amount_display") or ""),
            "source": str(cluster.get("top_amount_source") or "심사 결과"),
            "tone": "positive" if decision == "승인" else "neutral",
        })
    if cluster.get("avg_income") is not None or cluster.get("avg_income_display"):
        items.append({
            "label": "소득 구간",
            "value": f"{cluster.get('income_band') or '미상'} · {cluster.get('avg_income_display') or ''}".strip(" ·"),
            "source": str(cluster.get("top_income_source") or "고객 정보"),
            "tone": "neutral",
        })
    return [item for item in items if str(item.get("value") or "").strip()]


def _build_cluster_signal_answer_summary(query: str, selected_product: str, customer_clusters: list[dict]) -> dict[str, object]:
    product_label = _product_display_name(selected_product) or selected_product or "전체 상품"
    decision_focus = _extract_decision_focus(query)
    income_focus = _extract_income_band_focus(query)
    amount_focus = _extract_amount_band_focus(query)
    age_focus = _extract_age_band_focus(query)
    top_cluster = customer_clusters[0] if customer_clusters else {}
    focus_parts = [income_focus, age_focus, amount_focus, decision_focus]
    focus_label = " / ".join(part for part in focus_parts if part) or "요청 조건"
    if not top_cluster:
        return {
            "headline": f"{product_label} {focus_label} 고객군은 현재 로그에서 충분히 확인되지 않습니다.",
            "explanation": "질문에 들어온 조건을 그대로 적용했지만, 해당 조건의 고객군집이 없거나 표본이 부족합니다. 다른 소득구간이나 승인/거절 조건을 넓혀서 다시 보면 됩니다.",
            "highlights": [
                {"label": "상품", "value": product_label},
                {"label": "요청 조건", "value": focus_label},
                {"label": "고객군", "value": "0개"},
            ],
            "metric_summary": [],
            "source": "customer-cluster-signal",
            "source_model": "rule-based",
            "citations": [],
        }

    signal_items = _build_cluster_signal_items(top_cluster)
    top_signal_names = [str(item.get("label") or "") for item in signal_items[:3] if item.get("label")]
    cluster_label = " / ".join(
        part for part in [
            str(top_cluster.get("decision") or "").strip(),
            str(top_cluster.get("age_band") or "").strip(),
            str(top_cluster.get("income_band") or "").strip(),
            str(top_cluster.get("amount_band") or "").strip(),
        ]
        if part and part != "미상"
    )
    headline = f"{product_label} {cluster_label} 고객군의 공통 신호는 {' · '.join(top_signal_names) or '로그 지표'}입니다."
    signal_sentence = ", ".join(
        f"{item.get('label')} {item.get('value')}({item.get('source')})"
        for item in signal_items[:4]
    )
    explanation = (
        f"요청 조건에 맞는 고객군 {int(top_cluster.get('count') or 0):,}건을 기준으로 봤습니다. "
        f"공통으로 눈에 띄는 신호는 {signal_sentence or '충분한 지표가 없음'}입니다. "
        "실제 로그의 군집 지표 기준입니다."
    )
    highlights = [
        {"label": "상품", "value": product_label},
        {"label": "고객군", "value": cluster_label or str(top_cluster.get("label") or "-")},
        {"label": "표본", "value": f"{int(top_cluster.get('count') or 0):,}건"},
        *[
            {"label": str(item.get("label") or ""), "value": str(item.get("value") or "")}
            for item in signal_items[:4]
        ],
    ]
    metric_summary = [
        {
            "axis_key": str(item.get("label") or ""),
            "label": str(item.get("label") or ""),
            "feature_id": str(item.get("source") or ""),
            "display": str(item.get("value") or ""),
            "source": str(item.get("source") or ""),
        }
        for item in signal_items
    ]
    return {
        "headline": headline,
        "explanation": explanation,
        "highlights": highlights,
        "metric_summary": metric_summary,
        "source": "customer-cluster-signal",
        "source_model": "rule-based",
        "citations": [],
    }


def _update_workbench_job(job_id: str, stage_key: str, status: str, detail: str, tone: str = "info", meta: dict[str, object] | None = None) -> None:
    with _workbench_jobs_lock:
        job = _workbench_jobs.get(job_id)
        if job is None:
            return
        stage_timers = dict(job.get("stage_timers") or {})
        now_perf = time.perf_counter()
        stages = [dict(item) for item in (job.get("stages") or [])]
        stage_index = next((index for index, item in enumerate(stages) if item.get("key") == stage_key), -1)
        if stage_index < 0:
            return
        now = _iso_now()
        if status == "running":
            stage_timers.setdefault(stage_key, now_perf)
        stage_started_perf = float(stage_timers.get(stage_key) or now_perf)
        duration_ms = int((now_perf - stage_started_perf) * 1000)
        enriched_meta = dict(meta or {})
        if status == "completed":
            enriched_meta["duration_ms"] = duration_ms
        if status == "running":
            enriched_meta.setdefault("started_elapsed_ms", int((now_perf - float(job.get("started_monotonic") or now_perf)) * 1000))
        for index, item in enumerate(stages):
            if index < stage_index:
                item["status"] = "completed"
                item["progress"] = 100
                item["completed_at"] = item.get("completed_at") or now
            elif index == stage_index:
                item["status"] = status
                item["progress"] = 100 if status == "completed" else 66 if status == "running" else item.get("progress") or 0
                item["started_at"] = item.get("started_at") or now
                if status == "completed":
                    item["completed_at"] = now
                    item["duration_ms"] = duration_ms
                if enriched_meta:
                    item["meta"] = enriched_meta
            elif status == "running":
                item["status"] = "idle"
                item["progress"] = 0
        logs = list(job.get("logs") or [])
        logs.insert(0, _make_job_log(stage_key, detail, tone=tone, meta=enriched_meta))
        job["stages"] = stages
        job["stage_timers"] = stage_timers
        job["logs"] = logs[:18]
        job["updated_at"] = now
        if status == "running":
            job["status"] = "running"
            job["active_stage"] = stage_key
        elif status == "completed" and stage_index == len(stages) - 1:
            job["active_stage"] = stage_key


def _run_workbench_job(
    job_id: str,
    selected_product: str,
    query: str,
    feature_id: str,
    conversation_profile: dict[str, object] | None = None,
) -> None:
    try:
        result = _build_feature_workbench_payload(
            selected_product=selected_product,
            query=query,
            feature_id=feature_id,
            job_id=job_id,
            conversation_profile=conversation_profile,
        )
        with _workbench_jobs_lock:
            job = _workbench_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "completed"
            job["result"] = result
            job["completed_at"] = _iso_now()
            job["updated_at"] = job["completed_at"]
    except Exception as error:
        with _workbench_jobs_lock:
            job = _workbench_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "failed"
            job["error"] = str(error)
            job["completed_at"] = _iso_now()
            job["updated_at"] = job["completed_at"]
            stages = [dict(item) for item in (job.get("stages") or [])]
            active_key = str(job.get("active_stage") or "")
            for item in stages:
                if item.get("key") == active_key:
                    item["status"] = "failed"
                    item["completed_at"] = job["completed_at"]
            job["stages"] = stages
            logs = list(job.get("logs") or [])
            logs.insert(0, _make_job_log(active_key or "runtime", f"workbench query failed: {error}", tone="error"))
            job["logs"] = logs[:18]


def _read_json_file(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else {}


def _read_record_list(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    if isinstance(loaded, dict):
        records = loaded.get("records") or []
        return [item for item in records if isinstance(item, dict)]
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    return []


def _tokenize_text(value: str) -> list[str]:
    return [token for token in re.split(r"[^0-9A-Za-z_\u3131-\u318E\uAC00-\uD7A3]+", str(value or "").lower()) if token]


def _compact_search_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3131-\u318E\uAC00-\uD7A3]+", "", str(value or "").lower())


def _query_asks_influence_features(query: str) -> bool:
    compact_query = _compact_search_text(query)
    return (
        ("영향" in compact_query and ("feature" in compact_query or "피처" in compact_query or "특성" in compact_query))
        or "영향을주는feature" in compact_query
        or "영향feature" in compact_query
        or "중요feature" in compact_query
        or "중요한feature" in compact_query
    )


def _query_asks_cluster_signals(query: str) -> bool:
    compact_query = _compact_search_text(query)
    signal_markers = [
        "강한신호",
        "공통신호",
        "공통으로강한",
        "주요신호",
        "핵심신호",
        "공통특징",
        "주요특징",
        "공통패턴",
        "강한패턴",
        "strongsignal",
    ]
    cluster_markers = ["고객군", "고객군집", "군집", "세그먼트", "승인", "거절", "탈락"]
    return any(marker in compact_query for marker in signal_markers) and any(marker in compact_query for marker in cluster_markers)


def _extract_income_band_focus(query: str) -> str:
    compact_query = _compact_search_text(query)
    for label in ["초고소득", "저소득", "중소득", "고소득"]:
        if label in compact_query:
            return label
    return ""


def _extract_amount_band_focus(query: str) -> str:
    compact_query = _compact_search_text(query)
    for label in ["초대형", "소액", "중액", "고액", "대형"]:
        if label in compact_query:
            return "초대형" if label == "대형" and "초대형" in compact_query else label
    return ""


def _extract_decision_focus(query: str) -> str:
    compact_query = _compact_search_text(query)
    if any(marker in compact_query for marker in ["거절", "탈락", "부결", "reject", "rejected"]):
        return "거절"
    if any(marker in compact_query for marker in ["승인", "통과", "approve", "approved"]):
        return "승인"
    return ""


def _normalize_decision_label(value: str) -> str:
    compact_value = _compact_search_text(value)
    if any(marker in compact_value for marker in ["거절", "탈락", "부결", "reject", "rejected"]):
        return "거절"
    if any(marker in compact_value for marker in ["승인", "통과", "approve", "approved"]):
        return "승인"
    return ""


def _is_meaningful_reject_text(value: str) -> bool:
    compact_value = _compact_search_text(value)
    return compact_value not in {"", "없음", "해당없음", "미해당", "정상", "none", "nan", "null"}


def _infer_product_from_query(query: str) -> dict[str, object]:
    normalized_query = str(query or "").lower()
    compact_query = _compact_search_text(query)
    matches: list[dict[str, object]] = []
    for product_code, aliases in PRODUCT_QUERY_ALIASES.items():
        score = 0
        matched_terms: list[str] = []
        for alias in aliases:
            normalized_alias = str(alias or "").lower()
            compact_alias = _compact_search_text(normalized_alias)
            if normalized_alias and normalized_alias in normalized_query:
                score += max(6, len(compact_alias))
                matched_terms.append(alias)
            elif compact_alias and compact_alias in compact_query:
                score += max(5, len(compact_alias) - 1)
                matched_terms.append(alias)
        if product_code == "C6" and "이지신용" in compact_query and "대출" in compact_query:
            score += 12
            matched_terms.append("이지신용+대출")
        if score:
            matches.append({
                "product": product_code,
                "score": score,
                "matched_terms": _dedupe_text_items(matched_terms, limit=4),
            })
    matches.sort(key=lambda item: (-int(item.get("score") or 0), str(item.get("product") or "")))
    top = matches[0] if matches else {}
    second_score = int(matches[1].get("score") or 0) if len(matches) > 1 else 0
    top_score = int(top.get("score") or 0)
    return {
        "product": str(top.get("product") or ""),
        "confidence": "high" if top_score >= 6 and top_score - second_score >= 4 else ("medium" if top_score else "none"),
        "matches": matches[:3],
    }


def _product_display_name(product_code: str) -> str:
    return PRODUCT_DISPLAY_NAMES.get(str(product_code or "").strip(), str(product_code or "").strip())


def _replace_product_codes_for_display(text: str) -> str:
    rendered = str(text or "")
    if not rendered:
        return ""
    for code, label in PRODUCT_DISPLAY_NAMES.items():
        rendered = re.sub(rf"(?<![\w(]){re.escape(code)}(?![\w)])", label, rendered)
    return rendered


def _is_cross_product_feature_label(label: str, selected_product: str) -> bool:
    text = str(label or "")
    product_code = str(selected_product or "").strip()
    if not text or not product_code:
        return False
    other_product_code_hit = any(code != product_code and code in text for code in PRODUCT_DISPLAY_NAMES)
    return other_product_code_hit or any(term and term in text for term in PRODUCT_CROSS_AXIS_EXCLUSIONS.get(product_code, []))


def _feature_matches_selected_product(feature: dict[str, object], selected_product: str) -> bool:
    product_code = str(selected_product or "").strip()
    if not product_code:
        return True
    products = feature.get("products") or []
    return product_code in products


def _prioritize_reject_representative_features(
    representative_features: list[dict[str, object]],
    selected_product: str,
    all_features: list[dict[str, object]],
) -> list[dict[str, object]]:
    filtered = [
        item for item in representative_features
        if not _is_cross_product_feature_label(str(item.get("feature_name") or item.get("feature_id") or ""), selected_product)
    ]
    reject_code_feature = next(
        (
            item for item in all_features
            if _feature_matches_selected_product(item, selected_product)
            and str(item.get("feature_id") or "") == "decision.reject_reason_code"
        ),
        None,
    )
    reject_reason_feature = next(
        (
            item for item in all_features
            if _feature_matches_selected_product(item, selected_product)
            and "거절" in str(item.get("feature_name") or item.get("feature_id") or "")
            and not _is_cross_product_feature_label(str(item.get("feature_name") or item.get("feature_id") or ""), selected_product)
        ),
        None,
    )
    ordered = [item for item in [reject_code_feature, reject_reason_feature, *filtered] if isinstance(item, dict)]
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in ordered:
        key = str(item.get("feature_id") or item.get("feature_name") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 3:
            break
    return deduped or filtered


def _normalize_keyword_items(items: list[object], limit: int = 24) -> list[str]:
    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        normalized.extend(_tokenize_text(text))
    return _dedupe_text_items(normalized, limit=limit)


def _build_conversation_profile(payload: FeatureOntologyRuntimeRequest) -> dict[str, object]:
    session_id = str(payload.session_id or "").strip()
    turn_id = str(payload.turn_id or "").strip()
    history = list(payload.history or [])[-8:]
    answer_mode = str(getattr(payload, "answer_mode", "") or "general").strip()
    department = str(getattr(payload, "department", "") or "").strip()
    memo_notes = str(getattr(payload, "memo_notes", "") or "").strip()
    memo_seed = [answer_mode, department, memo_notes]
    memory_keywords = _normalize_keyword_items([*list(payload.memory_keywords or []), *memo_seed], limit=32)
    history_queries = [str(item.question or "").strip() for item in history if str(item.question or "").strip()]
    history_feature_ids = _dedupe_text_items([
        str(feature_id).strip()
        for item in history
        for feature_id in (item.selected_feature_ids or [])
        if str(feature_id).strip()
    ], limit=24)
    preferred_feature_ids = _dedupe_text_items([str(item).strip() for item in (payload.feedback.preferred_feature_ids or []) if str(item).strip()], limit=16)
    avoided_feature_ids = _dedupe_text_items([str(item).strip() for item in (payload.feedback.avoided_feature_ids or []) if str(item).strip()], limit=16)

    query_keywords = _tokenize_text(payload.query)
    session_memory_keywords: list[str] = []
    session_history_queries: list[str] = []
    session_history_feature_ids: list[str] = []
    if session_id:
        with _conversation_sessions_lock:
            session = dict(_conversation_sessions.get(session_id) or {})
            session_memory_keywords = [str(item).strip() for item in (session.get("memory_keywords") or []) if str(item).strip()]
            session_history_queries = [str(item).strip() for item in (session.get("history_queries") or []) if str(item).strip()]
            session_history_feature_ids = [str(item).strip() for item in (session.get("history_feature_ids") or []) if str(item).strip()]
            merged_memory_keywords = _dedupe_text_items(session_memory_keywords + memory_keywords + query_keywords, limit=32)
            merged_history_queries = _dedupe_text_items(session_history_queries + history_queries + ([payload.query] if str(payload.query or "").strip() else []), limit=12)
            merged_history_feature_ids = _dedupe_text_items(session_history_feature_ids + history_feature_ids + preferred_feature_ids, limit=24)
            _conversation_sessions[session_id] = {
                "memory_keywords": merged_memory_keywords,
                "history_queries": merged_history_queries,
                "history_feature_ids": merged_history_feature_ids,
                "answer_mode": answer_mode,
                "department": department,
                "memo_notes": memo_notes,
                "updated_at": _iso_now(),
            }
        memory_keywords = _dedupe_text_items(session_memory_keywords + memory_keywords + query_keywords, limit=32)
        history_queries = _dedupe_text_items(session_history_queries + history_queries, limit=12)
        history_feature_ids = _dedupe_text_items(session_history_feature_ids + history_feature_ids, limit=24)
    else:
        memory_keywords = _dedupe_text_items(memory_keywords + query_keywords, limit=32)

    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "memory_keywords": memory_keywords,
        "history_queries": history_queries,
        "history_feature_ids": history_feature_ids,
        "preferred_feature_ids": preferred_feature_ids,
        "avoided_feature_ids": avoided_feature_ids,
        "allow_clarification": bool(payload.allow_clarification),
        "clarification_budget": int(payload.clarification_budget),
        "answer_mode": answer_mode,
        "department": department,
        "memo_notes": memo_notes,
    }


def _score_feature_conversation_adjustment(feature: dict, conversation_profile: dict[str, object] | None = None) -> tuple[float, list[str]]:
    profile = conversation_profile or {}
    feature_id = str(feature.get("feature_id") or "").strip()
    if not feature_id:
        return 0.0, []
    score = 0.0
    reasons: list[str] = []
    preferred = {str(item).strip() for item in (profile.get("preferred_feature_ids") or []) if str(item).strip()}
    avoided = {str(item).strip() for item in (profile.get("avoided_feature_ids") or []) if str(item).strip()}
    history_feature_ids = {str(item).strip() for item in (profile.get("history_feature_ids") or []) if str(item).strip()}
    memory_keywords = [str(item).strip().lower() for item in (profile.get("memory_keywords") or []) if str(item).strip()]
    haystack = _feature_search_document(feature).lower()

    if feature_id in preferred:
        score += 6.0
        reasons.append("feedback.preferred")
    if feature_id in avoided:
        score -= 7.0
        reasons.append("feedback.avoided")
    if feature_id in history_feature_ids:
        score += 2.2
        reasons.append("history.feature-hit")

    keyword_hits = 0
    for token in memory_keywords[:24]:
        if token and token in haystack:
            keyword_hits += 1
    if keyword_hits:
        keyword_bonus = min(4.0, keyword_hits * 0.45)
        score += keyword_bonus
        reasons.append(f"memory.keyword-hit:{keyword_hits}")

    return round(float(score), 4), reasons[:4]


def _build_clarification_suggestion(
    query: str,
    ranked_features: list[tuple[float, dict]],
    primary_feature_selection: dict[str, object],
    conversation_profile: dict[str, object] | None = None,
    product_resolution: dict[str, object] | None = None,
    selected_product: str = "",
) -> dict[str, object]:
    profile = conversation_profile or {}
    allow_clarification = bool(profile.get("allow_clarification", True))
    clarification_budget = int(profile.get("clarification_budget") or 0)
    if not allow_clarification or clarification_budget <= 0:
        return {
            "needed": False,
            "reason": "clarification-disabled",
            "question": "",
            "options": [],
        }

    product_resolution = product_resolution or {}
    product_confidence = str(product_resolution.get("confidence") or "none")
    product_matches = [
        item for item in (product_resolution.get("matches") or [])
        if isinstance(item, dict) and str(item.get("product") or "").strip()
    ]
    if product_matches and product_confidence != "high":
        options = [
            {
                "type": "product",
                "product": str(item.get("product") or ""),
                "feature_id": str(item.get("product") or ""),
                "feature_name": f"상품 {str(item.get('product') or '')}",
                "axis_key": "product",
                "matched_terms": list(item.get("matched_terms") or []),
            }
            for item in product_matches[:3]
        ]
        return {
            "needed": True,
            "type": "product",
            "reason": "ambiguous-product-scope",
            "question": "질문에서 말한 상품 범위를 먼저 확인할까요?",
            "options": options,
            "budget_left": max(0, clarification_budget - 1),
        }

    top_ranked = ranked_features[:2]
    top_k = list((primary_feature_selection or {}).get("top_k") or [])[:3]
    if len(top_ranked) < 2 or not top_k:
        return {
            "needed": False,
            "reason": "insufficient-candidates",
            "question": "",
            "options": [],
        }

    top_score = float(top_ranked[0][0])
    second_score = float(top_ranked[1][0])
    score_gap = top_score - second_score
    short_query = len(_tokenize_text(query)) <= 2
    representative_features = [
        item for item in (primary_feature_selection or {}).get("representative_features") or []
        if isinstance(item, dict) and str(item.get("feature_id") or "").strip()
    ]
    representative_axis_keys = {
        str(item.get("axis_key") or _infer_feature_axis_key(item))
        for item in representative_features
    }
    top_axis_keys = {
        str(item.get("axis_key") or "")
        for item in top_k
        if str(item.get("axis_key") or "").strip()
    }
    explicit_metric_query = bool(_query_has_rate_intent(query) or _query_has_limit_intent(query))
    mixed_axis_candidates = len(top_axis_keys) >= 2
    ambiguous = score_gap < 2.6 or short_query or (mixed_axis_candidates and len(representative_axis_keys) <= 1 and not explicit_metric_query)
    if not ambiguous:
        return {
            "needed": False,
            "reason": "confident-selection",
            "question": "",
            "options": [],
            "score_gap": round(score_gap, 4),
        }

    options = [
        {
            "feature_id": str(item.get("feature_id") or ""),
            "feature_name": str(item.get("feature_name") or item.get("feature_id") or "feature"),
            "axis_key": str(item.get("axis_key") or ""),
        }
        for item in top_k[:3]
        if str(item.get("feature_id") or "").strip()
    ]
    if not options:
        return {
            "needed": False,
            "reason": "no-option",
            "question": "",
            "options": [],
            "score_gap": round(score_gap, 4),
        }
    option_labels = [str(item.get("feature_name") or "") for item in options[:2] if str(item.get("feature_name") or "")]
    question_text = "어떤 기준으로 다시 볼까요?"
    if option_labels:
        question_text = f"{', '.join(option_labels)} 중 어떤 기준으로 볼까요?"
    return {
        "needed": True,
        "type": "feature",
        "reason": "mixed-axis-candidates" if mixed_axis_candidates else ("small-score-gap" if score_gap < 2.6 else "short-query"),
        "question": question_text,
        "options": options,
        "score_gap": round(score_gap, 4),
        "budget_left": max(0, clarification_budget - 1),
    }


def _get_reject_code_mapping() -> dict[str, dict[str, str]]:
    return load_reject_code_mapping(ROOT / "data")


def _query_has_reject_intent(query: str, selected_feature: dict | None = None) -> bool:
    tokens = set(_tokenize_text(query))
    feature_text = " ".join([
        str((selected_feature or {}).get("feature_id") or ""),
        str((selected_feature or {}).get("feature_name") or ""),
        str((selected_feature or {}).get("category") or ""),
        " ".join(str(item) for item in ((selected_feature or {}).get("aliases") or [])),
    ]).lower()
    reject_tokens = {
        "거절", "거절사유", "거절코드", "탈락", "탈락사유", "사유", "불합격", "부결",
        "k코드", "ko", "knock", "knockout", "reject", "decision", "심사", "컷오프",
    }
    if tokens & reject_tokens:
        return True
    query_text = str(query or "").lower()
    if any(marker in query_text for marker in ["k코드", "knock-out", "knock out", "reject code", "거절 사유", "탈락 사유", "탈락사유"]):
        return True
    return any(marker in feature_text for marker in ["reject", "거절", "탈락", "사유", "knock", "k코드"])


def _score_reject_feature_boost(feature: dict, query: str) -> int:
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    category = str(feature.get("category") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, category, description, aliases])
    if not _query_has_reject_intent(query):
        return 0
    boost = 0
    if feature_id == "decision.reject_reason_code":
        boost += 44
    if any(marker in haystack for marker in ["reject_reason_code", "거절사유코드", "거절사유", "탈락사유", "k코드"]):
        boost += 26
    if any(marker in haystack for marker in ["knock-out", "knockout", "knock-out여부", "탈락", "취급불가사유"]):
        boost += 18
    if "decision" in category:
        boost += 5
    if "reject" in haystack or "거절" in haystack or "탈락" in haystack or "knock" in haystack:
        boost += 9
    return boost


def _score_reject_feature_routing(feature: dict) -> int:
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, description, aliases])
    score = 0
    if feature_id == "decision.reject_reason_code":
        score += 60
    if any(marker in haystack for marker in ["reject_reason_code", "거절사유코드", "거절사유", "탈락사유", "k코드"]):
        score += 28
    if any(marker in haystack for marker in ["reject", "거절", "탈락", "취급불가", "knock"]):
        score += 18
    return score


def _query_has_employment_intent(query: str) -> bool:
    tokens = set(_tokenize_text(query))
    employment_tokens = {
        "직장", "직장인", "직업", "재직", "회사", "회사원", "근로", "고용", "직위", "사업장",
    }
    if tokens & employment_tokens:
        return True
    query_text = str(query or "").lower()
    return any(marker in query_text for marker in employment_tokens)


def _query_has_age_intent(query: str) -> bool:
    age_markers = {"나이", "연령", "연령대", "age"}
    tokens = _tokenize_text(query)
    if any(any(marker in token for marker in age_markers) for token in tokens):
        return True
    query_text = str(query or "").lower()
    return any(marker in query_text for marker in age_markers)


def _extract_age_band_focus(query: str) -> str:
    query_text = str(query or "")
    explicit_band = re.search(r"([2-6]0)\s*대", query_text)
    if explicit_band:
        return f"{explicit_band.group(1)}대"
    explicit_age = re.search(r"(?<!\d)([2-6]\d)(?!\d)", query_text)
    if explicit_age:
        age = int(explicit_age.group(1))
        return f"{(age // 10) * 10}대"
    return ""


def _score_age_feature_boost(feature: dict, query: str) -> int:
    if not _query_has_age_intent(query):
        return 0
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    category = str(feature.get("category") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, category, description, aliases])
    boost = 0
    if feature_id == "applicant.age":
        boost += 18
    if any(marker in haystack for marker in ["연령", "나이", "applicant.age"]):
        boost += 10
    if "applicant" in category:
        boost += 4
    return boost


def _query_has_rate_intent(query: str) -> bool:
    rate_markers = {"금리", "이자율", "rate"}
    tokens = _tokenize_text(query)
    if any(any(marker in token for marker in rate_markers) for token in tokens):
        return True
    query_text = str(query or "").lower()
    return any(marker in query_text for marker in rate_markers)


def _score_rate_feature_boost(feature: dict, query: str) -> int:
    if not _query_has_rate_intent(query):
        return 0
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    category = str(feature.get("category") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, category, description, aliases])
    boost = 0
    if feature_id == "decision.applied_rate":
        boost += 18
    if any(marker in haystack for marker in ["금리", "이자율", "applied_rate", "적용금리", "산출금리"]):
        boost += 12
    if "decision" in category:
        boost += 4
    return boost


def _query_has_limit_intent(query: str) -> bool:
    limit_markers = {"한도", "limit", "가능금액", "대출가능금액"}
    tokens = _tokenize_text(query)
    if any(any(marker in token for marker in limit_markers) for token in tokens):
        return True
    query_text = str(query or "").lower()
    return any(marker in query_text for marker in limit_markers)


def _infer_limit_focus(query: str) -> str:
    query_text = str(query or "").lower()
    requested_markers = ["요청", "신청", "희망", "원하는", "requested", "requested_limit", "입력한 한도"]
    approved_markers = ["승인", "승인금액", "승인한도", "가능금액", "대출가능금액", "가능 한도", "나오는", "실제", "평균", "산출"]
    if any(marker in query_text for marker in requested_markers):
        return "requested"
    if any(marker in query_text for marker in approved_markers):
        return "approved"
    return "approved"


def _score_limit_feature_boost(feature: dict, query: str) -> int:
    if not _query_has_limit_intent(query):
        return 0
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    category = str(feature.get("category") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, category, description, aliases])
    limit_focus = _infer_limit_focus(query)
    boost = 0
    if feature_id == "decision.approved_amount":
        boost += 20 if limit_focus == "approved" else 8
    if feature_id == "loan.requested_limit":
        boost += 20 if limit_focus == "requested" else 6
    if any(marker in haystack for marker in ["한도", "limit", "가능금액", "승인가능금액", "대출가능금액", "requested_limit"]):
        boost += 10
    if category in {"decision", "loan"}:
        boost += 4
    return boost


def _score_employment_feature_boost(feature: dict, query: str) -> int:
    if not _query_has_employment_intent(query):
        return 0
    haystack = " ".join([
        str(feature.get("feature_id") or "").lower(),
        str(feature.get("feature_name") or "").lower(),
        str(feature.get("category") or "").lower(),
        str(feature.get("description") or "").lower(),
        " ".join(str(item) for item in (feature.get("aliases") or [])).lower(),
    ])
    boost = 0
    if any(marker in haystack for marker in ["직업", "occupation", "job"]):
        boost += 12
    if any(marker in haystack for marker in ["직장", "회사", "재직", "근로", "고용"]):
        boost += 10
    if any(marker in haystack for marker in ["직위", "종업원", "기업형태", "기업공개", "등록경과"]):
        boost += 6
    if any(marker in haystack for marker in ["annual_income", "연소득", "소득", "매출금액"]):
        boost -= 8
    return boost


def _score_employment_feature_routing(feature: dict) -> int:
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, description, aliases])
    score = 0
    if any(marker in feature_id or marker in feature_name for marker in ["직업", "직위", "재직", "현직장", "kcb직업", "occupation"]):
        score += 30
    if any(marker in haystack for marker in ["직업", "직장", "재직", "근로", "고용", "회사", "직위", "사업장"]):
        score += 16
    if any(marker in haystack for marker in ["종업원", "기업형태", "기업공개", "휴폐업", "등록경과"]):
        score += 8
    if any(marker in haystack for marker in ["annual_income", "연소득", "소득", "매출금액"]):
        score -= 18
    return score


def _score_rate_feature_routing(feature: dict) -> int:
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, description, aliases])
    score = 0
    if feature_id == "decision.applied_rate":
        score += 30
    if any(marker in haystack for marker in ["금리", "이자율", "적용금리", "산출금리", "applied_rate"]):
        score += 18
    return score


def _score_limit_feature_routing(feature: dict, query: str = "") -> int:
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    description = str(feature.get("description") or "").lower()
    aliases = " ".join(str(item) for item in (feature.get("aliases") or [])).lower()
    haystack = " ".join([feature_id, feature_name, description, aliases])
    limit_focus = _infer_limit_focus(query)
    score = 0
    if feature_id == "decision.approved_amount":
        score += 32 if limit_focus == "approved" else 16
    if feature_id == "loan.requested_limit":
        score += 30 if limit_focus == "requested" else 12
    if any(marker in haystack for marker in ["한도", "가능금액", "대출가능금액", "승인가능금액", "한도금액", "requested_limit"]):
        score += 14
    return score


def _feature_token_hit_breakdown(feature: dict, query: str) -> dict[str, object]:
    tokens = _tokenize_text(query)
    haystack = " ".join([
        str(feature.get("feature_id") or ""),
        str(feature.get("feature_name") or ""),
        str(feature.get("category") or ""),
        str(feature.get("description") or ""),
        " ".join(str(item) for item in (feature.get("aliases") or [])),
        " ".join(str(item) for item in (feature.get("products") or [])),
    ]).lower()
    feature_id = str(feature.get("feature_id") or "").lower()
    feature_name = str(feature.get("feature_name") or "").lower()
    haystack_tokens: list[str] = []
    feature_id_tokens: list[str] = []
    feature_name_tokens: list[str] = []
    for token in tokens:
      if token in haystack:
          haystack_tokens.append(token)
      if token in feature_id:
          feature_id_tokens.append(token)
      if token in feature_name:
          feature_name_tokens.append(token)
    return {
        "tokens": tokens,
        "haystack_tokens": haystack_tokens,
        "feature_id_tokens": feature_id_tokens,
        "feature_name_tokens": feature_name_tokens,
        "haystack_hits": len(haystack_tokens),
        "feature_id_hits": len(feature_id_tokens),
        "feature_name_hits": len(feature_name_tokens),
    }


def _compute_feature_semantic_scores(features: list[dict], query: str) -> dict[str, float]:
    if not query.strip():
        return {}
    feature_ids, matrix = _get_feature_embedding_matrix(features)
    if matrix is None or matrix.size == 0:
        return {}
    try:
        embeddings = get_embeddings()
        query_vector = np.asarray(embeddings.embed_query(query), dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if math.isclose(query_norm, 0.0):
            return {}
        query_vector = query_vector / query_norm
        similarities = matrix @ query_vector
        return {
            feature_ids[index]: float(score)
            for index, score in enumerate(similarities)
        }
    except Exception:
        return {}


def _build_feature_rank_breakdown(
    feature: dict,
    query: str,
    selected_product: str,
    semantic_score: float | None = None,
    combined_score: float | None = None,
    conversation_adjustment: float = 0.0,
    conversation_reasons: list[str] | None = None,
) -> dict[str, object]:
    token_breakdown = _feature_token_hit_breakdown(feature, query)
    coverage_bonus = min(6, int((feature.get("coverage") or {}).get("mapping_count") or 0) // 4)
    product_bonus = 3 if selected_product and selected_product in (feature.get("products") or []) else 0
    reject_boost = _score_reject_feature_boost(feature, query)
    employment_boost = _score_employment_feature_boost(feature, query)
    age_boost = _score_age_feature_boost(feature, query)
    rate_boost = _score_rate_feature_boost(feature, query)
    limit_boost = _score_limit_feature_boost(feature, query)
    lexical_score = (
        int(token_breakdown["haystack_hits"]) * 4
        + int(token_breakdown["feature_id_hits"]) * 5
        + int(token_breakdown["feature_name_hits"]) * 6
        + product_bonus
        + coverage_bonus
        + reject_boost
        + employment_boost
        + age_boost
        + rate_boost
        + limit_boost
    )
    semantic_value = float(semantic_score) if semantic_score is not None else None
    semantic_component = semantic_value * 20.0 if semantic_value is not None else 0.0
    conversation_component = float(conversation_adjustment or 0.0)
    return {
        "semantic_score": round(float(semantic_value), 4) if semantic_value is not None else None,
        "semantic_component": round(float(semantic_component), 4),
        "conversation_component": round(float(conversation_component), 4),
        "conversation_reasons": list(conversation_reasons or []),
        "lexical_score": lexical_score,
        "token_haystack_hits": int(token_breakdown["haystack_hits"]),
        "feature_id_hits": int(token_breakdown["feature_id_hits"]),
        "feature_name_hits": int(token_breakdown["feature_name_hits"]),
        "matched_tokens": list(token_breakdown["tokens"]),
        "matched_haystack_tokens": list(token_breakdown["haystack_tokens"]),
        "matched_feature_id_tokens": list(token_breakdown["feature_id_tokens"]),
        "matched_feature_name_tokens": list(token_breakdown["feature_name_tokens"]),
        "product_bonus": product_bonus,
        "coverage_bonus": coverage_bonus,
        "coverage_mapping_count": int((feature.get("coverage") or {}).get("mapping_count") or 0),
        "reject_boost": reject_boost,
        "employment_boost": employment_boost,
        "age_boost": age_boost,
        "rate_boost": rate_boost,
        "limit_boost": limit_boost,
        "combined_score": round(float(combined_score), 4) if combined_score is not None else round(float(semantic_component + lexical_score + conversation_component), 4),
        "formula": {
            "semantic_weight": 20,
            "haystack_hit_weight": 4,
            "feature_id_hit_weight": 5,
            "feature_name_hit_weight": 6,
            "product_match_bonus": 3,
            "coverage_bonus_cap": 6,
            "employment_boost_mode": "rule-based",
            "conversation_adjustment_mode": "feedback+memory",
        },
    }


def _build_query_token_feature_mapping(
    query: str,
    selected_product: str,
    ranked_features: list[tuple[float, dict]],
    primary_feature_selection: dict[str, object],
    all_features: list[dict] | None = None,
) -> list[dict[str, object]]:
    query_tokens = _tokenize_text(query)[:8]
    top_candidates = list((primary_feature_selection or {}).get("top_k") or [])[:3]
    ranked_feature_pool = [feature for _, feature in ranked_features[:200]]
    employment_feature_pool = list(all_features or ranked_feature_pool)
    semantic_feature_pool = list(all_features or ranked_feature_pool)
    mappings: list[dict[str, object]] = []

    for index, token in enumerate(query_tokens):
        normalized_token = str(token or "").strip().lower()
        feature_links: list[dict[str, object]] = []
        for candidate in top_candidates:
            matched_in: list[str] = []
            if normalized_token and normalized_token in {str(item).lower() for item in (candidate.get("matched_feature_name_tokens") or [])}:
                matched_in.append("feature_name")
            if normalized_token and normalized_token in {str(item).lower() for item in (candidate.get("matched_feature_id_tokens") or [])}:
                matched_in.append("feature_id")
            if normalized_token and normalized_token in {str(item).lower() for item in (candidate.get("matched_tokens") or [])}:
                matched_in.append("alias_or_description")
            if not matched_in:
                continue
            feature_links.append({
                "feature_id": str(candidate.get("feature_id") or ""),
                "feature_name": str(candidate.get("feature_name") or candidate.get("feature_id") or "feature"),
                "rank": int(candidate.get("rank") or 0),
                "hybrid_score": round(float(candidate.get("hybrid_score") or 0.0), 4),
                "matched_in": matched_in,
            })

        signal_type = "context-only"
        concept_label = ""
        if any(marker in normalized_token for marker in ["카드론", "대출"]):
            signal_type = "product"
            concept_label = selected_product or "ALL"
            product_feature = next(
                (
                    feature for feature in semantic_feature_pool
                    if str(feature.get("feature_id") or "") == "application.product_code"
                ),
                None,
            )
            if product_feature is not None and selected_product:
                feature_links.append({
                    "feature_id": "application.product_code",
                    "feature_name": _product_display_name(selected_product) or str(product_feature.get("feature_name") or "상품코드"),
                    "rank": 0,
                    "hybrid_score": 1.0,
                    "matched_in": ["product_alias"],
                })
        elif re.search(r"\d", normalized_token) or normalized_token.endswith("대"):
            signal_type = "age"
            concept_label = "applicant.age"
        elif any(marker in normalized_token for marker in ["금리", "이자율", "rate"]):
            signal_type = "rate"
            concept_label = "decision.applied_rate"
        elif any(marker in normalized_token for marker in ["한도", "가능금액", "limit"]):
            signal_type = "limit"
            concept_label = "loan.requested_limit" if _infer_limit_focus(query) == "requested" else "decision.approved_amount"
        elif any(marker in normalized_token for marker in ["거절", "부결", "탈락", "사유", "불합격", "reject", "knock"]):
            signal_type = "reject"
            concept_label = "decision.reject_reason_code"
        elif any(marker in normalized_token for marker in ["승인"]):
            signal_type = "decision"
            concept_label = "decision"
        elif any(marker in normalized_token for marker in ["직장", "직업", "재직", "회사", "근로", "고용", "직위"]):
            signal_type = "employment"
            concept_label = "employment"

        if feature_links:
            feature_links = [sorted(feature_links, key=lambda item: (-float(item.get("hybrid_score") or 0.0), int(item.get("rank") or 99)))[0]]
        elif signal_type == "employment":
            employment_candidates = [
                feature for feature in employment_feature_pool
                if _score_employment_feature_routing(feature) > 0
            ]
            employment_feature = (
                max(
                    employment_candidates,
                    key=lambda feature: (
                        _score_employment_feature_routing(feature) + (3 if selected_product and selected_product in (feature.get("products") or []) else 0),
                        int((feature.get("coverage") or {}).get("mapping_count") or 0),
                    ),
                )
                if employment_candidates
                else None
            )
            if employment_feature is not None:
                feature_links = [{
                    "feature_id": str(employment_feature.get("feature_id") or ""),
                    "feature_name": str(employment_feature.get("feature_name") or employment_feature.get("feature_id") or "feature"),
                    "rank": 0,
                    "hybrid_score": 0.0,
                    "matched_in": ["employment_domain"],
                }]
        elif signal_type == "rate":
            rate_candidates = [
                feature for feature in semantic_feature_pool
                if _score_rate_feature_routing(feature) > 0
            ]
            rate_feature = max(rate_candidates, key=_score_rate_feature_routing) if rate_candidates else None
            if rate_feature is not None:
                feature_links = [{
                    "feature_id": str(rate_feature.get("feature_id") or ""),
                    "feature_name": str(rate_feature.get("feature_name") or rate_feature.get("feature_id") or "feature"),
                    "rank": 0,
                    "hybrid_score": 0.0,
                    "matched_in": ["rate_domain"],
                }]
        elif signal_type == "limit":
            limit_candidates = [
                feature for feature in semantic_feature_pool
                if _score_limit_feature_routing(feature, query) > 0
            ]
            limit_feature = max(limit_candidates, key=lambda feature: _score_limit_feature_routing(feature, query)) if limit_candidates else None
            if limit_feature is not None:
                feature_links = [{
                    "feature_id": str(limit_feature.get("feature_id") or ""),
                    "feature_name": str(limit_feature.get("feature_name") or limit_feature.get("feature_id") or "feature"),
                    "rank": 0,
                    "hybrid_score": 0.0,
                    "matched_in": ["limit_domain"],
                }]
        elif signal_type == "reject":
            reject_candidates = [
                feature for feature in semantic_feature_pool
                if _score_reject_feature_routing(feature) > 0
            ]
            reject_feature = max(reject_candidates, key=_score_reject_feature_routing) if reject_candidates else None
            if reject_feature is not None:
                feature_links = [{
                    "feature_id": str(reject_feature.get("feature_id") or ""),
                    "feature_name": str(reject_feature.get("feature_name") or reject_feature.get("feature_id") or "feature"),
                    "rank": 0,
                    "hybrid_score": 0.0,
                    "matched_in": ["reject_domain"],
                }]

        if feature_links:
            matched_area_labels = {
                "feature_name": "feature 이름",
                "feature_id": "feature ID",
                "alias_or_description": "별칭/설명",
                "employment_domain": "직업/재직 도메인",
                "rate_domain": "금리 도메인",
                "limit_domain": "한도 도메인",
                "reject_domain": "거절/탈락 사유 도메인",
            }
            matched_areas: list[str] = []
            for link in feature_links:
                for area in link.get("matched_in") or []:
                    area_label = matched_area_labels.get(str(area), str(area))
                    if area_label not in matched_areas:
                        matched_areas.append(area_label)
            reason = (
                f"우선 연결 후보 {', '.join(link['feature_name'] for link in feature_links[:1])}의 "
                f"{', '.join(matched_areas) or '검색 문맥'}에 직접 닿는 토큰입니다."
            )
        elif signal_type == "product":
            reason = "상품 범위를 정하는 토큰입니다. feature에 직접 붙이기보다 Ontology Domain Select에서 검색 범위를 먼저 좁힙니다."
        elif signal_type == "age":
            reason = "연령대 표현입니다. 직접 hit가 없어도 Intent Router에서 연령 맥락 신호로 사용됩니다."
        elif signal_type == "decision":
            reason = "심사 결과 맥락 토큰입니다. 승인/거절 방향 해석에 쓰이지만 특정 feature에 직접 붙는 단어는 아닐 수 있습니다."
        elif signal_type == "reject":
            reason = "거절/탈락 사유 맥락 토큰입니다. 직접 hit가 약하면 거절사유코드 feature로 우선 라우팅합니다."
        elif signal_type == "employment":
            reason = "직업/재직 맥락 토큰입니다. 직접 hit가 약하면 고용 도메인 feature 1개로 우선 라우팅합니다."
        elif signal_type == "rate":
            reason = "금리 맥락 토큰입니다. 직접 hit가 약하면 금리 도메인 feature 1개로 우선 라우팅합니다."
        elif signal_type == "limit":
            reason = "한도 맥락 토큰입니다. 직접 hit가 약하면 한도 도메인 feature 1개로 우선 라우팅합니다."
        else:
            reason = "직접 hit가 없어 문장 전체 의미를 만드는 보조 토큰으로만 반영됩니다."

        mappings.append({
            "id": f"token-{index}",
            "token": token,
            "signal_type": signal_type,
            "concept_label": concept_label,
            "direct_feature_match": bool(feature_links),
            "primary_label": str(feature_links[0]["feature_name"] if feature_links else concept_label),
            "feature_links": feature_links[:1],
            "reason": reason,
        })

    return mappings


def _get_reject_code_descriptions(codes: list[str], reject_code_mapping: dict[str, dict[str, str]]) -> list[str]:
    descriptions: list[str] = []
    for code in codes:
        description = str((reject_code_mapping.get(str(code).strip().upper()) or {}).get("description") or "").strip()
        if description:
            descriptions.append(description)
    return descriptions


def _parse_named_lines(*texts: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for text in texts:
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            fragments = [line]
            if "," in line:
                fragments.extend(part.strip() for part in line.split(","))
            for fragment in fragments:
                if ":" not in fragment:
                    continue
                normalized_fragment = fragment[1:].strip() if fragment.startswith("-") else fragment
                label, value = normalized_fragment.split(":", 1)
                label = label.strip()
                value = value.strip()
                if label and value:
                    parsed[label] = value
    return parsed


def _to_number(value: str | None) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text or text in {"-", "nan", "None"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _bucketize(value: float | None, thresholds: list[tuple[float, str]], fallback: str = "미상") -> str:
    if value is None:
        return fallback
    for limit, label in thresholds:
        if value <= limit:
            return label
    return thresholds[-1][1] if thresholds else fallback


def _extract_scaled_measure(fields: dict[str, str], source_specs: list[tuple[str, int | None, int]]) -> tuple[float | None, str, float | None]:
    for field_name, small_value_threshold, multiplier in source_specs:
        raw_value = _to_number(fields.get(field_name))
        if raw_value is None:
            continue
        normalized_value = raw_value
        if small_value_threshold is not None and abs(normalized_value) < small_value_threshold:
            normalized_value *= multiplier
        return normalized_value, field_name, raw_value
    return None, "", None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return float(ordered[index])


def _round_threshold(value: float) -> int:
    magnitude = abs(float(value))
    if magnitude >= 10_000_000:
        step = 1_000_000
    elif magnitude >= 1_000_000:
        step = 100_000
    else:
        step = 10_000
    return max(step, int(round(value / step) * step))


def _derive_measure_band_thresholds(values: list[float], labels: list[str], fallback: list[tuple[float, str]]) -> list[tuple[float, str]]:
    if len(values) < 32:
        return list(fallback)
    quantiles = [0.25, 0.65, 0.85]
    limits: list[int] = []
    previous_limit = 0
    for quantile in quantiles:
        percentile_value = _percentile(values, quantile)
        if percentile_value is None:
            return list(fallback)
        rounded = _round_threshold(percentile_value)
        if rounded <= previous_limit:
            rounded = previous_limit + max(1_000, previous_limit // 10 or 1_000)
        limits.append(rounded)
        previous_limit = rounded
    return [
        (limits[0], labels[0]),
        (limits[1], labels[1]),
        (limits[2], labels[2]),
        (999_999_999_999, labels[3]),
    ]


def _serialize_thresholds(thresholds: list[tuple[float, str]]) -> list[dict[str, object]]:
    return [
        {"max_value": int(limit), "label": label}
        for limit, label in thresholds
    ]


def _format_krw_compact(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isclose(numeric_value, 0.0):
        return "0원"
    if abs(numeric_value) >= 10_000:
        manwon = numeric_value / 10_000
        if manwon >= 10_000:
            return f"약 {manwon / 10_000:.1f}억원"
        return f"약 {manwon:,.0f}만원"
    return f"약 {numeric_value:,.0f}원"


def _apply_profile_bands(profile: dict[str, object], income_thresholds: list[tuple[float, str]], amount_thresholds: list[tuple[float, str]]) -> dict[str, object]:
    profile["age_band"] = _bucketize(float(profile.get("age")) if profile.get("age") is not None else None, AGE_BAND_THRESHOLDS)
    profile["income_band"] = _bucketize(float(profile.get("income")) if profile.get("income") is not None else None, income_thresholds)
    profile["amount_band"] = _bucketize(float(profile.get("amount")) if profile.get("amount") is not None else None, amount_thresholds)
    profile["cluster_key"] = "|".join([
        str(profile.get("product") or "").strip(),
        str(profile.get("decision") or "미상").strip() or "미상",
        str(profile.get("age_band") or "미상"),
        str(profile.get("income_band") or "미상"),
        str(profile.get("amount_band") or "미상"),
        str((profile.get("reject_codes") or [None])[0] or ("정상" if not profile.get("reject_reason_text") else "사유있음")),
    ])
    return profile


def _format_cluster_reject_reason_summary(cluster: dict[str, object], limit: int = 2) -> str:
    descriptions = [
        str(item).strip()
        for item in (cluster.get("top_reject_descriptions") or [])
        if str(item).strip()
    ][:limit]
    if descriptions:
        return ", ".join(descriptions)
    codes = [
        str(item.get("code") or "").strip()
        for item in (cluster.get("top_reject_codes") or [])
        if isinstance(item, dict) and str(item.get("code") or "").strip()
    ][:limit]
    return ", ".join(codes)


def _build_reject_code_distribution(
    records_or_profiles: list[dict],
    selected_product: str,
    query: str,
    limit: int = 3,
) -> list[dict[str, object]]:
    """
    records_or_profiles: list of raw records or already-built profiles
    """
    def _is_profile(obj):
        return isinstance(obj, dict) and (
            "product" in obj and ("age" in obj or "age_band" in obj)
        )

    if records_or_profiles and all(_is_profile(r) for r in records_or_profiles):
        # 이미 profiles가 들어온 경우
        prepared_profiles = records_or_profiles
        reject_code_mapping = _get_reject_code_mapping()
        age_band_focus = _extract_age_band_focus(query)
    else:
        reject_code_mapping = _get_reject_code_mapping()
        age_band_focus = _extract_age_band_focus(query)
        profiles = [_build_record_profile(record, reject_code_mapping=reject_code_mapping) for record in records_or_profiles]
        income_thresholds = _derive_measure_band_thresholds(
            [float(item.get("income")) for item in profiles if item.get("income") is not None],
            ["저소득", "중소득", "고소득", "초고소득"],
            DEFAULT_INCOME_BAND_THRESHOLDS,
        )
        amount_thresholds = _derive_measure_band_thresholds(
            [float(item.get("amount")) for item in profiles if item.get("amount") is not None],
            ["소액", "중액", "고액", "초대형"],
            DEFAULT_AMOUNT_BAND_THRESHOLDS,
        )
        prepared_profiles = [
            _apply_profile_bands(dict(profile), income_thresholds, amount_thresholds)
            for profile in profiles
        ]

    def summarize(use_age_filter: bool) -> list[dict[str, object]]:
        counter: collections.Counter[str] = collections.Counter()
        rejected_count = 0
        for profile in prepared_profiles:
            if selected_product and profile.get("product") != selected_product:
                continue
            if use_age_filter and age_band_focus and str(profile.get("age_band") or "") != age_band_focus:
                continue
            codes = [
                str(code).strip().upper()
                for code in (profile.get("reject_codes") or [])
                if re.match(r"^K\d{3}$", str(code).strip().upper())
            ]
            if not codes:
                continue
            rejected_count += 1
            counter.update(codes)

        summary: list[dict[str, object]] = []
        for code, count in counter.most_common(limit):
            description = str((reject_code_mapping.get(code) or {}).get("description") or "").strip()
            summary.append({
                "code": code,
                "count": count,
                "description": description,
                "share": round(count / rejected_count, 4) if rejected_count else 0,
                "base_rejected_records": rejected_count,
                "age_filter": age_band_focus or "",
                "age_filter_used": bool(use_age_filter and age_band_focus),
                "scope": "age_band" if use_age_filter and age_band_focus else "product",
            })
        return summary

    age_summary = summarize(True)
    if age_summary or not age_band_focus:
        return age_summary
    # 1차 fallback: product 기준
    fallback_summary = summarize(False)
    for item in fallback_summary:
        item["fallback_reason"] = "no_age_band_reject_codes"
    if fallback_summary:
        return fallback_summary
    # 2차 fallback: 모든 거절 레코드에서 K코드 집계
    def summarize_any_reject():
        counter: collections.Counter[str] = collections.Counter()
        rejected_count = 0
        for profile in prepared_profiles:
            codes = [
                str(code).strip().upper()
                for code in (profile.get("reject_codes") or [])
                if re.match(r"^K\d{3}$", str(code).strip().upper())
            ]
            if not codes:
                continue
            rejected_count += 1
            counter.update(codes)
        summary: list[dict[str, object]] = []
        for code, count in counter.most_common(limit):
            description = str((reject_code_mapping.get(code) or {}).get("description") or "").strip()
            summary.append({
                "code": code,
                "count": count,
                "description": description,
                "share": round(count / rejected_count, 4) if rejected_count else 0,
                "base_rejected_records": rejected_count,
                "age_filter": "",
                "age_filter_used": False,
                "scope": "all_rejects",
                "fallback_reason": "no_product_reject_codes",
            })
        return summary
    any_reject_summary = summarize_any_reject()
    return any_reject_summary


def _extract_normalized_record_measures(record: dict) -> tuple[float | None, str, float | None, float | None, str, float | None, float | None]:
    normalized = dict(record.get("normalized_features") or record.get("features") or {})
    age = _to_number(normalized.get("age"))

    recognized_income = _to_number(normalized.get("recognized_income"))
    if recognized_income is not None:
        income = recognized_income
        income_source = str(normalized.get("recognized_income_source") or "recognized_income")
        raw_income = recognized_income
    else:
        income = None
        income_source = ""
        raw_income = None

    available_amount = _to_number(normalized.get("available_amount"))
    amount = available_amount
    amount_source = str(normalized.get("available_amount_source") or "available_amount") if available_amount is not None else ""
    raw_amount = available_amount
    return age, income, income_source, raw_income, amount, amount_source, raw_amount


def _is_missing_sentinel(value: float | None) -> bool:
    if value is None:
        return True
    return value in {8888888, 8888888.8, 88888888, 99999, 999999, 9999999}


def _extract_model_score(fields: dict[str, object]) -> tuple[float | None, str]:
    priority_terms = [
        "비대면연계대출스코어",
        "신용대출신청평점",
        "K-Score Score",
        "K2개인모델 스코어",
        "I index 스코어",
        "R-SCORE통합스코어",
        "NICE스코어",
        "PI플러스스코어",
    ]
    for term in priority_terms:
        for key, value in fields.items():
            if term in str(key):
                numeric = _to_number(value)
                if numeric is not None and not _is_missing_sentinel(numeric):
                    return numeric, str(key)
    for key, value in fields.items():
        key_text = str(key)
        if "스코어" in key_text or "평점" in key_text or "score" in key_text.lower():
            numeric = _to_number(value)
            if numeric is not None and not _is_missing_sentinel(numeric):
                return numeric, key_text
    return None, ""


def _extract_delinquency_rate(fields: dict[str, object]) -> tuple[float | None, str]:
    for key, value in fields.items():
        key_text = str(key)
        if "연체율" not in key_text and "부실률" not in key_text:
            continue
        numeric = _to_number(value)
        if numeric is not None and not _is_missing_sentinel(numeric):
            return numeric, key_text
    return None, ""


def _extract_delinquency_signal(fields: dict[str, object], reject_text: str = "", reject_descriptions: list[str] | None = None) -> bool:
    signal_terms = ["연체건수", "총연체금액", "최장연체일수", "합계연체일수", "연체일수"]
    for key, value in fields.items():
        key_text = str(key)
        if not any(term in key_text for term in signal_terms):
            continue
        numeric = _to_number(value)
        if numeric is not None and not _is_missing_sentinel(numeric) and numeric > 0:
            return True
    return False


def _build_record_profile(record: dict, reject_code_mapping: dict[str, dict[str, str]] | None = None, income_thresholds: list[tuple[float, str]] | None = None, amount_thresholds: list[tuple[float, str]] | None = None, decision_result: dict[str, object] | None = None) -> dict[str, object]:
    fields = _parse_named_lines(record.get("in_text"), record.get("out_text"), record.get("in_text2"), record.get("out_text2"))
    reject_code_mapping = reject_code_mapping or {}
    age, income, income_source, raw_income, amount, amount_source, raw_amount = _extract_normalized_record_measures(record)
    normalized = record.get("normalized_features") if isinstance(record.get("normalized_features"), dict) else {}
    rate = _to_number(normalized.get("applied_rate"))
    rate_source = "applied_rate" if rate is not None else ""
    raw_rate = rate
    if age is None:
        age = _to_number(fields.get("연령"))
    if income is None:
        income, income_source, raw_income = _extract_scaled_measure(fields, INCOME_MEASURE_SPECS)
    if amount is None:
        amount, amount_source, raw_amount = _extract_scaled_measure(fields, AMOUNT_MEASURE_SPECS)
    if rate is None:
        rate, rate_source, raw_rate = _extract_scaled_measure(fields, RATE_MEASURE_SPECS)
    reject_codes = [str(code).strip() for code in (record.get("reject_reason_codes") or []) if str(code).strip()]
    reject_text = str(record.get("reject_reason_text") or fields.get("거절사유") or "").strip()
    reject_descriptions = _get_reject_code_descriptions(reject_codes, reject_code_mapping)
    decision_result = decision_result or {}
    raw_decision = str(fields.get("승인 여부") or "").strip()
    normalized_decision = _normalize_decision_label(raw_decision)
    has_reject_signal = bool(reject_codes or reject_descriptions or _is_meaningful_reject_text(reject_text))
    decision = str(decision_result.get("decision") or normalized_decision or ("거절" if has_reject_signal else "승인"))
    active_reject_codes = [
        str(code).strip().upper()
        for code in (decision_result.get("active_reject_codes") or (reject_codes if decision == "거절" else []))
        if str(code).strip()
    ]
    receipt_no = str(fields.get("접수번호") or fields.get("신청서접수번호") or "").strip()
    model_score, model_score_source = _extract_model_score(fields)
    delinquency_rate, delinquency_rate_source = _extract_delinquency_rate(fields)
    delinquency_signal = _extract_delinquency_signal(fields, reject_text, reject_descriptions)
    profile = {
        "record_id": receipt_no or f"{record.get('product')}-{hash(str(record.get('full_text2') or record.get('full_text') or ''))}",
        "product": str(record.get("product") or "").strip(),
        "product_display": str(record.get("product_display") or record.get("product") or "").strip(),
        "decision": decision,
        "age": age,
        "income": income,
        "income_source": income_source,
        "income_raw": raw_income,
        "amount": amount,
        "amount_source": amount_source,
        "amount_raw": raw_amount,
        "rate": rate,
        "rate_source": rate_source,
        "rate_raw": raw_rate,
        "model_score": model_score,
        "model_score_source": model_score_source,
        "delinquency_rate": delinquency_rate,
        "delinquency_rate_source": delinquency_rate_source,
        "delinquency_signal": delinquency_signal,
        "reject_codes": active_reject_codes,
        "raw_reject_codes": reject_codes,
        "reject_descriptions": reject_descriptions if decision == "거절" else [],
        "reject_reason_text": reject_text,
        "decision_risk_score": decision_result.get("risk_score"),
        "decision_source": decision_result.get("source") or "rule",
        "fields": fields,
        "search_text": " ".join(
            part for part in [
                str(record.get("full_text") or ""),
                str(record.get("full_text2") or ""),
                reject_text,
                " ".join(reject_codes),
                " ".join(reject_descriptions),
            ] if part
        ).lower(),
    }
    return _apply_profile_bands(
        profile,
        income_thresholds or DEFAULT_INCOME_BAND_THRESHOLDS,
        amount_thresholds or DEFAULT_AMOUNT_BAND_THRESHOLDS,
    )


def _score_feature(feature: dict, query: str, selected_product: str) -> int:
    haystack = " ".join([
        str(feature.get("feature_id") or ""),
        str(feature.get("feature_name") or ""),
        str(feature.get("category") or ""),
        str(feature.get("description") or ""),
        " ".join(str(item) for item in (feature.get("aliases") or [])),
        " ".join(str(item) for item in (feature.get("products") or [])),
    ]).lower()
    tokens = _tokenize_text(query)
    score = 0
    for token in tokens:
        if token in haystack:
            score += 4
        if token in str(feature.get("feature_id") or "").lower():
            score += 5
        if token in str(feature.get("feature_name") or "").lower():
            score += 6
    if selected_product and selected_product in (feature.get("products") or []):
        score += 3
    score += min(6, int((feature.get("coverage") or {}).get("mapping_count") or 0) // 4)
    score += _score_reject_feature_boost(feature, query)
    score += _score_employment_feature_boost(feature, query)
    score += _score_age_feature_boost(feature, query)
    score += _score_rate_feature_boost(feature, query)
    score += _score_limit_feature_boost(feature, query)
    return score


def _feature_search_document(feature: dict) -> str:
    return " ".join(
        part
        for part in [
            str(feature.get("feature_id") or ""),
            str(feature.get("feature_name") or ""),
            str(feature.get("category") or ""),
            str(feature.get("description") or ""),
            " ".join(str(item) for item in (feature.get("aliases") or [])),
            " ".join(str(item) for item in (feature.get("products") or [])),
            " ".join(str(item) for item in (feature.get("directions") or [])),
        ]
        if part
    )


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _intent_classifier_signature() -> str:
    digest = hashlib.sha1()
    for intent_id in sorted(INTENT_CLASSIFIER_PROTOTYPES):
        spec = INTENT_CLASSIFIER_PROTOTYPES[intent_id]
        digest.update(intent_id.encode("utf-8"))
        digest.update(str(spec.get("description") or "").encode("utf-8"))
        for example in spec.get("examples") or []:
            digest.update(str(example).encode("utf-8"))
            digest.update(b"\x1e")
    return digest.hexdigest()


def _build_intent_embedding_bundle() -> tuple[list[str], dict[str, int], np.ndarray | None]:
    signature = _intent_classifier_signature()
    with _intent_embedding_cache_lock:
        cached_matrix = _intent_embedding_cache.get("matrix")
        if (
            _intent_embedding_cache.get("signature") == signature
            and isinstance(cached_matrix, np.ndarray)
        ):
            return (
                list(_intent_embedding_cache.get("intent_ids") or []),
                dict(_intent_embedding_cache.get("example_counts") or {}),
                cached_matrix,
            )

    intent_ids: list[str] = []
    documents: list[str] = []
    example_counts: dict[str, int] = {}
    for intent_id, spec in INTENT_CLASSIFIER_PROTOTYPES.items():
        examples = [str(item).strip() for item in (spec.get("examples") or []) if str(item).strip()]
        example_counts[intent_id] = len(examples)
        for example in examples:
            intent_ids.append(intent_id)
            documents.append(
                " ".join([
                    str(spec.get("label") or ""),
                    str(spec.get("description") or ""),
                    example,
                ]).strip()
            )

    if not documents:
        return [], {}, None

    try:
        embeddings = get_embeddings()
        matrix = np.asarray(embeddings.embed_documents(documents), dtype=np.float32)
        matrix = _normalize_matrix(matrix)
    except Exception:
        matrix = None

    with _intent_embedding_cache_lock:
        _intent_embedding_cache["signature"] = signature
        _intent_embedding_cache["intent_ids"] = list(intent_ids)
        _intent_embedding_cache["example_counts"] = dict(example_counts)
        _intent_embedding_cache["matrix"] = matrix
    return intent_ids, example_counts, matrix


def _score_intents_with_embeddings(query: str) -> list[dict[str, object]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []
    intent_ids, _example_counts, matrix = _build_intent_embedding_bundle()
    if matrix is None or matrix.size == 0 or not intent_ids:
        return []
    try:
        embeddings = get_embeddings()
        query_vector = np.asarray(embeddings.embed_query(normalized_query), dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if math.isclose(query_norm, 0.0):
            return []
        query_vector = query_vector / query_norm
        similarities = matrix @ query_vector
    except Exception:
        return []

    scores_by_intent: dict[str, list[float]] = {intent_id: [] for intent_id in INTENT_CLASSIFIER_PROTOTYPES}
    for index, intent_id in enumerate(intent_ids):
        scores_by_intent.setdefault(intent_id, []).append(float(similarities[index]))

    ranked: list[dict[str, object]] = []
    for intent_id, scores in scores_by_intent.items():
        if not scores:
            continue
        max_score = max(scores)
        avg_top_score = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
        combined_score = max_score * 0.7 + avg_top_score * 0.3
        ranked.append(
            {
                "intent": intent_id,
                "label": str((INTENT_CLASSIFIER_PROTOTYPES.get(intent_id) or {}).get("label") or intent_id),
                "score": round(float(combined_score), 4),
                "max_score": round(float(max_score), 4),
            }
        )
    ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return ranked


def _query_has_cluster_vector_intent(query: str) -> bool:
    compact_query = _compact_search_text(query)
    markers = [
        "군집", "벡터", "유사", "유사도", "faiss", "vector", "embedding", "similarity",
        "cluster", "segment", "세그먼트",
    ]
    return _query_asks_cluster_signals(query) or any(marker in compact_query for marker in markers)


def _rule_based_query_intent(
    query: str,
    selected_feature: dict | None,
    *,
    regulation_first_route: bool = False,
    strategy_simulation_route: bool = False,
) -> str:
    if not str(query or "").strip():
        return "general_fallback"
    if regulation_first_route:
        return "regulation_policy"
    if strategy_simulation_route:
        return "strategy_simulation"
    if _query_has_reject_intent(query, selected_feature):
        return "reject_reason"
    if _query_asks_average_metrics(query) or query_has_segment_metric_intent(query):
        return "rate_limit"
    if _query_has_cluster_vector_intent(query):
        return "cluster_vector"
    if _query_asks_influence_features(query):
        return "approval_factor"
    return ""


def _classify_query_intent(
    query: str,
    selected_feature: dict | None,
    *,
    regulation_first_route: bool = False,
    strategy_simulation_route: bool = False,
) -> dict[str, object]:
    embedding_candidates = _score_intents_with_embeddings(query)
    rule_intent = _rule_based_query_intent(
        query,
        selected_feature,
        regulation_first_route=regulation_first_route,
        strategy_simulation_route=strategy_simulation_route,
    )
    embedding_intent = str((embedding_candidates[0] or {}).get("intent") or "") if embedding_candidates else ""
    embedding_score = float((embedding_candidates[0] or {}).get("score") or 0.0) if embedding_candidates else 0.0

    if rule_intent:
        intent = rule_intent
        method = "rule+embedding" if embedding_candidates else "rule"
        confidence = max(0.72, embedding_score if embedding_intent == rule_intent else 0.0)
    elif embedding_intent and embedding_score >= 0.18:
        intent = embedding_intent
        method = "embedding"
        confidence = embedding_score
    else:
        intent = "general_fallback"
        method = "fallback"
        confidence = embedding_score

    spec = INTENT_CLASSIFIER_PROTOTYPES.get(intent) or {}
    return {
        "intent": intent,
        "label": str(spec.get("label") or intent),
        "confidence": round(float(confidence), 4),
        "method": method,
        "rule_intent": rule_intent,
        "embedding_intent": embedding_intent,
        "top_candidates": embedding_candidates[:4],
        "output_categories": list(spec.get("output_categories") or []),
    }


def _classify_output_categories(
    *,
    input_intent: str,
    answer_summary: dict[str, object],
    agentic_workspace: dict[str, object],
    ollama_runtime: dict[str, object],
    regulation_evidence: list[dict[str, object]],
    reject_code_summary: list[dict[str, object]],
) -> dict[str, object]:
    categories = ["answer_summary"]
    configured = list((INTENT_CLASSIFIER_PROTOTYPES.get(input_intent) or {}).get("output_categories") or [])
    categories.extend(str(item) for item in configured)

    source = str(answer_summary.get("source") or "").strip().lower()
    guardrail = str(ollama_runtime.get("guardrail") or "").strip().lower()
    active_tools = [
        str(tool.get("id") or "").strip().lower()
        for tool in (((agentic_workspace.get("version_1") or {}).get("active_tools") or []))
        if isinstance(tool, dict)
    ]
    if reject_code_summary or guardrail == "reject_code_grounding":
        categories.append("reject_code_analysis")
    if regulation_evidence or "regulation" in guardrail:
        categories.extend(["policy_regulation", "citation_answer"])
    if "average_metric" in guardrail:
        categories.append("metric_summary")
    if "cluster_signal" in guardrail:
        categories.append("cluster_analysis")
    if "strategy" in active_tools or input_intent == "strategy_simulation":
        categories.extend(["strategy_simulation", "product_strategy"])
    if "explainability" in active_tools:
        categories.append("feature_explainability")
    if "ollama" in source:
        categories.append("llm_generated_answer")
    if "evidence-blocked" in source:
        categories.append("evidence_blocked")

    deduped = _dedupe_text_items([item for item in categories if item], limit=12)
    primary = next((item for item in deduped if item != "answer_summary"), "answer_summary")
    return {
        "primary": primary,
        "categories": deduped,
        "source": str(answer_summary.get("source") or ""),
        "guardrail": str(ollama_runtime.get("guardrail") or ""),
        "active_tools": active_tools,
    }


def _prepare_feature_embedding_payload(features: list[dict]) -> tuple[list[str], list[str], str]:
    feature_ids: list[str] = []
    documents: list[str] = []
    digest = hashlib.sha1()
    for feature in features:
        feature_id = str(feature.get("feature_id") or "")
        document = _feature_search_document(feature)
        feature_ids.append(feature_id)
        documents.append(document)
        digest.update(feature_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(document.encode("utf-8"))
        digest.update(b"\x1e")
    return feature_ids, documents, digest.hexdigest()


def _store_full_feature_embedding_bundle(signature: str, feature_ids: list[str], matrix: np.ndarray) -> None:
    with _feature_embedding_cache_lock:
        _feature_embedding_cache["full_signature"] = signature
        _feature_embedding_cache["full_ids"] = list(feature_ids)
        _feature_embedding_cache["full_index_by_id"] = {
            feature_id: index
            for index, feature_id in enumerate(feature_ids)
        }
        _feature_embedding_cache["full_matrix"] = matrix
        _feature_embedding_cache["subset_matrices"] = collections.OrderedDict()


def _store_subset_feature_embedding_matrix(signature: str, feature_ids: list[str], matrix: np.ndarray) -> None:
    with _feature_embedding_cache_lock:
        subset_matrices = _feature_embedding_cache.get("subset_matrices")
        if not isinstance(subset_matrices, collections.OrderedDict):
            subset_matrices = collections.OrderedDict()
            _feature_embedding_cache["subset_matrices"] = subset_matrices
        subset_matrices[signature] = {
            "ids": list(feature_ids),
            "matrix": matrix,
        }
        subset_matrices.move_to_end(signature)
        while len(subset_matrices) > FEATURE_EMBEDDING_CACHE_LIMIT:
            subset_matrices.popitem(last=False)


def _load_persisted_full_feature_embedding_bundle(signature: str) -> tuple[list[str], np.ndarray] | None:
    if not FEATURE_EMBEDDING_CACHE_PATH.exists():
        return None
    try:
        with np.load(FEATURE_EMBEDDING_CACHE_PATH, allow_pickle=False) as loaded:
            cached_signature = str(loaded["signature"].item())
            if cached_signature != signature:
                return None
            feature_ids = loaded["feature_ids"].astype(str).tolist()
            matrix = np.asarray(loaded["matrix"], dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(feature_ids):
                return None
            return feature_ids, matrix
    except Exception:
        return None


def _persist_full_feature_embedding_bundle(signature: str, feature_ids: list[str], matrix: np.ndarray) -> None:
    try:
        FEATURE_EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = FEATURE_EMBEDDING_CACHE_PATH.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp_path,
            signature=np.array(signature),
            feature_ids=np.asarray(feature_ids, dtype=str),
            matrix=np.asarray(matrix, dtype=np.float32),
        )
        temp_path.replace(FEATURE_EMBEDDING_CACHE_PATH)
    except Exception:
        return


def _build_full_feature_embedding_bundle(features: list[dict]) -> tuple[list[str], np.ndarray | None, str]:
    feature_ids, documents, signature = _prepare_feature_embedding_payload(features)
    with _feature_embedding_cache_lock:
        cached_signature = _feature_embedding_cache.get("full_signature")
        cached_ids = _feature_embedding_cache.get("full_ids") or []
        cached_matrix = _feature_embedding_cache.get("full_matrix")
        if signature == cached_signature and feature_ids == cached_ids and isinstance(cached_matrix, np.ndarray):
            return feature_ids, cached_matrix, signature

    persisted = _load_persisted_full_feature_embedding_bundle(signature)
    if persisted is not None:
        persisted_ids, persisted_matrix = persisted
        _store_full_feature_embedding_bundle(signature, persisted_ids, persisted_matrix)
        return persisted_ids, persisted_matrix, signature

    try:
        embeddings = get_embeddings()
        matrix = np.asarray(embeddings.embed_documents(documents), dtype=np.float32)
        matrix = _normalize_matrix(matrix)
        _store_full_feature_embedding_bundle(signature, feature_ids, matrix)
        _persist_full_feature_embedding_bundle(signature, feature_ids, matrix)
        return feature_ids, matrix, signature
    except Exception:
        return feature_ids, None, signature


def _invalidate_feature_embedding_cache() -> None:
    with _feature_embedding_cache_lock:
        _feature_embedding_cache["full_signature"] = None
        _feature_embedding_cache["full_ids"] = []
        _feature_embedding_cache["full_index_by_id"] = {}
        _feature_embedding_cache["full_matrix"] = None
        _feature_embedding_cache["subset_matrices"] = collections.OrderedDict()


def _get_feature_embedding_matrix(features: list[dict], all_features: list[dict] | None = None) -> tuple[list[str], np.ndarray | None]:
    feature_ids, documents, subset_signature = _prepare_feature_embedding_payload(features)
    with _feature_embedding_cache_lock:
        subset_matrices = _feature_embedding_cache.get("subset_matrices")
        if isinstance(subset_matrices, collections.OrderedDict):
            cached_subset = subset_matrices.get(subset_signature)
            if isinstance(cached_subset, dict):
                cached_ids = cached_subset.get("ids") or []
                cached_matrix = cached_subset.get("matrix")
                if cached_ids == feature_ids and isinstance(cached_matrix, np.ndarray):
                    subset_matrices.move_to_end(subset_signature)
                    return feature_ids, cached_matrix

    source_features = all_features if all_features is not None else features
    full_ids, full_matrix, full_signature = _build_full_feature_embedding_bundle(source_features)
    if full_matrix is not None:
        with _feature_embedding_cache_lock:
            full_index_by_id = dict(_feature_embedding_cache.get("full_index_by_id") or {})
        indices = [full_index_by_id.get(feature_id) for feature_id in feature_ids]
        if all(index is not None for index in indices):
            if feature_ids == full_ids:
                return feature_ids, full_matrix
            subset_matrix = np.asarray(full_matrix[indices], dtype=np.float32)
            derived_signature = hashlib.sha1(f"{full_signature}|{subset_signature}".encode("utf-8")).hexdigest()
            _store_subset_feature_embedding_matrix(derived_signature, feature_ids, subset_matrix)
            _store_subset_feature_embedding_matrix(subset_signature, feature_ids, subset_matrix)
            return feature_ids, subset_matrix

    try:
        embeddings = get_embeddings()
        matrix = np.asarray(embeddings.embed_documents(documents), dtype=np.float32)
        matrix = _normalize_matrix(matrix)
        _store_subset_feature_embedding_matrix(subset_signature, feature_ids, matrix)
        return feature_ids, matrix
    except Exception:
        return feature_ids, None


def _semantic_rank_features(
    features: list[dict],
    query: str,
    selected_product: str,
    all_features: list[dict] | None = None,
    conversation_profile: dict[str, object] | None = None,
) -> tuple[list[tuple[float, dict]], str, dict[str, float], dict[str, float], dict[str, list[str]]]:
    conversation_adjustments: dict[str, float] = {}
    conversation_reasons: dict[str, list[str]] = {}
    if not query.strip():
        ranked: list[tuple[float, dict]] = []
        for feature in features:
            feature_id = str(feature.get("feature_id") or "")
            base_score = float(_score_feature(feature, query, selected_product))
            adjustment, reasons = _score_feature_conversation_adjustment(feature, conversation_profile)
            conversation_adjustments[feature_id] = adjustment
            conversation_reasons[feature_id] = reasons
            ranked.append((base_score + adjustment, feature))
        return ranked, "lexical", {}, conversation_adjustments, conversation_reasons

    feature_ids, matrix = _get_feature_embedding_matrix(features, all_features=all_features)
    if matrix is None or matrix.size == 0:
        ranked = []
        for feature in features:
            feature_id = str(feature.get("feature_id") or "")
            base_score = float(_score_feature(feature, query, selected_product))
            adjustment, reasons = _score_feature_conversation_adjustment(feature, conversation_profile)
            conversation_adjustments[feature_id] = adjustment
            conversation_reasons[feature_id] = reasons
            combined = base_score + adjustment
            if combined <= 0:
                continue
            ranked.append((combined, feature))
        return ranked, "lexical-fallback", {}, conversation_adjustments, conversation_reasons

    try:
        embeddings = get_embeddings()
        query_vector = np.asarray(embeddings.embed_query(query), dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if math.isclose(query_norm, 0.0):
            raise ValueError("empty query embedding")
        query_vector = query_vector / query_norm
        similarities = matrix @ query_vector
        semantic_scores = {
            feature_ids[index]: float(score)
            for index, score in enumerate(similarities)
        }
        ranked = []
        for index, feature in enumerate(features):
            feature_id = str(feature.get("feature_id") or "")
            semantic_score = float(similarities[index])
            lexical_score = float(_score_feature(feature, query, selected_product))
            adjustment, reasons = _score_feature_conversation_adjustment(feature, conversation_profile)
            conversation_adjustments[feature_id] = adjustment
            conversation_reasons[feature_id] = reasons
            combined = semantic_score * 20.0 + lexical_score + adjustment
            if lexical_score <= 0 and semantic_score < 0.18:
                continue
            ranked.append((combined, feature))
        return ranked, "embedding", semantic_scores, conversation_adjustments, conversation_reasons
    except Exception:
        ranked = []
        for feature in features:
            feature_id = str(feature.get("feature_id") or "")
            base_score = float(_score_feature(feature, query, selected_product))
            adjustment, reasons = _score_feature_conversation_adjustment(feature, conversation_profile)
            conversation_adjustments[feature_id] = adjustment
            conversation_reasons[feature_id] = reasons
            combined = base_score + adjustment
            if combined <= 0:
                continue
            ranked.append((combined, feature))
        return ranked, "lexical-fallback", {}, conversation_adjustments, conversation_reasons


def _build_related_features(selected_feature: dict, features: list[dict], selected_product: str, limit: int = 8, all_features: list[dict] | None = None) -> list[dict]:
    if not selected_feature:
        return []
    base_products = set(selected_feature.get("products") or [])
    base_directions = set(selected_feature.get("directions") or [])
    base_tokens = set(_tokenize_text(" ".join([
        str(selected_feature.get("feature_name") or ""),
        str(selected_feature.get("description") or ""),
        " ".join(selected_feature.get("aliases") or []),
    ])))
    ranked = []
    semantic_mode = False
    feature_ids, matrix = _get_feature_embedding_matrix(features, all_features=all_features)
    semantic_scores: dict[str, float] = {}
    if matrix is not None and feature_ids:
        try:
            selected_index = feature_ids.index(str(selected_feature.get("feature_id") or ""))
            similarities = matrix @ matrix[selected_index]
            semantic_scores = {
                feature_ids[index]: float(score)
                for index, score in enumerate(similarities)
            }
            semantic_mode = True
        except Exception:
            semantic_scores = {}
    for feature in features:
        if feature.get("feature_id") == selected_feature.get("feature_id"):
            continue
        products = set(feature.get("products") or [])
        directions = set(feature.get("directions") or [])
        overlap_products = sorted(base_products & products)
        overlap_directions = sorted(base_directions & directions)
        feature_tokens = set(_tokenize_text(" ".join([
            str(feature.get("feature_name") or ""),
            str(feature.get("description") or ""),
            " ".join(feature.get("aliases") or []),
        ])))
        token_overlap = sorted(base_tokens & feature_tokens)
        score = 0
        if selected_feature.get("category") and feature.get("category") == selected_feature.get("category"):
            score += 5
        score += len(overlap_products) * 4
        score += len(overlap_directions) * 2
        score += min(4, len(token_overlap))
        semantic_score = semantic_scores.get(str(feature.get("feature_id") or ""), 0.0)
        score += semantic_score * 10
        if selected_product and selected_product in products:
            score += 2
        if score <= 0:
            continue
        ranked.append({
            "feature_id": feature.get("feature_id"),
            "feature_name": feature.get("feature_name") or feature.get("feature_id"),
            "category": feature.get("category") or "unclassified",
            "description": feature.get("description") or "",
            "score": round(float(score), 3),
            "semantic_score": round(float(semantic_score), 4),
            "relation_mode": "embedding+rule" if semantic_mode else "rule",
            "shared_products": overlap_products,
            "shared_directions": overlap_directions,
            "shared_tokens": token_overlap[:6],
        })
    return sorted(ranked, key=lambda item: (-int(item["score"]), str(item["feature_name"])))[:limit]


def _build_feature_graph_edges(
    candidate_features: list[dict],
    all_features: list[dict] | None = None,
) -> tuple[dict[str, list[dict[str, object]]], str]:
    if len(candidate_features) < 2:
        return {}, "rule"

    feature_ids, matrix = _get_feature_embedding_matrix(candidate_features, all_features=all_features)
    similarity_by_pair: dict[tuple[str, str], float] = {}
    relation_mode = "rule"
    if matrix is not None and len(feature_ids) == len(candidate_features):
        relation_mode = "embedding+rule"
        for left_index, left_id in enumerate(feature_ids):
            for right_index in range(left_index + 1, len(feature_ids)):
                similarity_by_pair[(left_id, feature_ids[right_index])] = float(matrix[left_index] @ matrix[right_index])

    graph_edges: dict[str, list[dict[str, object]]] = {}
    for left_index, left_feature in enumerate(candidate_features):
        left_id = str(left_feature.get("feature_id") or "")
        if not left_id:
            continue
        left_products = set(left_feature.get("products") or [])
        left_directions = set(left_feature.get("directions") or [])
        left_tokens = set(_tokenize_text(" ".join([
            str(left_feature.get("feature_name") or ""),
            str(left_feature.get("description") or ""),
            " ".join(str(item) for item in (left_feature.get("aliases") or [])),
        ])))
        for right_index in range(left_index + 1, len(candidate_features)):
            right_feature = candidate_features[right_index]
            right_id = str(right_feature.get("feature_id") or "")
            if not right_id:
                continue
            right_products = set(right_feature.get("products") or [])
            right_directions = set(right_feature.get("directions") or [])
            right_tokens = set(_tokenize_text(" ".join([
                str(right_feature.get("feature_name") or ""),
                str(right_feature.get("description") or ""),
                " ".join(str(item) for item in (right_feature.get("aliases") or [])),
            ])))
            shared_products = sorted(left_products & right_products)
            shared_directions = sorted(left_directions & right_directions)
            shared_tokens = sorted(left_tokens & right_tokens)
            shared_category = bool(left_feature.get("category") and left_feature.get("category") == right_feature.get("category"))
            semantic_similarity = similarity_by_pair.get((left_id, right_id)) or similarity_by_pair.get((right_id, left_id)) or 0.0
            weight = 0.0
            reasons: list[str] = []
            if shared_category:
                weight += 1.4
                reasons.append(f"같은 분류({left_feature.get('category')})")
            if shared_products:
                weight += min(1.8, len(shared_products) * 0.6)
                reasons.append(f"같은 상품군 {', '.join(shared_products[:2])}")
            if shared_directions:
                weight += min(1.0, len(shared_directions) * 0.4)
                reasons.append(f"같은 방향 {', '.join(shared_directions[:2])}")
            if shared_tokens:
                weight += min(1.2, len(shared_tokens) * 0.3)
                reasons.append(f"같이 등장한 의미어 {', '.join(shared_tokens[:3])}")
            if semantic_similarity >= 0.33:
                weight += semantic_similarity * 2.4
                reasons.append(f"벡터 유사도 {semantic_similarity:.2f}")
            if weight <= 0:
                continue
            edge = {
                "target_feature_id": right_id,
                "target_feature_name": right_feature.get("feature_name") or right_id,
                "weight": round(float(weight), 4),
                "reasons": reasons[:4],
                "shared_products": shared_products[:3],
                "shared_directions": shared_directions[:3],
                "shared_tokens": shared_tokens[:4],
                "shared_category": str(left_feature.get("category") or "") if shared_category else "",
                "semantic_similarity": round(float(semantic_similarity), 4),
            }
            graph_edges.setdefault(left_id, []).append(edge)
            reverse_edge = dict(edge)
            reverse_edge["target_feature_id"] = left_id
            reverse_edge["target_feature_name"] = left_feature.get("feature_name") or left_id
            graph_edges.setdefault(right_id, []).append(reverse_edge)

    for edges in graph_edges.values():
        edges.sort(key=lambda item: (-float(item.get("weight") or 0.0), str(item.get("target_feature_name") or "")))
    return graph_edges, relation_mode


def _infer_feature_axis_key(feature_like: dict[str, object] | None) -> str:
    feature_like = feature_like or {}
    search_text = " ".join([
        str(feature_like.get("feature_id") or ""),
        str(feature_like.get("feature_name") or ""),
        str(feature_like.get("category") or ""),
        " ".join(str(item) for item in (feature_like.get("aliases") or [])),
    ]).lower()
    axis_rules = [
        ("rate", ["rate", "금리", "이율", "applied_rate", "interest"]),
        ("limit", ["limit", "한도", "amount", "금액", "requested_limit", "approved_amount", "가능금액"]),
        ("reject", ["reject", "거절", "부결"]),
        ("employment", ["employment", "job", "직장", "재직", "company", "기업"]),
        ("age", ["age", "연령", "나이"]),
        ("income", ["income", "소득", "급여", "연봉"]),
        ("credit", ["credit", "신용", "score", "평점"]),
    ]
    for axis_key, keywords in axis_rules:
        if any(keyword in search_text for keyword in keywords):
            return axis_key
    category = str(feature_like.get("category") or "").strip().lower()
    if category:
        return f"category:{category}"
    feature_id = str(feature_like.get("feature_id") or "").strip().lower()
    return feature_id or "feature"


def _resolve_representative_features(
    primary_feature_selection: dict[str, object] | None,
    selected_feature: dict | None = None,
    limit: int = 3,
) -> list[dict[str, object]]:
    selection = primary_feature_selection or {}
    representatives = [
        dict(item)
        for item in (selection.get("representative_features") or [])
        if isinstance(item, dict) and str(item.get("feature_id") or "").strip()
    ]
    if representatives:
        return representatives[:limit]
    fallback = selected_feature or {}
    feature_id = str(fallback.get("feature_id") or "").strip()
    if not feature_id:
        return []
    return [{
        "feature_id": feature_id,
        "feature_name": str(fallback.get("feature_name") or feature_id),
        "category": str(fallback.get("category") or "unclassified"),
        "description": str(fallback.get("description") or ""),
        "axis_key": _infer_feature_axis_key(fallback),
        "base_score": 0.0,
        "intent_score": 0.0,
        "graph_score": 0.0,
        "hybrid_score": 0.0,
        "support_count": 0,
        "support_labels": [],
        "matched_tokens": [],
        "matched_feature_id_tokens": [],
        "matched_feature_name_tokens": [],
        "graph_edges": [],
        "products": list(fallback.get("products") or []),
    }]


def _build_related_features_multi(
    representative_features: list[dict[str, object]],
    features: list[dict],
    selected_product: str,
    limit: int = 8,
    all_features: list[dict] | None = None,
) -> list[dict[str, object]]:
    if not representative_features:
        return []
    representative_ids = {
        str(item.get("feature_id") or "")
        for item in representative_features
        if str(item.get("feature_id") or "").strip()
    }
    merged: dict[str, dict[str, object]] = {}
    per_axis_limit = max(limit, 4)
    for representative in representative_features[:3]:
        source_id = str(representative.get("feature_id") or "")
        if not source_id:
            continue
        if str(representative.get("axis_key") or _infer_feature_axis_key(representative)) == "product":
            continue
        base_feature = next((feature for feature in features if str(feature.get("feature_id") or "") == source_id), None)
        if base_feature is None:
            base_feature = representative
        for item in _build_related_features(base_feature, features, selected_product, limit=per_axis_limit, all_features=all_features):
            feature_id = str(item.get("feature_id") or "")
            if not feature_id or feature_id in representative_ids:
                continue
            existing = merged.get(feature_id)
            if existing is None:
                merged[feature_id] = {
                    **item,
                    "source_axes": [str(representative.get("feature_name") or source_id)],
                    "source_feature_ids": [source_id],
                }
                continue
            existing["score"] = round(max(float(existing.get("score") or 0.0), float(item.get("score") or 0.0)), 3)
            existing["semantic_score"] = round(max(float(existing.get("semantic_score") or 0.0), float(item.get("semantic_score") or 0.0)), 4)
            existing["shared_products"] = _dedupe_text_items(list(existing.get("shared_products") or []) + list(item.get("shared_products") or []), limit=6)
            existing["shared_directions"] = _dedupe_text_items(list(existing.get("shared_directions") or []) + list(item.get("shared_directions") or []), limit=6)
            existing["shared_tokens"] = _dedupe_text_items(list(existing.get("shared_tokens") or []) + list(item.get("shared_tokens") or []), limit=8)
            existing["source_axes"] = _dedupe_text_items(list(existing.get("source_axes") or []) + [str(representative.get("feature_name") or source_id)], limit=4)
            existing["source_feature_ids"] = _dedupe_text_items(list(existing.get("source_feature_ids") or []) + [source_id], limit=4)
    return sorted(
        merged.values(),
        key=lambda item: (-float(item.get("score") or 0.0), -len(item.get("source_feature_ids") or []), str(item.get("feature_name") or "")),
    )[:limit]


def _build_representative_axis_details(
    representative_features: list[dict[str, object]],
    primary_feature_selection: dict[str, object],
    related_features: list[dict[str, object]],
) -> list[dict[str, object]]:
    top_candidates = list((primary_feature_selection or {}).get("top_k") or [])
    details: list[dict[str, object]] = []
    for representative in representative_features[:3]:
        feature_id = str(representative.get("feature_id") or "")
        if not feature_id:
            continue
        candidate = next((item for item in top_candidates if str(item.get("feature_id") or "") == feature_id), representative)
        axis_related = [
            item for item in related_features
            if feature_id in {str(source_id) for source_id in (item.get("source_feature_ids") or [])}
        ]
        details.append({
            "feature_id": feature_id,
            "feature_name": str(candidate.get("feature_name") or representative.get("feature_name") or feature_id),
            "axis_key": str(candidate.get("axis_key") or representative.get("axis_key") or _infer_feature_axis_key(representative)),
            "description": str(candidate.get("description") or representative.get("description") or ""),
            "hybrid_score": round(float(candidate.get("hybrid_score") or candidate.get("base_score") or 0.0), 4),
            "intent_score": round(float(candidate.get("intent_score") or 0.0), 4),
            "graph_score": round(float(candidate.get("graph_score") or 0.0), 4),
            "matched_tokens": _dedupe_text_items(list(candidate.get("matched_feature_name_tokens") or []) + list(candidate.get("matched_tokens") or []), limit=6),
            "graph_supports": list(candidate.get("graph_edges") or [])[:3],
            "support_labels": _dedupe_text_items(list(candidate.get("support_labels") or []), limit=4),
            "related_features": axis_related[:3],
        })
    return details


def _feature_to_intent_representative(
    feature: dict,
    axis_key: str,
    matched_tokens: list[str],
    score: float,
) -> dict[str, object]:
    feature_id = str(feature.get("feature_id") or "")
    return {
        "rank": 0,
        "feature_id": feature_id,
        "feature_name": str(feature.get("feature_name") or feature_id or "feature"),
        "category": str(feature.get("category") or "unclassified"),
        "description": str(feature.get("description") or ""),
        "axis_key": axis_key,
        "base_score": 0.0,
        "intent_score": round(float(score), 4),
        "graph_score": 0.0,
        "hybrid_score": round(float(score), 4),
        "support_count": 0,
        "support_labels": [],
        "matched_tokens": matched_tokens,
        "matched_feature_id_tokens": [],
        "matched_feature_name_tokens": [],
        "graph_edges": [],
        "products": list(feature.get("products") or []),
        "injected_by": "metric_intent",
    }


def _augment_primary_feature_selection_with_metric_intents(
    primary_feature_selection: dict[str, object],
    query: str,
    features: list[dict],
) -> dict[str, object]:
    selection = dict(primary_feature_selection or {})
    representatives = [
        dict(item)
        for item in (selection.get("representative_features") or [])
        if isinstance(item, dict) and str(item.get("feature_id") or "").strip()
    ]
    top_k = [
        dict(item)
        for item in (selection.get("top_k") or [])
        if isinstance(item, dict) and str(item.get("feature_id") or "").strip()
    ]
    seen_feature_ids = {str(item.get("feature_id") or "") for item in representatives}
    seen_axis_keys = {str(item.get("axis_key") or _infer_feature_axis_key(item)) for item in representatives}
    injected: list[dict[str, object]] = []
    explicit_metric_axis_keys: set[str] = set()

    if _query_has_rate_intent(query):
        explicit_metric_axis_keys.add("rate")
    if _query_has_limit_intent(query):
        explicit_metric_axis_keys.add("limit")

    if "rate" in explicit_metric_axis_keys and "rate" not in seen_axis_keys:
        rate_candidates = [feature for feature in features if _score_rate_feature_routing(feature) > 0]
        rate_feature = max(rate_candidates, key=_score_rate_feature_routing) if rate_candidates else None
        if rate_feature is not None:
            injected.append(_feature_to_intent_representative(rate_feature, "rate", ["금리"], _score_rate_feature_routing(rate_feature)))

    if "limit" in explicit_metric_axis_keys and "limit" not in seen_axis_keys:
        limit_candidates = [feature for feature in features if _score_limit_feature_routing(feature, query) > 0]
        limit_feature = max(limit_candidates, key=lambda feature: _score_limit_feature_routing(feature, query)) if limit_candidates else None
        if limit_feature is not None:
            injected.append(_feature_to_intent_representative(limit_feature, "limit", ["한도"], _score_limit_feature_routing(limit_feature, query)))
    if "limit" in explicit_metric_axis_keys:
        requested_amount_feature = next(
            (feature for feature in features if str(feature.get("feature_id") or "") == "loan.requested_limit"),
            None,
        )
        if requested_amount_feature is not None and "loan.requested_limit" not in seen_feature_ids:
            injected.append(_feature_to_intent_representative(
                requested_amount_feature,
                "requested_limit",
                ["대출금액", "한도"],
                max(8.0, _score_limit_feature_routing(requested_amount_feature, query)),
            ))

    for candidate in injected:
        feature_id = str(candidate.get("feature_id") or "")
        axis_key = str(candidate.get("axis_key") or "feature")
        if not feature_id or feature_id in seen_feature_ids or axis_key in seen_axis_keys:
            continue
        representatives.append(candidate)
        seen_feature_ids.add(feature_id)
        seen_axis_keys.add(axis_key)
        if feature_id not in {str(item.get("feature_id") or "") for item in top_k}:
            top_k.append(candidate)

    if explicit_metric_axis_keys:
        representatives = [
            item for item in representatives
            if str(item.get("axis_key") or _infer_feature_axis_key(item)) in explicit_metric_axis_keys
            or str(item.get("feature_id") or "") in {"decision.approved_amount", "loan.requested_limit"}
        ]

    selected_feature_id = str(selection.get("selected_feature_id") or "")
    ordered_representatives: list[dict[str, object]] = []
    for group in (
        [item for item in representatives if str(item.get("feature_id") or "") == selected_feature_id],
        [item for item in representatives if str(item.get("injected_by") or "") == "metric_intent"],
        sorted(
            [item for item in representatives if str(item.get("feature_id") or "") != selected_feature_id and str(item.get("injected_by") or "") != "metric_intent"],
            key=lambda item: -float(item.get("hybrid_score") or 0.0),
        ),
    ):
        for item in group:
            feature_id = str(item.get("feature_id") or "")
            axis_key = str(item.get("axis_key") or _infer_feature_axis_key(item))
            if not feature_id or feature_id in {str(existing.get("feature_id") or "") for existing in ordered_representatives}:
                continue
            if axis_key in {str(existing.get("axis_key") or _infer_feature_axis_key(existing)) for existing in ordered_representatives}:
                continue
            ordered_representatives.append(item)
            if len(ordered_representatives) >= 3:
                break
        if len(ordered_representatives) >= 3:
            break

    selection["representative_features"] = ordered_representatives
    selection["top_k"] = top_k[:5]
    if injected:
        selection["metric_intent_features"] = injected
    return selection


def _build_product_representative_feature(selected_product: str, features: list[dict]) -> dict[str, object] | None:
    product_code = str(selected_product or "").strip()
    if not product_code:
        return None
    product_feature = next(
        (feature for feature in features if str(feature.get("feature_id") or "") == "application.product_code"),
        None,
    )
    if product_feature is None:
        return None
    feature_id = str(product_feature.get("feature_id") or "application.product_code")
    product_name = _product_display_name(product_code) or product_code
    return {
        "rank": 0,
        "feature_id": feature_id,
        "feature_name": product_name,
        "category": str(product_feature.get("category") or "application"),
        "description": f"{product_name}({product_code}) 상품 스코프를 고정하는 대표 feature",
        "axis_key": "product",
        "base_score": 0.0,
        "intent_score": 1.0,
        "graph_score": 0.0,
        "hybrid_score": 1.0,
        "support_count": 0,
        "support_labels": [product_code],
        "matched_tokens": [product_name, product_code],
        "matched_feature_id_tokens": [],
        "matched_feature_name_tokens": [product_name],
        "graph_edges": [],
        "products": [product_code],
        "product_code": product_code,
        "injected_by": "product_intent",
    }


def _augment_primary_feature_selection_with_product_intent(
    primary_feature_selection: dict[str, object],
    selected_product: str,
    features: list[dict],
) -> dict[str, object]:
    product_feature = _build_product_representative_feature(selected_product, features)
    if product_feature is None:
        return primary_feature_selection
    selection = dict(primary_feature_selection or {})
    representatives = [
        dict(item)
        for item in (selection.get("representative_features") or [])
        if isinstance(item, dict) and str(item.get("feature_id") or "").strip()
    ]
    top_k = [
        dict(item)
        for item in (selection.get("top_k") or [])
        if isinstance(item, dict) and str(item.get("feature_id") or "").strip()
    ]
    if not any(str(item.get("axis_key") or "") == "product" for item in representatives):
        representatives = [product_feature] + representatives
    if not any(str(item.get("feature_id") or "") == str(product_feature.get("feature_id") or "") for item in top_k):
        top_k = [product_feature] + top_k
    selection["representative_features"] = representatives[:3]
    selection["top_k"] = top_k[:6]
    selection["product_intent_feature"] = product_feature
    return selection


def _select_primary_feature_hybrid(
    ranked_features: list[tuple[float, dict]],
    query: str,
    selected_product: str,
    all_features: list[dict] | None = None,
    pinned_feature_id: str = "",
    top_k: int = 3,
) -> tuple[dict, dict[str, object]]:
    if pinned_feature_id:
        pinned_feature = next((feature for _, feature in ranked_features if str(feature.get("feature_id") or "") == pinned_feature_id), None)
        if pinned_feature is not None:
            representative_feature = {
                "feature_id": str(pinned_feature.get("feature_id") or ""),
                "feature_name": str(pinned_feature.get("feature_name") or pinned_feature.get("feature_id") or "대표 축"),
                "category": str(pinned_feature.get("category") or "unclassified"),
                "description": str(pinned_feature.get("description") or ""),
                "axis_key": _infer_feature_axis_key(pinned_feature),
                "base_score": 0.0,
                "intent_score": 0.0,
                "graph_score": 0.0,
                "hybrid_score": 0.0,
                "support_count": 0,
                "support_labels": [],
                "matched_tokens": [],
                "matched_feature_id_tokens": [],
                "matched_feature_name_tokens": [],
                "graph_edges": [],
                "products": list(pinned_feature.get("products") or []),
            }
            return pinned_feature, {
                "mode": "manual-override",
                "selected_feature_id": str(pinned_feature.get("feature_id") or ""),
                "selected_feature_name": str(pinned_feature.get("feature_name") or pinned_feature.get("feature_id") or "대표 축"),
                "headline": "사용자가 직접 대표 축을 지정했습니다.",
                "representative_features": [representative_feature],
                "top_k": [],
                "intent_tokens": _tokenize_text(query)[:8],
                "graph_relation_mode": "manual",
                "graph_result_explanation": [
                    {
                        "title": "수동 선택",
                        "summary": "자동 선택 대신 사용자가 지정한 feature 를 그대로 대표로 사용합니다.",
                        "details": [str(pinned_feature.get("feature_name") or pinned_feature.get("feature_id") or "")],
                    }
                ],
            }

    top_candidates = []
    for base_score, feature in ranked_features[:max(1, top_k)]:
        feature_id = str(feature.get("feature_id") or "")
        breakdown = _build_feature_rank_breakdown(
            feature,
            query,
            selected_product,
            None,
            float(base_score),
        )
        top_candidates.append({
            "feature": feature,
            "feature_id": feature_id,
            "feature_name": str(feature.get("feature_name") or feature_id or "feature"),
            "base_score": round(float(base_score), 4),
            "breakdown": breakdown,
        })

    if not top_candidates:
        return {}, {
            "mode": "topk-intent-graph-hybrid",
            "selected_feature_id": "",
            "selected_feature_name": "",
            "headline": "대표 축 후보가 없어 기본 응답을 사용합니다.",
            "top_k": [],
            "intent_tokens": _tokenize_text(query)[:8],
            "graph_relation_mode": "rule",
            "graph_result_explanation": [],
        }

    graph_edges, graph_relation_mode = _build_feature_graph_edges([item["feature"] for item in top_candidates], all_features=all_features)
    query_tokens = _tokenize_text(query)
    query_has_reject_intent = _query_has_reject_intent(query)
    explanation_cards: list[dict[str, object]] = []
    ranked_payload: list[dict[str, object]] = []
    winning_candidate = None
    winning_score = -1.0

    for rank_index, candidate in enumerate(top_candidates, start=1):
        feature = candidate["feature"]
        feature_id = candidate["feature_id"]
        breakdown = candidate["breakdown"]
        token_hits = int(breakdown.get("token_haystack_hits") or 0)
        name_hits = int(breakdown.get("feature_name_hits") or 0)
        id_hits = int(breakdown.get("feature_id_hits") or 0)
        intent_score = float(name_hits * 1.8 + id_hits * 1.2 + token_hits * 1.1)
        if selected_product and selected_product in (feature.get("products") or []):
            intent_score += 1.5
        if query_has_reject_intent and ("reject" in str(feature.get("feature_id") or "").lower() or "거절" in str(feature.get("feature_name") or "")):
            intent_score += 2.0

        candidate_edges = graph_edges.get(feature_id, [])
        graph_score = float(sum(float(item.get("weight") or 0.0) for item in candidate_edges[:3]))
        support_count = len(candidate_edges)
        support_labels = [str(item.get("target_feature_name") or item.get("target_feature_id") or "") for item in candidate_edges[:3] if str(item.get("target_feature_name") or item.get("target_feature_id") or "")]
        hybrid_score = float(candidate["base_score"] + intent_score + graph_score)

        candidate_payload = {
            "rank": rank_index,
            "feature_id": feature_id,
            "feature_name": candidate["feature_name"],
            "category": str(feature.get("category") or "unclassified"),
            "description": str(feature.get("description") or ""),
            "axis_key": _infer_feature_axis_key(feature),
            "base_score": round(float(candidate["base_score"]), 4),
            "intent_score": round(float(intent_score), 4),
            "graph_score": round(float(graph_score), 4),
            "hybrid_score": round(float(hybrid_score), 4),
            "support_count": support_count,
            "support_labels": support_labels,
            "matched_tokens": list(breakdown.get("matched_haystack_tokens") or []),
            "matched_feature_id_tokens": list(breakdown.get("matched_feature_id_tokens") or []),
            "matched_feature_name_tokens": list(breakdown.get("matched_feature_name_tokens") or []),
            "graph_edges": candidate_edges[:3],
            "products": list(feature.get("products") or []),
        }
        ranked_payload.append(candidate_payload)

        if hybrid_score > winning_score:
            winning_candidate = candidate_payload
            winning_score = hybrid_score
            winning_feature = feature

    winning_feature = winning_feature if 'winning_feature' in locals() else top_candidates[0]["feature"]
    winning_candidate = winning_candidate or ranked_payload[0]
    representative_candidates: list[dict[str, object]] = []
    seen_axis_keys: set[str] = set()
    sorted_candidates = sorted(ranked_payload, key=lambda item: (-float(item.get("hybrid_score") or 0.0), int(item.get("rank") or 0)))
    winning_score = float(winning_candidate.get("hybrid_score") or 0.0)
    # 승인률에 중요한 feature 우선 후보군 정의
    approval_critical_keywords = [
        "연소득", "소득", "income", "신용점수", "kcb", "nice", "등급", "score", "dsr", "dti", "잔액", "대출잔액", "기존대출", "연체", "부채", "부채비율"
    ]
    def is_approval_critical(candidate):
        fname = (candidate.get("feature_name") or "").lower()
        fid = (candidate.get("feature_id") or "").lower()
        desc = (candidate.get("description") or "").lower()
        return any(k in fname or k in fid or k in desc for k in approval_critical_keywords)

    # 1차: 승인에 중요한 feature만 추출
    critical_candidates = [c for c in sorted_candidates if is_approval_critical(c)]
    # 2차: 기존 방식대로 점수 높은 후보
    fallback_candidates = [c for c in sorted_candidates if c not in critical_candidates]
    representative_candidates = []
    seen_axis_keys: set[str] = set()
    # 승인에 중요한 feature에서 최대 3개
    for candidate in critical_candidates:
        axis_key = str(candidate.get("axis_key") or "feature")
        candidate_score = float(candidate.get("hybrid_score") or 0.0)
        evidence_count = len(candidate.get("matched_tokens") or []) + len(candidate.get("matched_feature_name_tokens") or []) + int(candidate.get("support_count") or 0)
        has_signal = candidate_score > 0.0 and (
            float(candidate.get("intent_score") or 0.0) >= 0.8
            or evidence_count > 0
            or candidate_score >= max(1.0, winning_score * 0.35)
        )
        if not has_signal or axis_key in seen_axis_keys:
            continue
        representative_candidates.append(candidate)
        seen_axis_keys.add(axis_key)
        if len(representative_candidates) >= 3:
            break
    # 만약 3개 미만이면 점수순으로 추가
    if len(representative_candidates) < 3:
        for candidate in fallback_candidates:
            axis_key = str(candidate.get("axis_key") or "feature")
            if axis_key in seen_axis_keys:
                continue
            representative_candidates.append(candidate)
            seen_axis_keys.add(axis_key)
            if len(representative_candidates) >= 3:
                break
    if not representative_candidates:
        representative_candidates = [winning_candidate]
    representative_names = [str(item.get("feature_name") or item.get("feature_id") or "") for item in representative_candidates]
    explanation_cards = [
        {
            "title": "1. Top-3 후보를 먼저 뽑았습니다",
            "summary": f"질문과 가장 가까운 feature 상위 {len(ranked_payload)}개를 먼저 후보로 두었습니다.",
            "details": [
                f"1위 후보는 {ranked_payload[0]['feature_name']}였습니다.",
                f"최종 선택은 {winning_candidate['feature_name']}이지만, top-3 전체를 함께 참고했습니다.",
            ],
        },
        {
            "title": "2. 질문 의도와 직접 맞는지 다시 봤습니다",
            "summary": "질문 단어가 feature 이름, 별칭, 설명에 직접 닿는 후보에 intent 점수를 더했습니다.",
            "details": [
                f"질문 토큰: {', '.join(query_tokens[:6]) or '없음'}",
                f"선택 feature 직접 매칭: {', '.join(winning_candidate.get('matched_feature_name_tokens') or winning_candidate.get('matched_tokens') or []) or '직접 매칭 없음'}",
            ],
        },
        {
            "title": "3. 그래프에서 주변 feature 지지도도 같이 봤습니다",
            "summary": "선택 후보가 다른 상위 후보들과 얼마나 자연스럽게 연결되는지도 같이 계산했습니다.",
            "details": [
                f"주변에서 밀어준 feature: {', '.join(winning_candidate.get('support_labels') or []) or '강한 연결 없음'}",
                f"그래프 연결 수: {int(winning_candidate.get('support_count') or 0)}개",
            ],
        },
        {
            "title": "4. 최종 대표 축을 정했습니다",
            "summary": f"top-3 후보의 base + intent + graph 를 함께 본 뒤 {', '.join(representative_names) or winning_candidate['feature_name']}를 대표 축으로 묶었습니다.",
            "details": [
                f"base {winning_candidate['base_score']:.2f}",
                f"intent {winning_candidate['intent_score']:.2f}",
                f"graph {winning_candidate['graph_score']:.2f}",
                f"hybrid {winning_candidate['hybrid_score']:.2f}",
            ],
        },
    ]

    return winning_feature, {
        "mode": "topk-intent-graph-hybrid",
        "selected_feature_id": str(winning_feature.get("feature_id") or ""),
        "selected_feature_name": str(winning_feature.get("feature_name") or winning_feature.get("feature_id") or "대표 축"),
        "headline": f"top-3 후보를 함께 비교한 뒤 {', '.join(representative_names) or winning_candidate['feature_name']}를 대표 축으로 묶었습니다.",
        "intent_tokens": query_tokens[:8],
        "graph_relation_mode": graph_relation_mode,
        "top_k": ranked_payload,
        "reference_features": ranked_payload[:3],
        "representative_features": representative_candidates,
        "graph_result_explanation": explanation_cards,
    }


def _summarize_customer_cluster(cluster_key: str, items: list[dict[str, object]], reject_code_mapping: dict[str, dict[str, str]] | None = None) -> dict[str, object]:
    reject_code_mapping = reject_code_mapping or {}
    reject_counter = collections.Counter(
        code
        for item in items
        if str(item.get("decision") or "") == "거절"
        for code in (item.get("reject_codes") or [])
    )
    top_codes = [code for code, _ in reject_counter.most_common(3)]
    top_descriptions = _get_reject_code_descriptions(top_codes, reject_code_mapping)
    product = str(items[0].get("product") or "")
    decision = str(items[0].get("decision") or "미상")
    age_band = str(items[0].get("age_band") or "미상")
    income_band = str(items[0].get("income_band") or "미상")
    amount_band = str(items[0].get("amount_band") or "미상")
    reject_summary = ", ".join(top_descriptions[:2])
    income_values = [float(item.get("income")) for item in items if item.get("income") is not None]
    amount_values = [float(item.get("amount")) for item in items if item.get("amount") is not None]
    rate_values = [float(item.get("rate")) for item in items if item.get("rate") is not None]
    model_score_values = [float(item.get("model_score")) for item in items if item.get("model_score") is not None]
    delinquency_rate_values = [float(item.get("delinquency_rate")) for item in items if item.get("delinquency_rate") is not None]
    delinquency_signal_count = sum(1 for item in items if item.get("delinquency_signal"))
    income_sources = collections.Counter(str(item.get("income_source") or "").strip() for item in items if str(item.get("income_source") or "").strip())
    amount_sources = collections.Counter(str(item.get("amount_source") or "").strip() for item in items if str(item.get("amount_source") or "").strip())
    rate_sources = collections.Counter(str(item.get("rate_source") or "").strip() for item in items if str(item.get("rate_source") or "").strip())
    model_score_sources = collections.Counter(str(item.get("model_score_source") or "").strip() for item in items if str(item.get("model_score_source") or "").strip())
    delinquency_rate_sources = collections.Counter(str(item.get("delinquency_rate_source") or "").strip() for item in items if str(item.get("delinquency_rate_source") or "").strip())
    avg_income = round(sum(income_values) / len(income_values), 2) if income_values else None
    avg_amount = round(sum(amount_values) / len(amount_values), 2) if amount_values else None
    avg_rate = round(sum(rate_values) / len(rate_values), 4) if rate_values else None
    avg_model_score = round(sum(model_score_values) / len(model_score_values), 2) if model_score_values else None
    avg_delinquency_rate = round(sum(delinquency_rate_values) / len(delinquency_rate_values), 4) if delinquency_rate_values else None
    delinquency_proxy_rate = round((delinquency_signal_count / len(items)) * 100, 2) if items else None
    label = f"{product} {decision} {age_band} · {income_band} · {amount_band}"
    if reject_summary:
        label = f"{label} / {reject_summary}"
    elif reject_counter:
        label = f"{label} / {reject_counter.most_common(1)[0][0]}"
    return {
        "cluster_id": cluster_key,
        "label": label,
        "product": product,
        "decision": decision,
        "count": len(items),
        "age_band": age_band,
        "income_band": income_band,
        "amount_band": amount_band,
        "avg_income": avg_income,
        "avg_amount": avg_amount,
        "avg_rate": avg_rate,
        "avg_model_score": avg_model_score,
        "avg_delinquency_rate": avg_delinquency_rate,
        "delinquency_proxy_rate": delinquency_proxy_rate,
        "delinquency_signal_count": delinquency_signal_count,
        "avg_income_display": _format_krw_compact(avg_income),
        "avg_amount_display": _format_krw_compact(avg_amount),
        "avg_rate_display": f"{avg_rate:.2f}%" if avg_rate is not None else "",
        "avg_model_score_display": f"{avg_model_score:,.0f}점" if avg_model_score is not None else "",
        "avg_delinquency_rate_display": f"{avg_delinquency_rate:.2f}%" if avg_delinquency_rate is not None else "",
        "delinquency_proxy_rate_display": f"{delinquency_proxy_rate:.1f}%" if delinquency_proxy_rate is not None else "",
        "top_income_source": income_sources.most_common(1)[0][0] if income_sources else "",
        "top_amount_source": amount_sources.most_common(1)[0][0] if amount_sources else "",
        "top_rate_source": rate_sources.most_common(1)[0][0] if rate_sources else "",
        "top_model_score_source": model_score_sources.most_common(1)[0][0] if model_score_sources else "",
        "top_delinquency_rate_source": delinquency_rate_sources.most_common(1)[0][0] if delinquency_rate_sources else "",
        "top_reject_codes": [{"code": code, "count": count} for code, count in reject_counter.most_common(3)],
        "top_reject_descriptions": top_descriptions,
        "reject_summary": reject_summary,
        "search_text": " ".join([
            label,
            " ".join(top_codes),
            " ".join(top_descriptions),
            decision,
        ]).lower(),
        "examples": [
            {
                "record_id": item.get("record_id"),
                "decision": item.get("decision"),
                "reject_reason_text": item.get("reject_reason_text"),
            }
            for item in items[:3]
        ],
    }


def _load_or_build_customer_cluster_cache(records: list[dict], force_rebuild: bool = False) -> dict[str, object]:
    source_mtime_ns = FULL_TEXT_RECORDS_PATH.stat().st_mtime_ns if FULL_TEXT_RECORDS_PATH.exists() else 0
    existing = _read_json_file(FEATURE_CLUSTER_CACHE_PATH)
    cache_meta = existing.get("meta") or {}
    if (
        not force_rebuild
        and
        existing
        and int(cache_meta.get("cache_version") or 0) == FEATURE_CLUSTER_CACHE_VERSION
        and int(cache_meta.get("source_mtime_ns") or 0) == int(source_mtime_ns)
        and int(cache_meta.get("record_count") or 0) == len(records)
    ):
        return existing

    reject_code_mapping = _get_reject_code_mapping()
    decision_results = resolve_product_decisions(records)
    profiles = [
        _build_record_profile(record, reject_code_mapping=reject_code_mapping, decision_result=decision_results.get(index))
        for index, record in enumerate(records)
    ]
    income_thresholds = _derive_measure_band_thresholds(
        [float(item.get("income")) for item in profiles if item.get("income") is not None],
        ["저소득", "중소득", "고소득", "초고소득"],
        DEFAULT_INCOME_BAND_THRESHOLDS,
    )
    amount_thresholds = _derive_measure_band_thresholds(
        [float(item.get("amount")) for item in profiles if item.get("amount") is not None],
        ["소액", "중액", "고액", "초대형"],
        DEFAULT_AMOUNT_BAND_THRESHOLDS,
    )
    profiles = [
        _apply_profile_bands(dict(profile), income_thresholds, amount_thresholds)
        for profile in profiles
    ]
    grouped_all: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    grouped_by_product: dict[str, dict[str, list[dict[str, object]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for profile in profiles:
        cluster_key = str(profile.get("cluster_key") or "미상")
        grouped_all[cluster_key].append(profile)
        grouped_by_product[str(profile.get("product") or "전체")][cluster_key].append(profile)

    cache_payload = {
        "meta": {
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "cache_version": FEATURE_CLUSTER_CACHE_VERSION,
            "source_mtime_ns": int(source_mtime_ns),
            "record_count": len(records),
            "source_path": str(FULL_TEXT_RECORDS_PATH.relative_to(ROOT)),
            "income_band_thresholds": _serialize_thresholds(income_thresholds),
            "amount_band_thresholds": _serialize_thresholds(amount_thresholds),
            "income_sources": [field_name for field_name, _, _ in INCOME_MEASURE_SPECS],
            "amount_sources": [field_name for field_name, _, _ in AMOUNT_MEASURE_SPECS],
        },
        "all": sorted(
            [_summarize_customer_cluster(cluster_key, items, reject_code_mapping=reject_code_mapping) for cluster_key, items in grouped_all.items()],
            key=lambda item: (-int(item["count"]), str(item["label"])),
        ),
        "products": {
            product: sorted(
                [_summarize_customer_cluster(cluster_key, items, reject_code_mapping=reject_code_mapping) for cluster_key, items in clusters.items()],
                key=lambda item: (-int(item["count"]), str(item["label"])),
            )
            for product, clusters in grouped_by_product.items()
        },
    }
    FEATURE_CLUSTER_CACHE_PATH.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_payload


def _score_customer_cluster(cluster: dict[str, object], query: str, selected_feature: dict | None = None, representative_features: list[dict[str, object]] | None = None) -> float:
    score = float(cluster.get("count") or 0)
    search_text = str(cluster.get("search_text") or "")
    age_band_focus = _extract_age_band_focus(query)
    if age_band_focus:
        if str(cluster.get("age_band") or "") == age_band_focus:
            score += 220.0
        else:
            score -= 90.0
    for token in _tokenize_text(query):
        if token and token in search_text:
            score += 4.0
    if _query_has_reject_intent(query, selected_feature):
        decision = str(cluster.get("decision") or "")
        reject_codes = [str(item.get("code") or "") for item in (cluster.get("top_reject_codes") or []) if isinstance(item, dict)]
        score += len(reject_codes) * 12.0
        if decision == "거절":
            score += max(250.0, float(cluster.get("count") or 0) * 1.5)
        else:
            score -= 80.0
    feature_candidates = representative_features or ([selected_feature] if selected_feature else [])
    for feature_candidate in feature_candidates[:3]:
        feature_text = " ".join([
            str(feature_candidate.get("feature_id") or ""),
            str(feature_candidate.get("feature_name") or ""),
            " ".join(str(item) for item in (feature_candidate.get("aliases") or [])),
        ]).lower()
        if any(token and token in search_text for token in _tokenize_text(feature_text)):
            score += 5.0
    axis_keys = {
        str(feature_candidate.get("axis_key") or _infer_feature_axis_key(feature_candidate))
        for feature_candidate in feature_candidates[:3]
    }
    if "rate" in axis_keys and cluster.get("avg_rate") is not None:
        score += 6.0
    if "limit" in axis_keys and cluster.get("avg_amount") is not None:
        score += 6.0
    return score


def _build_cluster_metric_summary(cluster: dict[str, object], representative_features: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    axis_keys = [
        str(item.get("axis_key") or _infer_feature_axis_key(item))
        for item in (representative_features or [])[:3]
    ]
    metrics: list[dict[str, object]] = []
    if "rate" in axis_keys:
        metrics.append({
            "axis_key": "rate",
            "label": "평균 금리",
            "feature_id": "decision.applied_rate",
            "value": cluster.get("avg_rate"),
            "display": str(cluster.get("avg_rate_display") or ""),
            "source": str(cluster.get("top_rate_source") or ""),
        })
    if "limit" in axis_keys:
        metrics.append({
            "axis_key": "limit",
            "label": "평균 한도",
            "feature_id": "decision.approved_amount",
            "value": cluster.get("avg_amount"),
            "display": str(cluster.get("avg_amount_display") or ""),
            "source": str(cluster.get("top_amount_source") or ""),
        })
    return [item for item in metrics if item.get("value") is not None or item.get("display")]


def _build_customer_clusters(records: list[dict], selected_product: str, query: str = "", selected_feature: dict | None = None, representative_features: list[dict[str, object]] | None = None, limit: int = 6) -> list[dict]:
    cache_payload = _load_or_build_customer_cluster_cache(records)
    if selected_product:
        clusters = list((cache_payload.get("products") or {}).get(selected_product) or [])
    else:
        clusters = list(cache_payload.get("all") or [])

    # 군집을 상품+연령+금액+탈락사유 조합별로 세분화
    def cluster_key(item):
        return (
            str(item.get("product") or ""),
            str(item.get("age_band") or ""),
            str(item.get("amount_band") or ""),
            str(item.get("income_band") or ""),
            str(item.get("decision") or ""),
            ",".join(sorted([str(code.get("code") or code) for code in (item.get("top_reject_codes") or [])]))
        )
    # 중복 조합 제거 및 대표 군집만 추출
    seen = set()
    unique_clusters = []
    for c in clusters:
        k = cluster_key(c)
        if k not in seen:
            seen.add(k)
            unique_clusters.append(c)

    age_band_focus = _extract_age_band_focus(query)
    income_band_focus = _extract_income_band_focus(query)
    amount_band_focus = _extract_amount_band_focus(query)
    decision_focus = _extract_decision_focus(query)
    if age_band_focus:
        unique_clusters = [item for item in unique_clusters if str(item.get("age_band") or "") == age_band_focus]
    if income_band_focus:
        unique_clusters = [item for item in unique_clusters if str(item.get("income_band") or "") == income_band_focus]
    if amount_band_focus:
        unique_clusters = [item for item in unique_clusters if str(item.get("amount_band") or "") == amount_band_focus]
    if decision_focus:
        unique_clusters = [item for item in unique_clusters if str(item.get("decision") or "") == decision_focus]

    reject_intent = _query_has_reject_intent(query, selected_feature)
    approval_metric_focus = _query_has_metric_intent(query) and (_query_has_rate_intent(query) or _query_has_limit_intent(query)) and not reject_intent
    unique_clusters.sort(
        key=lambda item: (
            0 if (reject_intent and str(item.get("decision") or "") == "거절")
            else (0 if approval_metric_focus and str(item.get("decision") or "") == "승인" else 1 if (reject_intent or approval_metric_focus) else 0),
            -_score_customer_cluster(item, query, selected_feature, representative_features=representative_features),
            -int(item.get("count") or 0),
            str(item.get("label") or ""),
        )
    )
    # 군집 수 제한 완화 (최대 20개까지 반환)
    max_limit = max(limit, 20)
    enriched_clusters: list[dict] = []
    for cluster in unique_clusters[:max_limit]:
        enriched = dict(cluster)
        enriched["metric_summary"] = _build_cluster_metric_summary(enriched, representative_features)
        enriched_clusters.append(enriched)
    return enriched_clusters


def _build_retrieval_results(records: list[dict], selected_product: str, query: str, selected_feature: dict | None, representative_features: list[dict[str, object]] | None = None, limit: int = 6) -> list[dict]:
    reject_code_mapping = _get_reject_code_mapping()
    profiles = [_build_record_profile(record, reject_code_mapping=reject_code_mapping) for record in records]
    feature_candidates = representative_features or ([selected_feature] if selected_feature else [])
    feature_terms: list[str] = []
    for feature_candidate in feature_candidates[:3]:
        feature_terms.extend([
            str(feature_candidate.get("feature_name") or ""),
            str(feature_candidate.get("feature_id") or ""),
            " ".join(feature_candidate.get("aliases") or []),
        ])
    tokens = _tokenize_text(" ".join(filter(None, [query, *feature_terms])))
    reject_intent = _query_has_reject_intent(query, selected_feature)
    age_band_focus = _extract_age_band_focus(query)
    scored_by_record: dict[str, dict[str, object]] = {}
    for profile in profiles:
        if selected_product and profile.get("product") != selected_product:
            continue
        if age_band_focus and str(profile.get("age_band") or "") != age_band_focus:
            continue
        score = 0
        if selected_product and profile.get("product") == selected_product:
            score += 2
        search_text = str(profile.get("search_text") or "")
        for token in tokens:
            if token and token in search_text:
                score += 3
        for feature_candidate in feature_candidates[:3]:
            category = str(feature_candidate.get("category") or "")
            if category and category in search_text:
                score += 1
        if reject_intent:
            if str(profile.get("decision") or "") != "거절":
                continue
            reject_codes = profile.get("reject_codes") or []
            reject_descriptions = profile.get("reject_descriptions") or []
            score += len(reject_codes) * 4
            score += len(reject_descriptions) * 2
            if str(profile.get("decision") or "") == "거절":
                score += 4
        if score <= 0:
            continue
        record_key = str(profile.get("record_id") or "")
        candidate = {
            "record_id": profile.get("record_id"),
            "product": profile.get("product"),
            "product_display": profile.get("product_display"),
            "decision": profile.get("decision"),
            "score": score,
            "rate": profile.get("rate"),
            "rate_source": profile.get("rate_source"),
            "amount": profile.get("amount"),
            "amount_source": profile.get("amount_source"),
            "reject_codes": profile.get("reject_codes") or [],
            "reject_descriptions": profile.get("reject_descriptions") or [],
            "snippet": str(profile.get("search_text") or "")[:260],
        }
        existing = scored_by_record.get(record_key)
        if existing is None or int(candidate["score"]) > int(existing["score"]):
            scored_by_record[record_key] = candidate
    scored = list(scored_by_record.values())
    return sorted(scored, key=lambda item: (-int(item["score"]), str(item["record_id"])))[:limit]


def _detect_financial_agent_intents(query: str, selected_feature: dict | None = None) -> list[str]:
    compact_query = _compact_search_text(query)
    intents: list[str] = []
    metric_intent = _query_has_metric_intent(query)
    strategy_intent = _query_requires_strategy_simulation(query)
    reason_intent = _query_has_reject_intent(query, selected_feature) or any(token in compact_query for token in ["왜", "이유", "설명", "사유", "탈락", "거절"])
    if _query_requires_regulation_first(query):
        intents.append("policy")
    if _query_asks_cluster_signals(query):
        intents.append("cluster")
    if metric_intent:
        intents.append("cluster")
    if reason_intent:
        intents.append("explainability")
    if any(token in compact_query for token in ["군집", "고객군", "고객군집", "유사고객", "비슷한고객", "cluster", "세그먼트"]):
        intents.append("cluster")
    if any(token in compact_query for token in ["정책", "규제", "룰", "rule", "ontology", "온톨로지", "충돌", "conflict"]):
        intents.append("policy")
    if any(token in compact_query for token in ["전략", "시뮬레이션", "완화", "신상품", "상품만들", "수익", "부실", "simulation", "strategy"]):
        intents.append("strategy")
    if any(token in compact_query for token in ["영업점", "신용기획", "솔루션", "운영", "부서", "관점"]):
        intents.append("persona")
    if strategy_intent:
        intents.append("strategy")
    if not intents:
        intents.append("cluster" if metric_intent else "explainability")
    return _dedupe_text_items(intents, limit=5)


def _query_requires_strategy_simulation(query: str) -> bool:
    compact_query = _compact_search_text(query)
    scenario_markers = [
        "예측", "예상", "가정", "변화", "변하면", "바꾸면", "올리면", "올렸을때",
        "낮추면", "내리면", "인상", "인하", "상향", "하향", "조정", "늘리면", "줄이면",
        "완화", "강화", "시뮬레이션", "simulation", "strategy", "whatif", "what-if",
        "어떻게될까", "어떻게돼", "어떻게될지", "영향", "impact",
    ]
    business_markers = [
        "금리", "한도", "dsr", "거절코드", "승인률", "승인율", "수익", "부실", "리스크",
        "전환", "심사기준", "대출", "상품",
    ]
    return (
        any(token in compact_query for token in scenario_markers)
        and any(token in compact_query for token in business_markers)
    )


def _query_has_metric_intent(query: str) -> bool:
    compact_query = _compact_search_text(query)
    metric_markers = ["평균", "금리", "한도", "연체율", "부실률", "승인율", "지표", "분포", "수준", "얼마", "rate", "limit", "delinquency", "default"]
    return any(marker in compact_query for marker in metric_markers)


def _normalize_percentages(items: list[dict[str, object]], value_key: str = "impact") -> list[dict[str, object]]:
    total = sum(max(0, float(item.get(value_key) or 0)) for item in items)
    if total <= 0:
        return items
    normalized: list[dict[str, object]] = []
    for item in items:
        next_item = dict(item)
        next_item[value_key] = round((max(0, float(item.get(value_key) or 0)) / total) * 100, 1)
        normalized.append(next_item)
    return normalized


def _build_explainability_agent_result(
    query: str,
    selected_product: str,
    selected_feature: dict | None,
    representative_features: list[dict[str, object]],
    customer_clusters: list[dict],
    reject_code_summary: list[dict[str, object]],
) -> dict[str, object]:
    return build_explainability_payload(
        query=query,
        selected_product=selected_product,
        selected_feature=selected_feature,
        representative_features=representative_features,
        customer_clusters=customer_clusters,
        reject_code_summary=reject_code_summary,
        has_reject_intent=_query_has_reject_intent,
        has_metric_intent=_query_has_metric_intent,
        has_rate_intent=_query_has_rate_intent,
        has_limit_intent=_query_has_limit_intent,
        asks_cluster_signals=_query_asks_cluster_signals,
        compact_search_text=_compact_search_text,
        normalize_percentages=_normalize_percentages,
        product_display_name=_product_display_name,
        is_cross_product_feature_label=_is_cross_product_feature_label,
    )


def _format_metric_value(value: object, suffix: str = "") -> str:
    if value is None or value == "":
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if suffix:
        return f"{numeric:,.1f}{suffix}"
    return f"{numeric:,.0f}"


def _cluster_metric_cards(clusters: list[dict[str, object]]) -> list[dict[str, object]]:
    total_records = sum(int(item.get("count") or item.get("records") or 0) for item in clusters)
    rate_values = [float(item.get("avg_rate")) for item in clusters if item.get("avg_rate") is not None]
    amount_values = [float(item.get("avg_amount")) for item in clusters if item.get("avg_amount") is not None]
    score_values = [float(item.get("avg_model_score")) for item in clusters if item.get("avg_model_score") is not None]
    delinquency_values = [
        float(item.get("avg_delinquency_rate") if item.get("avg_delinquency_rate") is not None else item.get("delinquency_proxy_rate"))
        for item in clusters
        if item.get("avg_delinquency_rate") is not None or item.get("delinquency_proxy_rate") is not None
    ]
    return [
        {"label": "고객군", "value": f"{len(clusters)}개", "tone": "neutral"},
        {"label": "대상 로그", "value": f"{total_records:,}건", "tone": "neutral"},
        {"label": "평균 금리", "value": f"{(sum(rate_values) / len(rate_values)):.2f}%" if rate_values else "-", "tone": "warning"},
        {"label": "평균 한도", "value": _format_krw_compact(sum(amount_values) / len(amount_values)) if amount_values else "-", "tone": "positive"},
        {"label": "모델 점수", "value": _format_metric_value(sum(score_values) / len(score_values), "점") if score_values else "-", "tone": "neutral"},
        {"label": "연체/부실 proxy", "value": f"{(sum(delinquency_values) / len(delinquency_values)):.1f}%" if delinquency_values else "-", "tone": "warning"},
    ]


def _build_cluster_visualization(clusters: list[dict[str, object]]) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for index, item in enumerate(clusters[:6]):
        avg_amount = float(item.get("avg_amount") or 0)
        avg_rate = float(item.get("avg_rate") or 0)
        points.append({
            "id": item.get("cluster_id") or f"cluster-{index + 1}",
            "label": item.get("display_label") or item.get("label") or f"고객군 {index + 1}",
            "x": round(avg_amount, 2),
            "x_display": item.get("avg_amount_display") or "-",
            "y": round(avg_rate, 4),
            "y_display": item.get("avg_rate_display") or "-",
            "size": int(item.get("count") or 0),
            "risk": item.get("avg_delinquency_rate_display") or item.get("delinquency_proxy_rate_display") or "-",
            "score": item.get("avg_model_score_display") or "-",
        })
    return {
        "type": "bubble",
        "x_label": "평균 한도",
        "y_label": "평균 금리",
        "points": points,
    }


def _build_cluster_shap_values(clusters: list[dict[str, object]]) -> list[dict[str, object]]:
    top_cluster = clusters[0] if clusters else {}
    candidates: list[dict[str, object]] = []
    if top_cluster.get("avg_model_score") is not None:
        candidates.append({
            "feature": "실제 모델 점수",
            "impact": 32,
            "direction": "model_score",
            "evidence": top_cluster.get("avg_model_score_display") or top_cluster.get("top_model_score_source") or "score",
        })
    if top_cluster.get("delinquency_proxy_rate") is not None or top_cluster.get("avg_delinquency_rate") is not None:
        candidates.append({
            "feature": "연체/부실 신호",
            "impact": 28,
            "direction": "risk_up",
            "evidence": top_cluster.get("avg_delinquency_rate_display") or top_cluster.get("delinquency_proxy_rate_display") or "proxy",
        })
    if top_cluster.get("avg_rate") is not None:
        candidates.append({
            "feature": "평균 금리",
            "impact": 22,
            "direction": "price",
            "evidence": top_cluster.get("avg_rate_display") or "",
        })
    if top_cluster.get("avg_amount") is not None:
        candidates.append({
            "feature": "평균 한도",
            "impact": 18,
            "direction": "limit",
            "evidence": top_cluster.get("avg_amount_display") or "",
        })
    if top_cluster.get("income_band"):
        candidates.append({
            "feature": str(top_cluster.get("income_band")),
            "impact": 12,
            "direction": "segment",
            "evidence": "고객군집",
        })
    return _normalize_percentages(candidates[:5])


def _build_cluster_intelligence_result(customer_clusters: list[dict], selected_product: str, query: str = "") -> dict[str, object]:
    clusters: list[dict[str, object]] = []
    metric_question = _query_has_metric_intent(query)
    reject_intent = _query_has_reject_intent(query, None)
    approval_metric_focus = metric_question and (_query_has_rate_intent(query) or _query_has_limit_intent(query)) and not reject_intent
    ranked_clusters = sorted(
        customer_clusters,
        key=lambda item: (
            0 if metric_question and (item.get("avg_rate") is not None or item.get("avg_amount") is not None) else 1,
            -int(item.get("count") or 0),
        ),
    )
    approved_ranked_clusters = [item for item in ranked_clusters if str(item.get("decision") or "") == "승인"]
    ranked_source = approved_ranked_clusters if (approval_metric_focus and approved_ranked_clusters) else ranked_clusters
    analysis_basis = "승인 고객군 기준" if ranked_source is approved_ranked_clusters else "전체 고객군 기준"
    for item in ranked_source[:4]:
        count = int(item.get("count") or 0)
        approval_rate = 100.0 if str(item.get("decision") or "") == "승인" else 0.0
        product_label = _product_display_name(str(item.get("product") or selected_product or "")) or str(item.get("product") or selected_product or "전체")
        display_label = " / ".join(
            part for part in [
                product_label,
                str(item.get("age_band") or "").strip(),
                str(item.get("income_band") or "").strip(),
                str(item.get("amount_band") or "").strip(),
            ]
            if part and part != "미상"
        )
        delinquency_display = str(item.get("avg_delinquency_rate_display") or item.get("delinquency_proxy_rate_display") or "-")
        clusters.append({
            "cluster_id": item.get("cluster_id"),
            "label": item.get("label") or item.get("cluster_id") or "고객군",
            "display_label": display_label or item.get("label") or item.get("cluster_id") or "고객군",
            "records": count,
            "decision": item.get("decision") or "미상",
            "approval_rate": approval_rate,
            "avg_rate": item.get("avg_rate_display") or "-",
            "avg_limit": item.get("avg_amount_display") or "-",
            "avg_model_score": item.get("avg_model_score_display") or "-",
            "model_score_source": item.get("top_model_score_source") or "",
            "delinquency_rate": delinquency_display,
            "delinquency_source": item.get("top_delinquency_rate_source") or ("연체/부실 로그 proxy" if item.get("delinquency_proxy_rate_display") else ""),
            "risk_pattern": (
                "승인군 금리/한도/소득 패턴"
                if str(item.get("decision") or "") == "승인"
                else (_format_cluster_reject_reason_summary(item) or "소득/한도/연령 패턴")
            ),
        })
    return {
        "id": "cluster",
        "title": "Customer Cluster Intelligence",
        "status": "ready" if clusters else "empty",
        "summary": (
            f"{_product_display_name(selected_product) or selected_product or '전체'} 기준으로 "
            f"{len(clusters)}개 고객군을 비교했습니다. ({analysis_basis})"
            if clusters
            else "조건에 맞는 고객군을 찾지 못했습니다."
        ),
        "llm_call": False,
        "analysis_basis": analysis_basis,
        "metrics": _cluster_metric_cards(ranked_source[:4]),
        "clusters": clusters,
        "shap_values": _build_cluster_shap_values(ranked_source[:4]),
        "visualization": _build_cluster_visualization(ranked_source[:4]),
    }


def _build_policy_conflict_result(
    regulation_evidence: list[dict[str, object]],
    regulation_evidence_reason: str,
    answer_summary: dict[str, object],
) -> dict[str, object]:
    citations = list((answer_summary.get("citations") or [])[:3])
    if not citations and regulation_evidence:
        citations = [_normalize_regulation_citation(item) for item in regulation_evidence[:3]]
    conflicts: list[dict[str, object]] = []
    if citations:
        conflicts.append({
            "level": "review",
            "title": "규제 근거와 답변 연결 확인 필요",
            "detail": "규제 문서 evidence가 답변 근거로 노출되어 내부 Rule과 해석 일치 여부를 확인할 수 있습니다.",
        })
    elif regulation_evidence_reason in {"not_answer_relevant", "no_query_regulation_intent"}:
        conflicts.append({
            "level": "clear",
            "title": "답변 관련 규제 충돌 없음",
            "detail": "현재 질문 답변에는 규제 문서 근거가 직접 연결되지 않아 citations를 숨겼습니다.",
        })
    return {
        "id": "policy",
        "title": "Policy/Ontology Reasoning",
        "status": "ready",
        "summary": "규제 문서, 내부 Rule, 실제 심사 결과의 연결 여부를 확인합니다.",
        "llm_call": False,
        "citations": citations,
        "conflicts": conflicts,
        "evidence_count": len(regulation_evidence),
    }


def _build_strategy_simulation_result(
    customer_clusters: list[dict],
    reject_code_summary: list[dict[str, object]],
    query: str = "",
) -> dict[str, object]:
    total_records = sum(int(item.get("count") or 0) for item in customer_clusters) or sum(int(item.get("base_rejected_records") or 0) for item in reject_code_summary[:1]) or 1
    rejected_records = sum(int(item.get("count") or 0) for item in customer_clusters if str(item.get("decision") or "") == "거절")
    first_reject = reject_code_summary[0] if reject_code_summary else {}
    top_reject_share = float(first_reject.get("share") or 0.18)
    estimated_uplift = round(min(8.5, max(1.2, top_reject_share * 7.5)), 1)
    estimated_profit = round(total_records * estimated_uplift * 0.018, 1)
    estimated_risk = round(min(4.8, max(0.4, estimated_uplift * 0.32)), 1)
    compact_query = _compact_search_text(query)
    weighted_count = max(1, total_records)
    weighted_rate_values = [
        (_to_number(str(item.get("avg_rate") or "")) or 0.0) * int(item.get("count") or 0)
        for item in customer_clusters
        if item.get("avg_rate") is not None
    ]
    baseline_rate = sum(weighted_rate_values) / weighted_count if weighted_rate_values else 12.0
    baseline_approval_rate = round(max(0.0, min(100.0, ((total_records - rejected_records) / weighted_count) * 100)), 1)
    if any(token in compact_query for token in ["금리", "rate", "이자"]):
        rate_direction = "인상" if any(token in compact_query for token in ["올리", "인상", "상향", "높이"]) else ("인하" if any(token in compact_query for token in ["낮추", "내리", "인하", "하향"]) else "조정")
        summary = f"금리 {rate_direction} 시 승인률, 예상 이자수익, 부실률 변화를 고객군 기준으로 가정합니다."
        scenario = f"금리 {rate_direction} 민감도 시뮬레이션"
        metric_labels = ["승인률 민감도", "예상 이자수익", "부실률 민감도"]
    elif any(token in compact_query for token in ["한도", "limit"]):
        limit_direction = "확대" if any(token in compact_query for token in ["늘리", "올리", "상향", "확대"]) else ("축소" if any(token in compact_query for token in ["줄이", "낮추", "하향", "축소"]) else "조정")
        summary = f"한도 {limit_direction} 시 승인 전환, 노출 금액, 부실률 변화를 고객군 기준으로 가정합니다."
        scenario = f"한도 {limit_direction} 민감도 시뮬레이션"
        metric_labels = ["승인 전환 변화", "예상 노출/수익", "부실률 민감도"]
    else:
        summary = "질문에 포함된 전략 조건을 바꿨을 때 승인률, 수익, 리스크 변화를 빠르게 가정합니다."
        scenario = "전략 조건 민감도 시뮬레이션"
        metric_labels = ["승인률 변화", "예상 수익 변화", "부실률 변화"]
    rate_question = any(token in compact_query for token in ["금리", "湲덈━", "rate", "?댁옄"])
    limit_question = any(token in compact_query for token in ["한도", "?쒕룄", "limit"])
    rate_down = any(token in compact_query for token in ["내리", "내리면", "인하", "하향", "낮추", "낮추면"])
    limit_down = any(token in compact_query for token in ["줄이", "줄이면", "축소", "감액", "낮추", "낮추면"])
    scenario_rows: list[dict[str, object]] = []
    if rate_question:
        signed_steps = [-0.2, -0.5, -1.0] if rate_down else [0.2, 0.5, 1.0]
        for step in signed_steps:
            approval_delta = round((-0.9 * step) if step > 0 else abs(step) * 1.1, 1)
            profit_delta = round(((step / max(1.0, baseline_rate)) * 100) + (approval_delta * 0.45), 1)
            risk_delta = round((0.45 * step) if step > 0 else abs(step) * 0.35, 1)
            scenario_rows.append({
                "label": f"{step:+.1f}%p",
                "change": round(step, 1),
                "change_type": "rate_pp",
                "approval_delta": approval_delta,
                "profit_delta": profit_delta,
                "risk_delta": risk_delta,
                "approval_rate_after": round(max(0.0, min(100.0, baseline_approval_rate + approval_delta)), 1),
                "note": "금리 인하" if step < 0 else "금리 인상",
            })
        scenario = "금리 인하 구간별 시뮬레이션" if rate_down else "금리 인상 구간별 시뮬레이션"
        summary = "금리 조정 폭별로 승인률, 예상 이자수익, 부실률 변화를 고객군 기준으로 가정합니다."
    elif limit_question:
        signed_steps = [-5, -10, -20] if limit_down else [5, 10, 20]
        for step in signed_steps:
            approval_delta = round((0.08 * step) if step > 0 else -0.05 * abs(step), 1)
            profit_delta = round((0.62 * step) if step > 0 else -0.42 * abs(step), 1)
            risk_delta = round((0.1 * step) if step > 0 else -0.06 * abs(step), 1)
            scenario_rows.append({
                "label": f"{step:+.0f}%",
                "change": round(step, 1),
                "change_type": "limit_pct",
                "approval_delta": approval_delta,
                "profit_delta": profit_delta,
                "risk_delta": risk_delta,
                "approval_rate_after": round(max(0.0, min(100.0, baseline_approval_rate + approval_delta)), 1),
                "note": "한도 축소" if step < 0 else "한도 확대",
            })
        scenario = "한도 축소 구간별 시뮬레이션" if limit_down else "한도 확대 구간별 시뮬레이션"
        summary = "한도 조정 폭별로 승인 전환, 예상 이자수익, 부실률 변화를 고객군 기준으로 가정합니다."
    if scenario_rows:
        selected_row = scenario_rows[-1]
        estimated_uplift = float(selected_row.get("approval_delta") or 0.0)
        estimated_profit = float(selected_row.get("profit_delta") or 0.0)
        estimated_risk = float(selected_row.get("risk_delta") or 0.0)
    segment_impacts: list[dict[str, object]] = []
    for cluster in customer_clusters[:3]:
        cluster_count = int(cluster.get("count") or 0)
        cluster_share = round((cluster_count / weighted_count) * 100, 1)
        segment_impacts.append({
            "label": str(cluster.get("label") or cluster.get("cluster_id") or "고객군"),
            "decision": str(cluster.get("decision") or "미상"),
            "records": cluster_count,
            "share": cluster_share,
            "avg_rate": str(cluster.get("avg_rate_display") or cluster.get("avg_rate") or "-"),
            "avg_limit": str(cluster.get("avg_amount_display") or cluster.get("avg_limit") or "-"),
            "manager_note": "표본 비중이 커서 전체 수익 민감도에 영향이 큼" if cluster_share >= 25 else "보조 고객군으로 방향성 확인",
        })
    return {
        "id": "strategy",
        "title": "Strategy Simulation",
        "status": "ready",
        "summary": summary,
        "llm_call": True,
        "scenario": scenario,
        "baseline_rejected_records": rejected_records,
        "baseline": {
            "approval_rate": baseline_approval_rate,
            "avg_rate": round(baseline_rate, 2),
            "record_count": total_records,
        },
        "scenario_rows": scenario_rows,
        "segment_impacts": segment_impacts,
        "metrics": [
            {"label": metric_labels[0], "value": f"{estimated_uplift:+.1f}%", "tone": "positive" if estimated_uplift >= 0 else "warning"},
            {"label": metric_labels[1], "value": f"{estimated_profit:+.1f} index", "tone": "positive" if estimated_profit >= 0 else "warning"},
            {"label": metric_labels[2], "value": f"{estimated_risk:+.1f}%", "tone": "warning" if estimated_risk >= 0 else "positive"},
        ],
    }


def _build_department_persona_result(answer_summary: dict[str, object], customer_clusters: list[dict]) -> dict[str, object]:
    top_cluster = customer_clusters[0] if customer_clusters else {}
    headline = str(answer_summary.get("headline") or "분석 결과")
    return {
        "id": "persona",
        "title": "Department Persona View",
        "status": "ready",
        "summary": "같은 semantic 결과를 부서별 언어로 재해석합니다.",
        "llm_call": False,
        "personas": [
            {"name": "영업점", "focus": "상담 전환", "view": "고객에게 설명 가능한 사유와 보완 액션을 우선 확인합니다."},
            {"name": "신용기획", "focus": "Rule 개선", "view": f"{headline} 기준으로 정책 임계값과 리스크 패턴을 점검합니다."},
            {"name": "솔루션 운영", "focus": "운영 안정성", "view": f"{top_cluster.get('cluster_id') or '상위 군집'} 처리 흐름과 예외 케이스를 확인합니다."},
        ],
    }


def _build_agentic_workspace_payload(
    query: str,
    selected_product: str,
    selected_feature: dict | None,
    representative_features: list[dict[str, object]],
    customer_clusters: list[dict],
    reject_code_summary: list[dict[str, object]],
    regulation_evidence: list[dict[str, object]],
    regulation_evidence_reason: str,
    answer_summary: dict[str, object],
) -> dict[str, object]:
    intents = _detect_financial_agent_intents(query, selected_feature)
    explainability = _build_explainability_agent_result(query, selected_product, selected_feature, representative_features, customer_clusters, reject_code_summary)
    cluster = _build_cluster_intelligence_result(customer_clusters, selected_product, query=query)
    policy = _build_policy_conflict_result(regulation_evidence, regulation_evidence_reason, answer_summary)
    strategy = _build_strategy_simulation_result(customer_clusters, reject_code_summary, query=query)
    persona = _build_department_persona_result(answer_summary, customer_clusters)
    tool_map = {
        "explainability": explainability,
        "cluster": cluster,
        "policy": policy,
        "strategy": strategy,
        "persona": persona,
    }
    default_tool_order = ["policy"] if intents == ["policy"] else (["cluster"] if "cluster" in intents and "explainability" not in intents else ["explainability", "cluster"])
    active_tool_ids = _dedupe_text_items([
        *(intent for intent in intents if intent in tool_map),
        *default_tool_order,
    ], limit=4)
    active_tools = [tool_map[item] for item in active_tool_ids if item in tool_map]
    return {
        "mode": "conversational_workspace",
        "version_1": {
            "name": "AI Character Conversational Workspace",
            "principle": "캐릭터와 대화하는 흐름 안에서 필요한 Tool Card만 노출합니다.",
            "active_tools": active_tools,
        },
        "version_2": {
            "name": "Strategy Analytics Workspace",
            "principle": "동일 Semantic Layer와 Agent Workflow를 전략/운영 관점으로 재배치합니다.",
            "panels": [explainability, cluster, policy, strategy, persona],
        },
        "agent_workflow": [
            {"step": 1, "agent": "Intent Detection", "type": "router", "llm_call": False, "status": "done", "output": ", ".join(intents)},
            {"step": 2, "agent": "GraphRAG Retrieval", "type": "tool", "llm_call": False, "status": "done", "output": "semantic_context"},
            {"step": 3, "agent": "Cluster Analysis", "type": "tool", "llm_call": False, "status": "done", "output": f"{len(customer_clusters)} clusters"},
            {"step": 4, "agent": "Policy/Ontology Check", "type": "tool", "llm_call": False, "status": "done", "output": f"{len(regulation_evidence)} policy evidence"},
            {"step": 5, "agent": "Explainability Analysis", "type": "tool", "llm_call": False, "status": "done", "output": f"{len(explainability.get('shap_values') or [])} factors"},
            {"step": 6, "agent": "Ollama Final Response", "type": "nlg", "llm_call": True, "status": "minimal", "output": str(answer_summary.get("source") or "summary")},
        ],
        "semantic_layer": {
            "graph": "GraphRAG/ontology payload",
            "vector": "FAISS prebuilt cache",
            "policy": "regulation evidence cache",
            "neo4j_ready": True,
            "runtime_policy": "Ollama는 최종 자연어 생성 중심으로만 사용",
        },
    }


def _build_customer_cluster_api_payload(selected_product: str = "", limit: int = 12, force_rebuild: bool = False) -> dict[str, object]:
    records = _read_record_list(FULL_TEXT_RECORDS_PATH)
    cache_payload = _load_or_build_customer_cluster_cache(records, force_rebuild=force_rebuild)
    products = cache_payload.get("products") or {}
    selected_clusters = list(products.get(selected_product) or []) if selected_product else list(cache_payload.get("all") or [])
    meta = cache_payload.get("meta") or {}
    return {
        "status": "ok",
        "input": {
            "product": selected_product,
            "limit": limit,
            "force_rebuild": force_rebuild,
        },
        "meta": {
            "built_at": str(meta.get("built_at") or ""),
            "record_count": int(meta.get("record_count") or 0),
            "source_path": str(meta.get("source_path") or ""),
            "path": str(FEATURE_CLUSTER_CACHE_PATH.relative_to(ROOT)),
            "product_count": len(products),
            "cluster_count": len(selected_clusters),
            "products": sorted(products.keys()),
        },
        "clusters": selected_clusters[:limit],
    }


def _build_segment_metric_cube_api_payload(force_rebuild: bool = False) -> dict[str, object]:
    if force_rebuild or not DEFAULT_SEGMENT_CUBE_PATH.exists():
        write_segment_metric_cube(FULL_TEXT_RECORDS_PATH, DEFAULT_SEGMENT_CUBE_PATH)
    cube_payload = load_segment_metric_cube(DEFAULT_SEGMENT_CUBE_PATH)
    segments = list(cube_payload.get("segments") or [])
    meta = dict(cube_payload.get("meta") or {})
    product_segments = [
        item
        for item in segments
        if set((item.get("dimensions") or {}).keys()) == {"product"}
    ]
    product_summaries = []
    for segment in sorted(product_segments, key=lambda item: str((item.get("dimensions") or {}).get("product") or "")):
        dimensions = dict(segment.get("dimensions") or {})
        product_code = str(dimensions.get("product") or "ALL")
        if product_code == "ALL":
            continue
        product_summaries.append({
            "product": product_code,
            "product_label": _product_display_name(product_code) or product_code,
            "count": int(segment.get("count") or 0),
            "approval_rate_percent": segment.get("approval_rate_percent"),
            "rejection_rate_percent": segment.get("rejection_rate_percent"),
            "avg_rate_display": str(segment.get("avg_rate_display") or ""),
            "avg_amount_display": str(segment.get("avg_amount_display") or ""),
            "delinquency_proxy_rate_display": str(segment.get("delinquency_proxy_rate_display") or ""),
            "top_reject_codes": list(segment.get("top_reject_codes") or [])[:3],
            "reliability": str(segment.get("reliability") or ""),
        })

    grain_counter: collections.Counter[str] = collections.Counter(str(item.get("grain") or "") for item in segments)
    reliability_counter: collections.Counter[str] = collections.Counter(str(item.get("reliability") or "unknown") for item in segments)
    records = _read_record_list(FULL_TEXT_RECORDS_PATH)
    cluster_payload = _load_or_build_customer_cluster_cache(records)
    cluster_products = dict(cluster_payload.get("products") or {})
    cluster_summary = {
        "total_clusters": len(cluster_payload.get("all") or []),
        "record_count": int((cluster_payload.get("meta") or {}).get("record_count") or 0),
        "products": [
            {
                "product": product,
                "product_label": _product_display_name(product) or product,
                "cluster_count": len(items or []),
                "top_clusters": [
                    {
                        "label": str(item.get("label") or item.get("cluster_id") or ""),
                        "decision": str(item.get("decision") or ""),
                        "count": int(item.get("count") or 0),
                        "avg_rate_display": str(item.get("avg_rate_display") or ""),
                        "avg_amount_display": str(item.get("avg_amount_display") or ""),
                        "delinquency_proxy_rate_display": str(item.get("delinquency_proxy_rate_display") or ""),
                    }
                    for item in list(items or [])[:3]
                ],
            }
            for product, items in sorted(cluster_products.items())
        ],
    }
    return {
        "status": "ok",
        "meta": {
            "generated_at": str(meta.get("generated_at") or ""),
            "source_path": str(meta.get("source_path") or ""),
            "record_count": int(meta.get("record_count") or 0),
            "segment_count": int(meta.get("segment_count") or len(segments)),
            "products": list(meta.get("products") or []),
            "dimensions": list(meta.get("dimensions") or []),
            "income_band_thresholds": list(meta.get("income_band_thresholds") or []),
            "amount_band_thresholds": list(meta.get("amount_band_thresholds") or []),
            "path": str(DEFAULT_SEGMENT_CUBE_PATH.relative_to(ROOT)),
        },
        "product_summaries": product_summaries,
        "grain_summary": [
            {"grain": grain, "count": count}
            for grain, count in grain_counter.most_common(12)
            if grain
        ],
        "reliability_summary": [
            {"label": label, "count": count}
            for label, count in reliability_counter.most_common()
        ],
        "query_examples": [
            "이지신용대출 평균 금리와 한도는?",
            "40대 카드론 승인률은?",
            "저소득 카드론 연체 위험은?",
            "이지신용대출 거절사유코드 상위 3개는?",
            "승인 고객군과 거절 고객군 평균 한도를 비교해줘",
        ],
        "available_metrics": [
            {"label": "승인률/거절률", "detail": "상품, 연령대, 소득구간, 한도구간, 거절사유코드별 비율"},
            {"label": "평균 금리/한도", "detail": "실제 심사 로그에서 산출된 금리와 승인 가능 금액 평균"},
            {"label": "연체 위험 신호율", "detail": "실제 연체율이 없는 구간은 로그 기반 proxy로 표시"},
            {"label": "상위 거절사유코드", "detail": "K로 시작하는 거절사유코드와 한글 설명 TOP"},
        ],
        "cluster_summary": cluster_summary,
    }


PRODUCT_DEVELOPMENT_DEPARTMENTS = {
    "solution": {
        "name": "금융솔루션부",
        "icon": "🧭",
        "default_concept": "신상품 총괄과 상품 포트폴리오 관점에서 규제, 뉴스, 시장 변화를 함께 봅니다.",
    },
    "credit": {
        "name": "신용기획부",
        "icon": "📊",
        "default_concept": "심사솔루션 기준으로 상품별 금리, 한도, 거절코드와 리스크 통제 기준을 봅니다.",
    },
    "sales": {
        "name": "금융영업부",
        "icon": "🚀",
        "default_concept": "취급량, 취급률, 전환율 목표를 중시하며 승인 가능 고객을 공격적으로 찾습니다.",
    },
    "it": {
        "name": "IT개발자",
        "icon": "🛠️",
        "default_concept": "KCB, NICE, 신정원, 공공마이데이터 연계와 개발공수, 정합성 검증, 운영 영향도를 봅니다.",
    },
}

PRODUCT_DEVELOPMENT_PRODUCT_NAMES = {
    "C6": "이지신용대출(C6)",
    "C9": "이지론(C9)",
    "C11": "개인사업자대출(C11)",
    "C12": "이지대환대출(C12)",
}


def _normalize_department_concepts(raw_concepts: object) -> list[dict[str, object]]:
    source = raw_concepts if isinstance(raw_concepts, dict) else {}
    normalized: list[dict[str, object]] = []
    for dept_id, base in PRODUCT_DEVELOPMENT_DEPARTMENTS.items():
        raw = source.get(dept_id) if isinstance(source, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        note = str(raw.get("note") or raw.get("notes") or "").strip()
        concept = str(raw.get("concept") or "").strip()
        default_concept = str(base.get("default_concept") or "")
        merged_concept = concept or default_concept
        if note:
            merged_concept = f"{merged_concept} 담당자 메모: {note}"
        normalized.append({
            "id": dept_id,
            "name": str(base.get("name") or dept_id),
            "icon": str(base.get("icon") or ""),
            "default_concept": default_concept,
            "note": note,
            "concept": merged_concept,
        })
    return normalized


def _product_dev_json_from_text(text: str) -> dict[str, object]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start:end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _build_product_development_context() -> dict[str, object]:
    cube_payload = _build_segment_metric_cube_api_payload(force_rebuild=False)
    product_summaries = []
    for item in list(cube_payload.get("product_summaries") or [])[:8]:
        product = str(item.get("product") or "")
        product_summaries.append({
            "product": product,
            "product_label": PRODUCT_DEVELOPMENT_PRODUCT_NAMES.get(product, product),
            "count": int(item.get("count") or 0),
            "approval_rate_percent": item.get("approval_rate_percent"),
            "rejection_rate_percent": item.get("rejection_rate_percent"),
            "avg_rate_display": str(item.get("avg_rate_display") or "-"),
            "avg_amount_display": str(item.get("avg_amount_display") or "-"),
            "delinquency_proxy_rate_display": str(item.get("delinquency_proxy_rate_display") or "-"),
            "top_reject_codes": list(item.get("top_reject_codes") or [])[:3],
        })

    cluster_summary = dict(cube_payload.get("cluster_summary") or {})
    cluster_products = []
    for item in list(cluster_summary.get("products") or [])[:8]:
        product = str(item.get("product") or "")
        cluster_products.append({
            "product": product,
            "product_label": PRODUCT_DEVELOPMENT_PRODUCT_NAMES.get(product, product),
            "cluster_count": int(item.get("cluster_count") or 0),
            "top_clusters": list(item.get("top_clusters") or [])[:3],
        })

    pattern_payload = load_product_pattern_summary(DEFAULT_SUMMARY_PATH)
    pattern_products = []
    for product, item in sorted(dict(pattern_payload.get("products") or {}).items()):
        totals = dict(item.get("totals") or {})
        pattern_products.append({
            "product": product,
            "product_label": PRODUCT_DEVELOPMENT_PRODUCT_NAMES.get(product, product),
            "approval_rate_percent": totals.get("approval_rate_percent"),
            "approval_cases": totals.get("approval_cases"),
            "rejection_cases": totals.get("rejection_cases"),
            "approval_patterns": list(item.get("approval_patterns") or [])[:3],
            "rejection_patterns": list(item.get("rejection_patterns") or [])[:3],
            "top_reject_reason_codes": list(item.get("top_reject_reason_codes") or [])[:3],
        })

    with state.lock:
        regulation_summary = str(state.latest_regulation_analysis or "").strip()
        news_summary = str(state.latest_news_briefing or "").strip()

    return {
        "cube_meta": dict(cube_payload.get("meta") or {}),
        "product_summaries": product_summaries,
        "cluster_products": cluster_products,
        "pattern_products": pattern_products,
        "regulation_summary": regulation_summary[:700],
        "news_summary": news_summary[:700],
        "semantic_refresh": _semantic_refresh_snapshot(),
    }


def _product_development_context_prompt(context: dict[str, object]) -> str:
    product_lines = []
    for item in list(context.get("product_summaries") or []):
        reject_codes = ", ".join(str(code.get("code") or code) for code in list(item.get("top_reject_codes") or []))
        product_lines.append(
            f"- {item.get('product_label')}: 표본 {item.get('count')}건, 승인률 {item.get('approval_rate_percent')}%, "
            f"평균금리 {item.get('avg_rate_display')}, 평균한도 {item.get('avg_amount_display')}, "
            f"연체가능성 {item.get('delinquency_proxy_rate_display')}, 상위 거절코드 {reject_codes or '-'}"
        )
    cluster_lines = []
    for item in list(context.get("cluster_products") or []):
        clusters = []
        for cluster in list(item.get("top_clusters") or []):
            clusters.append(
                f"{cluster.get('decision') or '군집'} {cluster.get('label') or '-'} "
                f"{cluster.get('count') or 0}건 금리 {cluster.get('avg_rate_display') or '-'} 한도 {cluster.get('avg_amount_display') or '-'}"
            )
        cluster_lines.append(f"- {item.get('product_label')}: {' / '.join(clusters) if clusters else '-'}")
    pattern_lines = []
    for item in list(context.get("pattern_products") or []):
        approvals = ", ".join(f"{p.get('feature')} {p.get('rule')}" for p in list(item.get("approval_patterns") or []))
        rejects = ", ".join(f"{p.get('feature')} {p.get('rule')}" for p in list(item.get("rejection_patterns") or []))
        pattern_lines.append(f"- {item.get('product_label')}: 승인패턴 [{approvals or '-'}], 거절패턴 [{rejects or '-'}]")
    return "\n".join([
        "[상품별 통계]",
        *product_lines,
        "",
        "[상위 군집]",
        *cluster_lines,
        "",
        "[상품 패턴]",
        *pattern_lines,
        "",
        f"[규제/뉴스 요약]\n규제: {context.get('regulation_summary') or '-'}\n뉴스: {context.get('news_summary') or '-'}",
    ])[:7000]


def _concepts_prompt(concepts: list[dict[str, object]]) -> str:
    return "\n".join(
        f"- {item['name']}: {item.get('concept') or item.get('default_concept')}"
        for item in concepts
    )


def _fallback_product_agendas(context: dict[str, object]) -> list[dict[str, object]]:
    products = list(context.get("product_summaries") or [])
    c6 = next((item for item in products if item.get("product") == "C6"), products[0] if products else {})
    c9 = next((item for item in products if item.get("product") == "C9"), products[0] if products else {})
    return [
        {
            "id": "agenda-new-product",
            "type": "new_product",
            "title": "씬파일 승인 전환형 소액 신용 상품",
            "summary": "승인률이 낮은 상품군에서 리스크가 낮은 세그먼트만 골라 소액·단기·상환행동 기반으로 테스트합니다.",
            "why": [
                f"{c6.get('product_label', 'C6')} 승인률 {c6.get('approval_rate_percent', '-')}% 구간에 개선 여지가 있습니다.",
                "상위 거절코드를 바로 완화하기보다 한도와 금리를 보수적으로 묶어 테스트하는 안건입니다.",
            ],
            "target": "중신용·중소득 고객 중 연체 proxy가 낮고 한도 요청이 과하지 않은 군집",
            "expected_effect": ["승인 전환율 개선", "초기 부실률 통제", "기존 상품 로직 재사용 가능"],
            "data_points": [f"{item.get('product_label')}: 승인률 {item.get('approval_rate_percent')}%" for item in products[:4]],
        },
        {
            "id": "agenda-logic-improvement",
            "type": "logic_improvement",
            "title": "이지론/이지신용대출 거절코드 세분화 보완",
            "summary": "K코드 상위 사유를 그대로 탈락 처리하지 않고, 한도 감액·금리 보정·추가정보 조회로 분기합니다.",
            "why": [
                f"{c9.get('product_label', 'C9')}은 승인률 {c9.get('approval_rate_percent', '-')}%로 모수가 충분해 보완 효과를 검증하기 좋습니다.",
                "IT 개발공수는 기존 상품 룰 엔진의 분기 추가 중심으로 줄입니다.",
            ],
            "target": "K코드 상위 사유에 걸렸지만 소득·상환여력 보완 신호가 있는 고객",
            "expected_effect": ["불필요한 거절 감소", "한도/금리 정합성 개선", "영업 전환 후보 확대"],
            "data_points": [f"{item.get('product_label')}: 평균금리 {item.get('avg_rate_display')} / 평균한도 {item.get('avg_amount_display')}" for item in products[:4]],
        },
    ]


def _build_product_agenda_prompt(context: dict[str, object], concepts: list[dict[str, object]]) -> str:
    return f"""
너는 금융솔루션부 상품기획 담당자다.
아래 통계 큐브/군집분석/상품 패턴만 근거로 4개 부서가 토론할 안건 2개를 만든다.
말투는 현업이 바로 이해할 수 있게 짧고 쉽고, 살짝 위트 있게 쓴다.

반드시 JSON만 출력한다.
스키마:
{{
  "agendas": [
    {{
      "id": "agenda-new-product",
      "type": "new_product",
      "title": "안건명",
      "summary": "한 문장 요약",
      "why": ["근거1", "근거2"],
      "target": "대상 고객군",
      "expected_effect": ["기대효과1", "기대효과2"],
      "data_points": ["통계 근거1", "통계 근거2"]
    }},
    {{
      "id": "agenda-logic-improvement",
      "type": "logic_improvement",
      "title": "안건명",
      "summary": "한 문장 요약",
      "why": ["근거1", "근거2"],
      "target": "대상 고객군",
      "expected_effect": ["기대효과1", "기대효과2"],
      "data_points": ["통계 근거1", "통계 근거2"]
    }}
  ]
}}

[부서 컨셉]
{_concepts_prompt(concepts)}

{_product_development_context_prompt(context)}
""".strip()


def _fallback_product_debate(selected_agenda: dict[str, object], context: dict[str, object], concepts: list[dict[str, object]]) -> dict[str, object]:
    title = str(selected_agenda.get("title") or "상품개발 안건")
    agenda_type = str(selected_agenda.get("type") or "").strip().lower()
    products = list(context.get("product_summaries") or [])
    product_cards = [
        {
            "product": item.get("product"),
            "product_label": item.get("product_label"),
            "approval_rate_percent": item.get("approval_rate_percent"),
            "avg_rate_display": item.get("avg_rate_display"),
            "avg_amount_display": item.get("avg_amount_display"),
            "delinquency_proxy_rate_display": item.get("delinquency_proxy_rate_display"),
            "top_reject_codes": item.get("top_reject_codes"),
        }
        for item in products[:4]
    ]
    messages = [
        {"speaker": "금융솔루션부", "tone": "moderator", "message": f"{title} 기준으로 의견을 정리합니다. 안건 타입에 맞는 제안만 남기겠습니다."},
        {"speaker": "신용기획팀", "tone": "risk", "message": "리스크 민감 구간을 먼저 점검하고 변동 폭을 제어하겠습니다."},
        {"speaker": "상품전략팀", "tone": "sales", "message": "전환 가능성이 높은 고객군 중심으로 실행안을 좁혀보겠습니다."},
        {"speaker": "IT개발팀", "tone": "tech", "message": "기존 룰과 충돌하지 않도록 최소 변경 구조로 제안합니다."},
    ]

    new_product_candidates = [
        {
            "name": "씬파일 승인 전환형 소액 신용 상품",
            "target": "승인률이 낮은 군에서 리스크 하위 세그먼트를 분리해 승인 전환을 확대",
            "core_logic": ["초기한도 보수 운영", "행동 데이터 확인 후 증액", "신용/소득 proxy 혼합 검증"],
            "limit_rate_policy": "초기 한도는 작게 시작하고 금리는 위험 등급별 차등 적용",
            "risk_guardrails": ["거절 코드 급증 감시", "DSR/연체 proxy 임계치 차단", "30일 롤링 모니터링"],
        },
        {
            "name": "금리-한도 탄력형 전환 상품",
            "target": "거절 이력이 많은 고객군에 소액/단기 상품을 제공해 점진적 전환 유도",
            "core_logic": ["금리 스텝 구조", "재신청 시 가산점 반영", "세그먼트별 한도 상한 분리"],
            "limit_rate_policy": "한도 상단을 제한하고 금리는 단계적으로 완화",
            "risk_guardrails": ["고위험 코드 선차단", "상환/이탈 경보 자동화", "구간별 성과 리포트"],
        },
    ]
    logic_improvements = [
        {
            "product": item.get("product_label"),
            "change": "거절 코드 기반 분기 로직을 한도/금리/재신청 조건에 연계",
            "expected_effect": f"승인률 {item.get('approval_rate_percent')}% 구간에서 전환 후보 확대",
            "dev_impact": "규칙 테이블 및 점수 매핑 확장",
        }
        for item in products[:4]
    ]
    while len(logic_improvements) < 2:
        logic_improvements.append(
            {
                "product": "이지론(C9)",
                "change": "소득 proxy 보정 및 금리 밴드 재정렬",
                "expected_effect": "리스크 급증 없이 승인 전환 폭 확대",
                "dev_impact": "feature 파이프라인/검증 규칙 보완",
            }
        )

    final_new_product_candidates = new_product_candidates[:2] if agenda_type == "new_product" else []
    final_logic_improvements = logic_improvements[:2] if agenda_type == "logic_improvement" else []
    primary_new_product = final_new_product_candidates[0] if final_new_product_candidates else new_product_candidates[0]

    return {
        "selected_agenda": selected_agenda,
        "messages": messages,
        "final": {
            "new_product": primary_new_product,
            "new_product_candidates": final_new_product_candidates,
            "product_logic_improvements": final_logic_improvements,
            "implementation_plan": ["2주: 후보 정의/데이터 점검", "3주: 규칙 반영 및 검증", "2주: 시범 운영 모니터링"],
            "kpis": ["승인률", "평균금리", "평균한도", "부실 proxy", "거절코드 전환율"],
        },
        "product_cards": product_cards,
        "concepts": concepts,
    }


def _build_department_persona_profiles() -> list[dict[str, object]]:
    return [
        {
            "name": "신프로",
            "department": "신용기획부",
            "role": "리스크 기준선 설계 및 심사정책 검증",
            "skills": ["DSR 분석", "신용등급 분석", "거절코드 분석"],
            "memory": ["DSR 40% 이상은 연체 위험이 높음"],
            "constraints": ["손실률 최소화", "정책 위반 금지", "리스크 구간 선제 차단"],
            "stance": "보수적",
            "goal": "손실률 최소화",
            "tone": "risk",
        },
        {
            "name": "금프로",
            "department": "금융솔루션부",
            "role": "상품 구조 설계와 출시 전략 총괄",
            "skills": ["상품기획", "고객세그먼트 분석", "수익성 분석"],
            "memory": ["소액 한도 상품은 초기 유입 효과가 좋음"],
            "constraints": ["실행 가능한 MVP 우선", "기존 심사체계와 충돌 최소화"],
            "stance": "공격적",
            "goal": "신규 상품 출시",
            "tone": "moderator",
        },
        {
            "name": "영프로",
            "department": "금융영업부",
            "role": "영업 전환율 및 실행액 관점 검증",
            "skills": ["승인률 개선", "현장 세일즈 시나리오", "고객 커뮤니케이션"],
            "memory": ["조건이 명확하면 상담 전환율이 높아짐"],
            "constraints": ["현장 실행 난이도 낮게", "고객 안내 문구 명확화"],
            "stance": "실행중심",
            "goal": "승인 전환율 확대",
            "tone": "sales",
        },
        {
            "name": "아프로",
            "department": "IT개발자",
            "role": "데이터 연계/개발공수/배포리스크 검증",
            "skills": ["데이터 정합성 검증", "외부기관 연계 설계", "배포 안정성 점검"],
            "memory": ["룰이 명확하면 배치 검증부터 빠르게 적용 가능"],
            "constraints": ["연계 실패 대비책 필요", "배포 안정성 우선"],
            "stance": "현실적",
            "goal": "안정적 구현",
            "tone": "tech",
        },
    ]


def _build_department_agent_prompt_blocks(personas: list[dict[str, object]]) -> str:
    blocks: list[str] = []
    for persona in personas:
        name = str(persona.get("name") or "").strip()
        role = str(persona.get("role") or "").strip()
        skills = ", ".join([str(item).strip() for item in list(persona.get("skills") or []) if str(item).strip()]) or "-"
        memory = ", ".join([str(item).strip() for item in list(persona.get("memory") or []) if str(item).strip()]) or "-"
        constraints = ", ".join([str(item).strip() for item in list(persona.get("constraints") or []) if str(item).strip()]) or "-"
        stance = str(persona.get("stance") or "").strip()
        goal = str(persona.get("goal") or "").strip()
        blocks.append(
            f"""[AGENT PROMPT · {name}]
너는 {name}이다.

역할:
{role}

보유 skill:
{skills}

기억:
{memory}

판단 제약:
{constraints}

의사결정 성향:
{stance}

목표:
{goal}

아래 안건에 대해 네 부서 관점에서 의견을 내라.
다른 부서 발언과 충돌하면 반박 근거를 짧게 제시하고, 마지막에는 합의 가능한 수정안을 1개 제안하라."""
        )
    return "\n\n".join(blocks)


def _build_product_debate_prompt(selected_agenda: dict[str, object], context: dict[str, object], concepts: list[dict[str, object]]) -> str:
    personas = _build_department_persona_profiles()
    persona_json = json.dumps({"agents": personas}, ensure_ascii=False, indent=2)
    persona_prompt_blocks = _build_department_agent_prompt_blocks(personas)
    agenda_type = str(selected_agenda.get("type") or "").strip().lower()
    if agenda_type == "new_product":
        agenda_focus_instruction = (
            "안건 타입은 new_product다. final.new_product_candidates에 정확히 2개를 작성하고, "
            "final.product_logic_improvements는 빈 배열([])로 유지한다."
        )
    elif agenda_type == "logic_improvement":
        agenda_focus_instruction = (
            "안건 타입은 logic_improvement다. final.product_logic_improvements에 정확히 2개를 작성하고, "
            "final.new_product_candidates는 빈 배열([])로 유지한다."
        )
    else:
        agenda_focus_instruction = (
            "안건 타입이 불명확하면 둘 중 하나 축만 선택해서 정확히 2개를 작성하고, 다른 축은 빈 배열로 둔다."
        )
    return f"""
너는 금융 상품개발 토론을 진행하는 AI다.
선택된 안건을 기준으로 4개 부서의 의견을 구성하고, 최종 합의안을 JSON으로만 반환한다.
실행 가능성과 근거 중심으로 작성한다.

[안건 타입 규칙]
{agenda_focus_instruction}

반드시 아래 JSON 스키마를 따른다:
{{
  "messages": [
    {{"speaker": "금융솔루션부", "tone": "moderator", "message": "요약"}},
    {{"speaker": "신용기획팀", "tone": "risk", "message": "요약"}},
    {{"speaker": "상품전략팀", "tone": "sales", "message": "요약"}},
    {{"speaker": "IT개발팀", "tone": "tech", "message": "요약"}}
  ],
  "final": {{
    "new_product": {{
      "name": "상품명",
      "target": "대상",
      "core_logic": ["핵심 로직"],
      "limit_rate_policy": "한도/금리 정책",
      "risk_guardrails": ["리스크 가드레일"]
    }},
    "new_product_candidates": [
      {{"name": "상품명", "target": "대상", "core_logic": ["핵심 로직"]}}
    ],
    "product_logic_improvements": [
      {{"product": "상품", "change": "개선안", "expected_effect": "기대효과", "dev_impact": "개발영향"}}
    ],
    "implementation_plan": ["단계"],
    "kpis": ["지표"]
  }}
}}

[선택 안건]
{json.dumps(selected_agenda, ensure_ascii=False)}

[부서 Persona JSON]
{persona_json}

[부서별 Agent Prompt]
{persona_prompt_blocks}

[부서 개념]
{_concepts_prompt(concepts)}

{_product_development_context_prompt(context)}
""".strip()


def _contains_non_ascii_text(value: object) -> bool:
    text = str(value or "")
    return any(ord(ch) > 127 for ch in text)


def _normalize_product_debate_final(
    selected_agenda: dict[str, object],
    final_payload: dict[str, object],
    fallback_final: dict[str, object],
) -> dict[str, object]:
    agenda_type = str(selected_agenda.get("type") or "").strip().lower()
    normalized = dict(final_payload or {})
    fallback_new = dict(fallback_final.get("new_product") or {})
    fallback_candidates = [item for item in list(fallback_final.get("new_product_candidates") or []) if isinstance(item, dict)]
    fallback_improvements = [item for item in list(fallback_final.get("product_logic_improvements") or []) if isinstance(item, dict)]

    candidates = [item for item in list(normalized.get("new_product_candidates") or []) if isinstance(item, dict)]
    improvements = [item for item in list(normalized.get("product_logic_improvements") or []) if isinstance(item, dict)]
    new_product = dict(normalized.get("new_product") or {})

    if agenda_type == "new_product":
        merged = candidates[:2]
        while len(merged) < 2 and len(fallback_candidates) > len(merged):
            merged.append(fallback_candidates[len(merged)])
        if len(merged) < 2:
            merged = fallback_candidates[:2]
        normalized["new_product_candidates"] = merged[:2]
        normalized["product_logic_improvements"] = []
        normalized["new_product"] = dict(merged[0] if merged else fallback_new)
    elif agenda_type == "logic_improvement":
        merged = improvements[:2]
        while len(merged) < 2 and len(fallback_improvements) > len(merged):
            merged.append(fallback_improvements[len(merged)])
        if len(merged) < 2:
            merged = fallback_improvements[:2]
        normalized["product_logic_improvements"] = merged[:2]
        normalized["new_product_candidates"] = []
        normalized["new_product"] = fallback_new

    # LLM이 영어 단일 결과를 줄 때 한글 fallback으로 교정
    current_new = dict(normalized.get("new_product") or {})
    if not _contains_non_ascii_text(current_new.get("name")):
        normalized["new_product"] = fallback_new
    if agenda_type == "new_product":
        fixed_candidates = []
        for idx, item in enumerate(list(normalized.get("new_product_candidates") or [])[:2]):
            if _contains_non_ascii_text(item.get("name")):
                fixed_candidates.append(item)
            elif idx < len(fallback_candidates):
                fixed_candidates.append(fallback_candidates[idx])
        while len(fixed_candidates) < 2 and len(fallback_candidates) > len(fixed_candidates):
            fixed_candidates.append(fallback_candidates[len(fixed_candidates)])
        normalized["new_product_candidates"] = fixed_candidates[:2]
        if fixed_candidates:
            normalized["new_product"] = dict(fixed_candidates[0])

    return normalized


def _product_ollama_generate_fast(prompt: str, wait_seconds: int) -> str:
    acquired = PRODUCT_DEBATE_CALL_SEMAPHORE.acquire(timeout=max(1, wait_seconds))
    if not acquired:
        raise TimeoutError("상품개발 토론 호출이 혼잡하여 대기 중입니다. 잠시 후 다시 시도해 주세요.")
    try:
        return str(
            lightweight_ollama_generate(
                prompt,
                timeout_seconds=max(wait_seconds + 3, 12),
                fail_fast_if_busy=True,
                priority_group="ontology",
            )
            or ""
        )
    finally:
        with contextlib.suppress(Exception):
            PRODUCT_DEBATE_CALL_SEMAPHORE.release()


def _generate_product_development_agendas(concepts: list[dict[str, object]]) -> dict[str, object]:
    context = _build_product_development_context()
    prompt = _build_product_agenda_prompt(context, concepts)
    fallback = _fallback_product_agendas(context)
    source = "fallback"
    response_text = ""
    agendas = fallback
    try:
        response_text = _product_ollama_generate_fast(prompt, wait_seconds=8)
        parsed = _product_dev_json_from_text(response_text)
        parsed_agendas = parsed.get("agendas") if isinstance(parsed, dict) else None
        if isinstance(parsed_agendas, list) and len(parsed_agendas) >= 2:
            agendas = [item for item in parsed_agendas[:2] if isinstance(item, dict)]
            source = "ollama"
    except Exception as error:
        response_text = str(error)
    return {
        "status": "ok",
        "source": source,
        "context": context,
        "concepts": concepts,
        "agendas": agendas,
        "llm": {
            "model": OLLAMA_LIGHTWEIGHT_MODEL,
            "prompt": prompt,
            "response_text": response_text,
        },
    }


def _generate_product_development_debate(
    selected_agenda: dict[str, object],
    concepts: list[dict[str, object]],
    *,
    require_autogen: bool | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, object]:
    context = _build_product_development_context()
    fallback = _fallback_product_debate(selected_agenda, context, concepts)
    personas = _build_department_persona_profiles()
    turn_wait_seconds = max(1, _env_int("PRODUCT_DEBATE_TURN_WAIT_SECONDS", 3, minimum=1))
    max_turns = max(2, _env_int("PRODUCT_DEBATE_MAX_TURNS", _env_int("PRODUCT_DEBATE_MAX_ROUNDS", 2, minimum=2), minimum=2))
    force_autogen = PRODUCT_DEBATE_FORCE_AUTOGEN if require_autogen is None else bool(require_autogen)
    if progress_callback:
        progress_callback("orchestration-start", "AutoGen 오케스트레이션을 시작합니다.")
    orchestrated = run_product_debate_orchestration(
        selected_agenda=selected_agenda,
        context=context,
        concepts=concepts,
        personas=personas,
        llm_call=lambda prompt: _product_ollama_generate_fast(prompt, wait_seconds=turn_wait_seconds),
        parse_json=_product_dev_json_from_text,
        fallback_result=fallback,
        memory_path=PRODUCT_DEBATE_MEMORY_PATH,
        max_rounds=max_turns,
        retries=max(0, _env_int("PRODUCT_DEBATE_RETRIES", 1, minimum=0)),
        consensus_threshold=float(os.environ.get("PRODUCT_DEBATE_CONSENSUS_THRESHOLD", "0.72") or 0.72),
        require_autogen=force_autogen,
        progress_callback=progress_callback,
    )
    # Safety fallback: if orchestration failed to create a valid result payload, preserve legacy behavior.
    result_payload = dict(orchestrated.get("result") or {})
    if isinstance(result_payload.get("final"), dict):
        result_payload["final"] = _normalize_product_debate_final(
            selected_agenda=selected_agenda,
            final_payload=dict(result_payload.get("final") or {}),
            fallback_final=dict(fallback.get("final") or {}),
        )
        orchestrated["result"] = result_payload
    if not isinstance(result_payload.get("final"), dict) or not isinstance(result_payload.get("messages"), list):
        if force_autogen:
            raise RuntimeError("AutoGen 토론 결과가 유효하지 않습니다. 커스텀 루프로 전환하지 않도록 설정되어 실패 처리합니다.")
        prompt = _build_product_debate_prompt(selected_agenda, context, concepts)
        response_text = ""
        result = fallback
        source = "fallback"
        try:
            response_text = _product_ollama_generate_fast(prompt, wait_seconds=8)
            parsed = _product_dev_json_from_text(response_text)
            if isinstance(parsed.get("final"), dict) and isinstance(parsed.get("messages"), list):
                normalized_final = _normalize_product_debate_final(
                    selected_agenda=selected_agenda,
                    final_payload=dict(parsed.get("final") or {}),
                    fallback_final=dict(fallback.get("final") or {}),
                )
                result = {
                    **fallback,
                    **parsed,
                    "final": normalized_final,
                    "selected_agenda": selected_agenda,
                    "product_cards": fallback.get("product_cards") or [],
                    "concepts": concepts,
                }
                source = "ollama"
        except Exception as error:
            response_text = str(error)
        return {
            "status": "ok",
            "source": source,
            "context": context,
            "result": result,
            "llm": {
                "model": OLLAMA_LIGHTWEIGHT_MODEL,
                "prompt": prompt,
                "response_text": response_text,
            },
        }
    return orchestrated


def _dedupe_text_items(values: list[object], limit: int = 8) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if len(deduped) >= limit:
            break
    return deduped


def _read_ontology_relation_overrides() -> dict[str, dict[str, object]]:
    payload = _read_json_file(ONTOLOGY_RELATIONS_PATH)
    relation_index = payload.get("features") or payload.get("relations") or payload
    if not isinstance(relation_index, dict):
        return {}
    return {
        str(feature_id): value
        for feature_id, value in relation_index.items()
        if isinstance(value, dict)
    }


def _infer_semantic_domain(selected_product: str, selected_feature: dict | None) -> str:
    if selected_product:
        return "Loan"
    category = str((selected_feature or {}).get("category") or "").strip().lower()
    if category in {"loan", "applicant", "income", "decision", "credit", "credit_bureau", "credit_model"}:
        return "Loan"
    return "General"


def _infer_entity_type(selected_feature: dict | None) -> str:
    category = str((selected_feature or {}).get("category") or "").strip().lower()
    return str(CATEGORY_ENTITY_TYPE_MAP.get(category) or "FeatureAttribute")


def _build_query_intent_parsing(query: str, selected_product: str, question_token_mappings: list[dict[str, object]]) -> dict[str, object]:
    tokens = _tokenize_text(query)[:8]
    hit_tokens = [
        str(item.get("token") or "")
        for item in question_token_mappings
        if (item.get("feature_links") or [])
    ]
    context_tokens = [
        str(item.get("token") or "")
        for item in question_token_mappings
        if not (item.get("feature_links") or [])
    ]
    routed_signals = [
        {
            "token": str(item.get("token") or ""),
            "signal_type": str(item.get("signal_type") or "context-only"),
            "label": str(item.get("primary_label") or item.get("concept_label") or ""),
        }
        for item in question_token_mappings[:6]
    ]
    return {
        "query": query,
        "tokens": tokens,
        "selected_product": selected_product or "ALL",
        "hit_tokens": _dedupe_text_items(hit_tokens, limit=6),
        "context_tokens": _dedupe_text_items(context_tokens, limit=6),
        "routed_signals": routed_signals,
    }


def _build_semantic_retrieval_result(
    selected_feature: dict | None,
    primary_feature_selection: dict[str, object],
    related_features: list[dict[str, object]],
    selected_product: str,
) -> dict[str, object]:
    representative_candidates = _resolve_representative_features(primary_feature_selection, selected_feature=selected_feature, limit=3)
    top_candidates = representative_candidates or list((primary_feature_selection or {}).get("top_k") or [])[:3]
    # top-3 후보 전체를 대표축으로 반환
    # 도메인/카테고리 우선순위 적용: 직장인은 직업/재직, 카드론은 loan/한도/금리/연체 도메인 feature 우선 포함
    def is_job_feature(item):
        name = (item.get("feature_name") or "").lower()
        category = (item.get("category") or "").lower()
        return any(token in name or token in category for token in ["직업", "재직", "고용", "직장"])

    def is_loan_core_feature(item):
        name = (item.get("feature_name") or "").lower()
        category = (item.get("category") or "").lower()
        return any(token in name or token in category for token in ["loan", "한도", "금리", "연체", "이자", "대출", "limit", "rate", "delinquency"])

    # 1. 도메인별 우선 feature 추출
    job_features = [item for item in top_candidates if is_job_feature(item)]
    loan_features = [item for item in top_candidates if is_loan_core_feature(item)]
    # 2. top-3에 도메인별 대표 feature가 반드시 포함되도록 보장
    selected = []
    for item in (job_features + loan_features):
        if item not in selected:
            selected.append(item)
        if len(selected) >= 3:
            break
    for item in top_candidates:
        if item not in selected:
            selected.append(item)
        if len(selected) >= 3:
            break
    primary_features = [
        {
            "feature_id": str(item.get("feature_id") or ""),
            "feature_name": str(item.get("feature_name") or item.get("feature_id") or ""),
            "score": round(float(item.get("hybrid_score") or item.get("base_score") or 0.0), 4),
        }
        for item in selected[:3]
        if str(item.get("feature_id") or "").strip()
    ]
    if not primary_features and selected_feature:
        primary_features = [{
            "feature_id": str(selected_feature.get("feature_id") or ""),
            "feature_name": str(selected_feature.get("feature_name") or selected_feature.get("feature_id") or ""),
            "score": 0.0,
        }]
    return {
        "primary_features": primary_features,
        "related_features": _dedupe_text_items([
            item.get("feature_name") or item.get("feature_id") or ""
            for item in related_features[:6]
        ], limit=6),
        "domain": _infer_semantic_domain(selected_product, selected_feature),
        "entity_type": _infer_entity_type(selected_feature),
    }


def _build_ontology_graph_expansion(
    selected_feature: dict | None,
    related_features: list[dict[str, object]],
    primary_feature_selection: dict[str, object],
    selected_product: str,
    relation_overrides: dict[str, dict[str, object]],
) -> dict[str, object]:
    feature = selected_feature or {}
    feature_id = str(feature.get("feature_id") or "")
    override = relation_overrides.get(feature_id) or {}
    aliases = _dedupe_text_items(list(feature.get("aliases") or []), limit=8)
    field_mappings = [
        mapping for mapping in (feature.get("field_mappings") or [])
        if not selected_product or str(mapping.get("product") or "") == selected_product
    ]
    field_scope = _dedupe_text_items([
        f"{str(mapping.get('product_name') or mapping.get('product') or '').strip()}:{str(mapping.get('field_code') or mapping.get('label') or '').strip()}"
        for mapping in field_mappings[:6]
    ], limit=6)
    graph_neighbors = _dedupe_text_items([
        item.get("feature_name") or item.get("feature_id") or ""
        for item in related_features[:6]
    ], limit=6)
    support_candidates = _dedupe_text_items([
        item.get("feature_name") or item.get("feature_id") or ""
        for item in _resolve_representative_features(primary_feature_selection, selected_feature=selected_feature, limit=3)[1:]
    ], limit=3)
    relations = {
        relation_name: _dedupe_text_items(list(values), limit=8)
        for relation_name, values in (override.get("relations") or {}).items()
        if isinstance(values, list)
    }
    if not relations:
        if graph_neighbors:
            relations["graph_neighbors"] = graph_neighbors
        if support_candidates:
            relations["topk_competitors"] = support_candidates
        if field_scope:
            relations["mapped_fields"] = field_scope
        if feature.get("products"):
            relations["shared_product_scope"] = _dedupe_text_items(list(feature.get("products") or []), limit=6)
        if feature.get("directions"):
            relations["used_as_direction"] = _dedupe_text_items(list(feature.get("directions") or []), limit=4)
    business_purpose = str(override.get("business_purpose") or "").strip() or "명시된 business purpose 없음"
    return {
        "feature_id": feature_id,
        "feature_name": str(feature.get("feature_name") or feature_id),
        "canonical_meaning": str(override.get("canonical_meaning") or feature.get("description") or feature.get("feature_name") or feature_id),
        "aliases": aliases,
        "relations": relations,
        "business_purpose": business_purpose,
        "relation_mode": "override" if override else "derived",
    }


def _build_semantic_pipeline(
    query_intent: dict[str, object],
    semantic_retrieval_result: dict[str, object],
    ontology_expansion: dict[str, object],
    retrieval_results: list[dict[str, object]],
    regulation_evidence: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    primary_features = list(semantic_retrieval_result.get("primary_features") or [])
    primary_feature_ids = [str(item.get("feature_id") or "") for item in primary_features if str(item.get("feature_id") or "").strip()]
    relations = dict(ontology_expansion.get("relations") or {})
    relation_labels = _dedupe_text_items([
        relation_name.replace("_", " ")
        for relation_name in relations.keys()
    ], limit=6)
    return [
        {
            "step": 1,
            "key": "intent_parsing",
            "title": "Query Intent Parsing",
            "detail": f"질문 토큰 {', '.join(query_intent.get('tokens') or []) or '없음'}를 intent signal로 정리했습니다.",
        },
        {
            "step": 2,
            "key": "domain_scoped_faiss_retrieval",
            "title": "Domain-scoped FAISS Retrieval",
            "detail": f"상품 범위 {query_intent.get('selected_product') or 'ALL'}에서 semantic retrieval 후보를 좁혔습니다.",
        },
        {
            "step": 3,
            "key": "topk_feature_selection",
            "title": "Top-K Feature Selection",
            "detail": f"상위 feature 후보 {', '.join(primary_feature_ids) or '없음'}를 선택했습니다.",
        },
        {
            "step": 4,
            "key": "ontology_graph_expansion",
            "title": "Ontology Graph Expansion (1-hop)",
            "detail": f"1-hop relation {', '.join(relation_labels) or '없음'}를 기준으로 주변 feature를 확장했습니다.",
        },
        {
            "step": 5,
            "key": "semantic_context_compression",
            "title": "Semantic Context Compression",
            "detail": f"hit token, top-k, relation, retrieval {len(retrieval_results)}건 + regulation {len(regulation_evidence or [])}건을 압축해 grounding context를 만들었습니다.",
        },
        {
            "step": 6,
            "key": "prompt_context_generation",
            "title": "Prompt Context Generation",
            "detail": "block-structured prompt로 LLM 입력을 생성했습니다.",
        },
    ]


def _compress_semantic_context(
    query_intent: dict[str, object],
    semantic_retrieval_result: dict[str, object],
    ontology_expansion: dict[str, object],
    customer_clusters: list[dict[str, object]],
    retrieval_results: list[dict[str, object]],
    regulation_evidence: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    top_cluster = customer_clusters[0] if customer_clusters else {}
    retrieval_ids = _dedupe_text_items([
        item.get("record_id") or item.get("product") or ""
        for item in retrieval_results[:4]
    ], limit=4)
    regulation_citations = [
        {
            "name": str(item.get("name") or "regulation"),
            "chunk_index": int(item.get("chunk_index") or 0),
            "score": round(float(item.get("score") or 0.0), 4),
            "snippet": str(item.get("snippet") or "")[:220],
            "feature_hits": list(item.get("feature_hits") or []),
        }
        for item in (regulation_evidence or [])[:4]
    ]
    return {
        "query_intent": query_intent,
        "semantic_retrieval_result": semantic_retrieval_result,
        "ontology_expansion": ontology_expansion,
        "retrieval_evidence": retrieval_ids,
        "regulation_citations": regulation_citations,
        "cluster_summary": {
            "decision": str(top_cluster.get("decision") or ""),
            "age_band": str(top_cluster.get("age_band") or ""),
            "income_band": str(top_cluster.get("income_band") or ""),
            "amount_band": str(top_cluster.get("amount_band") or ""),
            "avg_rate": top_cluster.get("avg_rate"),
            "avg_rate_display": str(top_cluster.get("avg_rate_display") or ""),
            "avg_amount": top_cluster.get("avg_amount"),
            "avg_amount_display": str(top_cluster.get("avg_amount_display") or ""),
            "metric_summary": list(top_cluster.get("metric_summary") or []),
        },
        "grounding_rules": list(DEFAULT_SEMANTIC_GROUNDING_RULES),
    }


REGULATION_RELEVANCE_TERMS = {
    "regulation",
    "regulatory",
    "citation",
    "policy",
    "compliance",
    "law",
    "legal",
    "dsr",
    "dti",
    "ltv",
    "stress",
    "규제",
    "규정",
    "법",
    "법령",
    "준수",
    "정책",
    "감독",
    "금감원",
    "금융위",
    "보도자료",
    "문서",
    "근거문서",
    "시행",
    "시행방안",
    "스트레스",
    "스트레스dsr",
    "dsr",
    "dti",
    "ltv",
    "규제",
    "규정",
    "법",
    "법령",
    "정책",
    "감독",
    "금융위원회",
    "금융감독원",
    "보도자료",
    "시행",
    "시행방안",
    "스트레스",
    "스트레스dsr",
    "3단계",
    "가계부채",
}


def _query_has_regulation_intent(query: str) -> bool:
    normalized = re.sub(r"\s+", "", str(query or "").lower())
    if not normalized:
        return False
    spaced = str(query or "").lower()
    # Keep a short explicit override list for high-frequency policy queries that
    # users type in many surface variations.
    explicit_terms = {
        "금리인하요구권",
        "금리 인하 요구권",
        "interest rate reduction request",
        "rate reduction request",
    }
    if any(term in normalized or term in spaced for term in explicit_terms):
        return True
    return any(term in normalized or term in spaced for term in REGULATION_RELEVANCE_TERMS)


def _filter_regulation_evidence_for_answer(
    query: str,
    regulation_evidence: list[dict[str, object]] | None,
    representative_feature_ids: list[str] | None = None,
    related_feature_ids: list[str] | None = None,
) -> tuple[list[dict[str, object]], str]:
    evidence = list(regulation_evidence or [])
    if not evidence:
        return [], "no_evidence"

    if _query_has_regulation_intent(query):
        return evidence[:4], "query_regulation_intent"

    allowed_feature_ids = {
        str(value).strip()
        for value in [*(representative_feature_ids or []), *(related_feature_ids or [])]
        if str(value).strip()
    }
    strong_matches: list[dict[str, object]] = []
    for item in evidence:
        score = float(item.get("score") or 0.0)
        feature_hits = {
            str(value).strip()
            for value in (item.get("feature_hits") or [])
            if str(value).strip()
        }
        if score >= 0.72 and feature_hits and feature_hits & allowed_feature_ids:
            strong_matches.append(item)

    if strong_matches:
        return strong_matches[:3], "strong_feature_overlap"

    return [], "not_answer_relevant"


REGULATION_FIRST_STOPWORDS = {
    "오늘",
    "현재",
    "최근",
    "요즘",
    "지금",
    "뭐야",
    "무엇",
    "어떤",
    "알려줘",
    "알려주세요",
    "추천",
    "해줘",
    "해주세요",
    "궁금",
    "관련",
    "정보",
    "search",
    "find",
    "tell",
    "about",
    "please",
}


UNDERWRITING_SCOPE_TERMS = {
    "승인",
    "거절",
    "탈락",
    "부결",
    "심사",
    "심사로그",
    "신청",
    "신청자",
    "고객",
    "고객군",
    "고객군집",
    "군집",
    "유사고객",
    "평균",
    "금리",
    "한도",
    "대출금액",
    "승인가능금액",
    "연체",
    "부실",
    "부실률",
    "리스크",
    "신용",
    "소득",
    "카드론",
    "이지론",
    "이지신용대출",
    "개인사업자대출",
    "이지대환대출",
    "상품",
    "사유",
    "사유코드",
    "거절사유코드",
    "feature",
    "피처",
    "reject",
    "approve",
    "approved",
    "cluster",
    "score",
    "kcb",
    "nice",
    "c6",
    "c9",
    "c11",
    "c12",
}


def _clean_regulation_query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for token in _tokenize_text(query):
        cleaned = _compact_search_text(token)
        if len(cleaned) < 2 or cleaned in REGULATION_FIRST_STOPWORDS:
            continue
        if cleaned.endswith("가") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        if cleaned.endswith("은") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        if cleaned.endswith("는") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        if cleaned.endswith("을") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        if cleaned.endswith("를") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        if cleaned and cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def _query_has_underwriting_scope(query: str) -> bool:
    compact_query = _compact_search_text(query)
    if not compact_query:
        return False
    if any(term in compact_query for term in UNDERWRITING_SCOPE_TERMS):
        return True
    product_resolution = _infer_product_from_query(query)
    return bool(product_resolution.get("product") and str(product_resolution.get("confidence") or "") == "high")


def _query_requires_regulation_first(query: str) -> bool:
    if _query_has_regulation_intent(query):
        return True
    return not _query_has_underwriting_scope(query)


def _regulation_keyword_overlap(query: str, evidence: dict[str, object]) -> list[str]:
    evidence_text = _compact_search_text(" ".join([
        str(evidence.get("name") or ""),
        str(evidence.get("title") or ""),
        str(evidence.get("snippet") or ""),
        str(evidence.get("text") or ""),
        str(evidence.get("content") or ""),
    ]))
    overlaps: list[str] = []
    for token in _clean_regulation_query_tokens(query):
        if token and token in evidence_text:
            overlaps.append(token)
    return overlaps


def _select_regulation_first_evidence(
    query: str,
    regulation_evidence: list[dict[str, object]] | None,
) -> tuple[list[dict[str, object]], str]:
    evidence = list(regulation_evidence or [])
    if not evidence:
        return [], "regulation_first_no_evidence"

    if _query_has_regulation_intent(query):
        return evidence[:4], "regulation_first_query_intent"

    matched: list[dict[str, object]] = []
    for item in evidence:
        overlaps = _regulation_keyword_overlap(query, item)
        score = float(item.get("score") or 0.0)
        if overlaps and score >= 0.7:
            next_item = dict(item)
            next_item["query_keyword_hits"] = overlaps
            matched.append(next_item)

    if matched:
        matched.sort(key=lambda item: (-len(item.get("query_keyword_hits") or []), -float(item.get("score") or 0.0)))
        return matched[:3], "regulation_first_keyword_overlap"
    return [], "regulation_first_no_relevant_document"


def _normalize_regulation_citation(item: dict[str, object]) -> dict[str, object]:
    file_name = str(item.get("document_name") or item.get("name") or item.get("title") or "regulation.pdf").strip()
    return {
        "name": str(item.get("name") or item.get("title") or "regulation document"),
        "file_name": file_name,
        "pdf_url": f"/regulation/files/{urllib.parse.quote(file_name)}",
        "chunk_index": int(item.get("chunk_index") or 0),
        "page": int(item.get("page") or 0) if str(item.get("page") or "").strip() else 0,
        "article": str(item.get("article") or "").strip(),
        "score": round(float(item.get("score") or 0.0), 4),
        "snippet": str(item.get("snippet") or item.get("text") or item.get("content") or "").strip()[:700],
        "feature_hits": list(item.get("feature_hits") or []),
        "query_keyword_hits": list(item.get("query_keyword_hits") or []),
    }


def _format_regulation_citation_value(citation: dict[str, object]) -> str:
    name = str(citation.get("name") or "regulation document").strip()
    page = int(citation.get("page") or 0)
    article = str(citation.get("article") or "").strip()
    if article:
        return f"{name} ({article})"
    if page > 0:
        return f"{name} (p.{page})"
    return name


def _unique_regulation_citation_values(citations: list[dict[str, object]], limit: int = 2) -> list[str]:
    return _dedupe_text_items([_format_regulation_citation_value(item) for item in citations], limit=limit)


def _build_regulation_first_answer_summary(query: str, regulation_evidence: list[dict[str, object]]) -> dict[str, object]:
    citations = [_normalize_regulation_citation(item) for item in regulation_evidence[:4]]
    compact_query = _compact_search_text(query)
    citation_value = ", ".join(_unique_regulation_citation_values(citations, limit=2)) or "규제문서"
    combined_text = " ".join(str(item.get("snippet") or "") for item in citations)

    if "dsr" in compact_query or "스트레스dsr" in compact_query:
        headline = "규제문서 기준으로 DSR 3단계 적용 내용을 확인했습니다."
        if "2025" in combined_text or "25.7.1" in combined_text or "7.1" in combined_text:
            explanation = (
                "규제문서에는 금융당국이 당초 예정대로 2025년 7월 1일부터 3단계 스트레스 DSR을 시행하기로 했다고 정리되어 있습니다. "
                "다만 2025년 6월 30일까지 입주자모집공고가 시행된 집단대출과 부동산 매매계약이 체결된 일반 주담대는 종전 2단계 스트레스 DSR 적용 대상으로 설명됩니다. "
                f"근거는 {citation_value} 입니다."
            )
        else:
            explanation = (
                "규제문서에서 DSR 3단계와 직접 연결된 근거를 찾았습니다. "
                f"상세 문구는 Citation의 {citation_value}에서 확인할 수 있습니다."
            )
    else:
        snippets = [
            str(item.get("snippet") or "").strip()
            for item in citations[:2]
            if str(item.get("snippet") or "").strip()
        ]
        headline = "규제문서에서 질문과 연결되는 근거를 찾았습니다."
        explanation = (
            "규제문서 검색 결과를 우선 기준으로 답변합니다. "
            + " ".join(snippet[:180] for snippet in snippets)
            + f" 근거는 {citation_value} 입니다."
        ).strip()

    return {
        "headline": headline,
        "explanation": explanation,
        "highlights": [
            {"label": "Answer Source", "value": "Regulation Document"},
            {"label": "Citation", "value": citation_value},
            {"label": "Evidence Count", "value": str(len(citations))},
        ],
        "citations": citations,
        "source": "regulation-document",
        "source_model": "regulation-search",
    }


def _build_regulation_first_prompt_pack(
    query: str,
    regulation_evidence: list[dict[str, object]],
    base_prompt_pack: dict[str, object] | None = None,
) -> dict[str, object]:
    citations = [_normalize_regulation_citation(item) for item in regulation_evidence[:4]]
    citation_lines = [
        f"- {_format_regulation_citation_value(item)} | {str(item.get('snippet') or '')[:220]}"
        for item in citations
    ] or ["- 규제문서 검색 결과 없음"]
    pack = dict(base_prompt_pack or {})
    semantic_context = dict(pack.get("semantic_context") or {})
    semantic_context["regulation_citations"] = citations
    pack.update({
        "available": True,
        "model": OLLAMA_LIGHTWEIGHT_MODEL,
        "system_prompt": "규제문서 검색 결과를 근거로만 답변합니다. Citation이 없는 내용은 단정하지 않습니다.",
        "user_prompt": "\n".join([
            "[USER QUERY]",
            query or "",
            "",
            "[REGULATION SEARCH RESULT]",
            *citation_lines,
            "",
            "[ANSWER RULE]",
            "- 답변은 한국어로 작성합니다.",
            "- 근거 문서명과 chunk 번호를 반드시 함께 표시합니다.",
            "- 규제문서에 없는 내용은 없다고 말합니다.",
        ]),
        "context_preview": [
            f"query: {query or '-'}",
            f"regulation citations: {', '.join(_format_regulation_citation_value(item) for item in citations) or 'none'}",
        ],
        "semantic_context": semantic_context,
        "answer_mode": "regulation_document_grounding",
    })
    return pack


def _build_general_fallback_prompt_pack(
    query: str,
    base_prompt_pack: dict[str, object] | None = None,
) -> dict[str, object]:
    pack = dict(base_prompt_pack or {})
    pack.update({
        "available": True,
        "model": OLLAMA_LIGHTWEIGHT_MODEL,
        "system_prompt": "\n".join([
            "당신은 금융 Agent의 일반 답변 fallback입니다.",
            "심사로그와 규제문서에서 답을 찾지 못했을 때만 호출됩니다.",
            "실시간 날씨, 현재 영업 여부, 최신 맛집처럼 외부 검색이 필요한 정보는 지어내지 말고 확인할 수 없다고 말합니다.",
            "Citation이 없는 경우 Citation 없음이라고 명확히 말합니다.",
            "답변은 한국어로 짧고 쉽게 작성합니다.",
        ]),
        "user_prompt": "\n".join([
            "[USER QUERY]",
            query or "",
            "",
            "[CONTEXT]",
            "심사로그로 답변할 수 없고, 규제문서에서도 직접 근거를 찾지 못했습니다.",
            "",
            "[ANSWER RULE]",
            "- Ollama 일반지식으로 답변하되 최신/실시간 정보는 단정하지 않습니다.",
            "- 문서 근거가 없으면 Citation 없음이라고 표시합니다.",
        ]),
        "context_preview": [
            f"query: {query or '-'}",
            "regulation citations: none",
            "route: general ollama fallback",
        ],
        "answer_mode": "general_fallback_after_regulation_search",
    })
    return pack


def _build_strategy_simulation_prompt_pack(
    query: str,
    selected_product: str,
    base_prompt_pack: dict[str, object] | None = None,
) -> dict[str, object]:
    pack = dict(base_prompt_pack or {})
    system_prompt = str(pack.get("system_prompt") or "").strip()
    semantic_context = dict(pack.get("semantic_context") or {})
    cluster_summary = dict(semantic_context.get("cluster_summary") or {})
    user_prompt = ""
    strategy_rules = [
        "[STRATEGY SIMULATION MODE]",
        "- 반드시 Ollama가 고객군/승인률/금리/한도/부실률 근거를 함께 해석해 답변한다.",
        "- 질문의 변경 조건을 그대로 사용한다. 금리 질문이면 DSR/거절코드 완화로 바꾸지 않는다.",
        "- 확정 예측처럼 말하지 말고, 현재 학습 데이터 기준의 민감도 가정으로 표현한다.",
        "- 승인률, 수익 또는 이자수익, 리스크/부실률 방향을 짧게 비교한다.",
        "- 근거가 약한 수치는 '가정' 또는 '방향성'이라고 표시한다.",
    ]
    pack["answer_mode"] = "strategy_simulation_ollama_required"
    pack["system_prompt"] = "\n".join([system_prompt, *strategy_rules]).strip()
    pack["user_prompt"] = "\n".join([
        user_prompt,
        "",
        f"[SIMULATION QUESTION] {query}",
        f"[SELECTED PRODUCT] {selected_product}",
        "이 질문은 예측/시뮬레이션 질문입니다. 위 근거 안에서 변경 조건별 영향만 답하세요.",
    ]).strip()
    pack["user_prompt"] = "\n".join([
        f"[SIMULATION QUESTION] {query}",
        f"[SELECTED PRODUCT] {selected_product}",
        "[CURRENT DATA SNAPSHOT]",
        json.dumps(
            {
                "query": query,
                "selected_product": selected_product,
                "cluster_summary": cluster_summary,
                "top_axes": list((semantic_context.get("top_axes") or [])[:5]) if isinstance(semantic_context.get("top_axes"), list) else [],
                "retrieval_evidence": list((semantic_context.get("retrieval_evidence") or [])[:5]) if isinstance(semantic_context.get("retrieval_evidence"), list) else [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "이 질문은 예측/시뮬레이션 질문입니다.",
        "답변은 반드시 금리 변경 조건이 승인률, 이자수익, 리스크/부실률에 주는 방향성으로 작성하세요.",
        "최빈 reject reason, K코드, 거절 사유를 최종 결론으로 삼지 마세요. 필요한 경우 리스크 보조 근거로만 짧게 언급하세요.",
    ]).strip()
    pack["context_preview"] = [
        *list(pack.get("context_preview") or []),
        f"strategy simulation required: {query[:120]}",
    ]
    return pack


def _build_strategy_simulation_answer_summary(
    query: str,
    selected_product: str,
    customer_clusters: list[dict],
    reject_code_summary: list[dict[str, object]],
) -> dict[str, object]:
    strategy = _build_strategy_simulation_result(customer_clusters, reject_code_summary, query=query)
    metrics = list(strategy.get("metrics") or [])
    product_name = _product_display_name(selected_product) or selected_product or "선택 상품"
    scenario = str(strategy.get("scenario") or "전략 조건 시뮬레이션")
    compact_query = _compact_search_text(query)
    rate_up = any(token in compact_query for token in ["올리", "올리면", "인상", "상향", "높이", "높이면"])
    rate_down = any(token in compact_query for token in ["내리", "내리면", "인하", "하향", "낮추", "낮추면"])
    if rate_up:
        metrics = [
            {"label": "승인률 방향", "value": "하락 가능", "tone": "warning"},
            {"label": "이자수익 방향", "value": "건당 수익 상승, 총수익은 승인 감소와 상쇄", "tone": "positive"},
            {"label": "리스크 방향", "value": "저수익/고위험군 이탈 가능, 고금리 부담은 점검", "tone": "neutral"},
        ]
        scenario = "금리 인상 민감도 시뮬레이션"
    elif rate_down:
        metrics = [
            {"label": "승인률 방향", "value": "상승 가능", "tone": "positive"},
            {"label": "이자수익 방향", "value": "건당 수익 하락, 승인 증가로 일부 보전", "tone": "warning"},
            {"label": "리스크 방향", "value": "유입 확대에 따른 부실률 점검 필요", "tone": "warning"},
        ]
        scenario = "금리 인하 민감도 시뮬레이션"
    metric_text = ", ".join(
        f"{item.get('label')}: {item.get('value')}"
        for item in metrics[:3]
        if isinstance(item, dict)
    )
    explanation = (
        f"{product_name}에 대해 '{query}' 조건을 시뮬레이션 질문으로 해석했습니다. "
        f"{scenario} 기준으로 승인률, 이자수익, 리스크 방향을 함께 봐야 합니다. "
        f"현재 데이터 기반 민감도 초안은 {metric_text or '계산 가능한 지표가 제한적입니다'} 입니다. "
        "정확한 확정 예측이 아니라 고객군집과 현재 심사 로그를 이용한 방향성 추정입니다."
    )
    return {
        "headline": f"{product_name} 금리/조건 변경 시뮬레이션",
        "explanation": explanation,
        "highlights": [
            {"label": "Question Type", "value": "Strategy Simulation"},
            {"label": "Scenario", "value": scenario},
            *[
                {"label": str(item.get("label") or f"Metric {index + 1}"), "value": str(item.get("value") or "-")}
                for index, item in enumerate(metrics[:3])
                if isinstance(item, dict)
            ],
        ],
        "source": "strategy-simulation-grounding",
        "citations": [],
    }


def _run_general_ollama_fallback(
    query: str,
    prompt_pack: dict[str, object],
    job_id: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    prompt = _build_ollama_prompt_text(prompt_pack)
    runtime: dict[str, object] = {
        "enabled": bool(prompt),
        "status": "skipped",
        "model": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
        "input": {
            "system_prompt": str(prompt_pack.get("system_prompt") or ""),
            "user_prompt": str(prompt_pack.get("user_prompt") or ""),
            "context_preview": list(prompt_pack.get("context_preview") or []),
            "answer_mode": str(prompt_pack.get("answer_mode") or "general_fallback_after_regulation_search"),
            "prompt": prompt,
            "prompt_char_count": len(prompt),
        },
        "output": {
            "response_text": "",
            "response_preview": "",
        },
        "error": "",
        "duration_ms": 0,
        "updated_at": _iso_now(),
        "used_in_final_answer": False,
        "guardrail": "regulation_first_then_general_ollama",
    }
    summary: dict[str, object] = {
        "headline": "규제문서 근거가 없어 일반 답변으로 전환했습니다.",
        "explanation": "규제문서에서 직접 근거를 찾지 못했습니다. Ollama 일반 답변으로 전환합니다. Citation 없음.",
        "highlights": [
            {"label": "Answer Source", "value": "OLLAMA Fallback"},
            {"label": "Citation", "value": "규제문서 근거 없음"},
        ],
        "citations": [],
        "source": "ollama-general",
        "source_model": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
    }
    if not prompt:
        runtime["error"] = "Ollama prompt is empty."
        return runtime, summary

    started_at = time.perf_counter()
    runtime["status"] = "running"
    if job_id:
        _update_workbench_job(job_id, "ollama", "running", "규제문서 근거가 없어 Ollama 일반 답변을 준비합니다.", meta={"prompt_char_count": len(prompt)})

    try:
        response_text = lightweight_ollama_generate(
            prompt,
            timeout_seconds=WORKBENCH_OLLAMA_TIMEOUT_SECONDS,
            fail_fast_if_busy=True,
            priority_group="general",
        )
        normalized_text = _replace_product_codes_for_display(_normalize_ollama_text(response_text))
        runtime["status"] = "completed"
        runtime["output"] = {
            "response_text": normalized_text,
            "response_preview": normalized_text[:280],
        }
        runtime["used_in_final_answer"] = True
        summary["explanation"] = normalized_text or str(summary.get("explanation") or "")
        if "citation" not in _compact_search_text(str(summary["explanation"])):
            summary["explanation"] = f"{summary['explanation']} Citation 없음."
    except Exception as error:
        runtime["status"] = "unavailable" if _is_ollama_unavailable_error(error) else "failed"
        runtime["error"] = str(error)
        summary["explanation"] = (
            "규제문서에서 직접 근거를 찾지 못했고, 현재 Ollama 일반 답변도 사용할 수 없습니다. "
            "날씨나 맛집처럼 실시간 정보가 필요한 질문은 외부 검색 또는 위치 정보가 필요합니다. Citation 없음."
        )
    finally:
        runtime["duration_ms"] = int((time.perf_counter() - started_at) * 1000)
        runtime["updated_at"] = _iso_now()
    return runtime, summary


def _build_ollama_prompt_pack(
    selected_feature: dict | None,
    related_features: list[dict],
    clusters: list[dict],
    retrieval_results: list[dict],
    query: str,
    selected_product: str,
    primary_feature_selection: dict[str, object],
    question_token_mappings: list[dict[str, object]],
    ontology_payload: dict[str, object],
    regulation_evidence: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    relation_overrides = _read_ontology_relation_overrides()
    is_influence_feature_query = _query_asks_influence_features(query)
    is_cluster_signal_query = _query_asks_cluster_signals(query)
    compact_query = _compact_search_text(query)
    is_ontology_detail_query = _query_has_regulation_intent(query) or any(
        token in compact_query
        for token in ["ontology", "온톨로지", "relation", "관계", "규제", "정책", "룰"]
    )
    query_intent = _build_query_intent_parsing(query, selected_product, question_token_mappings)
    semantic_retrieval_result = _build_semantic_retrieval_result(selected_feature, primary_feature_selection, related_features, selected_product)
    ontology_expansion = _build_ontology_graph_expansion(selected_feature, related_features, primary_feature_selection, selected_product, relation_overrides)
    semantic_pipeline = _build_semantic_pipeline(query_intent, semantic_retrieval_result, ontology_expansion, retrieval_results, regulation_evidence=regulation_evidence)
    semantic_context = _compress_semantic_context(query_intent, semantic_retrieval_result, ontology_expansion, clusters, retrieval_results, regulation_evidence=regulation_evidence)
    representative_features = _resolve_representative_features(primary_feature_selection, selected_feature=selected_feature, limit=3)
    amount_axis_ids = {
        str(item.get("feature_id") or "")
        for item in representative_features
        if str(item.get("axis_key") or _infer_feature_axis_key(item)) == "limit"
        or str(item.get("feature_id") or "") in {"decision.approved_amount", "loan.requested_limit"}
        or any(marker in str(item.get("feature_name") or "") for marker in ["승인가능금액", "대출금액", "한도"])
    }
    amount_axis_labels = {
        str(item.get("feature_id") or ""): str(item.get("feature_name") or item.get("feature_id") or "")
        for item in representative_features
        if str(item.get("feature_id") or "") in amount_axis_ids
    }
    influence_features = [
        item for item in related_features
        if not amount_axis_ids
        or amount_axis_ids & {str(source_id) for source_id in (item.get("source_feature_ids") or [])}
    ][:8]
    cluster_lines = [
        " / ".join(part for part in [
            str(item.get("product") or "").strip(),
            str(item.get("decision") or "").strip(),
            str(item.get("age_band") or "").strip(),
            str(item.get("income_band") or "").strip(),
            str(item.get("amount_band") or "").strip(),
            ", ".join(
                f"{metric.get('label')}: {metric.get('display')}"
                for metric in (item.get("metric_summary") or [])
                if metric.get("display")
            ),
            _format_cluster_reject_reason_summary(item),
        ] if part)
        for item in clusters[:3]
    ]
    relation_lines: list[str] = []
    for relation_name, values in (ontology_expansion.get("relations") or {}).items():
        relation_lines.append(f"{relation_name}:")
        relation_lines.extend(f"- {value}" for value in (values or []))
        relation_lines.append("")
    if relation_lines and not relation_lines[-1].strip():
        relation_lines.pop()
    prompt_pipeline_lines: list[str] = []
    for step in semantic_pipeline:
        prompt_pipeline_lines.append(f"{int(step.get('step') or 0)}. {str(step.get('title') or '')}")
        prompt_pipeline_lines.append(f"- {str(step.get('detail') or '')}")
    primary_feature_lines = [
        f"- {str(item.get('feature_id') or '')} ({str(item.get('feature_name') or item.get('feature_id') or '')}, score={float(item.get('score') or 0.0):.4f})"
        for item in (semantic_retrieval_result.get("primary_features") or [])
    ] or ["- 없음"]
    related_feature_lines = [
        f"- {value}"
        for value in (semantic_retrieval_result.get("related_features") or [])
    ] or ["- 없음"]
    influence_feature_lines = [
        "- "
        + " / ".join(_dedupe_text_items([
            amount_axis_labels.get(str(source_id), str(source_id))
            for source_id in (item.get("source_feature_ids") or [])
            if not amount_axis_ids or str(source_id) in amount_axis_ids
        ], limit=3) or ["승인가능금액/대출금액"])
        + f"에 영향: {str(item.get('feature_id') or '')} ({str(item.get('feature_name') or item.get('feature_id') or '')}, category={str(item.get('category') or '')}, score={float(item.get('score') or 0.0):.2f})"
        + (f" | shared={', '.join(str(token) for token in (item.get('shared_tokens') or [])[:4])}" if item.get("shared_tokens") else "")
        for item in influence_features
    ] or ["- 없음"]
    alias_lines = [f"- {value}" for value in (ontology_expansion.get("aliases") or [])] or ["- 없음"]
    retrieval_evidence_lines = [f"- {value}" for value in (semantic_context.get("retrieval_evidence") or [])] or ["- 없음"]
    regulation_evidence_lines = [
        f"- {item.get('name')}#chunk-{int(item.get('chunk_index') or 0)} score={float(item.get('score') or 0.0):.2f} | {str(item.get('snippet') or '')[:120]}"
        for item in (semantic_context.get("regulation_citations") or [])
    ] or ["- 없음"]
    cluster_context_lines = [f"- {line}" for line in cluster_lines] or ["- 없음"]
    if is_influence_feature_query:
        cluster_context_lines = ["- 영향 feature 질문입니다. 군집 요약은 보조 근거이며 답변 첫 문장에 쓰지 마세요."]
    business_constraint_lines = [f"- {rule}" for rule in DEFAULT_SEMANTIC_GROUNDING_RULES]
    business_constraint_lines.extend([
        "- 대표 축, Principal Axis, Principal Component의 일반 개념 설명은 답변에 쓰지 마세요.",
        "- 현업 사용자가 볼 답변에는 질문과 직접 연결된 심사 기준, 결과, 다음 확인 포인트만 남기세요.",
    ])
    answer_instruction_lines = [
        "답변은 현업 사용자가 바로 이해할 수 있게 2~4문장으로 작성하세요.",
        "대표 축, Principal Axis, ontology expansion, relation, retrieval evidence 같은 내부 처리 용어는 쓰지 마세요.",
        "질문과 직접 연결된 상품, 고객군, 결과 지표, 확인 포인트만 남기세요.",
        "확실하지 않은 부분은 추정 표현으로 제한할 것.",
        "답변은 한국어로 작성.",
    ]
    if is_ontology_detail_query:
        answer_instruction_lines = [
            "정책/온톨로지 질문일 때만 관계 정보를 설명하세요.",
            "relation 이름을 나열하기보다 실제 업무 의미를 쉬운 한국어로 풀어 쓰세요.",
            "질문과 직접 관련 없는 규제/정책 근거는 제외하세요.",
            "답변은 한국어로 작성.",
        ]
    if is_cluster_signal_query:
        answer_instruction_lines = [
            "답변은 고객군의 공통 신호만 짧게 요약하세요.",
            "질문에 있는 소득구간/승인·거절 조건을 반드시 지키세요.",
            "대표 축, ontology expansion, relation, retrieval evidence 같은 내부 처리 용어는 쓰지 마세요.",
            "실제 로그에서 확인된 평균 금리, 평균 한도, 모델 점수, 연체/부실 proxy 같은 지표만 사용하세요.",
            "답변은 한국어로 작성.",
        ]
    if is_influence_feature_query:
        answer_instruction_lines = [
            "답변은 반드시 영향 feature 목록으로 시작하세요.",
            "군집(고소득/30대/중액 등)을 결론 첫 문장으로 쓰지 마세요.",
            "승인가능금액과 대출금액에 영향을 주는 feature를 4~6개 bullet로 설명하세요.",
            "각 feature가 어떤 축(소득, 신용등급, 기존대출, 한도소진 등)으로 영향을 주는지 짧게 쓰세요.",
            "제공된 feature 밖의 새 feature를 만들지 마세요.",
            "답변은 한국어로 작성.",
        ]
    system_prompt = "\n".join([
        "[SYSTEM ROLE]",
        "",
        "당신은 신용 심사 ontology 기반 설명 시스템이다.",
        "반드시 제공된 semantic context만 기반으로 답변한다.",
        "존재하지 않는 정책, 모델, feature를 생성하지 않는다.",
    ])
    user_prompt = "\n".join([
        "--------------------------------------------------",
        "",
        "[USER QUERY]",
        "",
        query or "없음",
        "",
        "--------------------------------------------------",
        "",
        "[ANSWER TARGET]",
        "",
        "Primary Features:",
        *primary_feature_lines,
        "",
        "Influence Features for 승인가능금액/대출금액:",
        *influence_feature_lines,
        "",
        "Instruction:",
        *answer_instruction_lines,
        "",
        "--------------------------------------------------",
        "",
        "[SEMANTIC PIPELINE]",
        "",
        *prompt_pipeline_lines,
        "",
        "--------------------------------------------------",
        "",
        "[SEMANTIC RETRIEVAL RESULT]",
        "",
        "Primary Features:",
        *primary_feature_lines,
        "",
        "Related Features:",
        *related_feature_lines,
        "",
        "Influence Features for 승인가능금액/대출금액:",
        *influence_feature_lines,
        "",
        "Domain:",
        f"- {semantic_retrieval_result.get('domain') or 'General'}",
        "",
        "Entity Type:",
        f"- {semantic_retrieval_result.get('entity_type') or 'FeatureAttribute'}",
        "",
        "Retrieval Evidence:",
        *retrieval_evidence_lines,
        "",
        "Regulation Evidence:",
        *regulation_evidence_lines,
        "",
        "--------------------------------------------------",
        "",
        "[ONTOLOGY EXPANSION]",
        "",
        "Feature:",
        f"- {ontology_expansion.get('feature_id') or ''}",
        "",
        "Canonical Meaning:",
        f"- {ontology_expansion.get('canonical_meaning') or '없음'}",
        "",
        "Aliases:",
        *alias_lines,
        "",
        "Relations:",
        *(relation_lines or ["- 명시된 relation 없음"]),
        "",
        "Business Purpose:",
        f"- {ontology_expansion.get('business_purpose') or '명시된 business purpose 없음'}",
        "",
        "Cluster Context:",
        *cluster_context_lines,
        "",
        "--------------------------------------------------",
        "",
        "[BUSINESS CONSTRAINTS]",
        "",
        *business_constraint_lines,
        "",
        "--------------------------------------------------",
        "",
        "[ANSWER INSTRUCTION]",
        "",
        *answer_instruction_lines,
    ])
    context_preview = [
        f"query: {query or '없음'}",
        f"intent hits: {', '.join(query_intent.get('hit_tokens') or []) or '없음'}",
        f"primary features: {', '.join(item.get('feature_id') or '' for item in semantic_retrieval_result.get('primary_features') or []) or '없음'}",
        f"influence features: {', '.join(str(item.get('feature_id') or '') for item in influence_features[:6]) or '없음'}",
        f"relations: {', '.join((ontology_expansion.get('relations') or {}).keys()) or '없음'}",
        f"retrieval evidence: {', '.join(semantic_context.get('retrieval_evidence') or []) or '없음'}",
        "regulation citations: " + (
            ", ".join(
                f"{str(item.get('name') or 'regulation')}#{int(item.get('chunk_index') or 0)}"
                for item in (semantic_context.get("regulation_citations") or [])[:3]
            )
            or "없음"
        ),
    ]
    return {
        "available": True,
        "model": OLLAMA_LIGHTWEIGHT_MODEL,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "context_preview": context_preview,
        "semantic_pipeline": semantic_pipeline,
        "semantic_context": semantic_context,
        "answer_mode": "influence_features" if is_influence_feature_query else ("cluster_signal" if is_cluster_signal_query else "semantic_summary"),
        "influence_features": influence_features,
        "ontology_expansion": ontology_expansion,
        "ontology_source": {
            "commonfeature_path": str(COMMONFEATURE_PATH.relative_to(ROOT)),
            "ontology_path": str(ONTOLOGY_PATH.relative_to(ROOT)),
            "relation_override_path": str(ONTOLOGY_RELATIONS_PATH.relative_to(ROOT)),
            "ontology_generated_at": str(ontology_payload.get('generated_at') or ''),
        },
    }


def _build_ollama_prompt_text(prompt_pack: dict[str, object]) -> str:
    return "\n\n".join([
        str(prompt_pack.get("system_prompt") or "").strip(),
        str(prompt_pack.get("user_prompt") or "").strip(),
    ]).strip()


def _normalize_ollama_text(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return " ".join(lines).strip()


def _strip_principal_axis_generalities(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    chunks = re.split(r"(?=\s*\d+\.\s*)", normalized)
    if len(chunks) <= 1:
        chunks = re.split(r"(?<=[.!?。])\s+", normalized)
    filtered: list[str] = []
    for chunk in chunks:
        sentence = chunk.strip()
        compact = _compact_search_text(sentence)
        is_principal_axis_definition = (
            ("대표축" in compact or "principalaxis" in compact or "principalcomponent" in compact)
            and any(token in compact for token in ["데이터", "상관관계", "차원", "줄이", "분석", "모델구축"])
        )
        if is_principal_axis_definition:
            continue
        filtered.append(sentence)
    cleaned = " ".join(filtered).strip()
    return cleaned or normalized


def _infer_influence_axis_label(feature: dict[str, object]) -> str:
    text = " ".join([
        str(feature.get("feature_id") or ""),
        str(feature.get("feature_name") or ""),
        str(feature.get("category") or ""),
        " ".join(str(item) for item in (feature.get("shared_tokens") or [])),
    ]).lower()
    if any(token in text for token in ["income", "소득", "연소득", "인정소득"]):
        return "소득/상환여력"
    if any(token in text for token in ["grade", "등급", "score", "스코어", "kcb", "nice"]):
        return "신용등급/스코어"
    if any(token in text for token in ["balance", "잔액", "loan", "대출", "기관수"]):
        return "기존대출/부채"
    if any(token in text for token in ["한도", "limit", "소진"]):
        return "한도/소진율"
    return str(feature.get("category") or "심사 신호")


def _business_influence_bucket(feature: dict[str, object]) -> tuple[str, str]:
    text = " ".join([
        str(feature.get("feature_id") or ""),
        str(feature.get("feature_name") or ""),
        str(feature.get("category") or ""),
        " ".join(str(item) for item in (feature.get("shared_tokens") or [])),
    ]).lower()
    if any(token in text for token in ["balance", "loan_balance", "credit_loan", "잔액", "대출잔액", "기존대출", "부채"]):
        return ("기존대출·부채", "이미 빌린 금액이 크면 추가로 받을 수 있는 한도가 줄어들 수 있습니다.")
    if any(token in text for token in ["kcb", "nice", "grade", "score", "등급", "스코어", "신용"]):
        return ("신용등급·스코어", "신용등급과 외부 신용평가 정보가 한도 산정의 핵심 기준입니다.")
    if any(token in text for token in ["income", "annual_income", "recognized_income", "소득", "연소득", "인정소득", "상환"]):
        return ("소득·상환여력", "소득이 높고 상환 여력이 충분할수록 승인 가능 한도가 커집니다.")
    if any(token in text for token in ["limit", "amount", "approved_amount", "requested_limit", "한도", "승인가능금액", "대출금액"]):
        return ("신청금액·한도", "신청한 금액과 내부 한도 기준이 맞는지 함께 봅니다.")
    return ("심사 보조 신호", "승인 한도를 보정할 때 함께 참고하는 심사 신호입니다.")


def _build_business_friendly_influence_answer(prompt_pack: dict[str, object], selected_product: str) -> str:
    influence_features = [
        item for item in (prompt_pack.get("influence_features") or [])
        if isinstance(item, dict) and str(item.get("feature_name") or item.get("feature_id") or "").strip()
    ]
    product_label = _product_display_name(selected_product) or selected_product or "해당 상품"
    if not influence_features:
        return f"{product_label}의 승인 한도에 영향을 주는 핵심 feature를 충분히 좁히지 못했습니다. 먼저 신청금액, 소득, 신용등급, 기존대출 정보를 확인하는 것이 좋습니다."

    grouped: dict[str, dict[str, object]] = {}
    for feature in influence_features:
        bucket, explanation = _business_influence_bucket(feature)
        feature_name = str(feature.get("feature_name") or feature.get("feature_id") or "").strip()
        if not feature_name:
            continue
        item = grouped.setdefault(bucket, {"explanation": explanation, "features": []})
        if feature_name not in item["features"]:
            item["features"].append(feature_name)

    priority = ["기존대출·부채", "신용등급·스코어", "소득·상환여력", "신청금액·한도", "심사 보조 신호"]
    ordered = sorted(grouped.items(), key=lambda pair: priority.index(pair[0]) if pair[0] in priority else len(priority))
    lines = [f"{product_label}의 승인 한도는 크게 {min(len(ordered), 3)}가지를 보면 됩니다."]
    for index, (bucket, item) in enumerate(ordered[:3], 1):
        examples = ", ".join(_dedupe_text_items([str(value) for value in item.get("features", [])], limit=2))
        example_suffix = f" 예: {examples}" if examples else ""
        lines.append(f"{index}. {bucket}: {item.get('explanation')}{example_suffix}")
    lines.append("정리하면 현업에서는 이미 빌린 금액, 신용등급, 인정 가능한 소득 순서로 확인하면 됩니다.")
    return " ".join(lines)


def _build_grounded_influence_answer(prompt_pack: dict[str, object], selected_product: str) -> str:
    return _build_business_friendly_influence_answer(prompt_pack, selected_product)
    influence_features = [
        item for item in (prompt_pack.get("influence_features") or [])
        if isinstance(item, dict) and str(item.get("feature_id") or "").strip()
    ][:6]
    primary_features = list(((prompt_pack.get("semantic_context") or {}).get("semantic_retrieval_result") or {}).get("primary_features") or [])
    primary_label = " / ".join(_dedupe_text_items([
        str(item.get("feature_name") or item.get("feature_id") or "")
        for item in primary_features
    ], limit=3))
    lines = [
        f"{selected_product or 'ALL'} 기준 {primary_label or '승인가능금액/대출금액'}에 영향을 주는 feature는 아래 후보를 우선 봐야 합니다.",
    ]
    for index, feature in enumerate(influence_features, 1):
        feature_id = str(feature.get("feature_id") or "")
        feature_name = str(feature.get("feature_name") or feature_id)
        axis_label = _infer_influence_axis_label(feature)
        source_axes = " / ".join(_dedupe_text_items([
            str(item) for item in (feature.get("source_axes") or [])
        ], limit=2)) or "승인가능금액/대출금액"
        lines.append(f"{index}. {feature_name} ({feature_id}): {source_axes}에 연결된 {axis_label} 신호입니다.")
    if not influence_features:
        lines.append("제공된 semantic context 안에서는 영향 feature 후보가 충분히 잡히지 않았습니다.")
    lines.append("군집 결과는 이 질문의 결론이 아니라 보조 근거로만 사용했습니다.")
    return " ".join(lines)


def _run_ollama_for_workbench(prompt_pack: dict[str, object], answer_summary: dict[str, object], job_id: str | None = None) -> tuple[dict[str, object], dict[str, object]]:
    prompt = _build_ollama_prompt_text(prompt_pack)
    runtime: dict[str, object] = {
        "enabled": bool(prompt),
        "status": "skipped",
        "model": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
        "input": {
            "system_prompt": str(prompt_pack.get("system_prompt") or ""),
            "user_prompt": str(prompt_pack.get("user_prompt") or ""),
            "context_preview": list(prompt_pack.get("context_preview") or []),
            "answer_mode": str(prompt_pack.get("answer_mode") or "semantic_summary"),
            "influence_features": list(prompt_pack.get("influence_features") or []),
            "prompt": prompt,
            "prompt_char_count": len(prompt),
        },
        "output": {
            "response_text": "",
            "response_preview": "",
        },
        "error": "",
        "duration_ms": 0,
        "updated_at": _iso_now(),
        "used_in_final_answer": False,
    }
    merged_summary = dict(answer_summary or {})
    highlights = list(merged_summary.get("highlights") or [])

    if not prompt:
        runtime["error"] = "Ollama prompt 가 비어 있습니다."
        return runtime, merged_summary

    started_at = time.perf_counter()
    runtime["status"] = "running"
    if job_id:
        _update_workbench_job(job_id, "ollama", "running", "Ollama 입력을 조립했습니다.", meta={"prompt_char_count": len(prompt)})

    try:
        response_text = lightweight_ollama_generate(
            prompt,
            timeout_seconds=WORKBENCH_OLLAMA_TIMEOUT_SECONDS,
            fail_fast_if_busy=True,
            priority_group="ontology",
        )
        normalized_text = _normalize_ollama_text(response_text)
        normalized_text = _strip_principal_axis_generalities(normalized_text)
        if str(prompt_pack.get("answer_mode") or "") == "influence_features":
            prompt_product = str((((prompt_pack.get("semantic_context") or {}).get("query_intent") or {}).get("selected_product") or ""))
            normalized_text = _build_grounded_influence_answer(prompt_pack, prompt_product)
        normalized_text = _replace_product_codes_for_display(normalized_text)
        runtime["status"] = "completed"
        runtime["output"] = {
            "response_text": normalized_text,
            "response_preview": normalized_text[:280],
        }
        runtime["used_in_final_answer"] = True
        runtime["duration_ms"] = int((time.perf_counter() - started_at) * 1000)
        runtime["updated_at"] = _iso_now()
        merged_summary["explanation"] = normalized_text or str(merged_summary.get("explanation") or "")
        merged_summary["source"] = "ollama"
        merged_summary["source_model"] = str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)
        filtered_highlights = [item for item in highlights if item.get("label") not in {"Answer Source", "LLM Model"}]
        merged_summary["highlights"] = [
            {"label": "Answer Source", "value": "OLLAMA"},
            {"label": "LLM Model", "value": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)},
            *filtered_highlights,
        ]
        if job_id:
            _update_workbench_job(job_id, "ollama", "running", "Ollama가 최종 답변을 생성했습니다.", tone="success", meta={"duration_ms": runtime["duration_ms"]})
    except Exception as error:
        if not _is_ollama_unavailable_error(error):
            raise
        runtime["status"] = "unavailable"
        runtime["error"] = str(error)
        runtime["duration_ms"] = int((time.perf_counter() - started_at) * 1000)
        runtime["updated_at"] = _iso_now()
        merged_summary["source"] = "rule-based-fallback"
        merged_summary["source_model"] = str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)
        merged_summary["highlights"] = [
            {"label": "Answer Source", "value": "Fallback"},
            {"label": "LLM Model", "value": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)},
            *[item for item in highlights if item.get("label") not in {"Answer Source", "LLM Model"}],
        ]
        if job_id:
            _update_workbench_job(job_id, "ollama", "running", f"Ollama 미사용: {error}", tone="warning")
    except Exception as error:
        runtime["status"] = "failed"
        runtime["error"] = str(error)
        runtime["duration_ms"] = int((time.perf_counter() - started_at) * 1000)
        runtime["updated_at"] = _iso_now()
        merged_summary["source"] = "rule-based-fallback"
        merged_summary["source_model"] = str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)
        merged_summary["highlights"] = [
            {"label": "Answer Source", "value": "Fallback"},
            {"label": "LLM Model", "value": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)},
            *[item for item in highlights if item.get("label") not in {"Answer Source", "LLM Model"}],
        ]
        if job_id:
            _update_workbench_job(job_id, "ollama", "running", f"Ollama 실패: {error}", tone="warning")
    return runtime, merged_summary


def _build_feature_workbench_payload(
    selected_product: str = "",
    query: str = "",
    feature_id: str = "",
    job_id: str | None = None,
    conversation_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    if job_id:
        _update_workbench_job(job_id, "extraction", "running", "commonfeature.json 과 full_text_records.json 을 읽는 중입니다.")
    ontology = _read_json_file(ONTOLOGY_PATH)
    commonfeature = _read_json_file(COMMONFEATURE_PATH)
    records = _read_record_list(FULL_TEXT_RECORDS_PATH)
    # profiles 1회만 생성
    reject_code_mapping = _get_reject_code_mapping()
    profiles = [_build_record_profile(record, reject_code_mapping=reject_code_mapping) for record in records]
    if job_id:
        _update_workbench_job(job_id, "extraction", "completed", f"feature {len(commonfeature.get('common_features') or [])}건, record {len(records)}건을 읽었습니다.", meta={"record_count": len(records)})
    all_features = list(commonfeature.get("common_features") or [])
    features = list(all_features)
    requested_product = selected_product
    product_resolution = _infer_product_from_query(query)
    inferred_product = str(product_resolution.get("product") or "")
    if inferred_product and str(product_resolution.get("confidence") or "") == "high":
        selected_product = inferred_product
    if job_id:
        _update_workbench_job(job_id, "alias", "running", "선택 상품 기준으로 feature 후보를 필터링합니다.", meta={"product": selected_product or "ALL"})
    if selected_product:
        features = [feature for feature in features if selected_product in (feature.get("products") or [])]
    if job_id:
        _update_workbench_job(job_id, "alias", "completed", f"상품 범위 필터 후 feature {len(features)}건이 남았습니다.", meta={"feature_count": len(features)})

    if job_id:
        _update_workbench_job(job_id, "mapping", "running", "질문과 가장 가까운 feature 를 semantic rank 합니다.", meta={"query": query})
    ranked_features, semantic_search_mode, semantic_scores, conversation_adjustments, conversation_reasons = _semantic_rank_features(
        features,
        query,
        selected_product,
        all_features=all_features,
        conversation_profile=conversation_profile,
    )
    ranked_features.sort(key=lambda item: (-item[0], -(int((item[1].get("coverage") or {}).get("mapping_count") or 0)), str(item[1].get("feature_name") or item[1].get("feature_id") or "")))

    # 도메인/카테고리 우선순위 적용: 직장인은 직업/재직, 카드론은 loan/한도/금리/연체 도메인 feature 우선 포함
    def is_job_feature(item):
        name = (item.get("feature_name") or "").lower()
        category = (item.get("category") or "").lower()
        return any(token in name or token in category for token in ["직업", "재직", "고용", "직장"])

    def is_loan_core_feature(item):
        name = (item.get("feature_name") or "").lower()
        category = (item.get("category") or "").lower()
        return any(token in name or token in category for token in ["loan", "한도", "금리", "연체", "이자", "대출", "limit", "rate", "delinquency"])

    # 1. 도메인별 우선 feature 추출
    top_candidates = [item[1] for item in ranked_features[:12]]
    job_features = [item for item in top_candidates if is_job_feature(item)]
    loan_features = [item for item in top_candidates if is_loan_core_feature(item)]
    # 2. top-3에 도메인별 대표 feature가 반드시 포함되도록 보장
    selected = []
    for item in (job_features + loan_features):
        if item not in selected:
            selected.append(item)
        if len(selected) >= 3:
            break
    for item in top_candidates:
        if item not in selected:
            selected.append(item)
        if len(selected) >= 3:
            break
    # top_k 후보를 도메인 우선으로 재정의
    top_k_candidates = selected[:3]
    if job_id:
        _update_workbench_job(job_id, "mapping", "completed", f"semantic search mode={semantic_search_mode}, 상위 후보 {min(len(ranked_features), 12)}건을 정렬했습니다.", meta={"semantic_search_mode": semantic_search_mode})

    if job_id:
        _update_workbench_job(job_id, "ontology", "running", "top-k + intent + graph hybrid 방식으로 대표 축과 related feature 를 계산합니다.")
    # 도메인 우선 top_k_candidates를 hybrid 대표축 선정에 강제 반영
    # top_k_candidates가 3개 이상이면 이 후보만으로 대표축 선정
    if len(top_k_candidates) >= 3:
        ranked_features_for_hybrid = [(0, item) for item in top_k_candidates]
    else:
        ranked_features_for_hybrid = ranked_features
    selected_feature, primary_feature_selection = _select_primary_feature_hybrid(
        ranked_features_for_hybrid,
        query,
        selected_product,
        all_features=all_features,
        pinned_feature_id=feature_id,
    )
    if not feature_id:
        primary_feature_selection = _augment_primary_feature_selection_with_metric_intents(
            primary_feature_selection,
            query,
            all_features,
        )
        primary_feature_selection = _augment_primary_feature_selection_with_product_intent(
            primary_feature_selection,
            selected_product,
            all_features,
        )
    if not selected_feature and features:
        selected_feature = sorted(features, key=lambda item: -(int((item.get("coverage") or {}).get("mapping_count") or 0)))[0]
        primary_feature_selection = {
            "mode": "coverage-fallback",
            "selected_feature_id": str(selected_feature.get("feature_id") or ""),
            "selected_feature_name": str(selected_feature.get("feature_name") or selected_feature.get("feature_id") or "대표 축"),
            "headline": "hybrid 후보가 비어 coverage 기준 fallback 을 사용했습니다.",
            "representative_features": _resolve_representative_features({}, selected_feature=selected_feature, limit=1),
            "intent_tokens": _tokenize_text(query)[:8],
            "graph_relation_mode": "rule",
            "top_k": [],
            "graph_result_explanation": [],
        }
        primary_feature_selection = _augment_primary_feature_selection_with_metric_intents(
            primary_feature_selection,
            query,
            all_features,
        )
        primary_feature_selection = _augment_primary_feature_selection_with_product_intent(
            primary_feature_selection,
            selected_product,
            all_features,
        )

    representative_features = _resolve_representative_features(primary_feature_selection, selected_feature=selected_feature, limit=3)
    if _query_has_reject_intent(query, selected_feature):
        representative_features = _prioritize_reject_representative_features(
            representative_features,
            selected_product,
            all_features,
        )
        selected_feature_name = str((selected_feature or {}).get("feature_name") or (selected_feature or {}).get("feature_id") or "")
        if _is_cross_product_feature_label(selected_feature_name, selected_product) and representative_features:
            selected_feature = representative_features[0]
        primary_feature_selection = dict(primary_feature_selection or {})
        primary_feature_selection["representative_features"] = representative_features
        primary_feature_selection["selected_feature_id"] = str((selected_feature or {}).get("feature_id") or "")
        primary_feature_selection["selected_feature_name"] = str((selected_feature or {}).get("feature_name") or (selected_feature or {}).get("feature_id") or "")
    related_features = _build_related_features_multi(representative_features, features, selected_product, all_features=all_features)
    representative_axis_details = _build_representative_axis_details(representative_features, primary_feature_selection, related_features)
    clarification = _build_clarification_suggestion(
        query,
        ranked_features,
        primary_feature_selection,
        conversation_profile=conversation_profile,
        product_resolution=product_resolution,
        selected_product=selected_product,
    )
    question_token_mappings = _build_query_token_feature_mapping(
        query,
        selected_product,
        ranked_features,
        primary_feature_selection,
        all_features=all_features,
    )
    if job_id:
        _update_workbench_job(job_id, "ontology", "completed", f"{primary_feature_selection.get('mode') or 'hybrid'} 방식으로 대표 축 {len(representative_features)}건과 related feature {len(related_features)}건을 확정했습니다.", meta={"selected_feature": str((selected_feature or {}).get('feature_id') or ''), "selection_mode": str(primary_feature_selection.get('mode') or ''), "representative_feature_count": len(representative_features)})

    if job_id:
        _update_workbench_job(job_id, "faiss", "running", "고객군집 캐시와 cluster 후보를 계산합니다.")
    customer_cluster_cache = _load_or_build_customer_cluster_cache(records)
    customer_clusters = _build_customer_clusters(records, selected_product, query=query, selected_feature=selected_feature, representative_features=representative_features)
    # profiles를 넘겨서 중복 변환 방지
    reject_code_summary = _build_reject_code_distribution(profiles, selected_product, query, limit=3)
    if job_id:
        _update_workbench_job(job_id, "faiss", "completed", f"cluster {len(customer_clusters)}건을 계산했고 K코드 상위 {len(reject_code_summary)}건을 확인했습니다.", meta={"cluster_count": len(customer_clusters), "top_reject_codes": reject_code_summary})

    if job_id:
        _update_workbench_job(job_id, "retrieval", "running", "관련 레코드와 retrieval 결과를 생성합니다.")
    retrieval_results = _build_retrieval_results(records, selected_product, query, selected_feature, representative_features=representative_features)
    representative_feature_ids = [
        str(item.get("feature_id") or "").strip()
        for item in representative_features
        if str(item.get("feature_id") or "").strip()
    ]
    related_feature_ids = [
        str(item.get("feature_id") or "").strip()
        for item in related_features[:8]
        if str(item.get("feature_id") or "").strip()
    ]
    raw_regulation_evidence = search_regulation_evidence(
        query,
        k=6,
        preferred_feature_ids=representative_feature_ids,
        avoided_feature_ids=list((conversation_profile or {}).get("avoided_feature_ids") or []),
        expansion_feature_ids=related_feature_ids,
    )
    raw_regulation_evidence_count = len(raw_regulation_evidence)
    regulation_first_route = _query_requires_regulation_first(query)
    strategy_simulation_route = _query_requires_strategy_simulation(query)
    intent_classification = _classify_query_intent(
        query,
        selected_feature,
        regulation_first_route=regulation_first_route,
        strategy_simulation_route=strategy_simulation_route,
    )
    if regulation_first_route:
        regulation_evidence, regulation_evidence_reason = _select_regulation_first_evidence(
            query,
            raw_regulation_evidence,
        )
    else:
        regulation_evidence, regulation_evidence_reason = _filter_regulation_evidence_for_answer(
            query,
            raw_regulation_evidence,
            representative_feature_ids=representative_feature_ids,
            related_feature_ids=related_feature_ids,
        )
    reject_codes = sorted({str(code).strip() for record in records for code in (record.get("reject_reason_codes") or []) if str(code).strip()})
    if job_id:
        _update_workbench_job(
            job_id,
            "retrieval",
            "completed",
            f"retrieval 결과 {len(retrieval_results)}건과 규제 evidence {len(regulation_evidence)}건, reject code {len(reject_codes)}종을 정리했습니다.",
            meta={
                "retrieval_count": len(retrieval_results),
                "regulation_evidence_count": len(regulation_evidence),
                "raw_regulation_evidence_count": raw_regulation_evidence_count,
                "regulation_evidence_reason": regulation_evidence_reason,
                "regulation_first_route": regulation_first_route,
            },
        )

    if job_id:
        _update_workbench_job(job_id, "ollama", "running", "최종 화면용 answer summary 와 Ollama 입력을 정리합니다.")
    prompt_pack = _build_ollama_prompt_pack(
        selected_feature,
        related_features,
        customer_clusters,
        retrieval_results,
        query,
        selected_product,
        primary_feature_selection,
        question_token_mappings,
        ontology,
        regulation_evidence=regulation_evidence,
    )
    if conversation_profile:
        memo_notes = str(conversation_profile.get("memo_notes") or "").strip()
        department = str(conversation_profile.get("department") or "").strip()
        answer_mode = str(conversation_profile.get("answer_mode") or "").strip()
        memo_lines = [
            "[USER MEMO / DEPARTMENT PREFERENCE]",
            f"- mode: {answer_mode or 'general'}",
            f"- department: {department or 'not selected'}",
            f"- memo: {memo_notes or 'none'}",
            "- 답변에는 이 메모를 우선 고려하되, 실제 데이터 근거와 충돌하면 데이터 근거를 우선하세요.",
        ]
        prompt_pack["user_prompt"] = "\n".join([str(prompt_pack.get("user_prompt") or ""), "", *memo_lines]).strip()
        prompt_pack["context_preview"] = [
            *list(prompt_pack.get("context_preview") or []),
            f"user memo: {department or '-'} / {memo_notes[:80] or '-'}",
        ]
    evidence_gate = evaluate_runtime_evidence(
        query=query,
        answer_mode=str((conversation_profile or {}).get("answer_mode") or "general"),
        retrieval_results=retrieval_results,
        regulation_evidence=regulation_evidence,
        customer_clusters=customer_clusters,
    )
    if evidence_gate.get("allowed"):
        prompt_pack = apply_grounded_prompt_rules(prompt_pack, evidence_gate)
    server_answer_summary = _build_answer_summary(query, selected_product, selected_feature, representative_features, customer_clusters, retrieval_results, profiles, related_features=related_features, reject_code_summary=reject_code_summary)
    server_answer_summary["citations"] = list((prompt_pack.get("semantic_context") or {}).get("regulation_citations") or [])
    if not evidence_gate.get("allowed"):
        ollama_runtime, answer_summary = build_blocked_runtime_answer(
            query=query,
            evaluation=evidence_gate,
            model=str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
        )
        ollama_runtime["updated_at"] = _iso_now()
    elif regulation_first_route and regulation_evidence:
        prompt_pack = _build_regulation_first_prompt_pack(query, regulation_evidence, prompt_pack)
        grounded_summary = _build_regulation_first_answer_summary(query, regulation_evidence)
        ollama_runtime, answer_summary = _run_ollama_for_workbench(prompt_pack, grounded_summary, job_id=job_id)
        answer_summary["citations"] = list(grounded_summary.get("citations") or [])
    elif regulation_first_route:
        prompt_pack = _build_general_fallback_prompt_pack(query, prompt_pack)
        ollama_runtime, answer_summary = _run_general_ollama_fallback(query, prompt_pack, job_id=job_id)
    elif strategy_simulation_route:
        prompt_pack = _build_strategy_simulation_prompt_pack(query, selected_product, prompt_pack)
        strategy_answer_summary = _build_strategy_simulation_answer_summary(
            query,
            selected_product,
            customer_clusters,
            reject_code_summary,
        )
        ollama_runtime, answer_summary = _run_ollama_for_workbench(prompt_pack, strategy_answer_summary, job_id=job_id)
    elif _query_asks_average_metrics(query) or query_has_segment_metric_intent(query):
        answer_summary = _build_average_metric_answer_summary(query, selected_product, profiles, customer_clusters)
        ollama_runtime = {
            "enabled": False,
            "status": "skipped",
            "model": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
            "input": {
                "answer_mode": "average_metric_grounding",
                "query": query,
                "selected_product": selected_product,
            },
            "output": {
                "response_text": "",
                "response_preview": "",
            },
            "error": "",
            "duration_ms": 0,
            "updated_at": _iso_now(),
            "used_in_final_answer": False,
            "guardrail": "average_metric_grounding",
        }
    elif _query_asks_cluster_signals(query):
        answer_summary = _build_cluster_signal_answer_summary(query, selected_product, customer_clusters)
        ollama_runtime = {
            "enabled": False,
            "status": "skipped",
            "model": str(prompt_pack.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
            "input": {
                "answer_mode": "cluster_signal_grounding",
                "query": query,
                "selected_product": selected_product,
            },
            "output": {
                "response_text": "",
                "response_preview": "",
            },
            "error": "",
            "duration_ms": 0,
            "updated_at": _iso_now(),
            "used_in_final_answer": False,
            "guardrail": "cluster_signal_grounding",
        }
    else:
        ollama_runtime, answer_summary = _run_ollama_for_workbench(prompt_pack, server_answer_summary, job_id=job_id)
    if not list(answer_summary.get("citations") or []) and regulation_evidence:
        answer_summary["citations"] = [
            _normalize_regulation_citation(item)
            for item in regulation_evidence[:4]
        ]
    if evidence_gate.get("allowed"):
        answer_summary = decorate_grounded_answer_summary(answer_summary, evidence_gate)
    reject_code_line = _format_reject_code_summary(reject_code_summary)
    if reject_code_line and not strategy_simulation_route and _query_has_reject_intent(query, selected_feature):
        reject_code_scope_note = _format_reject_code_scope_note(reject_code_summary, selected_product)
        explanation_text = str(server_answer_summary.get("explanation") or answer_summary.get("explanation") or "").strip()
        if not all(str(item.get("code") or "") in explanation_text for item in reject_code_summary[:3]):
            scope_prefix = f"{reject_code_scope_note} " if reject_code_scope_note else ""
            explanation_text = f"{explanation_text} {scope_prefix}거절사유코드 상위 코드는 {reject_code_line} 입니다.".strip()
        answer_summary["headline"] = server_answer_summary.get("headline") or answer_summary.get("headline")
        answer_summary["explanation"] = explanation_text
        answer_summary["source"] = "server-grounded"
        answer_summary["source_model"] = str(ollama_runtime.get("model") or OLLAMA_LIGHTWEIGHT_MODEL)
        answer_summary["reject_code_summary"] = reject_code_summary
        answer_summary["citations"] = list(server_answer_summary.get("citations") or [])
        ollama_runtime["used_in_final_answer"] = False
        ollama_runtime["guardrail"] = "reject_code_grounding"
        existing_highlights = [
            item for item in (answer_summary.get("highlights") or [])
            if item.get("label") != "Top Reject Codes"
        ]
        insert_at = 2 if len(existing_highlights) >= 2 else len(existing_highlights)
        answer_summary["highlights"] = [
            *existing_highlights[:insert_at],
            {"label": "Top Reject Codes", "value": reject_code_line},
            *existing_highlights[insert_at:],
        ]
    answer_summary["headline"] = _replace_product_codes_for_display(str(answer_summary.get("headline") or ""))
    answer_summary["explanation"] = _replace_product_codes_for_display(str(answer_summary.get("explanation") or ""))
    for item in answer_summary.get("highlights") or []:
        if isinstance(item, dict) and "value" in item:
            item["value"] = _replace_product_codes_for_display(str(item.get("value") or ""))
    if job_id:
        ollama_status = str(ollama_runtime.get("status") or "unknown")
        _update_workbench_job(job_id, "ollama", "completed", f"화면 상단 answer summary 를 생성했습니다. ollama={ollama_status}", meta={"headline": answer_summary.get("headline"), "ollama_status": ollama_status})
    agentic_workspace = _build_agentic_workspace_payload(
        query=query,
        selected_product=selected_product,
        selected_feature=selected_feature,
        representative_features=representative_features,
        customer_clusters=customer_clusters,
        reject_code_summary=reject_code_summary,
        regulation_evidence=regulation_evidence,
        regulation_evidence_reason=regulation_evidence_reason,
        answer_summary=answer_summary,
    )
    output_classification = _classify_output_categories(
        input_intent=str(intent_classification.get("intent") or ""),
        answer_summary=answer_summary,
        agentic_workspace=agentic_workspace,
        ollama_runtime=ollama_runtime,
        regulation_evidence=regulation_evidence,
        reject_code_summary=reject_code_summary,
    )

    return {
        "status": "ok",
        "input": {
            "product": selected_product,
            "requested_product": requested_product,
            "product_resolution": product_resolution,
            "query": query,
            "feature_id": feature_id,
            "intent": str(intent_classification.get("intent") or ""),
            "intent_confidence": float(intent_classification.get("confidence") or 0.0),
        },
        "summary": {
            "feature_count": len(features),
            "record_count": len(records),
            "cluster_count": len(customer_clusters),
            "reject_code_count": len(reject_codes),
            "top_reject_codes": reject_code_summary,
            "selected_feature_name": str((selected_feature or {}).get("feature_name") or (selected_feature or {}).get("feature_id") or ""),
            "representative_feature_names": [
                str(item.get("feature_name") or item.get("feature_id") or "")
                for item in representative_features
            ],
            "semantic_search_mode": semantic_search_mode,
            "primary_feature_select_mode": str(primary_feature_selection.get("mode") or "topk-intent-graph-hybrid"),
            "ollama_status": str(ollama_runtime.get("status") or "skipped"),
            "ollama_model": str(ollama_runtime.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
            "semantic_rank_formula": {
                "semantic_weight": 20,
                "haystack_hit_weight": 4,
                "feature_id_hit_weight": 5,
                "feature_name_hit_weight": 6,
                "product_match_bonus": 3,
                "coverage_bonus_cap": 6,
                "reject_boost_mode": "rule-based",
            },
            "cluster_storage_mode": "file-cache",
            "cluster_cache_built_at": str((customer_cluster_cache.get("meta") or {}).get("built_at") or ""),
            "clarification_needed": bool(clarification.get("needed")),
            "products": sorted({str(record.get("product") or "").strip() for record in records if str(record.get("product") or "").strip()}),
            "intent": str(intent_classification.get("intent") or ""),
            "intent_confidence": float(intent_classification.get("confidence") or 0.0),
            "output_category": str(output_classification.get("primary") or ""),
        },
        "intent_classification": intent_classification,
        "output_classification": output_classification,
        "search_results": [
            {
                "score": round(float(score), 4),
                "feature_id": feature.get("feature_id"),
                "feature_name": feature.get("feature_name") or feature.get("feature_id"),
                "category": feature.get("category") or "unclassified",
                "description": feature.get("description") or "",
                "products": feature.get("products") or [],
                "directions": feature.get("directions") or [],
                "aliases": (feature.get("aliases") or [])[:8],
                "coverage": feature.get("coverage") or {},
                "score_breakdown": _build_feature_rank_breakdown(
                    feature,
                    query,
                    selected_product,
                    semantic_scores.get(str(feature.get("feature_id") or "")),
                    float(score),
                    conversation_adjustment=conversation_adjustments.get(str(feature.get("feature_id") or ""), 0.0),
                    conversation_reasons=conversation_reasons.get(str(feature.get("feature_id") or ""), []),
                ),
            }
            for score, feature in ranked_features[:12]
        ],
        "selected_feature": selected_feature or {},
        "representative_features": representative_features,
        "representative_axis_details": representative_axis_details,
        "primary_feature_selection": primary_feature_selection,
        "question_token_mappings": question_token_mappings,
        "related_features": related_features,
        "customer_clusters": customer_clusters,
        "cluster_labels": [
            {
                "cluster_id": item.get("cluster_id"),
                "label": item.get("label"),
                "reason": " / ".join(part for part in [
                    str(item.get('decision') or '').strip(),
                    str(item.get('income_band') or '').strip(),
                    str(item.get('amount_band') or '').strip(),
                    _format_cluster_reject_reason_summary(item),
                ] if part),
            }
            for item in customer_clusters
        ],
        "retrieval_results": retrieval_results,
        "regulation_evidence": regulation_evidence,
        "regulation_evidence_meta": {
            "raw_count": raw_regulation_evidence_count,
            "shown_count": len(regulation_evidence),
            "reason": regulation_evidence_reason,
        },
        "semantic_pipeline": list(prompt_pack.get("semantic_pipeline") or []),
        "semantic_context": dict(prompt_pack.get("semantic_context") or {}),
        "ontology_expansion": dict(prompt_pack.get("ontology_expansion") or {}),
        "clarification": clarification,
        "conversation": {
            "session_id": str((conversation_profile or {}).get("session_id") or ""),
            "turn_id": str((conversation_profile or {}).get("turn_id") or ""),
            "memory_keywords": list((conversation_profile or {}).get("memory_keywords") or []),
            "history_feature_ids": list((conversation_profile or {}).get("history_feature_ids") or []),
            "preferred_feature_ids": list((conversation_profile or {}).get("preferred_feature_ids") or []),
            "avoided_feature_ids": list((conversation_profile or {}).get("avoided_feature_ids") or []),
            "answer_mode": str((conversation_profile or {}).get("answer_mode") or ""),
            "department": str((conversation_profile or {}).get("department") or ""),
            "memo_notes": str((conversation_profile or {}).get("memo_notes") or ""),
        },
        "cluster_storage": {
            "path": str(FEATURE_CLUSTER_CACHE_PATH.relative_to(ROOT)),
            "built_at": str((customer_cluster_cache.get("meta") or {}).get("built_at") or ""),
            "record_count": int((customer_cluster_cache.get("meta") or {}).get("record_count") or 0),
            "income_band_thresholds": (customer_cluster_cache.get("meta") or {}).get("income_band_thresholds") or [],
            "amount_band_thresholds": (customer_cluster_cache.get("meta") or {}).get("amount_band_thresholds") or [],
            "income_sources": (customer_cluster_cache.get("meta") or {}).get("income_sources") or [],
            "amount_sources": (customer_cluster_cache.get("meta") or {}).get("amount_sources") or [],
        },
        "roadmap": [
            {"key": "feature_ontology", "title": "Feature Ontology API", "status": "ready", "detail": "commonfeature.json 기반 집계/검색/요약"},
            {"key": "semantic_search", "title": "Semantic Search UI", "status": "ready", "detail": f"{semantic_search_mode} feature semantic search"},
            {"key": "feature_relation", "title": "Feature Relation", "status": "ready", "detail": "공통 category, 상품, direction 기반 관계 계산"},
            {"key": "customer_cluster", "title": "Customer Cluster", "status": "ready", "detail": "결정/소득/금액/거절코드 기준 고객군집 파일 캐시"},
            {"key": "cluster_labeling", "title": "Cluster Labeling", "status": "ready", "detail": "군집 자동 라벨 생성"},
            {"key": "retrieval_layer", "title": "Retrieval Layer", "status": "ready", "detail": "feature/query 기반 관련 레코드 검색"},
            {"key": "ollama", "title": "OLLAMA", "status": "ready", "detail": "로컬 Ollama 실행 또는 prompt pack 전달"},
        ],
        "agentic_workspace": agentic_workspace,
        "agent_workflow": list(agentic_workspace.get("agent_workflow") or []),
        "semantic_financial_layer": dict(agentic_workspace.get("semantic_layer") or {}),
        "answer_summary": answer_summary,
        "ollama_runtime": ollama_runtime,
        "ollama_ready": prompt_pack,
    }


@app.get("/ontology/state")
def ontology_state() -> dict:
    ontology = _read_json_file(ONTOLOGY_PATH)
    commonfeature = _read_json_file(COMMONFEATURE_PATH)
    return {
        "status": "ok",
        "ontology": ontology,
        "commonfeature": commonfeature,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/feature-ontology/workbench")
def feature_ontology_workbench(
    product: str = Query(default=""),
    query: str = Query(default=""),
    feature_id: str = Query(default=""),
) -> dict:
    return _build_feature_workbench_payload(
        selected_product=str(product or "").strip(),
        query=str(query or "").strip(),
        feature_id=str(feature_id or "").strip(),
    )


@app.post("/feature-ontology/runtime-jobs")
def feature_ontology_runtime_job_start(payload: FeatureOntologyRuntimeRequest = Body(default_factory=FeatureOntologyRuntimeRequest)) -> dict[str, object]:
    selected_product = str(payload.product or "").strip()
    query = str(payload.query or "").strip()
    feature_id = str(payload.feature_id or "").strip()
    conversation_profile = _build_conversation_profile(payload)
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "queued",
        "input": {
            "product": selected_product,
            "query": query,
            "feature_id": feature_id,
            "session_id": str(conversation_profile.get("session_id") or ""),
            "turn_id": str(conversation_profile.get("turn_id") or ""),
        },
        "stages": _create_workbench_stage_statuses(),
        "logs": [_make_job_log("runtime", f"질문을 접수했습니다: {query or '기본 질의'}")],
        "active_stage": None,
        "result": None,
        "error": "",
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
        "completed_at": None,
        "started_monotonic": time.perf_counter(),
        "elapsed_ms": 0,
        "stage_timers": {},
    }
    _store_workbench_job(job_id, job)
    thread = threading.Thread(
        target=_run_workbench_job,
        args=(job_id, selected_product, query, feature_id, conversation_profile),
        daemon=True,
    )
    thread.start()
    return _snapshot_workbench_job(job_id)


@app.get("/feature-ontology/runtime-jobs/{job_id}")
def feature_ontology_runtime_job_status(job_id: str) -> dict[str, object]:
    return _snapshot_workbench_job(job_id)


@app.get("/feature-ontology/clusters")
def feature_ontology_clusters(
    product: str = Query(default=""),
    limit: int = Query(default=12, ge=1, le=100),
) -> dict:
    return _build_customer_cluster_api_payload(
        selected_product=str(product or "").strip(),
        limit=int(limit),
        force_rebuild=False,
    )


@app.post("/feature-ontology/clusters/rebuild")
def feature_ontology_clusters_rebuild(payload: dict[str, object] = Body(default_factory=dict)) -> dict:
    selected_product = str(payload.get("product") or "").strip()
    limit = int(payload.get("limit") or 12)
    return _build_customer_cluster_api_payload(
        selected_product=selected_product,
        limit=limit,
        force_rebuild=True,
    )


@app.get("/feature-ontology/segment-metric-cube")
def feature_ontology_segment_metric_cube(
    force_rebuild: bool = Query(default=False),
) -> dict[str, object]:
    return _build_segment_metric_cube_api_payload(force_rebuild=bool(force_rebuild))


@app.get("/feature-ontology/semantic-refresh-status")
def feature_ontology_semantic_refresh_status() -> dict[str, object]:
    return {
        "status": "ok",
        "refresh": _semantic_refresh_snapshot(),
    }


@app.post("/feature-ontology/semantic-refresh")
def feature_ontology_semantic_refresh() -> dict[str, object]:
    return {
        "status": "ok",
        "refresh": _run_semantic_refresh_once("manual"),
    }


@app.post("/feature-ontology/product-development/agendas")
def feature_ontology_product_development_agendas(payload: dict[str, object] = Body(default_factory=dict)) -> dict[str, object]:
    concepts = _normalize_department_concepts(payload.get("department_concepts") or payload.get("concepts") or {})
    return _generate_product_development_agendas(concepts)


def _update_product_debate_job(job_id: str, **updates: object) -> None:
    with PRODUCT_DEBATE_JOB_STATUS_LOCK:
        current = dict(PRODUCT_DEBATE_JOB_STATUS.get(job_id) or {})
        current.update(updates)
        current["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        PRODUCT_DEBATE_JOB_STATUS[job_id] = current


@app.post("/feature-ontology/product-development/debate-jobs")
def feature_ontology_product_development_debate_job_start(payload: dict[str, object] = Body(default_factory=dict)) -> dict[str, object]:
    selected_agenda = payload.get("selected_agenda") or payload.get("agenda") or {}
    if not isinstance(selected_agenda, dict):
        raise HTTPException(status_code=400, detail="selected_agenda is required")
    concepts = _normalize_department_concepts(payload.get("department_concepts") or payload.get("concepts") or {})
    job_id = f"pd-{uuid.uuid4().hex[:12]}"
    now_iso = datetime.datetime.now().isoformat(timespec="seconds")
    with PRODUCT_DEBATE_JOB_STATUS_LOCK:
        PRODUCT_DEBATE_JOB_STATUS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "stage": "queued",
            "detail": "토론 작업을 큐에 등록했습니다.",
            "created_at": now_iso,
            "updated_at": now_iso,
            "result": None,
            "error": "",
        }

    def _runner() -> None:
        def _progress(stage: str, detail: str) -> None:
            _update_product_debate_job(job_id, stage=stage, detail=str(detail or ""))

        try:
            result = _generate_product_development_debate(
                selected_agenda,
                concepts,
                require_autogen=True,
                progress_callback=_progress,
            )
            _update_product_debate_job(
                job_id,
                status="completed",
                stage="completed",
                detail="토론이 완료되었습니다.",
                result=result,
                error="",
            )
        except Exception as error:  # noqa: BLE001
            _update_product_debate_job(
                job_id,
                status="failed",
                stage="failed",
                detail="토론 생성 중 오류가 발생했습니다.",
                error=str(error),
            )

    threading.Thread(target=_runner, daemon=True).start()
    return {"status": "ok", "job_id": job_id}


@app.get("/feature-ontology/product-development/debate-jobs/{job_id}")
def feature_ontology_product_development_debate_job_status(job_id: str) -> dict[str, object]:
    with PRODUCT_DEBATE_JOB_STATUS_LOCK:
        payload = PRODUCT_DEBATE_JOB_STATUS.get(job_id)
        if not payload:
            raise HTTPException(status_code=404, detail="debate job not found")
        return {"status": "ok", **payload}


@app.post("/feature-ontology/product-development/debate")
def feature_ontology_product_development_debate(payload: dict[str, object] = Body(default_factory=dict)) -> dict[str, object]:
    selected_agenda = payload.get("selected_agenda") or payload.get("agenda") or {}
    if not isinstance(selected_agenda, dict):
        raise HTTPException(status_code=400, detail="selected_agenda is required")
    concepts = _normalize_department_concepts(payload.get("department_concepts") or payload.get("concepts") or {})
    return _generate_product_development_debate(selected_agenda, concepts, require_autogen=True)


@app.post("/feature-ontology/ollama")
def feature_ontology_ollama(payload: dict[str, object] = Body(default_factory=dict)) -> dict:
    selected_product = str(payload.get("product") or "").strip()
    query = str(payload.get("query") or "").strip()
    feature_id = str(payload.get("feature_id") or "").strip()
    workbench = _build_feature_workbench_payload(
        selected_product=selected_product,
        query=query,
        feature_id=feature_id,
    )
    prompt_pack = dict(workbench.get("ollama_ready") or {})
    prompt = _build_ollama_prompt_text(prompt_pack)
    if not prompt:
        raise HTTPException(status_code=400, detail="Ollama 프롬프트를 생성할 수 없습니다.")
    ollama_runtime = dict(workbench.get("ollama_runtime") or {})
    output = dict(ollama_runtime.get("output") or {})
    status = str(ollama_runtime.get("status") or "skipped")
    return {
        "status": "ok" if status == "completed" else status,
        "model": str(ollama_runtime.get("model") or OLLAMA_LIGHTWEIGHT_MODEL),
        "prompt": prompt,
        "response_text": str(output.get("response_text") or ""),
        "detail": str(ollama_runtime.get("error") or ""),
        "workbench": workbench,
    }


@app.post("/settings/ollama-gpu")
def settings_ollama_gpu(payload: OllamaRuntimeToggleRequest) -> dict:
    preferences = set_ollama_gpu_enabled(bool(payload.enabled))
    with state.lock:
        state.ollama_gpu_enabled = bool(preferences.get("ollama_gpu_enabled"))
    return {"status": "ok", **preferences}


@app.post("/settings/ontology-query-priority")
def settings_ontology_query_priority(payload: OllamaRuntimeToggleRequest) -> dict:
    preferences = set_ontology_query_priority_enabled(bool(payload.enabled))
    with state.lock:
        state.ontology_query_priority_enabled = bool(preferences.get("ontology_query_priority_enabled"))
    return {"status": "ok", **preferences}


@app.post("/ontology/save")
def ontology_save(payload: OntologySaveRequest) -> dict:
    ontology = dict(payload.ontology or {})
    commonfeature = _rebuild_commonfeature_from_ontology(ontology, dict(payload.commonfeature or {}))

    ontology["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    ontology["source"] = {
        **(ontology.get("source") or {}),
        "commonfeature_path": str(COMMONFEATURE_PATH.relative_to(ROOT)),
    }

    ONTOLOGY_PATH.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8")
    COMMONFEATURE_PATH.write_text(json.dumps(commonfeature, ensure_ascii=False, indent=2), encoding="utf-8")
    _invalidate_feature_embedding_cache()

    return {
        "status": "ok",
        "detail": "ontology/commonfeature 저장 완료",
        "ontology": ontology,
        "commonfeature": commonfeature,
    }


def _rebuild_commonfeature_from_ontology(ontology: dict, commonfeature: dict) -> dict:
    feature_seed = {
        str(item.get("feature_id") or ""): item
        for item in (commonfeature.get("common_features") or [])
        if str(item.get("feature_id") or "").strip()
    }

    grouped: dict[str, dict] = {}
    products = ontology.get("products") or {}
    for product_code, product_payload in products.items():
        product_name = str(product_payload.get("product_name") or product_code)
        for section_name, direction in (("input_fields", "input"), ("output_fields", "output")):
            for field_code, mapping in (product_payload.get(section_name) or {}).items():
                feature_id = str((mapping or {}).get("feature_id") or "").strip()
                if not feature_id:
                    continue

                seed = feature_seed.get(feature_id, {})
                entry = grouped.setdefault(
                    feature_id,
                    {
                        "feature_id": feature_id,
                        "feature_name": str((mapping or {}).get("feature_name") or seed.get("feature_name") or feature_id),
                        "category": str((mapping or {}).get("category") or seed.get("category") or "unclassified"),
                        "description": str(seed.get("description") or ""),
                        "directions": set(seed.get("directions") or []),
                        "aliases": set(seed.get("aliases") or []),
                        "products": set(seed.get("products") or []),
                        "field_mappings": [],
                        "sample_values": {},
                    },
                )
                entry["feature_name"] = str((mapping or {}).get("feature_name") or entry["feature_name"])
                entry["category"] = str((mapping or {}).get("category") or entry["category"])
                entry["directions"].add(direction)
                entry["products"].add(product_code)
                label = str((mapping or {}).get("label") or "").strip()
                if label:
                    entry["aliases"].add(label)
                entry["field_mappings"].append(
                    {
                        "product": product_code,
                        "product_name": product_name,
                        "direction": direction,
                        "field_code": field_code,
                        "label": label,
                        "observed_count": int((mapping or {}).get("observed_count") or 0),
                    }
                )
                for sample in (mapping or {}).get("sample_values") or []:
                    value = str((sample or {}).get("value") or "").strip()
                    if not value:
                        continue
                    entry["sample_values"][value] = max(
                        int(entry["sample_values"].get(value) or 0),
                        int((sample or {}).get("count") or 0),
                    )

        reject_meta = product_payload.get("reject_reason") or {}
        reject_feature_id = str(reject_meta.get("feature_id") or "decision.reject_reason_code")
        reject_entry = grouped.setdefault(
            reject_feature_id,
            {
                "feature_id": reject_feature_id,
                "feature_name": str(reject_meta.get("feature_name") or "거절사유코드"),
                "category": "decision",
                "description": str(reject_meta.get("description") or "K코드 기반 거절사유 taxonomy"),
                "directions": {"reject"},
                "aliases": {"거절사유코드", "K코드"},
                "products": set(),
                "field_mappings": [],
                "sample_values": {},
            },
        )
        reject_entry["products"].add(product_code)

    common_features = []
    fallback_count = 0
    for feature_id, item in grouped.items():
        sample_values = [
            {"value": value, "count": count}
            for value, count in sorted(item["sample_values"].items(), key=lambda kv: (-int(kv[1]), kv[0]))[:10]
        ]
        payload = {
            "feature_id": feature_id,
            "feature_name": item["feature_name"],
            "category": item["category"],
            "description": item["description"],
            "directions": sorted(item["directions"]),
            "aliases": sorted(alias for alias in item["aliases"] if str(alias).strip()),
            "products": sorted(item["products"]),
            "coverage": {
                "product_count": len(item["products"]),
                "mapping_count": len(item["field_mappings"]),
            },
            "field_mappings": sorted(
                item["field_mappings"],
                key=lambda row: (row.get("product") or "", row.get("direction") or "", row.get("field_code") or ""),
            ),
            "sample_values": sample_values,
        }
        if payload["category"] == "unclassified":
            fallback_count += 1
        common_features.append(payload)

    common_features.sort(key=lambda row: (row.get("category") == "unclassified", row.get("feature_id") or ""))
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": {
            "mapping_snapshot_path": str((ROOT / "data" / "product_mapping_snapshot.json").relative_to(ROOT)),
            "ontology_path": str(ONTOLOGY_PATH.relative_to(ROOT)),
            "reject_taxonomy_code_count": len(((ontology.get("reject_reason_taxonomy") or {}).get("codes") or {})),
        },
        "statistics": {
            "product_count": len(products),
            "common_feature_count": len(common_features),
            "fallback_feature_count": fallback_count,
            "classified_feature_count": len(common_features) - fallback_count,
        },
        "common_features": common_features,
    }


@app.on_event("startup")
def app_startup() -> None:
    hydrate_state_from_existing_artifacts()
    persisted_regulation = _load_persisted_regulation_state()
    if persisted_regulation:
        with state.lock:
            state.regulation_upload_summary_enabled = bool(
                persisted_regulation.get("regulation_upload_summary_enabled", False)
            )
            summary = str(persisted_regulation.get("summary") or "").strip()
            updated_at = str(persisted_regulation.get("updated_at") or "").strip()
            detail = str(persisted_regulation.get("detail") or "").strip()
            persisted_files = persisted_regulation.get("files") or []
            state.regulation_files = [
                str(item).strip()
                for item in persisted_files
                if str(item).strip()
            ]
            persisted_file_stats = persisted_regulation.get("file_stats") or []
            state.regulation_file_stats = [
                item
                for item in persisted_file_stats
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ]
            if summary:
                state.latest_regulation_analysis = summary
            if updated_at:
                try:
                    state.last_regulation_analysis_time = datetime.datetime.fromisoformat(updated_at)
                except Exception:
                    state.last_regulation_analysis_time = None
            statuses = dict(state.agent_statuses or {})
            statuses["regulation_agent"] = {
                "status": "completed",
                "detail": detail or "규제 문서 학습 내역이 저장되어 복원되었습니다.",
                "updated_at": updated_at or _iso_now(),
            }
            state.agent_statuses = statuses
    if AUTO_START_WORKER:
        worker.update_interval(WORKER_START_INTERVAL_SECONDS)
        worker.start()
        record_activity_event(
            "startup_sequence",
            "running",
            f"Background worker auto-start enabled. interval={worker.interval_seconds}s",
            update_status=True,
        )
    else:
        record_activity_event(
            "startup_sequence",
            "completed",
            "Background worker auto-start skipped. Use /worker/start or AUTO_START_WORKER=1 when needed.",
            update_status=True,
        )

    if AUTO_START_SEMANTIC_REFRESH:
        _start_semantic_refresh_scheduler()
    else:
        _update_semantic_refresh_status(
            enabled=False,
            status="idle",
            message="자동 갱신 비활성화 상태입니다. /feature-ontology/semantic-refresh 수동 실행은 가능합니다.",
            next_run_at="",
        )
        record_activity_event(
            "startup_sequence",
            "completed",
            "Semantic refresh scheduler auto-start skipped.",
            update_status=True,
        )


@app.on_event("shutdown")
def app_shutdown() -> None:
    _stop_semantic_refresh_scheduler()


def build_ws_snapshot(snapshot: dict) -> dict:
    return {
        "news": snapshot.get("news", []),
        "recent_news_fallback": snapshot.get("recent_news_fallback", []),
        "issues": snapshot.get("issues", []),
        "vector_count": snapshot.get("vector_count"),
        "vector_events": snapshot.get("vector_events", []),
        "agent_activity_log": snapshot.get("agent_activity_log", []),
        "agent_statuses": snapshot.get("agent_statuses", {}),
        "ollama_runtime": snapshot.get("ollama_runtime", {}),
        "cardloan_debate": snapshot.get("cardloan_debate", {}),
        "latest_news_prompt_input": snapshot.get("latest_news_prompt_input", {}),
        "last_news_prompt_input_time": snapshot.get("last_news_prompt_input_time"),
        "latest_log_prompt_input": snapshot.get("latest_log_prompt_input", {}),
        "last_log_prompt_input_time": snapshot.get("last_log_prompt_input_time"),
        "log_prompt_template_override": snapshot.get("log_prompt_template_override"),
        "latest_regulation_analysis": snapshot.get("latest_regulation_analysis"),
        "last_regulation_analysis_time": snapshot.get("last_regulation_analysis_time"),
        "latest_news_briefing": snapshot.get("latest_news_briefing"),
        "last_news_briefing_time": snapshot.get("last_news_briefing_time"),
        "latest_log_briefing": snapshot.get("latest_log_briefing"),
        "last_log_briefing_time": snapshot.get("last_log_briefing_time"),
        "last_news_time": snapshot.get("last_news_time"),
        "last_new_item_time": snapshot.get("last_new_item_time"),
        "news_crawl_running": snapshot.get("news_crawl_running"),
        "news_crawl_target_count": snapshot.get("news_crawl_target_count"),
        "news_crawl_success_count": snapshot.get("news_crawl_success_count"),
        "news_crawl_failure_count": snapshot.get("news_crawl_failure_count"),
        "last_news_crawl_time": snapshot.get("last_news_crawl_time"),
        "last_news_crawl_error": snapshot.get("last_news_crawl_error"),
        "news_pipeline_stats": snapshot.get("news_pipeline_stats", {}),
        "last_faiss_time": snapshot.get("last_faiss_time"),
        "backend_diagnostics": snapshot.get("backend_diagnostics", {}),
    }


def _is_expected_websocket_disconnect(error: Exception) -> bool:
    if isinstance(error, WebSocketDisconnect):
        return True
    if ConnectionClosed and isinstance(error, ConnectionClosed):
        return True
    if isinstance(error, OSError) and getattr(error, "winerror", None) in {64, 10054}:
        return True

    message = str(error).lower()
    return (
        "websocket is not connected" in message
        or "closing handshake" in message
        or "connection closed" in message
    )


@app.get("/health")
def health() -> dict:
    # 화면에서 가장 먼저 확인하는 상태 API입니다.
    # 서버가 살아있는지, 워커가 도는지, 최근 분석 상태가 어떤지 전달합니다.
    hydrate_state_from_existing_artifacts()
    snapshot = state.snapshot(include_faiss_items=False, include_vector_count=False)
    snapshot["backend_diagnostics"] = build_backend_diagnostics(
        worker_running=worker.running,
        worker_interval_seconds=worker.interval_seconds,
    )
    snapshot["background_startup"] = {
        "auto_start_worker": AUTO_START_WORKER,
        "worker_start_interval_seconds": WORKER_START_INTERVAL_SECONDS,
        "auto_start_semantic_refresh": AUTO_START_SEMANTIC_REFRESH,
        "semantic_refresh_interval_seconds": SEMANTIC_REFRESH_INTERVAL_SECONDS,
        "health_check_ollama": HEALTH_CHECK_OLLAMA,
    }
    snapshot["ollama_health"] = (
        _probe_ollama_health()
        if HEALTH_CHECK_OLLAMA
        else {
            "status": "skipped",
            "detail": "Set HEALTH_CHECK_OLLAMA=1 or call /health/ollama for a live Ollama probe.",
        }
    )
    return {"status": "ok", "worker_running": worker.running, **snapshot}


@app.get("/health/ollama")
def health_ollama() -> dict[str, object]:
    return _probe_ollama_health()


@app.post("/news/collect", response_model=NewsCollectResponse)
def news_collect() -> NewsCollectResponse:
    news, issues = collect_news_bundle(accumulate=True)
    snapshot = state.snapshot(include_faiss_items=False)
    return NewsCollectResponse(
        news=news,
        issues=issues,
        count=len(news),
        last_new_item_time=snapshot.get("last_new_item_time"),
    )


@app.post("/chat/openai")
def chat_openai_proxy(payload: dict[str, object] = Body(default_factory=dict)) -> dict[str, object]:
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    normalized_messages: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip() or "user"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        normalized_messages.append({"role": role, "content": content})

    if not normalized_messages:
        raise HTTPException(status_code=400, detail="messages has no valid items")

    model = str(payload.get("model") or "gpt-4o-mini")
    temperature = float(payload.get("temperature") or 0.7)
    max_tokens = int(payload.get("max_tokens") or 500)
    completion = _openai_chat_completion(
        normalized_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "status": "ok",
        "model": model,
        "content": str((((completion.get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or "").strip(),
        "raw": completion,
    }


@app.post("/logs/analyze", response_model=LogAnalyzeResponse)
def logs_analyze(payload: LogAnalyzeRequest) -> LogAnalyzeResponse:
    results, file_count = analyze_logs_bundle(
        raw_logs=payload.raw_logs, log_dir=payload.log_dir
    )
    return LogAnalyzeResponse(file_count=file_count, results=enrich_results(results))


@app.post("/faiss/build")
def faiss_build(payload: FaissBuildRequest) -> dict:
    try:
        count = build_faiss_bundle(logs=payload.logs, news=payload.news)
        return {"status": "ok", "vector_count": count}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/faiss/search")
def faiss_search(payload: SearchRequest) -> dict:
    try:
        return search_faiss(payload.query, payload.k, payload.store_name)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/chat/strategy", response_model=StrategyChatResponse)
def chat_strategy(payload: StrategyChatRequest) -> StrategyChatResponse:
    try:
        return StrategyChatResponse(
            **ask_strategy(
                payload.question,
                news_prompt_template=payload.news_prompt_template,
                log_prompt_template=payload.log_prompt_template,
            )
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/chat/cardloan-debate", response_model=CardloanDebateResponse)
def chat_cardloan_debate(payload: CardloanDebateRequest) -> CardloanDebateResponse:
    try:
        return CardloanDebateResponse(
            **ask_cardloan_debate(
                payload.question,
                reviewer_prompts=payload.reviewer_prompts,
                reviewer_settings=payload.reviewer_settings,
            )
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/settings/log-prompt")
def settings_log_prompt(payload: LogPromptTemplateRequest) -> dict:
    from agent.strategy_chat import DEFAULT_LOG_AGENT_PROMPT_TEMPLATE

    template = str(payload.template or "").strip() or None
    with state.lock:
        state.log_prompt_template_override = (
            None
            if template in {None, DEFAULT_LOG_AGENT_PROMPT_TEMPLATE}
            else template
        )
    return {
        "status": "ok",
        "log_prompt_template_override": state.log_prompt_template_override,
    }


@app.post("/settings/news-prompt")
def settings_news_prompt(payload: NewsPromptTemplateRequest) -> dict:
    from agent.strategy_chat import DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE

    template = str(payload.template or "").strip() or None
    with state.lock:
        state.news_prompt_template_override = (
            None
            if template in {None, DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE}
            else template
        )
    return {
        "status": "ok",
        "news_prompt_template_override": state.news_prompt_template_override,
    }


@app.post("/settings/news-agent-ollama")
def settings_news_agent_ollama(payload: AgentOllamaToggleRequest) -> dict:
    with state.lock:
        state.news_agent_ollama_enabled = bool(payload.enabled)
    return {
        "status": "ok",
        "news_agent_ollama_enabled": bool(state.news_agent_ollama_enabled),
    }


@app.post("/settings/log-agent-ollama")
def settings_log_agent_ollama(payload: AgentOllamaToggleRequest) -> dict:
    with state.lock:
        state.log_agent_ollama_enabled = bool(payload.enabled)
    return {
        "status": "ok",
        "log_agent_ollama_enabled": bool(state.log_agent_ollama_enabled),
    }


@app.post("/settings/regulation-upload-summary")
def settings_regulation_upload_summary(payload: AgentOllamaToggleRequest) -> dict:
    enabled = bool(payload.enabled)
    with state.lock:
        state.regulation_upload_summary_enabled = enabled
    _persist_regulation_summary_toggle(enabled)
    return {
        "status": "ok",
        "regulation_upload_summary_enabled": bool(state.regulation_upload_summary_enabled),
    }


@app.post("/analysis/run", response_model=FullAnalysisResponse)
def analysis_run(
    log_dir: str = "data/logs",
    collect_news: bool = True,
) -> FullAnalysisResponse:
    try:
        snapshot = run_full_analysis(log_dir=log_dir, collect_news=collect_news)
        return FullAnalysisResponse(**snapshot)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/analysis/status", response_model=FullAnalysisResponse)
def analysis_status() -> FullAnalysisResponse:
    hydrate_state_from_existing_artifacts()
    snapshot = state.snapshot(include_faiss_items=False)
    snapshot["results"] = enrich_results(snapshot["results"])
    snapshot["backend_diagnostics"] = build_backend_diagnostics(
        worker_running=worker.running,
        worker_interval_seconds=worker.interval_seconds,
    )
    return FullAnalysisResponse(**snapshot)


@app.get("/diagnostics/status")
def diagnostics_status() -> dict:
    hydrate_state_from_existing_artifacts()
    snapshot = state.snapshot(include_faiss_items=False)
    diagnostics = build_backend_diagnostics(
        worker_running=worker.running,
        worker_interval_seconds=worker.interval_seconds,
    )
    return {
        "status": "ok",
        "diagnostics": diagnostics,
        "last_run_time": snapshot.get("last_run_time"),
        "last_faiss_time": snapshot.get("last_faiss_time"),
        "vector_count": snapshot.get("vector_count"),
    }


@app.get("/charts")
def charts_all() -> dict:
    # 메인 대시보드의 4개 차트가 한 번에 가져가는 스냅샷 API입니다.
    snapshot = state.snapshot(include_faiss_items=False)
    return {
        "status": "ok",
        "last_chart_time": snapshot.get("last_chart_time"),
        "charts": get_chart_payloads(),
    }


@app.get("/charts/{chart_name}")
def charts_one(chart_name: str) -> dict:
    payloads = get_chart_payloads()
    if chart_name not in payloads:
        raise HTTPException(status_code=404, detail=f"unknown chart: {chart_name}")
    snapshot = state.snapshot(include_faiss_items=False)
    return {
        "status": "ok",
        "last_chart_time": snapshot.get("last_chart_time"),
        "chart_name": chart_name,
        "data": payloads[chart_name],
    }


@app.get("/faiss/entries")
def faiss_entries(limit: int = 200, store_name: str | None = None, type: str | None = None) -> dict:
    try:
        from rag.vector_db import get_vector_count, list_vectors

        items = list_vectors(limit=limit, store_name=store_name)
        if type:
            normalized_type = str(type).strip().lower()
            items = [item for item in items if str(item.get("type") or "").strip().lower() == normalized_type]
        return {
            "status": "ok",
            "count": len(items),
            "total_count": get_vector_count(store_name),
            "store_name": store_name,
            "type": type,
            "items": items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/entry")
def faiss_entry(doc_id: str) -> dict:
    try:
        from rag.vector_db import get_vector_by_id

        item = get_vector_by_id(doc_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"unknown vector id: {doc_id}")
        return {
            "status": "ok",
            "id": item.get("id"),
            "metadata": item.get("metadata") or {},
            "page_content": item.get("page_content") or "",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/export")
def faiss_export(format: str = "json", limit: int = 200, store_name: str | None = None):
    try:
        from rag.vector_db import list_vectors
        import json, csv, io

        items = list_vectors(limit=limit, store_name=store_name)
        if format == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["id", "store", "type", "product", "agent", "source", "name", "snippet"])
            writer.writeheader()
            for row in items:
                writer.writerow(row)
            return Response(content=output.getvalue(), media_type="text/csv")
        else:
            return {"status": "ok", "count": len(items), "store_name": store_name, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/entry/{doc_id}")
def faiss_entry(doc_id: str):
    try:
        from rag.vector_db import get_vector_by_id

        item = get_vector_by_id(doc_id)
        if item is None:
            raise HTTPException(status_code=404, detail="entry not found")
        return {"status": "ok", "item": item}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/news/evidence-detail")
def news_evidence_detail(
    link: str | None = None,
    title: str | None = None,
    summary: str | None = None,
) -> dict:
    try:
        from rag.vector_db import get_vector_by_id, list_vectors

        normalized_link = str(link or "").strip().lower()
        normalized_title = str(title or "").strip().lower()
        normalized_summary = str(summary or "").strip()

        candidates = list_vectors(limit=1600, store_name="news")
        best_item: dict[str, Any] | None = None
        best_score = -1.0

        for candidate in candidates:
            doc_id = str(candidate.get("id") or "").strip()
            if not doc_id:
                continue
            full_item = get_vector_by_id(doc_id)
            if not full_item:
                continue
            metadata = dict(full_item.get("metadata") or {})
            meta_link = str(metadata.get("link") or "").strip().lower()
            meta_title = str(metadata.get("title") or metadata.get("name") or "").strip().lower()
            page_content = str(full_item.get("page_content") or "")
            doc_type = str(metadata.get("type") or full_item.get("type") or "").strip().lower()
            if doc_type not in {"news", "signal_news", "generated_news"}:
                continue

            score = 0.0
            if normalized_link and meta_link and normalized_link == meta_link:
                score += 1.0
            if normalized_title and meta_title:
                if normalized_title == meta_title:
                    score += 0.9
                elif normalized_title in meta_title or meta_title in normalized_title:
                    score += 0.6
            if normalized_title and normalized_title in page_content.lower():
                score += 0.3
            if score > best_score:
                best_score = score
                best_item = full_item

        metadata = dict((best_item or {}).get("metadata") or {})
        snippet = str((best_item or {}).get("page_content") or "")
        resolved_summary = (
            str(metadata.get("summary") or "").strip()
            or normalized_summary
            or str(metadata.get("evidence_sentence") or "").strip()
        )
        evidence_sentences = metadata.get("evidence_sentences")
        if not isinstance(evidence_sentences, list) or not evidence_sentences:
            single = str(metadata.get("evidence_sentence") or "").strip()
            evidence_sentences = [single] if single else []

        scored_metadata = {
            "rule_score": float(metadata.get("rule_score") or metadata.get("low_cost_score") or 0.0),
            "embed_score": float(metadata.get("embed_score") or metadata.get("embed_relevance") or 0.0),
            "importance_score": float(metadata.get("importance_score") or 0.0),
            "impact_cardloan": str(metadata.get("impact_cardloan") or ""),
            "impact_product": str(metadata.get("impact_product") or ""),
            "sentiment": str(metadata.get("sentiment") or metadata.get("impact_direction") or "neutral"),
            "evidence_sentences": evidence_sentences,
            "source": str(metadata.get("source") or metadata.get("publisher") or ""),
            "published_at": str(metadata.get("published_at") or ""),
            "link": str(metadata.get("link") or ""),
            "title": str(metadata.get("title") or ""),
            "chunk_id": str(metadata.get("chunk_id") or ""),
        }
        return {
            "status": "ok",
            "matched": bool(best_item),
            "summary": resolved_summary,
            "metadata": scored_metadata,
            "snippet": snippet[:2000],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/stats")
def faiss_stats() -> dict:
    """Compute per-product aggregates from FAISS stored vectors.

    Returns average applied rate, average available amount, approval rate,
    common rejection reasons, and average credit grades per product code.
    """
    try:
        from rag.vector_db import FAISS_STORE_LOGS, list_vectors

        snapshot = state.snapshot()
        latest_event = (snapshot.get("vector_events") or [{}])[0]
        cache_version = (
            latest_event.get("timestamp"),
            snapshot.get("vector_count"),
        )
        now = time.time()

        if (
            _faiss_stats_cache.get("data") is not None
            and _faiss_stats_cache.get("version") == cache_version
            and (now - float(_faiss_stats_cache.get("cached_at") or 0.0)) < 30
        ):
            return {
                "status": "ok",
                "products": _faiss_stats_cache.get("data"),
                "cached": True,
            }

        items = list_vectors(limit=10000, store_name=FAISS_STORE_LOGS)
        per_prod: dict[str, list[dict]] = {}
        for it in items:
            prod = it.get("product") or "UNKNOWN"
            per_prod.setdefault(prod, []).append(it)

        stats: dict = {}
        for prod, product_items in per_prod.items():
            cnt = 0
            sum_rate = 0.0
            rate_count = 0
            sum_limit = 0.0
            limit_count = 0
            approvals = 0
            approval_count = 0
            reject_reasons: dict[str, int] = {}
            kcb_scores = []
            nice_scores = []

            for item in product_items:
                features = item.get("features", {}) or {}
                meta = {
                    "in_fields": item.get("in_fields") or {},
                    "out_fields": item.get("out_fields") or {},
                }

                # applied_rate
                ar = None
                try:
                    ar = features.get("applied_rate") if isinstance(features, dict) else None
                except Exception:
                    ar = None
                if ar is not None:
                    try:
                        sum_rate += float(ar)
                        rate_count += 1
                    except Exception:
                        pass

                # available_amount
                aa = None
                try:
                    aa = features.get("available_amount") if isinstance(features, dict) else None
                except Exception:
                    aa = None
                if aa is not None:
                    try:
                        sum_limit += float(aa)
                        limit_count += 1
                    except Exception:
                        pass

                # approval detection: look into metadata in_fields/out_fields
                approved = None
                for field_container in (meta.get("out_fields") or {}, meta.get("in_fields") or {}, features or {}):
                    if not isinstance(field_container, dict):
                        continue
                    for k, v in field_container.items():
                        sval = str(v).lower() if v is not None else ""
                        if any(tok in sval for tok in ("승인", "승", "approve", "approved", "ok", " 승인 ")):
                            approved = True
                            break
                        if any(tok in sval for tok in ("거절", "불가", "reject", "denied", "불허")):
                            approved = False
                            # capture reason nearby
                            if len(sval) > 2:
                                reject_reasons[sval[:200]] = reject_reasons.get(sval[:200], 0) + 1
                            break
                    if approved is not None:
                        break

                if approved is True:
                    approvals += 1
                if approved is not None:
                    approval_count += 1

                # credit scores
                try:
                    cs = features.get("credit_score")
                    if cs is not None:
                        nice_scores.append(float(cs))
                except Exception:
                    pass
                try:
                    cg = features.get("credit_grade")
                    if cg is not None:
                        # map grades A,B,C -> numeric fallback
                        if isinstance(cg, str) and cg.isalpha():
                            # A=4,B=3,C=2,D=1,S=5
                            mapping = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                            val = mapping.get(cg.upper())
                            if val is not None:
                                kcb_scores.append(val)
                        else:
                            try:
                                kcb_scores.append(float(cg))
                            except Exception:
                                pass
                except Exception:
                    pass

                cnt += 1

            stats[prod] = {
                "count": cnt,
                "avg_applied_rate": (sum_rate / rate_count) if rate_count else None,
                "avg_available_amount": (sum_limit / limit_count) if limit_count else None,
                "approval_rate": (approvals / approval_count) if approval_count else None,
                "top_reject_reasons": sorted(reject_reasons.items(), key=lambda x: -x[1])[:5],
                "avg_kcb_grade": (sum(kcb_scores) / len(kcb_scores)) if kcb_scores else None,
                "avg_credit_score": (sum(nice_scores) / len(nice_scores)) if nice_scores else None,
            }

        _faiss_stats_cache["version"] = cache_version
        _faiss_stats_cache["data"] = stats
        _faiss_stats_cache["cached_at"] = now

        return {"status": "ok", "products": stats, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/search_features")
def faiss_search_features(type: str | None = None, feature_key: str | None = None, feature_value: str | None = None, limit: int = 200, store_name: str | None = None) -> dict:
    """Search FAISS entries by metadata `type` and feature key/value.

    - `type`: optional metadata type filter (e.g., 'log', 'news', 'signal_news')
    - `feature_key`: feature field name to match (only relevant for items with `features` metadata)
    - `feature_value`: substring to match against the feature value (optional)
    """
    try:
        from rag.vector_db import list_vectors, get_vector_by_id

        items = list_vectors(limit=10000, store_name=store_name)
        results: list[dict] = []
        count = 0
        for it in items:
            if type and (it.get("type") or "") != type:
                continue
            doc_id = it.get("id")
            if not doc_id:
                continue
            doc = get_vector_by_id(str(doc_id)) or {}
            meta = doc.get("metadata", {}) or {}
            features = meta.get("features", {}) or {}

            # If feature_key provided, require it exists
            if feature_key:
                if not isinstance(features, dict) or feature_key not in features:
                    continue
                if feature_value:
                    fv = features.get(feature_key)
                    if fv is None:
                        continue
                    # match substring (case-insensitive)
                    if str(feature_value).lower() not in str(fv).lower():
                        continue

            results.append({
                "id": str(doc_id),
                "type": it.get("type"),
                "product": it.get("product"),
                "snippet": it.get("snippet"),
                "features": features,
            })
            count += 1
            if count >= limit:
                break

        return {"status": "ok", "count": len(results), "store_name": store_name, "items": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/faiss/similar_logs")
def faiss_similar_logs(query: str, limit: int = 8) -> dict:
    try:
        from rag.vector_db import search_similar_log_items

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {
                "status": "ok",
                "query": "",
                "count": 0,
                "store_name": "logs",
                "items": [],
            }

        items = search_similar_log_items(normalized_query, k=max(1, int(limit or 8)))
        return {
            "status": "ok",
            "query": normalized_query,
            "count": len(items),
            "store_name": "logs",
            "items": items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/product-pattern-summary", response_model=ProductSummaryResponse)
def product_pattern_summary() -> ProductSummaryResponse:
    payload = load_product_pattern_summary(DEFAULT_SUMMARY_PATH)
    if not isinstance(payload, dict):
        payload = {}
    return ProductSummaryResponse(status="ok", payload=payload)


@app.post("/regulation/upload", response_model=RegulationUploadResponse)
async def regulation_upload(files: list[UploadFile] = File(...)) -> RegulationUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    with state.lock:
        statuses = dict(state.agent_statuses or {})
        statuses["regulation_agent"] = {
            "status": "running",
            "detail": "규제 문서 분석 실행 중...",
            "updated_at": started_at,
        }
        state.agent_statuses = statuses
    record_activity_event(
        "regulation_agent",
        "running",
        f"규제 문서 업로드 시작: {len(files)}건",
        update_status=True,
    )

    files_data: list[tuple[str, bytes]] = []
    file_names: list[str] = []
    REGULATION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for upload in files:
        raw_bytes = await upload.read()
        file_name = str(upload.filename or "unknown").strip() or "unknown"
        files_data.append((file_name, raw_bytes))
        file_names.append(file_name)
        try:
            safe_name = pathlib.Path(file_name).name
            (REGULATION_UPLOAD_DIR / safe_name).write_bytes(raw_bytes)
        except Exception:
            pass

    try:
        ingest_report = dict(ingest_files_with_report(files_data, doc_type="regulation") or {})
        after_count = int(ingest_report.get("vector_count") or 0)
        added_count = int(ingest_report.get("added_count") or 0)
        file_stats = list(ingest_report.get("file_stats") or [])
        summary = ""
        with state.lock:
            summary_enabled = bool(state.regulation_upload_summary_enabled)
        summary_mode = "disabled"
        if summary_enabled:
            _, _, rules_found = search_context("규제", k=6)
            rule_context = "\n\n".join(rules_found)
            summary_mode = "llm"
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        regulation_agent,
                        rule_context,
                        "",
                        "업로드된 규제 문서 분석 및 요약을 작성하라",
                    )
                    summary = str(future.result(timeout=45) or "").strip()
            except concurrent.futures.TimeoutError:
                summary_mode = "fallback-timeout"
            except Exception:
                summary_mode = "fallback-error"

            if not summary:
                context_preview = [str(item or "").strip() for item in rules_found[:3] if str(item or "").strip()]
                summary = "규제 문서가 업로드되어 벡터 인덱스에 반영되었습니다. "
                if context_preview:
                    summary += "핵심 근거: " + " | ".join(text[:90] for text in context_preview)
                else:
                    summary += "현재 추출 가능한 규제 컨텍스트가 부족해 요약은 축약되었습니다."
        else:
            summary = "규제 문서를 벡터에 적재했습니다. (요약 생성 꺼짐)"

        updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        with state.lock:
            statuses = dict(state.agent_statuses or {})
            statuses["regulation_agent"] = {
                "status": "completed",
                "detail": f"문서 {len(file_names)}건 분석 완료 · 벡터 {added_count}건 추가 · summary={summary_mode}",
                "updated_at": updated_at,
            }
            state.agent_statuses = statuses
            state.latest_regulation_analysis = summary
            state.regulation_files = list(file_names)
            state.regulation_file_stats = list(file_stats)
            state.last_regulation_analysis_time = datetime.datetime.fromisoformat(updated_at)

        _persist_regulation_state(
            {
                "status": "ok",
                "detail": f"문서 {len(file_names)}건 분석 완료 · 벡터 {added_count}건 추가 · summary={summary_mode}",
                "summary": summary,
                "updated_at": updated_at,
                "files": file_names,
                "vector_count": after_count,
                "added_count": added_count,
                "summary_mode": summary_mode,
                "regulation_upload_summary_enabled": summary_enabled,
                "file_stats": file_stats,
            }
        )

        record_vector_event(
            "regulation_agent",
            "upload",
            max(0, int(after_count) - int(added_count)),
            after_count,
            f"규제 문서 {len(file_names)}건 벡터 적재",
        )
        record_activity_event(
            "regulation_agent",
            "completed",
            f"규제 문서 {len(file_names)}건 분석 완료",
            update_status=True,
        )
        return RegulationUploadResponse(
            status="ok",
            detail="규제 문서 분석 완료",
            vector_count=after_count,
            added_count=added_count,
            summary=summary,
            updated_at=updated_at,
            files=file_names,
            file_stats=file_stats,
        )
    except Exception as error:
        failed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        with state.lock:
            statuses = dict(state.agent_statuses or {})
            statuses["regulation_agent"] = {
                "status": "failed",
                "detail": str(error),
                "updated_at": failed_at,
            }
            state.agent_statuses = statuses
        record_activity_event(
            "regulation_agent",
            "failed",
            f"규제 문서 업로드 실패: {error}",
            update_status=True,
        )
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/regulation/files/{file_name:path}")
def regulation_uploaded_file(file_name: str) -> FileResponse:
    target_path = _resolve_regulation_uploaded_file(file_name)
    return FileResponse(
        str(target_path),
        media_type="application/pdf",
        filename=target_path.name,
    )


@app.websocket("/ws/faiss")
async def websocket_faiss_updates(websocket: WebSocket):
    await websocket.accept()
    last_sent = 0
    last_signature = None
    try:
        while True:
            snapshot = state.snapshot(include_faiss_items=False)
            snapshot["backend_diagnostics"] = build_backend_diagnostics(
                worker_running=worker.running,
                worker_interval_seconds=worker.interval_seconds,
            )
            events = snapshot.get("vector_events", []) or []
            to_send = []
            for ev in reversed(events):
                ts = ev.get("timestamp") or ""
                try:
                    tval = int(float(ts.replace(".", ""))) if ts else 0
                except Exception:
                    tval = 0
                if tval > last_sent:
                    to_send.append(ev)
                    last_sent = max(last_sent, tval)

            signature = (
                (events[0].get("timestamp") if events else None),
                (snapshot.get("agent_activity_log", [{}])[0].get("timestamp") if snapshot.get("agent_activity_log") else None),
                snapshot.get("last_news_prompt_input_time"),
                snapshot.get("last_log_prompt_input_time"),
                snapshot.get("last_news_briefing_time"),
                snapshot.get("last_log_briefing_time"),
                snapshot.get("last_news_time"),
                snapshot.get("last_new_item_time"),
                snapshot.get("news_crawl_running"),
                snapshot.get("news_crawl_success_count"),
                snapshot.get("news_crawl_failure_count"),
                snapshot.get("last_news_crawl_time"),
                (snapshot.get("ollama_runtime", {}) or {}).get("updated_at"),
                (snapshot.get("ollama_runtime", {}) or {}).get("status"),
            )

            for ev in to_send:
                try:
                    payload = {
                        "type": "vector_event",
                        "event": ev,
                        "snapshot": build_ws_snapshot(snapshot),
                    }
                except Exception:
                    payload = {"type": "vector_event", "event": ev}
                await websocket.send_json(payload)

            if signature != last_signature:
                last_signature = signature
                await websocket.send_json(
                    {
                        "type": "state_update",
                        "snapshot": build_ws_snapshot(snapshot),
                    }
                )
            await asyncio.sleep(1)
    except Exception as error:
        if _is_expected_websocket_disconnect(error):
            return
        raise


@app.post("/worker/start", response_model=GenericMessage)
def worker_start(payload: WorkerConfigRequest) -> GenericMessage:
    worker.update_interval(payload.interval_seconds)
    started = worker.start()
    detail = f"worker interval={worker.interval_seconds}s"
    return GenericMessage(
        status="started" if started else "already_running", detail=detail
    )


@app.post("/worker/stop", response_model=GenericMessage)
def worker_stop() -> GenericMessage:
    stopped = worker.stop()
    return GenericMessage(status="stopped" if stopped else "already_stopped")
