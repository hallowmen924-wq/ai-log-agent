import logging
import os
import time
import threading
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)


class _SuppressTransformersPathAliasFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Accessing `__path__` from" not in message


for _logger_name in ("transformers", "transformers.utils.import_utils"):
    _hf_logger = logging.getLogger(_logger_name)
    if not any(
        isinstance(existing_filter, _SuppressTransformersPathAliasFilter)
        for existing_filter in _hf_logger.filters
    ):
        _hf_logger.addFilter(_SuppressTransformersPathAliasFilter())

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.faiss_customer_db import (
    CUSTOMER_SEARCH_TYPES,
    format_customer_search_results,
)
from rag.product_pattern_summary import build_product_pattern_summary_documents
from rag.faiss_logs_db import (
    build_log_documents,
    build_log_ingest_preview,
    format_log_search_results,
    prepare_log_records,
)
from rag.faiss_news_db import (
    NEWS_LIKE_TYPES,
    RULE_LIKE_TYPES,
    build_news_documents,
    split_news_search_results,
)

import io
import json
import re
import shutil
from typing import Any, Iterable


_embeddings = None
_embeddings_lock = threading.Lock()
_embeddings_warmed = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_LOCAL_EMBEDDING_MODEL_DIR = os.environ.get(
    "LOCAL_EMBEDDING_MODEL_DIR",
    os.path.join(
        _PROJECT_ROOT,
        "models",
        "sentence-transformers",
        "all-MiniLM-L6-v2",
    ),
)

# 모듈 수준 파일 로거 설정 (RAG ingest 로그 저장)
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
try:
    os.makedirs(_LOG_DIR, exist_ok=True)
except Exception:
    pass
_LOG_FILE = os.environ.get(
    "RAG_INGEST_LOG_FILE", os.path.join(_LOG_DIR, "rag_ingest.log")
)
ingest_logger = logging.getLogger("rag_ingest")
if not ingest_logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ingest_logger.addHandler(_fh)
    ingest_logger.setLevel(logging.INFO)


def get_local_embedding_model_path() -> str:
    model_path = os.path.abspath(_LOCAL_EMBEDDING_MODEL_DIR)
    if os.path.isdir(model_path):
        return model_path
    raise FileNotFoundError(
        "로컬 임베딩 모델을 찾을 수 없습니다. "
        f"모델 경로를 준비하세요: {model_path}"
    )


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                model_path = get_local_embedding_model_path()
                _embeddings = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={
                        "local_files_only": True,
                    },
                )
    return _embeddings


def warmup_embeddings() -> dict[str, int]:
    global _embeddings_warmed

    load_started_at = time.perf_counter()
    embeddings = get_embeddings()
    load_elapsed_ms = int((time.perf_counter() - load_started_at) * 1000)

    warmup_elapsed_ms = 0
    if not _embeddings_warmed:
        with _embeddings_lock:
            if not _embeddings_warmed:
                warmup_started_at = time.perf_counter()
                try:
                    embeddings.embed_query("faiss warmup")
                except Exception:
                    embeddings.embed_documents(["faiss warmup"])
                warmup_elapsed_ms = int((time.perf_counter() - warmup_started_at) * 1000)
                _embeddings_warmed = True

    return {
        "load_ms": load_elapsed_ms,
        "warmup_ms": warmup_elapsed_ms,
        "total_ms": load_elapsed_ms + warmup_elapsed_ms,
    }


FAISS_STORE_LOGS = "logs"
FAISS_STORE_NEWS = "news"
FAISS_STORE_CUSTOMER = "customer"
FAISS_STORE_DOCUMENT = "document"

FAISS_STORE_PATHS = {
    FAISS_STORE_LOGS: os.path.join(_PROJECT_ROOT, "faiss_logs"),
    FAISS_STORE_NEWS: os.path.join(_PROJECT_ROOT, "faiss_news"),
    FAISS_STORE_CUSTOMER: os.path.join(_PROJECT_ROOT, "faiss_customer"),
    FAISS_STORE_DOCUMENT: os.path.join(_PROJECT_ROOT, "faiss_document"),
}

LEGACY_FAISS_PATH = os.path.join(_PROJECT_ROOT, "faiss_db")
EMPTY_STORE_MARKER = ".empty_store"

_db_registry: dict[str, FAISS | None] = {
    FAISS_STORE_LOGS: None,
    FAISS_STORE_NEWS: None,
    FAISS_STORE_CUSTOMER: None,
    FAISS_STORE_DOCUMENT: None,
}

_COMMONFEATURE_PATH = os.path.join(_PROJECT_ROOT, "data", "commonfeature.json")
_ONTOLOGY_RELATIONS_PATH = os.path.join(_PROJECT_ROOT, "data", "ontology_relations.json")
_ontology_entity_index_cache: dict[str, Any] | None = None
_ontology_entity_index_lock = threading.Lock()


def _tokenize_entity_text(text: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^0-9a-zA-Z가-힣_\.]+", str(text or "").lower())
        if len(token) >= 2
    ]


def _load_ontology_entity_index() -> dict[str, Any]:
    global _ontology_entity_index_cache
    if _ontology_entity_index_cache is not None:
        return _ontology_entity_index_cache

    with _ontology_entity_index_lock:
        if _ontology_entity_index_cache is not None:
            return _ontology_entity_index_cache

        feature_map: dict[str, dict[str, Any]] = {}
        token_to_feature_ids: dict[str, set[str]] = {}
        relations_map: dict[str, set[str]] = {}

        try:
            with open(_COMMONFEATURE_PATH, "r", encoding="utf-8") as common_file:
                payload = json.load(common_file)
            for item in (payload.get("common_features") or []):
                feature_id = str(item.get("feature_id") or "").strip()
                if not feature_id:
                    continue
                feature_name = str(item.get("feature_name") or feature_id).strip()
                category = str(item.get("category") or "").strip()
                aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()]
                feature_map[feature_id] = {
                    "feature_id": feature_id,
                    "feature_name": feature_name,
                    "category": category,
                }
                terms = set(_tokenize_entity_text(feature_id))
                terms.update(_tokenize_entity_text(feature_name))
                for alias in aliases[:16]:
                    terms.update(_tokenize_entity_text(alias))
                for term in terms:
                    token_to_feature_ids.setdefault(term, set()).add(feature_id)
        except Exception:
            feature_map = {}
            token_to_feature_ids = {}

        try:
            with open(_ONTOLOGY_RELATIONS_PATH, "r", encoding="utf-8") as relation_file:
                relation_payload = json.load(relation_file)
            for feature_id, relation_item in ((relation_payload.get("features") or {}).items()):
                key = str(feature_id or "").strip()
                if not key:
                    continue
                linked: set[str] = set()
                for values in ((relation_item or {}).get("relations") or {}).values():
                    if not isinstance(values, list):
                        continue
                    for value in values:
                        value_text = str(value or "").strip()
                        if value_text:
                            linked.add(value_text)
                relations_map[key] = linked
        except Exception:
            relations_map = {}

        _ontology_entity_index_cache = {
            "feature_map": feature_map,
            "token_to_feature_ids": token_to_feature_ids,
            "relations_map": relations_map,
        }
        return _ontology_entity_index_cache


def _extract_ontology_entities_from_text(text: str, limit: int = 8) -> tuple[list[dict[str, str]], list[str]]:
    index_bundle = _load_ontology_entity_index()
    feature_map: dict[str, dict[str, Any]] = dict(index_bundle.get("feature_map") or {})
    token_to_feature_ids: dict[str, set[str]] = dict(index_bundle.get("token_to_feature_ids") or {})
    relations_map: dict[str, set[str]] = dict(index_bundle.get("relations_map") or {})

    lowered = str(text or "").lower()
    token_hits: dict[str, int] = {}
    for token in _tokenize_entity_text(lowered):
        for feature_id in token_to_feature_ids.get(token, set()):
            token_hits[feature_id] = token_hits.get(feature_id, 0) + 1

    for feature_id in feature_map.keys():
        if feature_id and feature_id.lower() in lowered:
            token_hits[feature_id] = token_hits.get(feature_id, 0) + 3

    ranked_ids = sorted(
        token_hits.keys(),
        key=lambda feature_id: (-int(token_hits.get(feature_id) or 0), feature_id),
    )[: max(1, int(limit or 8))]

    entities: list[dict[str, str]] = []
    linked_feature_ids: set[str] = set()
    for feature_id in ranked_ids:
        feature = feature_map.get(feature_id) or {}
        entities.append(
            {
                "feature_id": feature_id,
                "feature_name": str(feature.get("feature_name") or feature_id),
                "category": str(feature.get("category") or ""),
            }
        )
        linked_feature_ids.update(relations_map.get(feature_id, set()))

    return entities, sorted(linked_feature_ids)


def normalize_store_name(store_name: str | None) -> str:
    candidate = str(store_name or FAISS_STORE_LOGS).strip().lower()
    if candidate in FAISS_STORE_PATHS:
        return candidate
    raise ValueError(f"unknown FAISS store: {store_name}")


def get_store_path(store_name: str) -> str:
    return FAISS_STORE_PATHS[normalize_store_name(store_name)]


def infer_store_from_metadata(metadata: dict[str, Any] | None) -> str:
    meta = metadata or {}
    explicit_store = str(meta.get("store") or "").strip().lower()
    if explicit_store in FAISS_STORE_PATHS:
        return explicit_store

    doc_type = str(meta.get("type") or "").strip().lower()
    agent_name = str(meta.get("agent") or "").strip().lower()

    if doc_type in {"log", "generated_log"}:
        return FAISS_STORE_LOGS
    if doc_type in {"news", "signal_news", "generated_news"}:
        return FAISS_STORE_NEWS
    if doc_type in {"regulation", "rule", "generated_regulation", "document"}:
        return FAISS_STORE_DOCUMENT
    if doc_type in {"customer_pattern", "product_pattern_summary", "sales_strategy", "generated_decision", "generated_customer"}:
        return FAISS_STORE_CUSTOMER
    if agent_name in {"log", "log_agent"}:
        return FAISS_STORE_LOGS
    if agent_name in {"news", "news_agent"}:
        return FAISS_STORE_NEWS
    if agent_name in {"regulation", "regulation_agent", "document"}:
        return FAISS_STORE_DOCUMENT
    return FAISS_STORE_CUSTOMER


def infer_store_from_doc_type(doc_type: str | None) -> str:
    dummy_meta = {"type": str(doc_type or "").strip().lower()}
    return infer_store_from_metadata(dummy_meta)


def infer_store_from_generated_item(report: dict[str, Any]) -> str:
    return infer_store_from_metadata(
        {
            "type": report.get("type"),
            "agent": report.get("agent"),
            "store": report.get("store"),
        }
    )


def _load_local_db(path: str) -> FAISS | None:
    if not os.path.exists(path):
        return None
    index_path = os.path.join(path, "index.faiss")
    metadata_path = os.path.join(path, "index.pkl")
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        return None
    try:
        return FAISS.load_local(
            path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def _iter_docstore_documents(local_db: FAISS | None):
    if local_db is None:
        return []
    doc_map = getattr(local_db.docstore, "_dict", {}) or {}
    return list(doc_map.values())


def load_existing_generated_docs(store_name: str):
    normalized_store = normalize_store_name(store_name)
    target_db = _load_local_db(get_store_path(normalized_store))
    source_db = target_db
    if source_db is None and os.path.exists(LEGACY_FAISS_PATH):
        source_db = _load_local_db(LEGACY_FAISS_PATH)

    if source_db is None:
        return []

    try:
        return [
            doc
            for doc in _iter_docstore_documents(source_db)
            if _is_generated_doc_type(getattr(doc, "metadata", {}).get("type", ""))
            and infer_store_from_metadata(getattr(doc, "metadata", {}) or {})
            == normalized_store
        ]
    except Exception:
        return []


def _is_generated_doc_type(doc_type: str | None) -> bool:
    normalized = str(doc_type or "").strip().lower()
    return normalized.startswith("generated_") or normalized == "signal_news"


def _parse_signal_news_payload(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_generated_doc_payload(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    title = str(report.get("title", "generated document")).strip()
    content = str(report.get("content", "")).strip()
    agent_name = str(report.get("agent", "agent")).strip()
    doc_type = str(report.get("type") or f"generated_{agent_name}").strip().lower()

    metadata: dict[str, Any] = {
        "type": doc_type,
        "agent": agent_name,
    }
    if doc_type == "generated_log":
        structured_payload = report.get("structured_payload") or {}
        if isinstance(structured_payload, dict):
            structured_text = str(structured_payload.get("text") or "").strip()
            structured_metadata = structured_payload.get("metadata") or {}
            if structured_text:
                metadata["features"] = (
                    structured_metadata if isinstance(structured_metadata, dict) else {}
                )
                metadata["raw_content"] = content
                return structured_text, metadata
    if doc_type == "signal_news":
        payload = _parse_signal_news_payload(content)
        tags = [str(item).strip() for item in payload.get("tags") or [] if str(item).strip()]
        signal_summary = str(payload.get("signal_summary") or "").strip()
        search_text = str(payload.get("search_text") or "").strip()
        risk_signal = [str(item).strip() for item in payload.get("risk_signal") or [] if str(item).strip()]
        opportunity_signal = [str(item).strip() for item in payload.get("opportunity_signal") or [] if str(item).strip()]
        linked_decision = [str(item).strip() for item in payload.get("linked_decision") or [] if str(item).strip()]

        optimized_text = search_text or content
        metadata["features"] = {
            "tags": tags,
            "signal_summary": signal_summary,
            "risk_signal": risk_signal,
            "opportunity_signal": opportunity_signal,
            "linked_decision": linked_decision,
        }
        metadata["raw_content"] = content
        return f"제목: {title}\n내용: {optimized_text}", metadata

    return f"제목: {title}\n내용: {content}", metadata


def should_preserve_existing_doc(store_name: str, metadata: dict[str, Any] | None) -> bool:
    meta = metadata or {}
    doc_type = str(meta.get("type") or "").strip().lower()
    source = str(meta.get("source") or "").strip().lower()

    if store_name == FAISS_STORE_LOGS:
        return False
    if store_name == FAISS_STORE_NEWS:
        return source == "upload" or doc_type in {
            "news",
            "signal_news",
        }
    if store_name == FAISS_STORE_DOCUMENT:
        return source == "upload" or doc_type in {
            "regulation",
            "rule",
            "generated_regulation",
        }
    if store_name == FAISS_STORE_CUSTOMER:
        return doc_type == "sales_strategy"
    return False


def load_preserved_store_docs(store_name: str):
    normalized_store = normalize_store_name(store_name)
    target_db = _load_local_db(get_store_path(normalized_store))
    source_db = target_db
    if source_db is None and os.path.exists(LEGACY_FAISS_PATH):
        source_db = _load_local_db(LEGACY_FAISS_PATH)

    if source_db is None:
        return []

    try:
        return [
            doc
            for doc in _iter_docstore_documents(source_db)
            if infer_store_from_metadata(getattr(doc, "metadata", {}) or {})
            == normalized_store
            and should_preserve_existing_doc(
                normalized_store, getattr(doc, "metadata", {}) or {}
            )
        ]
    except Exception:
        return []

_NULL_LIKE_VALUES = {"", "NULL", "NONE", "NAN", "N/A", "NA"}


def is_placeholder_numeric_text(text: str) -> bool:
    compact = str(text or "").replace(",", "").strip()
    if not compact:
        return False

    sign = ""
    if compact.startswith(("+", "-")):
        sign = compact[0]
        compact = compact[1:]

    if not compact:
        return False

    if "." in compact:
        integer_part, fractional_part = compact.split(".", 1)
        digits_only = f"{integer_part}{fractional_part}"
    else:
        digits_only = compact

    if not digits_only.isdigit():
        return False

    if sign == "-":
        return True

    return len(digits_only) >= 6 and set(digits_only) == {"9"}


def normalize_zero_like_text(text: str) -> str:
    compact = text.replace(",", "")
    if re.fullmatch(r"[+-]?0+(?:\.0+)?", compact):
        return "0"
    return text


def normalize_numeric_text(text: str) -> str:
    compact = text.replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
        return text

    sign = ""
    if compact.startswith(("+", "-")):
        sign = compact[0]
        compact = compact[1:]

    if "." in compact:
        integer_part, fractional_part = compact.split(".", 1)
        integer_part = integer_part.lstrip("0") or "0"
        return f"{sign}{integer_part}.{fractional_part}"

    return f"{sign}{compact.lstrip('0') or '0'}"


def clean_faiss_text(value):
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\b1\.\s*0\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = normalize_zero_like_text(text)
    return normalize_numeric_text(text)


def is_ignorable_faiss_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value == 0.0:
            return True
        return numeric_value < 0 or is_placeholder_numeric_text(str(value))

    text = clean_faiss_text(value)
    if not text:
        return True
    if text.upper() in _NULL_LIKE_VALUES:
        return True

    numeric_candidate = text.replace(",", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", numeric_candidate):
        if is_placeholder_numeric_text(numeric_candidate):
            return True
        try:
            return float(numeric_candidate) == 0.0
        except Exception:
            return False

    return False


def sanitize_faiss_fields(fields: dict, drop_keys: set[str] | None = None) -> dict:
    if not fields:
        return {}

    cleaned = {}
    for key, value in fields.items():
        if drop_keys and key in drop_keys:
            continue
        if is_ignorable_faiss_value(value):
            continue

        cleaned_key = clean_faiss_text(key)
        cleaned_value = clean_faiss_text(value)
        if not cleaned_key or is_ignorable_faiss_value(cleaned_value):
            continue
        cleaned[cleaned_key] = cleaned_value

    return cleaned


def sanitize_faiss_mapping(mapping: dict) -> dict:
    if not mapping:
        return {}

    cleaned = {}
    for key, value in mapping.items():
        cleaned_key = clean_faiss_text(key)
        cleaned_value = clean_faiss_text(value)
        if not cleaned_key:
            continue
        cleaned[cleaned_key] = cleaned_value or cleaned_key

    return cleaned


def find_globally_ignorable_field_keys(logs, field_name: str) -> set[str]:
    key_states: dict[str, bool] = {}

    for log in logs:
        fields = (log or {}).get(field_name, {}) or {}
        for key, value in fields.items():
            current = key_states.setdefault(key, True)
            key_states[key] = current and is_ignorable_faiss_value(value)

    return {key for key, always_ignorable in key_states.items() if always_ignorable}


def apply_mapping(fields, mapping):
    result = []

    for k, v in fields.items():
        if is_ignorable_faiss_value(v):
            continue

        key_text = clean_faiss_text(k)
        value_text = clean_faiss_text(v)
        meaning = clean_faiss_text(mapping.get(k, key_text) or key_text)
        if not key_text or not meaning or is_ignorable_faiss_value(value_text):
            continue

        result.append(f"{meaning}: {value_text}")

    return ", ".join(result)


def map_fields(fields: dict, mapping: dict) -> dict:
    """Return a new dict where keys are replaced by mapping.get(key, key)."""
    if not fields:
        return {}
    try:
        mapped = {}
        for k, v in fields.items():
            if is_ignorable_faiss_value(v):
                continue
            key_text = clean_faiss_text(k)
            value_text = clean_faiss_text(v)
            mapped_key = clean_faiss_text(mapping.get(k, key_text) or key_text)
            if not mapped_key or is_ignorable_faiss_value(value_text):
                continue
            mapped[mapped_key] = value_text
        return mapped
    except Exception:
        # fallback: return original
        return sanitize_faiss_fields(fields)


def should_skip_faiss_log(log_item: dict) -> bool:
    product_code = str(log_item.get("product") or log_item.get("product_code") or "").strip().upper()
    if not product_code:
        return False
    if re.fullmatch(r"S\d{4}", product_code):
        return True
    if product_code.startswith("W"):
        return True
    return False


def build_vector_db(
    logs,
    news,
    rebuild_stores: Iterable[str] | None = None,
):
    start = time.perf_counter()
    print("벡터 생성 시작")

    requested_stores = {
        normalize_store_name(store_name)
        for store_name in (rebuild_stores or FAISS_STORE_PATHS.keys())
    }
    if not requested_stores:
        return get_vector_count()

    warmup_timing = warmup_embeddings()

    log_documents = []
    news_documents = []
    customer_documents = []
    document_documents = []

    if FAISS_STORE_LOGS in requested_stores:
        log_documents.extend(load_preserved_store_docs(FAISS_STORE_LOGS))
    if FAISS_STORE_NEWS in requested_stores:
        news_documents.extend(load_preserved_store_docs(FAISS_STORE_NEWS))
    if FAISS_STORE_CUSTOMER in requested_stores:
        customer_documents.extend(load_preserved_store_docs(FAISS_STORE_CUSTOMER))
    if FAISS_STORE_DOCUMENT in requested_stores:
        document_documents.extend(load_preserved_store_docs(FAISS_STORE_DOCUMENT))

    needs_prepared_logs = bool({FAISS_STORE_LOGS, FAISS_STORE_CUSTOMER} & requested_stores)
    prepared_logs = []
    if needs_prepared_logs:
        prepared_logs = prepare_log_records(
            logs,
            ingest_logger,
            show_progress=True,
            should_skip_log=should_skip_faiss_log,
            sanitize_fields=sanitize_faiss_fields,
            sanitize_mapping=sanitize_faiss_mapping,
            find_ignorable_keys=find_globally_ignorable_field_keys,
            apply_mapping=apply_mapping,
            map_fields=map_fields,
            clean_text=clean_faiss_text,
        )

    if FAISS_STORE_LOGS in requested_stores:
        log_documents.extend(build_log_documents(prepared_logs, FAISS_STORE_LOGS))
    if FAISS_STORE_NEWS in requested_stores:
        news_documents.extend(
            build_news_documents(
                news,
                ingest_logger,
                clean_text=clean_faiss_text,
                store_name=FAISS_STORE_NEWS,
            )
        )
    if FAISS_STORE_CUSTOMER in requested_stores:
        customer_documents.extend(
            build_product_pattern_summary_documents(
                logs,
                clean_text=clean_faiss_text,
                store_name=FAISS_STORE_CUSTOMER,
            )
        )

    store_documents = {
        FAISS_STORE_LOGS: log_documents,
        FAISS_STORE_NEWS: news_documents,
        FAISS_STORE_CUSTOMER: customer_documents,
        FAISS_STORE_DOCUMENT: document_documents,
    }
    store_timings: dict[str, dict[str, int]] = {}
    for store_name in requested_stores:
        store_timings[store_name] = _rebuild_store(store_name, store_documents[store_name])

    counts = {
        current_store: get_vector_count(current_store) for current_store in FAISS_STORE_PATHS
    }
    total_count = sum(counts.values())

    try:
        ingest_logger.info(
            "FAISS stores saved: logs=%d news=%d customer=%d document=%d total=%d warmup_ms=%d stores=%s",
            counts[FAISS_STORE_LOGS],
            counts[FAISS_STORE_NEWS],
            counts[FAISS_STORE_CUSTOMER],
            counts[FAISS_STORE_DOCUMENT],
            total_count,
            warmup_timing.get("total_ms", 0),
            json.dumps(store_timings, ensure_ascii=False),
        )
    except Exception:
        ingest_logger.info("FAISS stores saved: total=%d", total_count)

    print(
        "FAISS warmup load "
        f"{warmup_timing.get('load_ms', 0)}ms / query {warmup_timing.get('warmup_ms', 0)}ms"
    )
    for store_name in requested_stores:
        timing = store_timings.get(store_name, {})
        print(
            f"{store_name} split {timing.get('split_ms', 0)}ms / "
            f"embed+index {timing.get('build_ms', 0)}ms / save {timing.get('save_ms', 0)}ms / "
            f"total {timing.get('total_ms', 0)}ms"
        )
    print(f"완료: {time.perf_counter() - start:.2f}초")
    return total_count


def prepare_log_ingest_preview(logs) -> dict[str, Any]:
    prepared_logs = prepare_log_records(
        logs,
        ingest_logger,
        show_progress=False,
        should_skip_log=should_skip_faiss_log,
        sanitize_fields=sanitize_faiss_fields,
        sanitize_mapping=sanitize_faiss_mapping,
        find_ignorable_keys=find_globally_ignorable_field_keys,
        apply_mapping=apply_mapping,
        map_fields=map_fields,
        clean_text=clean_faiss_text,
    )
    return build_log_ingest_preview(prepared_logs, FAISS_STORE_LOGS)


def _clear_store(store_name: str) -> None:
    normalized_store = normalize_store_name(store_name)
    _db_registry[normalized_store] = None
    store_path = get_store_path(normalized_store)
    if os.path.exists(store_path):
        shutil.rmtree(store_path, ignore_errors=True)


def _ensure_empty_store(store_name: str) -> None:
    store_path = get_store_path(store_name)
    os.makedirs(store_path, exist_ok=True)
    marker_path = os.path.join(store_path, EMPTY_STORE_MARKER)
    with open(marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write(store_name)


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50)
    split_docs: list[Document] = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return split_docs


def _rebuild_store(store_name: str, documents: list[Document]) -> dict[str, int]:
    normalized_store = normalize_store_name(store_name)
    started_at = time.perf_counter()
    split_docs = _split_documents(documents)
    split_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    print(f"{normalized_store} document 개수: {len(documents)} / chunk 수: {len(split_docs)}")

    if not split_docs:
        _clear_store(normalized_store)
        _ensure_empty_store(normalized_store)
        total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "document_count": len(documents),
            "chunk_count": 0,
            "split_ms": split_elapsed_ms,
            "build_ms": 0,
            "save_ms": 0,
            "total_ms": total_elapsed_ms,
            "vector_count": 0,
        }

    build_started_at = time.perf_counter()
    local_db = FAISS.from_documents(split_docs, get_embeddings())
    build_elapsed_ms = int((time.perf_counter() - build_started_at) * 1000)

    save_started_at = time.perf_counter()
    local_db.save_local(get_store_path(normalized_store))
    save_elapsed_ms = int((time.perf_counter() - save_started_at) * 1000)
    _db_registry[normalized_store] = local_db
    vector_count = len(getattr(local_db, "index_to_docstore_id", []) or [])
    total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    ingest_logger.info(
        "FAISS rebuild store=%s docs=%d chunks=%d split_ms=%d embed_index_ms=%d save_ms=%d total_ms=%d",
        normalized_store,
        len(documents),
        len(split_docs),
        split_elapsed_ms,
        build_elapsed_ms,
        save_elapsed_ms,
        total_elapsed_ms,
    )
    return {
        "document_count": len(documents),
        "chunk_count": len(split_docs),
        "split_ms": split_elapsed_ms,
        "build_ms": build_elapsed_ms,
        "save_ms": save_elapsed_ms,
        "total_ms": total_elapsed_ms,
        "vector_count": vector_count,
    }


def load_db(store_name: str | None = None):
    if store_name is None:
        for current_store in FAISS_STORE_PATHS:
            load_db(current_store)
        return

    normalized_store = normalize_store_name(store_name)
    if _db_registry[normalized_store] is None:
        store_path = get_store_path(normalized_store)
        index_path = os.path.join(store_path, "index.faiss")
        metadata_path = os.path.join(store_path, "index.pkl")
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"FAISS 로드: {normalized_store}")
            _db_registry[normalized_store] = FAISS.load_local(
                store_path,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )


def _get_loaded_db(store_name: str):
    normalized_store = normalize_store_name(store_name)
    load_db(normalized_store)
    return _db_registry[normalized_store]


def _similarity_search(
    store_name: str,
    query: str,
    k: int,
    allowed_types: set[str] | None = None,
):
    local_db = _get_loaded_db(store_name)
    if local_db is None:
        return []

    fetch_k = max(int(k or 5) * 4, 12)
    docs = local_db.similarity_search(query, k=fetch_k)
    if not allowed_types:
        return docs[:k]

    matched = []
    for doc in docs:
        doc_type = str((getattr(doc, "metadata", {}) or {}).get("type") or "").strip().lower()
        if doc_type in allowed_types:
            matched.append(doc)
        if len(matched) >= k:
            break
    return matched


def search_logs_context(query: str, k: int = 5) -> list[str]:
    return format_log_search_results(
        _similarity_search(FAISS_STORE_LOGS, query, k, {"log"}),
        apply_mapping,
    )


def search_context(query, k=5):
    logs, news, rules = [], [], []

    logs = search_logs_context(query, k=k)

    news_docs = _similarity_search(
        FAISS_STORE_NEWS,
        query,
        k,
        NEWS_LIKE_TYPES,
    )
    rule_docs = _similarity_search(
        FAISS_STORE_DOCUMENT,
        query,
        k,
        RULE_LIKE_TYPES,
    )
    news, _ = split_news_search_results(news_docs)
    _, rules = split_news_search_results(rule_docs)

    return logs, news, rules


def search_news_context(query: str, k: int = 5) -> tuple[list[str], list[str]]:
    news_docs = _similarity_search(
        FAISS_STORE_NEWS,
        query,
        k,
        NEWS_LIKE_TYPES,
    )
    rule_docs = _similarity_search(
        FAISS_STORE_DOCUMENT,
        query,
        k,
        RULE_LIKE_TYPES,
    )
    news, _ = split_news_search_results(news_docs)
    _, rules = split_news_search_results(rule_docs)
    return news, rules


def search_regulation_evidence(
    query: str,
    k: int = 6,
    preferred_feature_ids: list[str] | None = None,
    avoided_feature_ids: list[str] | None = None,
    expansion_feature_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []

    preferred_set = {
        str(item).strip() for item in (preferred_feature_ids or []) if str(item).strip()
    }
    avoided_set = {
        str(item).strip() for item in (avoided_feature_ids or []) if str(item).strip()
    }
    expansion_set = {
        str(item).strip() for item in (expansion_feature_ids or []) if str(item).strip()
    }
    target_feature_set = preferred_set | expansion_set

    docs = _similarity_search(
        FAISS_STORE_DOCUMENT,
        normalized_query,
        max(12, int(k or 6) * 5),
        RULE_LIKE_TYPES,
    )
    query_tokens = set(_tokenize_entity_text(normalized_query))
    scored: list[dict[str, Any]] = []

    for rank, doc in enumerate(docs):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        text = str(getattr(doc, "page_content", "") or "")
        text_lower = text.lower()
        feature_ids_raw = str(metadata.get("ontology_feature_ids") or "")
        linked_feature_ids_raw = str(metadata.get("ontology_linked_feature_ids") or "")
        feature_ids = [item.strip() for item in feature_ids_raw.split("|") if item.strip()]
        linked_feature_ids = [item.strip() for item in linked_feature_ids_raw.split("|") if item.strip()]

        base_score = max(0.0, 2.8 - (rank * 0.16))
        token_hit_count = sum(1 for token in query_tokens if token and token in text_lower)
        token_bonus = min(0.9, token_hit_count * 0.08)
        feature_hit_count = len(set(feature_ids) & target_feature_set)
        linked_hit_count = len(set(linked_feature_ids) & target_feature_set)
        avoided_hit_count = len(set(feature_ids) & avoided_set)

        score = (
            base_score
            + min(3.0, feature_hit_count * 0.75)
            + min(2.0, linked_hit_count * 0.45)
            + token_bonus
            - min(3.0, avoided_hit_count * 0.9)
        )

        if score <= 0:
            continue

        snippet = " ".join(part for part in text.replace("\n", " ").split(" ") if part).strip()[:260]
        scored.append(
            {
                "score": round(float(score), 4),
                "name": str(metadata.get("name") or "regulation"),
                "doc_type": str(metadata.get("type") or ""),
                "chunk_index": int(metadata.get("chunk_index") or 0),
                "feature_hits": sorted(set(feature_ids) & target_feature_set),
                "linked_hits": sorted(set(linked_feature_ids) & target_feature_set),
                "source": str(metadata.get("source") or ""),
                "snippet": snippet,
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            str(item.get("name") or ""),
            int(item.get("chunk_index") or 0),
        )
    )
    return scored[: max(1, int(k or 6))]


def search_customer_context(query: str, k: int = 5) -> list[str]:
    return format_customer_search_results(
        _similarity_search(FAISS_STORE_CUSTOMER, query, k, CUSTOMER_SEARCH_TYPES)
    )


def search_similar_logs(query, k: int = 3):
    return _similarity_search(FAISS_STORE_LOGS, query, k, {"log"})


def get_vector_count(store_name: str | None = None):
    if store_name is None:
        return sum(get_vector_count(current_store) for current_store in FAISS_STORE_PATHS)
    local_db = _get_loaded_db(store_name)
    if local_db is None:
        return 0
    return len(getattr(local_db, "index_to_docstore_id", []) or [])


def get_store_document_count(
    store_name: str,
    allowed_types: set[str] | None = None,
) -> int:
    local_db = _get_loaded_db(store_name)
    if local_db is None:
        return 0

    normalized_types = {
        str(doc_type or "").strip().lower() for doc_type in (allowed_types or set())
    }
    count = 0
    for doc in _iter_docstore_documents(local_db):
        doc_type = str((getattr(doc, "metadata", {}) or {}).get("type") or "").strip().lower()
        if normalized_types and doc_type not in normalized_types:
            continue
        count += 1
    return count


def get_store_news_keys(store_name: str = FAISS_STORE_NEWS) -> set[tuple[str, str]]:
    local_db = _get_loaded_db(store_name)
    if local_db is None:
        return set()

    news_keys: set[tuple[str, str]] = set()
    for doc in _iter_docstore_documents(local_db):
        metadata = getattr(doc, "metadata", {}) or {}
        doc_type = str(metadata.get("type") or "").strip().lower()
        if doc_type != "news":
            continue
        title = str(metadata.get("title") or "").strip()
        link = str(metadata.get("link") or "").strip()
        news_keys.add((title, link))
    return news_keys


def _append_documents_to_store(store_name: str, documents: list[Document]) -> int:
    normalized_store = normalize_store_name(store_name)
    split_docs = _split_documents(documents)
    return _append_split_documents_to_store(normalized_store, split_docs)


def _append_split_documents_to_store(store_name: str, split_docs: list[Document]) -> int:
    normalized_store = normalize_store_name(store_name)
    if not split_docs:
        return get_vector_count(normalized_store)

    local_db = _get_loaded_db(normalized_store)
    if local_db is None:
        local_db = FAISS.from_documents(split_docs, get_embeddings())
    else:
        local_db.add_documents(split_docs)

    local_db.save_local(get_store_path(normalized_store))
    _db_registry[normalized_store] = local_db
    return len(getattr(local_db, "index_to_docstore_id", []) or [])


def append_structured_log_documents(logs: list[dict[str, Any]]) -> tuple[int, int]:
    if not logs:
        return 0, get_vector_count(FAISS_STORE_LOGS)

    prepared_logs = prepare_log_records(
        logs,
        ingest_logger,
        show_progress=False,
        should_skip_log=should_skip_faiss_log,
        sanitize_fields=sanitize_faiss_fields,
        sanitize_mapping=sanitize_faiss_mapping,
        find_ignorable_keys=find_globally_ignorable_field_keys,
        apply_mapping=apply_mapping,
        map_fields=map_fields,
        clean_text=clean_faiss_text,
    )
    documents = build_log_documents(prepared_logs, FAISS_STORE_LOGS)
    return len(documents), _append_documents_to_store(FAISS_STORE_LOGS, documents)


def append_news_documents(news_items: list[dict[str, Any]]) -> tuple[int, int]:
    if not news_items:
        return 0, get_vector_count(FAISS_STORE_NEWS)

    documents = build_news_documents(
        news_items,
        ingest_logger,
        clean_text=clean_faiss_text,
        store_name=FAISS_STORE_NEWS,
    )
    return len(documents), _append_documents_to_store(FAISS_STORE_NEWS, documents)


def save_generated_documents(items, store_name: str | None = None):
    if not items:
        return get_vector_count()

    documents_by_store: dict[str, list[Document]] = {
        FAISS_STORE_LOGS: [],
        FAISS_STORE_NEWS: [],
        FAISS_STORE_CUSTOMER: [],
        FAISS_STORE_DOCUMENT: [],
    }
    for report in items:
        try:
            ingest_logger.info(
                "---- RAG INGEST: generated document ----\n%s",
                json.dumps(report, ensure_ascii=False, indent=2),
            )
        except Exception:
            ingest_logger.info("---- RAG INGEST: generated document ---- %s", str(report))

        content = str(report.get("content", "")).strip()
        if not content:
            continue
        target_store = normalize_store_name(store_name) if store_name else infer_store_from_generated_item(report)
        page_content, metadata = _build_generated_doc_payload(report)
        documents_by_store[target_store].append(
            Document(
                page_content=page_content,
                metadata={**metadata, "store": target_store},
            )
        )

    if not any(documents_by_store.values()):
        return get_vector_count()

    counts = {
        current_store: _append_documents_to_store(current_store, current_docs)
        for current_store, current_docs in documents_by_store.items()
    }
    try:
        ingest_logger.info(
            "FAISS saved (generated_docs): logs=%d news=%d customer=%d document=%d total=%d",
            counts[FAISS_STORE_LOGS],
            counts[FAISS_STORE_NEWS],
            counts[FAISS_STORE_CUSTOMER],
            counts[FAISS_STORE_DOCUMENT],
            get_vector_count(),
        )
    except Exception:
        ingest_logger.info("FAISS saved (generated_docs): total=%d", get_vector_count())
    return get_vector_count()


def ingest_files(
    files_data: list[tuple[str, bytes]],
    doc_type: str = "regulation",
    store_name: str | None = None,
) -> int:
    """
    files_data: list of (name, raw_bytes)
    Adds split chunks of provided files into the FAISS DB with metadata type `doc_type`.
    Returns number of vectors after ingest.
    """
    documents = []
    target_store = normalize_store_name(store_name) if store_name else infer_store_from_doc_type(doc_type)
    for name, raw in files_data:
        text = ""
        try:
            try:
                text = raw.decode("utf-8")
            except Exception:
                try:
                    import PyPDF2

                    reader = PyPDF2.PdfReader(io.BytesIO(raw))
                    pages = [p.extract_text() or "" for p in reader.pages]
                    text = "\n".join(pages)
                except Exception:
                    text = ""
        except Exception:
            text = ""

        if not text:
            text = f"[파일 {name}의 텍스트 추출에 실패했습니다]"

        # 기록
        try:
            ingest_logger.info(
                "---- RAG INGEST: uploaded file ----\n%s",
                json.dumps({"name": name, "size": len(raw)}, ensure_ascii=False),
            )
        except Exception:
            ingest_logger.info("---- RAG INGEST: uploaded file ---- %s", name)

        # create Document
        documents.append(
            Document(
                page_content=f"제목: {name}\n내용: {text}",
                metadata={
                    "type": doc_type,
                    "source": "upload",
                    "name": name,
                    "store": target_store,
                },
            )
        )

    if not documents:
        return get_vector_count()

    split_docs = _split_documents(documents)

    if not split_docs:
        return get_vector_count(target_store)

    if target_store == FAISS_STORE_DOCUMENT and str(doc_type or "").strip().lower() in RULE_LIKE_TYPES:
        for chunk_index, chunk_doc in enumerate(split_docs):
            entities, linked_feature_ids = _extract_ontology_entities_from_text(
                getattr(chunk_doc, "page_content", ""),
                limit=8,
            )
            metadata = dict(getattr(chunk_doc, "metadata", {}) or {})
            metadata["chunk_index"] = int(chunk_index)
            metadata["ontology_feature_ids"] = "|".join(item["feature_id"] for item in entities)
            metadata["ontology_feature_names"] = "|".join(item["feature_name"] for item in entities)
            metadata["ontology_linked_feature_ids"] = "|".join(linked_feature_ids[:20])
            metadata["ontology_entity_count"] = int(len(entities))
            chunk_doc.metadata = metadata

    count_after = _append_split_documents_to_store(target_store, split_docs)
    try:
        ingest_logger.info(
            "FAISS saved (file_ingest:%s): %d vectors", target_store, count_after
        )
    except Exception:
        ingest_logger.info("FAISS saved (file_ingest): unknown vector count")

    ingest_logger.info(
        "Ingested %d chunks for %d files", len(split_docs), len(files_data)
    )
    return count_after


def _qualify_doc_id(store_name: str, doc_id: str) -> str:
    return f"{store_name}:{doc_id}"


def _split_qualified_doc_id(doc_id: str) -> tuple[str | None, str]:
    text = str(doc_id)
    if ":" not in text:
        return None, text
    prefix, raw_id = text.split(":", 1)
    if prefix in FAISS_STORE_PATHS:
        return prefix, raw_id
    return None, text


def _list_vectors_for_store(store_name: str, limit: int = 200) -> list[dict]:
    try:
        local_db = _get_loaded_db(store_name)
        if local_db is None:
            return []

        items = []
        ids = []
        try:
            raw_ids = getattr(local_db, "index_to_docstore_id", []) or []
            if isinstance(raw_ids, dict):
                ids = [doc_id for _, doc_id in sorted(raw_ids.items())]
            else:
                ids = list(raw_ids)
        except Exception:
            ids = []

        if not ids:
            doc_map = getattr(local_db.docstore, "_dict", {}) or {}
            ids = list(doc_map.keys())

        if limit and limit > 0:
            ids = ids[-limit:]

        for raw_doc_id in ids:
            try:
                doc_map = getattr(local_db.docstore, "_dict", {}) or {}
                doc = doc_map.get(raw_doc_id)
                if doc is None:
                    for key, value in doc_map.items():
                        if str(key) == str(raw_doc_id):
                            doc = value
                            break
                if doc is None:
                    continue
                meta = getattr(doc, "metadata", {}) or {}
                items.append(
                    {
                        "id": _qualify_doc_id(store_name, str(raw_doc_id)),
                        "store": store_name,
                        "type": meta.get("type"),
                        "product": meta.get("product"),
                        "agent": meta.get("agent"),
                        "source": meta.get("source"),
                        "name": meta.get("name"),
                        "features": meta.get("features") or {},
                        "in_fields": meta.get("in_fields") or {},
                        "out_fields": meta.get("out_fields") or {},
                        "reject_reason_codes": meta.get("reject_reason_codes") or [],
                        "reject_reason_details": meta.get("reject_reason_details") or [],
                        "snippet": (getattr(doc, "page_content", "") or "")[:400],
                    }
                )
            except Exception:
                continue
        return items
    except Exception:
        return []


def list_vectors(limit: int = 200, store_name: str | None = None) -> list[dict]:
    if store_name is not None:
        return _list_vectors_for_store(store_name, limit=limit)

    per_store_limit = max(limit, 200) if limit and limit > 0 else 200
    items: list[dict] = []
    for current_store in (FAISS_STORE_LOGS, FAISS_STORE_NEWS, FAISS_STORE_CUSTOMER):
        items.extend(_list_vectors_for_store(current_store, limit=per_store_limit))
    if limit and limit > 0:
        return items[-limit:]
    return items


def get_vector_by_id(doc_id: str) -> dict | None:
    store_name, raw_doc_id = _split_qualified_doc_id(doc_id)
    candidate_stores = [store_name] if store_name else list(FAISS_STORE_PATHS.keys())

    for current_store in candidate_stores:
        if current_store is None:
            continue
        try:
            local_db = _get_loaded_db(current_store)
            if local_db is None:
                continue
            doc_map = getattr(local_db.docstore, "_dict", {}) or {}
            doc = doc_map.get(raw_doc_id)
            if doc is None:
                for key, value in doc_map.items():
                    if str(key) == str(raw_doc_id):
                        doc = value
                        break
            if doc is None:
                continue
            return {
                "id": _qualify_doc_id(current_store, str(raw_doc_id)),
                "page_content": getattr(doc, "page_content", ""),
                "metadata": getattr(doc, "metadata", {}) or {},
            }
        except Exception:
            continue
    return None
