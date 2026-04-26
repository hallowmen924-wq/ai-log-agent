import os
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.faiss_customer_db import (
    CUSTOMER_SEARCH_TYPES,
    build_customer_documents,
    format_customer_search_results,
)
from rag.faiss_logs_db import (
    build_log_documents,
    format_log_search_results,
    prepare_log_records,
)
from rag.faiss_news_db import (
    NEWS_LIKE_TYPES,
    RULE_LIKE_TYPES,
    build_news_documents,
    split_news_search_results,
)

_embeddings = None
import io
import json
import logging
import re
import shutil
from typing import Any

# 모듈 수준 파일 로거 설정 (RAG ingest 로그 저장)
_LOG_DIR = os.path.join("logs")
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


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


FAISS_STORE_LOGS = "logs"
FAISS_STORE_NEWS = "news"
FAISS_STORE_CUSTOMER = "customer"

FAISS_STORE_PATHS = {
    FAISS_STORE_LOGS: "faiss_logs",
    FAISS_STORE_NEWS: "faiss_news",
    FAISS_STORE_CUSTOMER: "faiss_customer",
}

LEGACY_FAISS_PATH = "faiss_db"

_db_registry: dict[str, FAISS | None] = {
    FAISS_STORE_LOGS: None,
    FAISS_STORE_NEWS: None,
    FAISS_STORE_CUSTOMER: None,
}


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

    if doc_type == "log":
        return FAISS_STORE_LOGS
    if doc_type in {"news", "regulation", "rule"}:
        return FAISS_STORE_NEWS
    if doc_type in {"customer_pattern", "sales_strategy"}:
        return FAISS_STORE_CUSTOMER
    if doc_type.startswith("agent_report"):
        if agent_name in {"log", "log_agent"}:
            return FAISS_STORE_LOGS
        if agent_name in {"news", "news_agent", "regulation", "regulation_agent"}:
            return FAISS_STORE_NEWS
        return FAISS_STORE_CUSTOMER
    return FAISS_STORE_CUSTOMER


def infer_store_from_doc_type(doc_type: str | None) -> str:
    dummy_meta = {"type": str(doc_type or "").strip().lower()}
    return infer_store_from_metadata(dummy_meta)


def infer_store_from_report(report: dict[str, Any]) -> str:
    return infer_store_from_metadata(
        {
            "type": "agent_report",
            "agent": report.get("agent"),
            "store": report.get("store"),
        }
    )


def _load_local_db(path: str) -> FAISS | None:
    if not os.path.exists(path):
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


def load_existing_agent_report_docs(store_name: str):
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
            if getattr(doc, "metadata", {}).get("type", "").startswith("agent_report")
            and infer_store_from_metadata(getattr(doc, "metadata", {}) or {})
            == normalized_store
        ]
    except Exception:
        return []


def should_preserve_existing_doc(store_name: str, metadata: dict[str, Any] | None) -> bool:
    meta = metadata or {}
    doc_type = str(meta.get("type") or "").strip().lower()
    source = str(meta.get("source") or "").strip().lower()

    if store_name == FAISS_STORE_LOGS:
        return doc_type.startswith("agent_report")
    if store_name == FAISS_STORE_NEWS:
        return doc_type.startswith("agent_report") or source == "upload" or doc_type in {
            "regulation",
            "rule",
        }
    if store_name == FAISS_STORE_CUSTOMER:
        return doc_type.startswith("agent_report") or doc_type == "sales_strategy"
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
        return float(value) == 0.0

    text = clean_faiss_text(value)
    if not text:
        return True
    if text.upper() in _NULL_LIKE_VALUES:
        return True

    numeric_candidate = text.replace(",", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", numeric_candidate):
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


def build_vector_db(logs, news):
    start = time.time()
    print("벡터 생성 시작")

    log_documents = []
    news_documents = []
    customer_documents = []

    log_documents.extend(load_preserved_store_docs(FAISS_STORE_LOGS))
    news_documents.extend(load_preserved_store_docs(FAISS_STORE_NEWS))
    customer_documents.extend(load_preserved_store_docs(FAISS_STORE_CUSTOMER))
    prepared_logs = prepare_log_records(
        logs,
        ingest_logger,
        should_skip_log=should_skip_faiss_log,
        sanitize_fields=sanitize_faiss_fields,
        sanitize_mapping=sanitize_faiss_mapping,
        find_ignorable_keys=find_globally_ignorable_field_keys,
        apply_mapping=apply_mapping,
        map_fields=map_fields,
        clean_text=clean_faiss_text,
    )
    log_documents.extend(build_log_documents(prepared_logs, FAISS_STORE_LOGS))
    news_documents.extend(
        build_news_documents(
            news,
            ingest_logger,
            clean_text=clean_faiss_text,
            store_name=FAISS_STORE_NEWS,
        )
    )
    customer_documents.extend(
        build_customer_documents(
            prepared_logs,
            clean_text=clean_faiss_text,
            store_name=FAISS_STORE_CUSTOMER,
        )
    )

    counts = {
        FAISS_STORE_LOGS: _rebuild_store(FAISS_STORE_LOGS, log_documents),
        FAISS_STORE_NEWS: _rebuild_store(FAISS_STORE_NEWS, news_documents),
        FAISS_STORE_CUSTOMER: _rebuild_store(FAISS_STORE_CUSTOMER, customer_documents),
    }
    total_count = sum(counts.values())

    try:
        ingest_logger.info(
            "FAISS stores saved: logs=%d news=%d customer=%d total=%d",
            counts[FAISS_STORE_LOGS],
            counts[FAISS_STORE_NEWS],
            counts[FAISS_STORE_CUSTOMER],
            total_count,
        )
    except Exception:
        ingest_logger.info("FAISS stores saved: total=%d", total_count)

    print(f"완료: {time.time() - start:.2f}초")
    return total_count


def _clear_store(store_name: str) -> None:
    normalized_store = normalize_store_name(store_name)
    _db_registry[normalized_store] = None
    store_path = get_store_path(normalized_store)
    if os.path.exists(store_path):
        shutil.rmtree(store_path, ignore_errors=True)


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs: list[Document] = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return split_docs


def _rebuild_store(store_name: str, documents: list[Document]) -> int:
    normalized_store = normalize_store_name(store_name)
    split_docs = _split_documents(documents)
    print(f"{normalized_store} document 개수: {len(documents)} / chunk 수: {len(split_docs)}")

    if not split_docs:
        _clear_store(normalized_store)
        return 0

    local_db = FAISS.from_documents(split_docs, get_embeddings())
    local_db.save_local(get_store_path(normalized_store))
    _db_registry[normalized_store] = local_db
    return len(getattr(local_db, "index_to_docstore_id", []) or [])


def load_db(store_name: str | None = None):
    if store_name is None:
        for current_store in FAISS_STORE_PATHS:
            load_db(current_store)
        return

    normalized_store = normalize_store_name(store_name)
    if _db_registry[normalized_store] is None:
        store_path = get_store_path(normalized_store)
        if os.path.exists(store_path):
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


def search_context(query, k=5):
    logs, news, rules = [], [], []

    logs = format_log_search_results(
        _similarity_search(FAISS_STORE_LOGS, query, k, {"log"}),
        apply_mapping,
    )

    news, rules = split_news_search_results(_similarity_search(
        FAISS_STORE_NEWS,
        query,
        k,
        NEWS_LIKE_TYPES | RULE_LIKE_TYPES,
    ))

    return logs, news, rules


def search_news_context(query: str, k: int = 5) -> tuple[list[str], list[str]]:
    docs = _similarity_search(
        FAISS_STORE_NEWS,
        query,
        k,
        NEWS_LIKE_TYPES | RULE_LIKE_TYPES,
    )
    return split_news_search_results(docs)


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


def _append_documents_to_store(store_name: str, documents: list[Document]) -> int:
    normalized_store = normalize_store_name(store_name)
    if not documents:
        return get_vector_count(normalized_store)

    local_db = _get_loaded_db(normalized_store)
    if local_db is None:
        local_db = FAISS.from_documents(documents, get_embeddings())
    else:
        local_db.add_documents(documents)

    local_db.save_local(get_store_path(normalized_store))
    _db_registry[normalized_store] = local_db
    return len(getattr(local_db, "index_to_docstore_id", []) or [])


def save_agent_reports(reports, store_name: str | None = None):
    if not reports:
        return get_vector_count()

    documents_by_store: dict[str, list[Document]] = {
        FAISS_STORE_LOGS: [],
        FAISS_STORE_NEWS: [],
        FAISS_STORE_CUSTOMER: [],
    }
    for report in reports:
        try:
            ingest_logger.info(
                "---- RAG INGEST: agent report ----\n%s",
                json.dumps(report, ensure_ascii=False, indent=2),
            )
        except Exception:
            ingest_logger.info("---- RAG INGEST: agent report ---- %s", str(report))

        title = str(report.get("title", "agent report")).strip()
        content = str(report.get("content", "")).strip()
        agent_name = str(report.get("agent", "agent")).strip()
        if not content:
            continue
        target_store = normalize_store_name(store_name) if store_name else infer_store_from_report(report)
        doc_type = str(report.get("type") or f"agent_report_{agent_name}").strip().lower()
        documents_by_store[target_store].append(
            Document(
                page_content=f"제목: {title}\n내용: {content}",
                metadata={
                    "type": doc_type,
                    "agent": agent_name,
                    "store": target_store,
                },
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
            "FAISS saved (agent_report): logs=%d news=%d customer=%d total=%d",
            counts[FAISS_STORE_LOGS],
            counts[FAISS_STORE_NEWS],
            counts[FAISS_STORE_CUSTOMER],
            get_vector_count(),
        )
    except Exception:
        ingest_logger.info("FAISS saved (agent_report): total=%d", get_vector_count())
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

    count_after = _append_documents_to_store(target_store, split_docs)
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
