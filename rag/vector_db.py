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

_embeddings = None
import io
import json
import logging
import re

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


def load_existing_agent_report_docs():
    if not os.path.exists("faiss_db"):
        return []

    try:
        existing_db = FAISS.load_local(
            "faiss_db",
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        doc_map = getattr(existing_db.docstore, "_dict", {})
        return [
            doc
            for doc in doc_map.values()
            if getattr(doc, "metadata", {}).get("type") == "agent_report"
        ]
    except Exception:
        return []


db = None


def apply_mapping(fields, mapping):
    result = []

    for k, v in fields.items():
        meaning = mapping.get(k, k)  # 매핑 없으면 코드 그대로

        result.append(f"{meaning}: {v}")

    return ", ".join(result)


def build_vector_db(logs, news):
    global db

    start = time.time()
    print("🚀 벡터 생성 시작")

    documents = []
    documents.extend(load_existing_agent_report_docs())

    # helper: 숫자/퍼센트 파싱
    def _parse_number(text: str):
        if not text:
            return None
        m = re.search(r"[-+]?[0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?%?", str(text))
        if not m:
            return None
        s = m.group(0)
        if s.endswith("%"):
            try:
                return float(s[:-1].replace(",", ""))
            except Exception:
                return None
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None

    def _extract_features(log_item: dict):
        features = {
            "available_amount": None,
            "applied_rate": None,
            "ko_codes": [],
            "case_id": None,
            "product_code": log_item.get("product") or log_item.get("product_code"),
            # 추가 추출 항목
            "loan_term_months": None,
            "loan_term_raw": None,
            "credit_grade": None,
            "credit_score": None,
            "annual_income": None,
            "purpose": None,
            "collateral": None,
            "interest_type": None,
        }

        in_fields = log_item.get("in_fields", {}) or {}
        out_fields = log_item.get("out_fields", {}) or {}
        in_mapping = log_item.get("in_mapping", {}) or {}
        out_mapping = log_item.get("out_mapping", {}) or {}

        # 수집할 필드 목록: (key, label, value)
        scan_fields = []
        for src, mapping in ((in_fields, in_mapping), (out_fields, out_mapping)):
            for k, v in src.items():
                label = str(mapping.get(k, k))
                scan_fields.append((k, label, v))
                # case id 후보
                if features["case_id"] is None and str(k).lower() in (
                    "case_id",
                    "id",
                    "req_no",
                    "request_id",
                    "caseid",
                ):
                    features["case_id"] = str(v)

        # 보조 키워드 및 단위 처리
        for k, label, value in scan_fields:
            val_str = "" if value is None else str(value)
            l_low = label.lower()
            v_low = val_str.lower()

            # 금액 추출
            if features["available_amount"] is None and (
                any(
                    tok in l_low
                    for tok in ("대출", "한도", "금액", "limit", "available")
                )
                or re.search(r"\b(원|만원|억|천원|만)\b", v_low)
            ):
                num = _parse_number(val_str)
                if num is not None:
                    multiplier = 1
                    if "만원" in v_low or (
                        "만" in v_low and re.search(r"\d+만", v_low)
                    ):
                        multiplier = 10000
                    elif "억" in v_low:
                        multiplier = 100000000
                    elif "천" in v_low and "원" in v_low:
                        multiplier = 1000
                    try:
                        features["available_amount"] = int(num * multiplier)
                    except Exception:
                        features["available_amount"] = int(float(num) * multiplier)

            # 대출기간 추출 (개월/년)
            if features["loan_term_months"] is None and (
                "개월" in v_low
                or "년" in v_low
                or any(tok in l_low for tok in ("기간", "term", "months", "years"))
            ):
                # 숫자+단위(예: 36개월, 3년)
                m = re.search(
                    r"(\d+(?:\.\d+)?)\s*(개월|개월|개월|월|년|yr|y|months|years)?",
                    v_low,
                )
                if m:
                    val = float(m.group(1))
                    unit = m.group(2) or ""
                    unit = unit.strip()
                    months = None
                    if unit in ("년", "y", "yr", "years"):
                        months = int(val * 12)
                    elif unit in ("개월", "월", "months"):
                        months = int(val)
                    else:
                        # 단위 없을 경우 가정: 숫자가 크면 개월로, 작으면 년으로 추정하지 않음
                        months = int(val)
                    features["loan_term_months"] = months
                    features["loan_term_raw"] = val_str

            # 금리 추출
            if features["applied_rate"] is None and (
                any(tok in l_low for tok in ("금리", "rate", "이율", "percent"))
                or "%" in v_low
            ):
                num = _parse_number(val_str)
                if num is not None:
                    features["applied_rate"] = float(num)

            # 신용등급/점수 추출
            if features["credit_grade"] is None and any(
                tok in l_low for tok in ("등급", "grade", "신용")
            ):
                # 등급 문자(A,B,C,S) 또는 숫자(300-900)
                g = re.search(r"\b([A-D][+-]?|S|[0-9]{3,4})\b", val_str, re.I)
                if g:
                    gval = g.group(1)
                    if gval.isdigit():
                        try:
                            features["credit_score"] = int(gval)
                        except Exception:
                            features["credit_score"] = float(gval)
                    else:
                        features["credit_grade"] = gval.upper()

            # 연소득/소득 추출
            if features["annual_income"] is None and any(
                tok in l_low for tok in ("소득", "연소득", "income", "salary")
            ):
                num = _parse_number(val_str)
                if num is not None:
                    multiplier = 1
                    if "만원" in v_low or (
                        "만" in v_low and re.search(r"\d+만", v_low)
                    ):
                        multiplier = 10000
                    elif "억" in v_low:
                        multiplier = 100000000
                    features["annual_income"] = int(num * multiplier)

            # 용도, 담보, 이자유형
            if features["purpose"] is None and any(
                tok in l_low for tok in ("용도", "purpose")
            ):
                features["purpose"] = val_str
            if features["collateral"] is None and any(
                tok in l_low for tok in ("담보", "collateral")
            ):
                features["collateral"] = val_str
            if features["interest_type"] is None and any(
                tok in l_low for tok in ("변동", "고정", "fixed", "variable")
            ):
                features["interest_type"] = val_str

            # KO 코드 탐지
            if re.match(r"^(K|KO)[0-9A-Za-z_\-]*$", str(k), re.I) or re.match(
                r"^(K|KO)[0-9A-Za-z_\-]*$", label, re.I
            ):
                features["ko_codes"].append(str(k))

            for m in re.findall(r"\b(KO?-?[0-9A-Za-z_]+)\b", val_str):
                features["ko_codes"].append(m)

        # dedupe
        features["ko_codes"] = list(dict.fromkeys(features["ko_codes"]))
        return features

    logger = ingest_logger
    # 로그
    for i, log in enumerate(logs):
        # 원본 로그 전체를 파일로 기록
        try:
            logger.info(
                "---- RAG INGEST: original log ----\n%s",
                json.dumps(log, ensure_ascii=False, indent=2),
            )
        except Exception:
            logger.info("---- RAG INGEST: original log ---- %s", str(log))

        print(f"📄 로그 처리 중... {i+1}/{len(logs)}")

        # 🔥 매핑 적용
        in_text = apply_mapping(
            log.get("in_fields", {}),
            log.get("in_mapping", {}),
        )

        out_text = apply_mapping(
            log.get("out_fields", {}),
            log.get("out_mapping", {}),
        )

        full_text = f"""
        [상품] {log.get("product")} [IN] {in_text} [OUT] {out_text}
        """

        print(f"변환된 로그:\n{full_text[:300]}")

        # feature 추출
        features = _extract_features(log)

        documents.append(
            Document(
                page_content=full_text[:2000],
                metadata={
                    "type": "log",
                    "product": log.get("product"),
                    "in_fields": log.get("in_fields", {}),
                    "out_fields": log.get("out_fields", {}),
                    "in_mapping": log.get("in_mapping", {}),
                    "out_mapping": log.get("out_mapping", {}),
                    "features": features,
                },
            )
        )

    # 뉴스
    for i, n in enumerate(news):
        # 원본 뉴스 아이템 전체 파일로 기록
        try:
            ingest_logger.info(
                "---- RAG INGEST: original news ----\n%s",
                json.dumps(n, ensure_ascii=False, indent=2),
            )
        except Exception:
            ingest_logger.info("---- RAG INGEST: original news ---- %s", str(n))

        print(f"📰 뉴스 처리 중... {i+1}/{len(news)}")

        title = n.get("title", "")
        content = (n.get("content") or n.get("summary") or "")[:1000]

        text = f"제목: {title}\n내용: {content}"

        documents.append(Document(page_content=text, metadata={"type": "news"}))

    print(f"✅ document 개수: {len(documents)}")

    # chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    print(f"🔥 총 chunk 수: {len(split_docs)}")

    print("🧠 embedding 시작")

    db = FAISS.from_documents(split_docs, get_embeddings())

    db.save_local("faiss_db")

    print(f"⏱️ 완료: {time.time() - start:.2f}초")


def load_db():
    global db

    if db is None:
        if os.path.exists("faiss_db"):
            print("📦 FAISS 로드")

            db = FAISS.load_local(
                "faiss_db",
                get_embeddings(),
                allow_dangerous_deserialization=True,  # 🔥 핵심
            )


def search_context(query, k=5):
    load_db()  # 🔥 추가

    docs = db.similarity_search(query, k=k)

    logs, news, rules = [], [], []

    for d in docs:
        if d.metadata["type"] == "log":
            # 메타데이터에 원본 필드/매핑이 있으면 사람이 읽기 좋은 형태로 재구성합니다.
            meta = d.metadata
            in_fields = meta.get("in_fields") or {}
            out_fields = meta.get("out_fields") or {}
            in_mapping = meta.get("in_mapping") or {}
            out_mapping = meta.get("out_mapping") or {}

            in_text = apply_mapping(in_fields, in_mapping)
            out_text = apply_mapping(out_fields, out_mapping)

            formatted = f"[상품] {meta.get('product')} [IN] {in_text} [OUT] {out_text}"
            logs.append(formatted)
        elif d.metadata["type"] == "news":
            news.append(d.page_content)
        else:
            rules.append(d.page_content)

    return logs, news, rules


def search_similar_logs(query):
    load_db()

    return db.similarity_search(query, k=3)


def get_vector_count():

    if not os.path.exists("faiss_db"):
        return 0

    db = FAISS.load_local(
        "faiss_db", get_embeddings(), allow_dangerous_deserialization=True
    )

    return len(db.index_to_docstore_id)


def save_agent_reports(reports):
    global db

    if not reports:
        return get_vector_count()

    if db is None:
        load_db()

    documents = []
    for report in reports:
        # 에이전트 리포트도 저장 전 원본을 파일로 기록합니다.
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
        documents.append(
            Document(
                page_content=f"제목: {title}\n내용: {content}",
                metadata={"type": "agent_report", "agent": agent_name},
            )
        )

    if not documents:
        return get_vector_count()

    if db is None:
        db = FAISS.from_documents(documents, get_embeddings())
    else:
        db.add_documents(documents)

    db.save_local("faiss_db")
    return len(db.index_to_docstore_id)


def ingest_files(
    files_data: list[tuple[str, bytes]], doc_type: str = "regulation"
) -> int:
    """
    files_data: list of (name, raw_bytes)
    Adds split chunks of provided files into the FAISS DB with metadata type `doc_type`.
    Returns number of vectors after ingest.
    """
    global db

    documents = []
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
                metadata={"type": doc_type, "source": "upload", "name": name},
            )
        )

    if not documents:
        return get_vector_count()

    # split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    if not split_docs:
        return get_vector_count()

    # ingest into or create FAISS
    if db is None:
        db = FAISS.from_documents(split_docs, get_embeddings())
    else:
        db.add_documents(split_docs)

    db.save_local("faiss_db")

    ingest_logger.info(
        "Ingested %d chunks for %d files", len(split_docs), len(files_data)
    )
    return len(db.index_to_docstore_id)


def list_vectors(limit: int = 200) -> list[dict]:
    """Return metadata summary for vectors stored in FAISS (limit recent items)."""
    try:
        if not os.path.exists("faiss_db"):
            return []
        local_db = FAISS.load_local(
            "faiss_db",
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        doc_map = getattr(local_db.docstore, "_dict", {})
        items = []
        for doc_id, doc in list(doc_map.items())[:limit]:
            meta = getattr(doc, "metadata", {}) or {}
            items.append(
                {
                    "id": doc_id,
                    "type": meta.get("type"),
                    "product": meta.get("product"),
                    "agent": meta.get("agent"),
                    "source": meta.get("source"),
                    "name": meta.get("name"),
                    "snippet": (getattr(doc, "page_content", "") or "")[:400],
                }
            )
        return items
    except Exception:
        return []
