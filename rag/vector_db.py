import os
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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

db=None
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

    # 로그
    for i, log in enumerate(logs):

        print(f"📄 로그 처리 중... {i+1}/{len(logs)}")

        # 🔥 매핑 적용
        in_text = apply_mapping(
        log.get("in_fields", {}),
        log.get("in_mapping", {})
         )

        out_text = apply_mapping(
        log.get("out_fields", {}),
        log.get("out_mapping", {})
        )

        full_text = f"""
        [상품] {log.get("product")} [IN] {in_text} [OUT] {out_text}
        """

        print(f"변환된 로그:\n{full_text[:300]}")

        documents.append(
            Document(
                page_content=full_text[:2000],
                metadata={
                "type": "log",
                "product": log.get("product")
            }
        )
    )

    # 뉴스
    for i, n in enumerate(news):
        print(f"📰 뉴스 처리 중... {i+1}/{len(news)}")

        title = n.get("title", "")
        content = (n.get("content") or n.get("summary") or "")[:1000]

        text = f"제목: {title}\n내용: {content}"

        documents.append(
            Document(
                page_content=text,
                metadata={"type": "news"}
            )
        )

    print(f"✅ document 개수: {len(documents)}")

    # chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

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
                allow_dangerous_deserialization=True   # 🔥 핵심
            )

def search_context(query, k=5):
    load_db()   # 🔥 추가

    docs = db.similarity_search(query, k=k)

    logs, news, rules = [], [], []

    for d in docs:
        if d.metadata["type"] == "log":
            logs.append(d.page_content)
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
        "faiss_db",
        get_embeddings(),
        allow_dangerous_deserialization=True
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
