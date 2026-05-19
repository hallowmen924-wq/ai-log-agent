from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document

try:
    from langchain.retrievers import EnsembleRetriever
except Exception:  # pragma: no cover
    EnsembleRetriever = None  # type: ignore[assignment]

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:  # pragma: no cover
    BM25Retriever = None  # type: ignore[assignment]


def build_faiss_retriever(local_db: Any, k: int = 10):
    if local_db is None:
        return None
    return local_db.as_retriever(search_kwargs={"k": max(1, int(k))})


def build_bm25_retriever(documents: list[Document], k: int = 10):
    if BM25Retriever is None or not documents:
        return None
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = max(1, int(k))
    return retriever


def build_ensemble_retriever(faiss_retriever: Any, bm25_retriever: Any):
    if EnsembleRetriever is None:
        return None
    retrievers = [item for item in [bm25_retriever, faiss_retriever] if item is not None]
    if not retrievers:
        return None
    if len(retrievers) == 1:
        return retrievers[0]
    return EnsembleRetriever(retrievers=retrievers, weights=[0.6, 0.4])


def _dedupe_key_for_doc(doc: Document) -> str:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source = str(metadata.get("source") or "")
    page = str(metadata.get("page") or "")
    article = str(metadata.get("article") or "")
    chunk_id = str(metadata.get("chunk_id") or metadata.get("chunk_index") or "")
    if source or page or article or chunk_id:
        return f"{source}|{page}|{article}|{chunk_id}"
    name = str(metadata.get("name") or metadata.get("document_name") or "")
    content = str(getattr(doc, "page_content", "") or "")
    hashed = hashlib.sha1(f"{name}|{content[:300]}".encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"fallback|{hashed}"


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    deduped: list[Document] = []
    for doc in documents:
        key = _dedupe_key_for_doc(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def optional_rerank_documents(documents: list[Document], query: str) -> list[Document]:
    # TODO: add local reranker or external reranker here when latency/cost budget allows.
    # 현재는 no-op으로 유지합니다.
    _ = query
    return list(documents)


def build_multi_query_retriever(base_retriever: Any):
    # TODO: MultiQueryRetriever(LMM query expansion) optional hook.
    # 이번 파이프라인에는 연결하지 않습니다.
    return base_retriever


def format_retrieval_debug_info(documents: list[Document], query: str) -> str:
    lines = [f"[regulation_retrieval] query={query!r} result_count={len(documents)}"]
    for index, doc in enumerate(documents, start=1):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        lines.append(
            f"  #{index} name={metadata.get('document_name') or metadata.get('name') or '-'} "
            f"article={metadata.get('article') or '-'} page={metadata.get('page') or '-'} "
            f"source={metadata.get('source') or '-'} chunk_id={metadata.get('chunk_id') or metadata.get('chunk_index') or '-'}"
        )
    return "\n".join(lines)


def retrieve_relevant_documents(
    *,
    query: str,
    local_db: Any,
    candidate_documents: list[Document],
    faiss_k: int = 10,
    bm25_k: int = 10,
    top_k: int = 6,
) -> list[Document]:
    faiss_retriever = build_faiss_retriever(local_db, k=faiss_k)
    bm25_retriever = build_bm25_retriever(candidate_documents, k=bm25_k)
    retriever = build_ensemble_retriever(faiss_retriever, bm25_retriever)
    if retriever is None:
        # Fallback when EnsembleRetriever is unavailable in current langchain package:
        # union FAISS/BM25 results and dedupe.
        docs: list[Document] = []
        for candidate in [bm25_retriever, faiss_retriever]:
            if candidate is None:
                continue
            current_docs = (
                candidate.invoke(query)
                if hasattr(candidate, "invoke")
                else candidate.get_relevant_documents(query)
            )
            docs.extend([doc for doc in current_docs if isinstance(doc, Document)])
    else:
        docs = (
            retriever.invoke(query)
            if hasattr(retriever, "invoke")
            else retriever.get_relevant_documents(query)
        )
    docs = [doc for doc in docs if isinstance(doc, Document)]
    docs = deduplicate_documents(docs)
    docs = optional_rerank_documents(docs, query)
    return docs[: max(1, int(top_k))]
