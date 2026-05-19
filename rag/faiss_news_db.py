from __future__ import annotations

import datetime
import hashlib
import json
import re
from urllib.parse import urlparse
from typing import Any, Callable

import numpy as np
from langchain_core.documents import Document


NEWS_LIKE_TYPES = {"news", "signal_news", "generated_news"}
RULE_LIKE_TYPES = {"regulation", "rule", "generated_regulation"}
NEWS_PIPELINE_KEYWORDS = (
    "카드론", "대출", "신용대출", "심사", "금리", "연체", "부실", "dsr", "규제",
    "여신", "가계대출", "금리인하요구권", "신상품", "한도",
)
LAST_NEWS_PIPELINE_STATS: dict[str, Any] = {}


def _normalize_news_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_news_link(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}".rstrip("/")


def _extract_publisher(news_item: dict[str, Any]) -> str:
    publisher = str(news_item.get("publisher") or "").strip()
    if publisher:
        return publisher
    link = str(news_item.get("link") or "")
    host = (urlparse(link).netloc or "").lower()
    return host.replace("www.", "")


def _extract_date(news_item: dict[str, Any]) -> str:
    raw = str(
        news_item.get("published_at")
        or news_item.get("published")
        or news_item.get("collected_at")
        or ""
    ).strip()
    if not raw:
        return ""
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
    except Exception:
        return raw


def _keyword_hits(text: str) -> list[str]:
    lowered = _normalize_news_text(text)
    return [keyword for keyword in NEWS_PIPELINE_KEYWORDS if keyword in lowered]


def _compute_low_cost_score(news_item: dict[str, Any]) -> float:
    title = str(news_item.get("title") or "")
    summary = str(news_item.get("summary") or "")
    merged = f"{title} {summary}".strip()
    hits = _keyword_hits(merged)
    score = min(1.0, len(hits) * 0.14)
    if len(_normalize_news_text(title)) >= 14:
        score += 0.08
    if "카드론" in _normalize_news_text(merged):
        score += 0.2
    return min(1.0, score)


def _safe_embed_texts(embeddings: Any, texts: list[str]) -> np.ndarray | None:
    if embeddings is None or not texts:
        return None
    try:
        vectors = embeddings.embed_documents(texts)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
    except Exception:
        return None


def run_news_pipeline(
    news_items: list[dict[str, Any]],
    *,
    embeddings: Any = None,
    existing_news_docs: list[Document] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_count = len(news_items)
    stage1_items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for news_item in news_items:
        title = str(news_item.get("title") or "").strip()
        summary = str(news_item.get("summary") or "").strip()
        link = _normalize_news_link(news_item.get("link") or "")
        if not title or not link:
            continue
        dedupe_key = hashlib.sha1(f"{title.lower()}|{link}".encode("utf-8", errors="ignore")).hexdigest()
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        normalized = dict(news_item)
        normalized["title"] = title
        normalized["summary"] = summary
        normalized["link"] = str(news_item.get("link") or "").strip()
        normalized["publisher"] = _extract_publisher(news_item)
        normalized["published_at"] = _extract_date(news_item)
        normalized["industry"] = "loan"
        normalized["base_keywords"] = _keyword_hits(f"{title} {summary}")
        normalized["low_cost_score"] = _compute_low_cost_score(normalized)
        stage1_items.append(normalized)

    # low-cost filter
    stage2_items = [item for item in stage1_items if float(item.get("low_cost_score") or 0.0) >= 0.16]
    if not stage2_items:
        return [], {
            "raw_count": raw_count,
            "stage1_count": len(stage1_items),
            "stage2_count": 0,
            "stage3_count": 0,
            "llm_candidate_count": 0,
        }

    # cosine duplicate reduction (within batch)
    stage2_texts = [
        _normalize_news_text(f"{item.get('title', '')} {item.get('summary', '')}")
        for item in stage2_items
    ]
    stage2_matrix = _safe_embed_texts(embeddings, stage2_texts)
    stage2_deduped: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    if stage2_matrix is not None:
        for index, item in enumerate(stage2_items):
            is_duplicate = False
            for selected_index in selected_indices:
                sim = float(stage2_matrix[index] @ stage2_matrix[selected_index])
                if sim >= 0.95:
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected_indices.append(index)
                stage2_deduped.append(item)
    else:
        stage2_deduped = stage2_items

    # medium filter (embedding relevance + cluster + existing important similarity)
    stage3_items = list(stage2_deduped)
    stage3_texts = [
        _normalize_news_text(f"{item.get('title', '')} {item.get('summary', '')} {item.get('content', '')[:600]}")
        for item in stage3_items
    ]
    stage3_matrix = _safe_embed_texts(embeddings, stage3_texts)
    anchor_texts = [
        "카드론 심사 영향 규제 금리 연체 리스크",
        "신상품 개발 영향 출시 전략 한도 금리",
    ]
    anchor_matrix = _safe_embed_texts(embeddings, anchor_texts)

    important_vectors: np.ndarray | None = None
    if embeddings is not None and existing_news_docs:
        important_texts: list[str] = []
        for doc in existing_news_docs:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            if float(metadata.get("importance_score") or 0.0) < 0.75:
                continue
            important_texts.append(
                _normalize_news_text(
                    f"{metadata.get('title', '')} {str(getattr(doc, 'page_content', '') or '')[:400]}"
                )
            )
            if len(important_texts) >= 24:
                break
        important_vectors = _safe_embed_texts(embeddings, important_texts) if important_texts else None

    for index, item in enumerate(stage3_items):
        embed_relevance = 0.0
        if stage3_matrix is not None and anchor_matrix is not None:
            embed_relevance = float(max(stage3_matrix[index] @ anchor_matrix[0], stage3_matrix[index] @ anchor_matrix[1]))
        cluster_id = index
        if stage3_matrix is not None:
            for prev in range(index):
                if float(stage3_matrix[index] @ stage3_matrix[prev]) >= 0.86:
                    cluster_id = int(stage3_items[prev].get("cluster_id") or prev)
                    break
        existing_similarity = 0.0
        if stage3_matrix is not None and important_vectors is not None and len(important_vectors):
            existing_similarity = float(np.max(important_vectors @ stage3_matrix[index]))
        low_score = float(item.get("low_cost_score") or 0.0)
        importance_score = max(0.0, min(1.0, (low_score * 0.45) + (embed_relevance * 0.4) + (existing_similarity * 0.15)))
        item["embed_relevance"] = round(embed_relevance, 4)
        item["cluster_id"] = int(cluster_id)
        item["existing_similarity"] = round(existing_similarity, 4)
        item["importance_score"] = round(importance_score, 4)
        item["impact_cardloan"] = "high" if embed_relevance >= 0.52 else ("medium" if embed_relevance >= 0.36 else "low")
        item["impact_product"] = "high" if embed_relevance >= 0.58 else ("medium" if embed_relevance >= 0.4 else "low")
        item["impact_direction"] = "neutral"
        item["llm_candidate"] = bool(importance_score >= 0.72 or (0.45 <= importance_score <= 0.56))
        snippet = str(item.get("content") or item.get("summary") or "").strip()
        item["evidence_sentence"] = snippet[:240]

    return stage3_items, {
        "raw_count": raw_count,
        "stage1_count": len(stage1_items),
        "stage2_count": len(stage2_deduped),
        "stage3_count": len(stage3_items),
        "llm_candidate_count": sum(1 for item in stage3_items if item.get("llm_candidate")),
    }


def get_last_news_pipeline_stats() -> dict[str, Any]:
    return dict(LAST_NEWS_PIPELINE_STATS or {})


def build_news_documents(
    news_items: list[dict[str, Any]],
    logger,
    *,
    clean_text: Callable[[Any], str],
    store_name: str,
    embeddings: Any = None,
    existing_news_docs: list[Document] | None = None,
) -> list[Document]:
    filtered_items, stats = run_news_pipeline(
        news_items,
        embeddings=embeddings,
        existing_news_docs=existing_news_docs,
    )
    global LAST_NEWS_PIPELINE_STATS
    LAST_NEWS_PIPELINE_STATS = dict(stats or {})
    try:
        logger.info("news pipeline stats: %s", json.dumps(stats, ensure_ascii=False))
    except Exception:
        pass
    documents: list[Document] = []
    for index, news_item in enumerate(filtered_items):
        try:
            logger.info(
                "---- RAG INGEST: original news ----\n%s",
                json.dumps(news_item, ensure_ascii=False, indent=2),
            )
        except Exception:
            logger.info("---- RAG INGEST: original news ---- %s", str(news_item))

        print(f"뉴스 처리 중... {index + 1}/{len(news_items)}")
        title = news_item.get("title", "")
        content = (news_item.get("content") or news_item.get("summary") or "")[:1000]
        text = clean_text(f"제목: {title} 내용: {content}")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "type": "news",
                    "store": store_name,
                    "title": str(title).strip(),
                    "link": str(news_item.get("link", "")).strip(),
                    "source": str(news_item.get("publisher") or ""),
                    "publisher": str(news_item.get("publisher") or ""),
                    "published_at": str(news_item.get("published_at") or ""),
                    "industry": str(news_item.get("industry") or "loan"),
                    "keywords": "|".join(news_item.get("base_keywords") or []),
                    "low_cost_score": float(news_item.get("low_cost_score") or 0.0),
                    "embed_relevance": float(news_item.get("embed_relevance") or 0.0),
                    "cluster_id": int(news_item.get("cluster_id") or 0),
                    "existing_similarity": float(news_item.get("existing_similarity") or 0.0),
                    "importance_score": float(news_item.get("importance_score") or 0.0),
                    "impact_cardloan": str(news_item.get("impact_cardloan") or ""),
                    "impact_product": str(news_item.get("impact_product") or ""),
                    "impact_direction": str(news_item.get("impact_direction") or "neutral"),
                    "evidence_sentence": str(news_item.get("evidence_sentence") or ""),
                    "llm_candidate": bool(news_item.get("llm_candidate")),
                },
            )
        )
    return documents


def split_news_search_results(docs: list[Document]) -> tuple[list[str], list[str]]:
    news_items: list[str] = []
    rule_items: list[str] = []
    for doc in docs:
        doc_type = str((getattr(doc, "metadata", {}) or {}).get("type") or "").strip().lower()
        if doc_type in NEWS_LIKE_TYPES:
            news_items.append(getattr(doc, "page_content", ""))
        else:
            rule_items.append(getattr(doc, "page_content", ""))
    return news_items, rule_items
