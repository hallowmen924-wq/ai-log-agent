from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.documents import Document


NEWS_LIKE_TYPES = {"news", "agent_report_news"}
RULE_LIKE_TYPES = {"regulation", "rule", "agent_report_regulation"}


def build_news_documents(
    news_items: list[dict[str, Any]],
    logger,
    *,
    clean_text: Callable[[Any], str],
    store_name: str,
) -> list[Document]:
    documents: list[Document] = []
    for index, news_item in enumerate(news_items):
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
                metadata={"type": "news", "store": store_name},
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
