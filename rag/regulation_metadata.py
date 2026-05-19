from __future__ import annotations

import io
import re
from typing import Any

from langchain_core.documents import Document


_ARTICLE_PATTERN = re.compile(r"(제\s*\d+\s*조(?:의\s*\d+)?)")
_DATE_PATTERN = re.compile(r"((?:19|20)\d{2}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}(?:일)?)")


def _normalize_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    return text


def _extract_article(text: str) -> str:
    match = _ARTICLE_PATTERN.search(text or "")
    return str(match.group(1)).strip() if match else ""


def _extract_effective_date(text: str) -> str:
    match = _DATE_PATTERN.search(text or "")
    return str(match.group(1)).strip() if match else ""


def extract_text_pages(raw: bytes, file_name: str) -> list[dict[str, Any]]:
    try:
        decoded = raw.decode("utf-8")
        normalized = _normalize_text(decoded)
        if normalized:
            return [{"page": 1, "text": normalized}]
    except Exception:
        pass

    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        pages: list[dict[str, Any]] = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = _normalize_text(page.extract_text() or "")
            if page_text:
                pages.append({"page": index, "text": page_text})
        if pages:
            return pages
    except Exception:
        pass

    return [{"page": 1, "text": f"[파일 {file_name}의 텍스트 추출에 실패했습니다]"}]


def build_upload_documents(
    *,
    name: str,
    raw: bytes,
    doc_type: str,
    target_store: str,
) -> list[Document]:
    pages = extract_text_pages(raw, name)
    documents: list[Document] = []
    for page_item in pages:
        page = int(page_item.get("page") or 1)
        text = str(page_item.get("text") or "")
        article = _extract_article(text)
        effective_date = _extract_effective_date(text)
        documents.append(
            Document(
                page_content=f"제목: {name}\n페이지: {page}\n내용: {text}",
                metadata={
                    "type": doc_type,
                    "source": "upload",
                    "name": name,
                    "document_name": name,
                    "store": target_store,
                    "page": page,
                    "article": article,
                    "effective_date": effective_date,
                },
            )
        )
    return documents

