from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"
_INDEX_PATH = _DATA_DIR / "regulation_knowledge_index.json"

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_./%-]+")
_REJECT_CODE_PATTERN = re.compile(r"\bK\d{3}\b", flags=re.IGNORECASE)
_PRODUCT_CODE_PATTERN = re.compile(r"\bC\d{1,2}\b", flags=re.IGNORECASE)
_DATE_PATTERN = re.compile(r"((?:19|20)\d{2}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}(?:일)?)")

_POLICY_TAG_RULES: list[tuple[str, list[str]]] = [
    ("dsr", ["dsr", "총부채원리금상환비율", "스트레스dsr", "stress dsr"]),
    ("dti", ["dti"]),
    ("ltv", ["ltv", "담보인정비율"]),
    ("rate_reduction", ["금리인하요구권", "금리 인하 요구권", "rate reduction request"]),
    ("limit_policy", ["한도", "limit", "승인가능금액", "가능금액"]),
    ("interest_policy", ["금리", "이자율", "interest rate"]),
    ("underwriting", ["심사", "승인", "거절", "부결", "탈락", "reject", "approval"]),
    ("compliance", ["규제", "준수", "compliance", "정책", "내부rule", "rule"]),
]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _extract_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for matched in _TOKEN_PATTERN.findall(str(text or "").lower()):
        token = matched.strip()
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def _extract_date(text: str) -> str:
    matched = _DATE_PATTERN.search(str(text or ""))
    return str(matched.group(1)).strip() if matched else ""


def _detect_policy_tags(text: str) -> list[str]:
    lowered = str(text or "").lower()
    tags: list[str] = []
    for tag, cues in _POLICY_TAG_RULES:
        if any(cue.lower() in lowered for cue in cues):
            tags.append(tag)
    return tags


def _entry_key(entry: dict[str, Any]) -> str:
    return "|".join(
        [
            str(entry.get("document_name") or ""),
            str(entry.get("page") or ""),
            str(entry.get("article") or ""),
            str(entry.get("chunk_id") or ""),
        ]
    )


def _load_index() -> dict[str, Any]:
    if not _INDEX_PATH.exists():
        return {"updated_at": "", "entries": [], "stats": {}}
    try:
        payload = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": "", "entries": [], "stats": {}}
    if not isinstance(payload, dict):
        return {"updated_at": "", "entries": [], "stats": {}}
    payload.setdefault("entries", [])
    payload.setdefault("stats", {})
    payload.setdefault("updated_at", "")
    return payload


def _save_index(payload: dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    keyword_counter: dict[str, int] = {}
    tag_counter: dict[str, int] = {}
    reject_counter: dict[str, int] = {}
    product_counter: dict[str, int] = {}

    for item in entries:
        for token in (item.get("keywords") or [])[:24]:
            token_text = str(token).strip().lower()
            if not token_text:
                continue
            keyword_counter[token_text] = int(keyword_counter.get(token_text, 0)) + 1
        for tag in item.get("policy_tags") or []:
            tag_text = str(tag).strip()
            if not tag_text:
                continue
            tag_counter[tag_text] = int(tag_counter.get(tag_text, 0)) + 1
        for code in item.get("reject_codes") or []:
            code_text = str(code).strip().upper()
            if not code_text:
                continue
            reject_counter[code_text] = int(reject_counter.get(code_text, 0)) + 1
        for product in item.get("product_codes") or []:
            product_text = str(product).strip().upper()
            if not product_text:
                continue
            product_counter[product_text] = int(product_counter.get(product_text, 0)) + 1

    def _top(counter: dict[str, int], limit: int = 12) -> list[dict[str, Any]]:
        ranked = sorted(counter.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:limit]
        return [{"value": key, "count": int(value)} for key, value in ranked]

    return {
        "entry_count": int(len(entries)),
        "top_keywords": _top(keyword_counter),
        "top_policy_tags": _top(tag_counter),
        "top_reject_codes": _top(reject_counter),
        "top_product_codes": _top(product_counter),
    }


def build_knowledge_entries_from_documents(documents: list[Document]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for document in documents:
        metadata = dict(getattr(document, "metadata", {}) or {})
        text = _normalize_text(getattr(document, "page_content", "") or "")
        doc_name = str(metadata.get("document_name") or metadata.get("name") or "").strip()
        if not doc_name:
            continue
        page = int(metadata.get("page") or 0)
        article = str(metadata.get("article") or "").strip()
        effective_date = str(metadata.get("effective_date") or "").strip() or _extract_date(text)
        chunk_id = str(metadata.get("chunk_id") or metadata.get("chunk_index") or "").strip()
        reject_codes = sorted(
            {
                matched.group(0).upper()
                for matched in _REJECT_CODE_PATTERN.finditer(text)
            }
        )
        product_codes = sorted(
            {
                matched.group(0).upper()
                for matched in _PRODUCT_CODE_PATTERN.finditer(text)
            }
        )
        keywords = _extract_tokens(text)
        policy_tags = _detect_policy_tags(text)
        entry = {
            "document_name": doc_name,
            "page": page,
            "article": article,
            "effective_date": effective_date,
            "chunk_id": chunk_id,
            "reject_codes": reject_codes,
            "product_codes": product_codes,
            "policy_tags": policy_tags,
            "keywords": keywords[:80],
            "snippet": text[:320],
            "source": str(metadata.get("source") or "upload"),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        entries.append(entry)
    return entries


def update_regulation_knowledge_index(
    entries: list[dict[str, Any]],
    *,
    replaced_document_names: list[str] | None = None,
) -> dict[str, Any]:
    current = _load_index()
    current_entries = [
        item for item in list(current.get("entries") or [])
        if isinstance(item, dict)
    ]
    replaced_names = {
        str(name).strip()
        for name in (replaced_document_names or [])
        if str(name).strip()
    }
    if replaced_names:
        current_entries = [
            item
            for item in current_entries
            if str(item.get("document_name") or "").strip() not in replaced_names
        ]

    merged = current_entries + [item for item in entries if isinstance(item, dict)]

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in merged:
        key = _entry_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    payload = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "entries": deduped,
        "stats": _compute_stats(deduped),
    }
    _save_index(payload)
    return payload


def get_regulation_knowledge_index() -> dict[str, Any]:
    return _load_index()


def search_regulation_knowledge(query: str, k: int = 6) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query).lower()
    if not normalized_query:
        return []
    query_tokens = set(_extract_tokens(normalized_query))
    payload = _load_index()
    entries = list(payload.get("entries") or [])
    scored: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        keyword_set = {str(token).strip().lower() for token in (entry.get("keywords") or []) if str(token).strip()}
        tag_set = {str(tag).strip().lower() for tag in (entry.get("policy_tags") or []) if str(tag).strip()}
        reject_codes = {str(code).strip().upper() for code in (entry.get("reject_codes") or []) if str(code).strip()}
        product_codes = {str(code).strip().upper() for code in (entry.get("product_codes") or []) if str(code).strip()}

        token_hits = sorted(query_tokens & keyword_set)
        tag_hits = sorted(tag for tag in tag_set if tag in normalized_query)
        reject_hits = sorted(code for code in reject_codes if code.lower() in normalized_query)
        product_hits = sorted(code for code in product_codes if code.lower() in normalized_query)

        score = (
            float(len(token_hits) * 0.24)
            + float(len(tag_hits) * 0.52)
            + float(len(reject_hits) * 0.9)
            + float(len(product_hits) * 0.45)
        )
        if score <= 0:
            continue
        scored.append(
            {
                **entry,
                "knowledge_score": round(score, 4),
                "token_hits": token_hits[:10],
                "tag_hits": tag_hits[:6],
                "reject_hits": reject_hits[:6],
                "product_hits": product_hits[:6],
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item.get("knowledge_score") or 0.0),
            str(item.get("document_name") or ""),
            int(item.get("page") or 0),
        )
    )
    return scored[: max(1, int(k or 6))]


def summarize_regulation_knowledge(document_names: list[str] | None = None) -> dict[str, Any]:
    payload = _load_index()
    entries = [item for item in list(payload.get("entries") or []) if isinstance(item, dict)]
    doc_names = {str(name).strip() for name in (document_names or []) if str(name).strip()}
    if doc_names:
        entries = [item for item in entries if str(item.get("document_name") or "").strip() in doc_names]
    return {
        "updated_at": str(payload.get("updated_at") or ""),
        "document_count": len(
            {str(item.get("document_name") or "").strip() for item in entries if str(item.get("document_name") or "").strip()}
        ),
        "entry_count": len(entries),
        "stats": _compute_stats(entries),
    }
