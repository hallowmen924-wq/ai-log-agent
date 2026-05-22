from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Callable

import requests

from mapper.reject_code_mapper import format_reject_reason_details
from rag.product_pattern_summary import (
    DEFAULT_SUMMARY_PATH,
    format_product_pattern_summary_for_prompt,
    load_product_pattern_summary,
)
from rag.vector_db import (
    get_vector_count,
    save_generated_documents,
    search_news_context,
    search_similar_logs,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_LIGHTWEIGHT_MODEL = os.environ.get("OLLAMA_LIGHTWEIGHT_MODEL", "mistral")
OLLAMA_EXECUTION_LOCK = threading.RLock()
OLLAMA_RUNTIME_STATE_LOCK = threading.RLock()
OLLAMA_GPU_ENABLED = True
ONTOLOGY_QUERY_PRIORITY_ENABLED = True
_PENDING_ONTOLOGY_REQUESTS = 0
_PENDING_PRODUCT_DEBATE_REQUESTS = 0

DEFAULT_LOG_AGENT_PROMPT_TEMPLATE = """당신은 금융 심사 로그 분석가입니다.

[상품별 승인/거절 패턴 요약]
{product_pattern_summary}

출력 형식:
1) 심사시스템 상태: 정상/과부하/지연/오류 등
2) 이상 징후: 한도/금리/승인 여부 중심으로 1~2문장
3) 조치: 1~2개
""".strip()

DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE = """너는 금융 뉴스 분석 AI다.

다음 뉴스에서 "대출 심사 영향 신호"를 추출하라.

[규칙]
- 반드시 JSON만 출력
- 설명 금지
- 항목 수 제한 준수

[출력 형식]
{
  "태그": [],
  "요약": "",
  "위험신호": [],
  "검색텍스트": "",
  "영향점수": 0,        // 1~5 (심사 영향도)
  "긴급점수": 0,       // 1~5 (심사시스템 단기 영향 여부)
  "비즈니스점수": 0       // 1~5 (신상품 기회 또는 수익 영향)
}

[평가 기준]
- 영향점수: 심사 기준 변화 가능성
- 긴급점수: 심사시스템 단기 영향 여부
- 비즈니스점수: 신상품 기회 또는 수익 영향

뉴스:
{news_text}""".strip()


def get_ollama_runtime_preferences() -> dict[str, bool]:
    with OLLAMA_RUNTIME_STATE_LOCK:
        return {
            "ollama_gpu_enabled": bool(OLLAMA_GPU_ENABLED),
            "ontology_query_priority_enabled": bool(ONTOLOGY_QUERY_PRIORITY_ENABLED),
        }


def set_ollama_gpu_enabled(enabled: bool) -> dict[str, bool]:
    global OLLAMA_GPU_ENABLED
    with OLLAMA_RUNTIME_STATE_LOCK:
        OLLAMA_GPU_ENABLED = bool(enabled)
        return get_ollama_runtime_preferences()


def set_ontology_query_priority_enabled(enabled: bool) -> dict[str, bool]:
    global ONTOLOGY_QUERY_PRIORITY_ENABLED
    with OLLAMA_RUNTIME_STATE_LOCK:
        ONTOLOGY_QUERY_PRIORITY_ENABLED = bool(enabled)
        return get_ollama_runtime_preferences()


def _begin_ontology_priority_scope(priority_group: str) -> bool:
    global _PENDING_ONTOLOGY_REQUESTS, _PENDING_PRODUCT_DEBATE_REQUESTS
    with OLLAMA_RUNTIME_STATE_LOCK:
        if priority_group == "product_debate":
            _PENDING_PRODUCT_DEBATE_REQUESTS += 1
            return True
        if priority_group != "ontology" or not ONTOLOGY_QUERY_PRIORITY_ENABLED:
            return False
        _PENDING_ONTOLOGY_REQUESTS += 1
        return True


def _end_ontology_priority_scope(enabled: bool) -> None:
    global _PENDING_ONTOLOGY_REQUESTS, _PENDING_PRODUCT_DEBATE_REQUESTS
    if not enabled:
        return
    with OLLAMA_RUNTIME_STATE_LOCK:
        if _PENDING_PRODUCT_DEBATE_REQUESTS > 0:
            _PENDING_PRODUCT_DEBATE_REQUESTS = max(0, _PENDING_PRODUCT_DEBATE_REQUESTS - 1)
        else:
            _PENDING_ONTOLOGY_REQUESTS = max(0, _PENDING_ONTOLOGY_REQUESTS - 1)


def _has_pending_ontology_priority() -> bool:
    with OLLAMA_RUNTIME_STATE_LOCK:
        return bool(ONTOLOGY_QUERY_PRIORITY_ENABLED and _PENDING_ONTOLOGY_REQUESTS > 0)


def _has_pending_product_debate_priority() -> bool:
    with OLLAMA_RUNTIME_STATE_LOCK:
        return _PENDING_PRODUCT_DEBATE_REQUESTS > 0


def _build_lightweight_ollama_options() -> dict[str, Any]:
    num_ctx = int(os.environ.get("OLLAMA_LIGHTWEIGHT_NUM_CTX", "1536") or 1536)
    num_predict = int(os.environ.get("OLLAMA_LIGHTWEIGHT_NUM_PREDICT", "180") or 180)
    temperature = float(os.environ.get("OLLAMA_LIGHTWEIGHT_TEMPERATURE", "0.1") or 0.1)
    options: dict[str, Any] = {
        "num_ctx": max(512, num_ctx),
        "num_predict": max(64, num_predict),
        "temperature": max(0.0, min(1.0, temperature)),
    }
    with OLLAMA_RUNTIME_STATE_LOCK:
        gpu_enabled = bool(OLLAMA_GPU_ENABLED)
    if not gpu_enabled:
        options["num_gpu"] = 0
    return options


class OllamaUnavailableError(RuntimeError):
    pass


OllamaProgressCallback = Callable[[str, dict[str, Any]], None]


def _build_ollama_unavailable_message() -> str:
    return (
        "Ollama 서버에 연결할 수 없습니다. "
        "Ollama가 실행 중인지 확인하고, 모델이 준비된 뒤 다시 시도하세요. "
        f"대상 주소: {OLLAMA_URL}"
    )


def _ollama_generate(
    model: str,
    prompt: str,
    *,
    options: dict[str, Any] | None = None,
    progress_callback: OllamaProgressCallback | None = None,
    timeout_seconds: int | float = 180,
    fail_fast_if_busy: bool = False,
    priority_group: str = "default",
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    if options:
        payload["options"] = options

    if progress_callback is not None:
        progress_callback("start", {"model": model, "prompt": prompt})

    parts: list[str] = []
    lock_acquired = False
    busy_message = (
        "[lock-busy] Ollama가 다른 작업을 처리 중입니다. "
        "workbench 응답은 fallback summary 로 먼저 표시하고, 잠시 뒤 다시 시도하세요."
    )
    priority_message = "[priority-busy] 온톨로지 질의가 우선 처리 중입니다. 현재 요청은 잠시 뒤 다시 시도하세요."
    product_debate_priority_message = "[debate-busy] 상품개발 토론 결론 생성이 우선 처리 중입니다. 잠시 뒤 다시 시도하세요."
    priority_scope_enabled = _begin_ontology_priority_scope(priority_group)
    try:
        if priority_group != "product_debate" and _has_pending_product_debate_priority():
            if progress_callback is not None:
                progress_callback(
                    "failed",
                    {"model": model, "error": product_debate_priority_message},
                )
            raise OllamaUnavailableError(product_debate_priority_message)
        if priority_group != "ontology" and _has_pending_ontology_priority():
            if progress_callback is not None:
                progress_callback(
                    "failed",
                    {"model": model, "error": priority_message},
                )
            raise OllamaUnavailableError(priority_message)

        if fail_fast_if_busy:
            lock_acquired = OLLAMA_EXECUTION_LOCK.acquire(blocking=False)
            if not lock_acquired:
                if progress_callback is not None:
                    progress_callback(
                        "failed",
                        {"model": model, "error": busy_message},
                    )
                raise OllamaUnavailableError(busy_message)
        else:
            OLLAMA_EXECUTION_LOCK.acquire()
            lock_acquired = True

        try:
            with requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout_seconds,
                stream=True,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    try:
                        chunk_payload = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    error_message = str(chunk_payload.get("error") or "").strip()
                    if error_message:
                        raise RuntimeError(error_message)

                    chunk_text = str(chunk_payload.get("response") or "")
                    if chunk_text:
                        parts.append(chunk_text)
                        if progress_callback is not None:
                            progress_callback(
                                "chunk",
                                {
                                    "model": model,
                                    "chunk": chunk_text,
                                    "text": "".join(parts),
                                },
                            )

                    if chunk_payload.get("done"):
                        break
        finally:
            if lock_acquired:
                OLLAMA_EXECUTION_LOCK.release()
    except requests.Timeout as error:
        timeout_message = (
            "[timeout] Ollama 응답 지연(timeout)으로 요청이 종료되었습니다. "
            "모델 추론 시간이 길어졌거나 현재 부하가 높습니다."
        )
        if progress_callback is not None:
            progress_callback(
                "failed",
                {"model": model, "error": timeout_message},
            )
        raise OllamaUnavailableError(timeout_message) from error
    except requests.ConnectionError as error:
        connection_message = f"[connection] {_build_ollama_unavailable_message()}"
        if progress_callback is not None:
            progress_callback(
                "failed",
                {"model": model, "error": connection_message},
            )
        raise OllamaUnavailableError(connection_message) from error
    except requests.HTTPError as error:
        http_status = int(getattr(getattr(error, "response", None), "status_code", 0) or 0)
        http_message = f"[http-error] Ollama HTTP 오류가 발생했습니다. (status={http_status})"
        if progress_callback is not None:
            progress_callback(
                "failed",
                {"model": model, "error": http_message},
            )
        raise OllamaUnavailableError(http_message) from error
    except requests.RequestException as error:
        unknown_request_message = (
            "[request-error] Ollama 요청 중 네트워크 오류가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )
        if progress_callback is not None:
            progress_callback(
                "failed",
                {"model": model, "error": unknown_request_message},
            )
        raise OllamaUnavailableError(unknown_request_message) from error
    except Exception as error:
        if progress_callback is not None:
            progress_callback(
                "failed",
                {"model": model, "error": str(error)},
            )
        raise
    finally:
        _end_ontology_priority_scope(priority_scope_enabled)

    answer = "".join(parts).strip()
    if not answer:
        if progress_callback is not None:
            progress_callback(
                "failed",
                {"model": model, "error": "Ollama 응답 본문이 비어 있습니다."},
            )
        raise RuntimeError("Ollama 응답 본문이 비어 있습니다.")

    if progress_callback is not None:
        progress_callback("completed", {"model": model, "text": answer})
    return answer


def mistral_generate(
    prompt: str,
    progress_callback: OllamaProgressCallback | None = None,
    timeout_seconds: int | float = 180,
    fail_fast_if_busy: bool = False,
    priority_group: str = "default",
) -> str:
    return _ollama_generate(
        "mistral",
        prompt,
        progress_callback=progress_callback,
        timeout_seconds=timeout_seconds,
        fail_fast_if_busy=fail_fast_if_busy,
        priority_group=priority_group,
    )


def lightweight_ollama_generate(
    prompt: str,
    progress_callback: OllamaProgressCallback | None = None,
    timeout_seconds: int | float = 180,
    fail_fast_if_busy: bool = False,
    priority_group: str = "default",
) -> str:
    return _ollama_generate(
        OLLAMA_LIGHTWEIGHT_MODEL,
        prompt,
        options=_build_lightweight_ollama_options(),
        progress_callback=progress_callback,
        timeout_seconds=timeout_seconds,
        fail_fast_if_busy=fail_fast_if_busy,
        priority_group=priority_group,
    )


def emit_agent_event(
    event_callback: Callable[[str, str, str], None] | None,
    agent: str,
    status: str,
    detail: str,
) -> None:
    if event_callback is not None:
        event_callback(agent, status, detail)


def emit_vector_event(
    vector_callback: Callable[[str, str, int, int, str], None] | None,
    source: str,
    action: str,
    before_count: int,
    after_count: int,
    detail: str,
) -> None:
    if vector_callback is not None:
        vector_callback(source, action, before_count, after_count, detail)


def trim_context(items: list[str], limit: int = 3) -> str:
    if not items:
        return "관련 데이터가 없습니다."
    return "\n\n".join(items[:limit])


def map_field_items(
    fields: dict[str, Any], mapping: dict[str, str], limit: int = 3
) -> list[str]:
    if not fields:
        return []
    items = []
    for key, value in list(fields.items())[:limit]:
        label = str(mapping.get(key, key))
        items.append(f"{label}={value}")
    return items


def map_all_field_items(fields: dict[str, Any], mapping: dict[str, str]) -> list[str]:
    if not fields:
        return []
    items = []
    for key, value in fields.items():
        label = str(mapping.get(key, key))
        items.append(f"{label}={value}")
    return items


INPUT_PRIORITY_GROUPS = [
    ["나이", "연령", "age"],
    ["등급", "신용등급", "grade", "rating"],
    ["대출잔액", "잔액", "여신잔액", "loan balance", "balance"],
    ["외국인", "국적", "foreign", "foreigner"],
    ["신용카드잔액", "카드잔액", "카드론잔액", "card balance"],
    ["연체고객", "연체", "delinquent", "overdue"],
]


OUTPUT_PRIORITY_GROUPS = [
    ["한도", "대출가능금액", "가능금액", "limit", "available amount"],
    ["금리", "이율", "rate", "interest"],
    ["승인", "거절", "승인여부", "approval", "decision"],
    ["최종대출가능금액", "최종한도", "최종 가능 금액", "final available amount"],
    ["추정소득", "소득추정", "estimated income", "income"],
]


def pick_ordered_field_items(
    fields: dict[str, Any],
    mapping: dict[str, str],
    priority_groups: list[list[str]],
    limit: int,
) -> list[str]:
    if not fields:
        return []

    entries: list[tuple[str, str, str]] = []
    for key, value in fields.items():
        label = str(mapping.get(key, key))
        entries.append((str(key), label, f"{label}={value}"))

    picked: list[str] = []
    used_texts: set[str] = set()

    for group in priority_groups:
        group_lower = [token.lower() for token in group]
        for key, label, rendered in entries:
            haystacks = [key.lower(), label.lower()]
            if rendered in used_texts:
                continue
            if any(token in hay for token in group_lower for hay in haystacks):
                picked.append(rendered)
                used_texts.add(rendered)
                break
        if len(picked) >= limit:
            return picked[:limit]

    for _, _, rendered in entries:
        if rendered in used_texts:
            continue
        picked.append(rendered)
        if len(picked) >= limit:
            break
    return picked[:limit]


def pick_priority_field_items(
    fields: dict[str, Any], mapping: dict[str, str], limit: int = 3
) -> list[str]:
    if not fields:
        return []
    priority_keywords = [
        "한도",
        "대출",
        "금리",
        "이율",
        "승인",
        "거절",
        "신용",
        "점수",
        "등급",
        "소득",
        "dsr",
    ]
    scored: list[tuple[int, str]] = []
    for key, value in fields.items():
        label = str(mapping.get(key, key))
        score = 0
        lower_label = label.lower()
        for idx, keyword in enumerate(priority_keywords, start=1):
            if keyword.lower() in lower_label:
                score = max(score, len(priority_keywords) - idx + 1)
        scored.append((score, f"{label}={value}"))
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked = [item for _, item in scored[:limit] if item]
    if picked:
        return picked
    return map_field_items(fields, mapping, limit=limit)


def pick_representative_input_items(
    fields: dict[str, Any], mapping: dict[str, str], limit: int = 6
) -> list[str]:
    return pick_ordered_field_items(
        fields,
        mapping,
        priority_groups=INPUT_PRIORITY_GROUPS,
        limit=limit,
    )


def pick_representative_output_items(
    fields: dict[str, Any], mapping: dict[str, str], limit: int = 5
) -> list[str]:
    return pick_ordered_field_items(
        fields,
        mapping,
        priority_groups=OUTPUT_PRIORITY_GROUPS,
        limit=limit,
    )


def trim_news_items(news_items: list[dict[str, Any]], limit: int = 1) -> str:
    if not news_items:
        return "관련 데이터가 없습니다."

    crawled_items = [
        item
        for item in news_items
        if str(item.get("content", "")).strip()
        or str(item.get("summary", "")).strip()
        or str(item.get("title", "")).strip()
    ]
    selected_items = crawled_items[:limit]

    if not selected_items:
        return "관련 데이터가 없습니다. 아직 기사 요약이 준비되지 않았습니다."

    snippets = []
    for item in selected_items:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        content = str(item.get("content", "")).strip()
        body = summary or content
        compact_body = " ".join(body.split())[:280]
        snippets.append(f"제목: {title}\n요약: {compact_body}")
    return "\n\n".join(snippets)


def trim_log_results(log_items: list[dict[str, Any]], limit: int = 1) -> str:
    if not log_items:
        return "관련 데이터가 없습니다."

    snippets = []
    for item in log_items[:limit]:
        product = str(item.get("product", "N/A"))
        in_fields = item.get("in_fields", {}) or {}
        out_fields = item.get("out_fields", {}) or {}
        in_mapping = item.get("in_mapping", {}) or {}
        out_mapping = item.get("out_mapping", {}) or {}
        reject_reason_details = item.get("reject_reason_details", []) or []

        rep_in_items = map_all_field_items(in_fields, in_mapping)
        rep_out_items = map_all_field_items(out_fields, out_mapping)
        reject_items = format_reject_reason_details(reject_reason_details, limit=5)

        snippets.append(
            f"상품: {product}\n"
            f"대표 입력: {', '.join(rep_in_items) or '-'}\n"
            f"대표 출력: {', '.join(rep_out_items) or '-'}\n"
            f"거절 사유: {', '.join(reject_items) or '-'}"
        )
    return "\n\n".join(snippets)


PRODUCT_MAP = {
    "C6": "신용대출",
    "C9": "카드론",
    "C11": "개인사업자대출",
    "C12": "대환대출",
}


def group_logs_by_product(log_items: list[dict[str, Any]] | list[str], limit_per_product: int = 1) -> str:
    """상품 코드별로 로그를 묶어 프롬프트에 들어갈 텍스트를 생성합니다.

    각 상품 섹션은 대표 케이스를 최대 `limit_per_product`개까지 보여주고,
    가능하면 한도/금리/승인 관련 필드를 요약해서 표시합니다.
    """
    if not log_items:
        return "관련 데이터가 없습니다."

    if not isinstance(log_items[0], dict):
        return trim_context([str(item) for item in log_items], limit=1)

    by_prod: dict[str, list[dict[str, Any]]] = {}
    for it in log_items:
        prod = str(it.get("product") or it.get("product_code") or "UNKNOWN")
        by_prod.setdefault(prod, []).append(it)

    sections = []
    for prod, items in by_prod.items():
        prod_name = PRODUCT_MAP.get(prod, prod)
        header = f"[상품 코드: {prod}] {prod_name} — {len(items)}건"
        parts = [header]
        for i, item in enumerate(items[:limit_per_product], start=1):
            in_fields = item.get("in_fields", {}) or {}
            out_fields = item.get("out_fields", {}) or {}
            in_mapping = item.get("in_mapping", {}) or {}
            out_mapping = item.get("out_mapping", {}) or {}
            reject_reason_details = item.get("reject_reason_details", []) or []

            # 한도, 금리, 승인 여부 추출 시도 — 매핑된(사람 읽기) 키와 영어 키 모두 검사
            limit_val = (
                in_fields.get("limit")
                or in_fields.get("amount")
                or in_fields.get("한도")
                or in_fields.get("대출금액")
                or out_fields.get("limit")
                or out_fields.get("amount")
                or out_fields.get("한도")
                or "-"
            )
            rate_val = (
                in_fields.get("rate")
                or in_fields.get("interest")
                or in_fields.get("금리")
                or in_fields.get("이율")
                or out_fields.get("rate")
                or out_fields.get("interest")
                or out_fields.get("금리")
                or "-"
            )
            approve_val = (
                item.get("decision")
                or item.get("approval")
                or in_fields.get("approval")
                or in_fields.get("승인")
                or out_fields.get("approval")
                or out_fields.get("승인")
                or "-"
            )

            mapped_in_snippet = ", ".join(map_all_field_items(in_fields, in_mapping))
            mapped_out_snippet = ", ".join(map_all_field_items(out_fields, out_mapping))
            reject_reason_snippet = ", ".join(
                format_reject_reason_details(reject_reason_details, limit=5)
            )
            short = (
                f"  {i}. case_id={item.get('case_id','-')} limit={limit_val} rate={rate_val} approval={approve_val}\n"
                f"     입력필드={mapped_in_snippet or '-'}\n"
                f"     출력필드={mapped_out_snippet or '-'}\n"
                f"     거절사유={reject_reason_snippet or '-'}"
            )
            parts.append(short)
        sections.append("\n".join(parts))

    return "\n\n".join(sections)


def build_log_context_from_similar_cases(query: str, k: int = 1) -> str:
    try:
        docs = search_similar_logs(query)
    except Exception:
        docs = []

    if not docs:
        return "관련 데이터가 없습니다."

    structured_items: list[dict[str, Any]] = []
    for doc in docs[:k]:
        meta = getattr(doc, "metadata", {}) or {}
        structured_items.append(
            {
                "product": meta.get("product") or meta.get("product_code") or "UNKNOWN",
                "in_fields": meta.get("in_fields") or {},
                "out_fields": meta.get("out_fields") or {},
                "reject_reason_details": meta.get("reject_reason_details") or [],
                # FAISS metadata is already stored with mapped Korean labels.
                "in_mapping": {},
                "out_mapping": {},
                "case_id": (meta.get("features") or {}).get("case_id", "-"),
            }
        )
    return group_logs_by_product(structured_items, limit_per_product=1)


def extract_product_codes_from_log_context(log_context: str) -> list[str]:
    matches = re.findall(r"\[상품 코드: (C\d+)\]", str(log_context or ""), flags=re.I)
    ordered: list[str] = []
    seen: set[str] = set()
    for match in matches:
        product_code = str(match).upper()
        if product_code in seen:
            continue
        seen.add(product_code)
        ordered.append(product_code)
    return ordered


def build_product_pattern_context(log_context: str) -> str:
    summary = load_product_pattern_summary(DEFAULT_SUMMARY_PATH)
    product_codes = extract_product_codes_from_log_context(log_context)
    return format_product_pattern_summary_for_prompt(
        summary,
        product_codes=product_codes,
    )


def build_log_agent_prompt(
    log_context: str,
    user_input: str,
    prompt_template: str | None = None,
    similar_cases_text: str | None = None,
    product_pattern_context: str | None = None,
) -> str:
    summary_context = str(
        product_pattern_context
        if product_pattern_context is not None
        else build_product_pattern_context(log_context)
    ).strip()
    template = str(prompt_template or DEFAULT_LOG_AGENT_PROMPT_TEMPLATE).strip()
    prompt = template.replace("{user_input}", user_input)
    prompt = prompt.replace("{product_pattern_summary}", summary_context)
    prompt = prompt.replace("{log_text}", log_context)
    prompt = prompt.replace(
        "{similar_cases}",
        str(similar_cases_text or "(유사 사례는 실행 시점에 자동 주입됩니다.)"),
    )
    return prompt.strip()


def build_news_agent_prompt(
    news_context: str,
    user_input: str,
    prompt_template: str | None = None,
) -> str:
    template = str(prompt_template or DEFAULT_NEWS_AGENT_PROMPT_TEMPLATE).strip()
    prompt = template.replace("{news_text}", news_context)
    prompt = prompt.replace("{user_input}", user_input)
    return prompt.strip()


def build_news_fallback_briefing(news_items: list[dict[str, Any]]) -> str:
    headlines: list[str] = []
    for item in news_items[:3]:
        title = str(item.get("title") or item.get("summary") or "").strip()
        if title:
            headlines.append(title)

    headline_text = ", ".join(headlines) if headlines else "수집된 주요 기사 제목 없음"
    return (
        "{\n"
        '  "tags": ["뉴스수집", "수동검토"],\n'
        '  "signal_summary": "Ollama 연결 실패로 자동 신호 추출을 완료하지 못했습니다.",\n'
        '  "risk_signal": ["자동 뉴스 신호 추출 실패", "수동 검토 필요"],\n'
        '  "opportunity_signal": ["수집된 기사 제목 기반으로 수동 태깅 가능"],\n'
        '  "linked_decision": ["규제/금리/대출 키워드 포함 기사 우선 검토"],\n'
        f'  "search_text": "[뉴스수집][수동검토] 주요 기사 {headline_text} 자동 신호 추출 실패로 수동 검토가 필요합니다."\n'
        "}"
    )


def build_log_fallback_briefing(log_items: list[dict[str, Any]]) -> str:
    product_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    reject_code_counts: dict[str, int] = {}

    for item in log_items:
        product = str(item.get("product") or item.get("product_code") or "").strip().upper()
        if product:
            product_counts[product] = product_counts.get(product, 0) + 1

        features = item.get("features") or {}
        decision = str(
            features.get("심사결과")
            or features.get("decision")
            or item.get("decision")
            or ""
        ).strip()
        if decision:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        for code in (item.get("reject_reason_codes") or features.get("reject_reason_codes") or []):
            normalized = str(code).strip()
            if normalized:
                reject_code_counts[normalized] = reject_code_counts.get(normalized, 0) + 1

    top_products = ", ".join(
        f"{key} {value}건"
        for key, value in sorted(product_counts.items(), key=lambda entry: (-entry[1], entry[0]))[:3]
    ) or "상품 분포 없음"
    top_decisions = ", ".join(
        f"{key} {value}건"
        for key, value in sorted(decision_counts.items(), key=lambda entry: (-entry[1], entry[0]))[:3]
    ) or "심사결과 정보 없음"
    top_reject_codes = ", ".join(
        f"{key} {value}건"
        for key, value in sorted(reject_code_counts.items(), key=lambda entry: (-entry[1], entry[0]))[:5]
    ) or "대표 거절코드 없음"

    return "\n".join([
        "[fallback] 로그 에이전트 Ollama 호출이 비활성화되어 통계형 브리핑으로 대체했습니다.",
        f"- 최근 로그 수: {len(log_items)}건",
        f"- 상품 분포: {top_products}",
        f"- 심사결과 분포: {top_decisions}",
        f"- 상위 거절코드: {top_reject_codes}",
        "- 자연어 추론 대신 현재 수집된 로그 메타데이터만 사용했습니다.",
    ])


def build_agent_prompt_input(
    agent: str,
    context_text: str,
    user_input: str,
    source: str,
    news_prompt_template: str | None = None,
    log_prompt_template: str | None = None,
    log_product_pattern_summary: str | None = None,
) -> dict[str, str]:
    prompt = (
        build_log_agent_prompt(
            context_text,
            user_input,
            prompt_template=log_prompt_template,
            product_pattern_context=log_product_pattern_summary,
        )
        if agent == "log_agent"
        else build_news_agent_prompt(
            context_text,
            user_input,
            prompt_template=news_prompt_template,
        )
    )
    return {
        "agent": agent,
        "source": source,
        "user_input": user_input,
        "context": context_text,
        "prompt": prompt,
        "template_mode": (
            "custom"
            if agent == "log_agent" and str(log_prompt_template or "").strip()
            else (
                "custom"
                if agent == "news_agent" and str(news_prompt_template or "").strip()
                else "default"
            )
        ),
    }


def log_agent(
    log_context: str,
    user_input: str,
    prompt_template: str | None = None,
    product_pattern_context: str | None = None,
    progress_callback: OllamaProgressCallback | None = None,
) -> str:
    prompt = build_log_agent_prompt(
        log_context,
        user_input,
        prompt_template=prompt_template,
        product_pattern_context=product_pattern_context,
    )
    return lightweight_ollama_generate(prompt, progress_callback=progress_callback)


def case_based_log_inference(
    query: str,
    extra_context: str | None = None,
    k: int = 5,
    prompt_template: str | None = None,
) -> str:
    """Search similar cases in the Vector DB and call Ollama to infer cause and remediation.

    Returns a concise text with '원인' and '조치'.
    """
    candidates = []
    try:
        docs = search_similar_logs(query)
        candidates = docs[:k]
    except Exception:
        candidates = []

    # format candidate cases
    case_texts = []
    for idx, d in enumerate(candidates, start=1):
        meta = getattr(d, "metadata", {}) or {}
        features = meta.get("features", {})
        prod = meta.get("product") or meta.get("product_code") or "-"
        case_lines = [f"Case {idx}: product={prod}"]
        if features:
            # include key feature values
            fa = features.get("available_amount")
            fr = features.get("applied_rate")
            ft = features.get("loan_term_months")
            kg = features.get("ko_codes")
            if fa is not None:
                case_lines.append(f" available_amount={fa}")
            if fr is not None:
                case_lines.append(f" applied_rate={fr}")
            if ft is not None:
                case_lines.append(f" loan_term_months={ft}")
            if kg:
                case_lines.append(f" ko_codes={kg}")
        # include short content
        content = (
            (getattr(d, "page_content", "") or "").strip().replace("\n", " ")[:400]
        )
        case_lines.append(f" content={content}")
        case_texts.append(";".join(case_lines))

    cases_block = "\n\n".join(case_texts) if case_texts else "(유사 사례 없음)"
    product_pattern_context = build_product_pattern_context(extra_context or "")

    prompt = build_log_agent_prompt(
        extra_context or "(없음)",
        query,
        prompt_template=prompt_template,
        similar_cases_text=cases_block,
    )

    return mistral_generate(prompt)


def build_log_generated_payload(
    analysis: str,
    title: str,
    product_name: str | None,
    features: dict[str, Any] | None,
) -> dict[str, Any]:
    feature_map = features or {}
    decision = str(
        feature_map.get("심사결과")
        or feature_map.get("decision")
        or "심사판단미상"
    ).strip()
    knockout_reasons = feature_map.get("KNOCK-OUT 사유") or []
    if not isinstance(knockout_reasons, list):
        knockout_reasons = [str(knockout_reasons)] if knockout_reasons else []
    primary_reason = str(knockout_reasons[0]).strip() if knockout_reasons else "사유미상"
    product_label = str(product_name or title or "대출상품").strip()
    recognized_income = str(feature_map.get("인정소득") or "").strip()
    applied_rate = str(feature_map.get("금리") or "").strip()
    kcb_score = str(feature_map.get("KCB점수") or "").strip()
    nice_score = str(feature_map.get("NICE점수") or "").strip()
    dti_ratio = str(feature_map.get("dti") or "").strip()
    dsr_ratio = str(feature_map.get("dsr비율") or "").strip()
    summary = " ".join(str(analysis or "").split())[:220]
    text = (
        f"[대출심사][{decision}][{primary_reason}] 고객은 {product_label} 심사시 "
        f"연소득 {recognized_income or '-'}원, 신용점수 KCB {kcb_score or '-'} / NICE {nice_score or '-'}, "
        f"DTI {dti_ratio or '-'}%, DSR {dsr_ratio or '-'}로 대출 심사에서 {decision}됨. "
        f"금리 {applied_rate or '-'} 수준으로 검토되었고, 주요 사유는 {primary_reason}이다. "
        f"요약: {summary or '로그 에이전트 분석 결과.'}"
    )
    return {
        "text": text,
        "metadata": {
            "인정소득": recognized_income,
            "금리": applied_rate,
            "KCB점수": kcb_score,
            "NICE점수": nice_score,
            "dti": dti_ratio,
            "dsr비율": dsr_ratio,
            "심사결과": decision,
            "KNOCK-OUT 사유": knockout_reasons,
        },
    }


def news_agent(
    news_context: str,
    user_input: str,
    prompt_template: str | None = None,
    progress_callback: OllamaProgressCallback | None = None,
) -> str:
    prompt = build_news_agent_prompt(
        news_context,
        user_input,
        prompt_template=prompt_template,
    )
    return lightweight_ollama_generate(prompt, progress_callback=progress_callback)


def extract_product_codes_from_log_items(log_items: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in log_items:
        product_code = str(item.get("product") or item.get("product_code") or "").strip().upper()
        if not product_code or product_code in seen:
            continue
        seen.add(product_code)
        ordered.append(product_code)
    return ordered


def build_product_pattern_context_from_log_items(log_items: list[dict[str, Any]]) -> str:
    summary = load_product_pattern_summary(DEFAULT_SUMMARY_PATH)
    product_codes = extract_product_codes_from_log_items(log_items)
    return format_product_pattern_summary_for_prompt(
        summary,
        product_codes=product_codes,
    )


def parse_news_signal_payload(news_result: str) -> dict[str, Any]:
    text = str(news_result or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def format_news_signal_for_decision(news_result: str) -> str:
    payload = parse_news_signal_payload(news_result)
    if not payload:
        return str(news_result or "").strip() or "뉴스 신호 데이터가 없습니다."

    tags = ", ".join(str(item).strip() for item in payload.get("tags") or [] if str(item).strip()) or "-"
    risk_signal = "; ".join(str(item).strip() for item in payload.get("risk_signal") or [] if str(item).strip()) or "-"
    opportunity_signal = "; ".join(str(item).strip() for item in payload.get("opportunity_signal") or [] if str(item).strip()) or "-"
    linked_decision = "; ".join(str(item).strip() for item in payload.get("linked_decision") or [] if str(item).strip()) or "-"
    signal_summary = str(payload.get("signal_summary") or "").strip() or "-"

    return (
        f"- 핵심 변화: {signal_summary}\n"
        f"- 검색 태그: {tags}\n"
        f"- 리스크 신호: {risk_signal}\n"
        f"- 기회 신호: {opportunity_signal}\n"
        f"- 심사/상품 연결 판단: {linked_decision}"
    )


def regulation_agent(rule_context: str, log_context: str, user_input: str) -> str:
    prompt = f"""
당신은 금융 규제 IT 전문가입니다.

[사용자 질문]
{user_input}

[규제]
{rule_context}

[로그]
{log_context}

반드시 아래 형식으로 답하세요.
1. 위반 가능 항목
2. RCLIPS 솔루션에서 고쳐야 할 항목
3. RCLIPS 솔루션 외 프로그램에서 고쳐야 할 항목
4. 규제 대응 포인트 3개
"""
    return mistral_generate(prompt)


def _pick_primary_sentence(text: str, fallback: str) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return fallback

    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    for part in parts:
        candidate = str(part).strip()
        if candidate:
            return candidate[:180]
    return fallback


def synthesize_final_decision(log_result: str, news_result: str, rule_result: str) -> str:
    combined = "\n".join([str(log_result or ""), str(news_result or ""), str(rule_result or "")])
    verdict = "판단 보류"
    if "거절" in combined:
        verdict = "거절"
    elif "조건부 승인" in combined:
        verdict = "조건부 승인"
    elif "승인" in combined:
        verdict = "승인"

    log_summary = _pick_primary_sentence(log_result, "로그 분석 요약이 없습니다.")
    news_summary = _pick_primary_sentence(
        format_news_signal_for_decision(news_result),
        "뉴스 영향 요약이 없습니다.",
    )
    rule_summary = _pick_primary_sentence(rule_result, "규제 판단 요약이 없습니다.")

    return (
        f"1. 최종 판단: {verdict}\n"
        f"2. 이유: 로그 분석은 '{log_summary}'로 요약됩니다.\n"
        f"3. 리스크 요약: 뉴스/외부 환경은 '{news_summary}' 입니다.\n"
        f"4. 대응 전략: 규제 판단 기준 '{rule_summary}'를 우선 반영합니다.\n"
        "5. 추가 확인 필요 항목: 최종 승인 전 한도, 규제 적용 대상, 최신 수집 로그를 재확인하세요."
    )


def render_report(
    question: str,
    log_result: str,
    news_result: str,
    rule_result: str,
    final_result: str,
) -> str:
    return f"""
질문: {question}

📄 로그 분석
{log_result}

📰 뉴스 영향
{news_result}

⚖️ 규제 판단
{rule_result}

🧠 최종 결론
{final_result}
""".strip()


def strategy_chat(
    user_input: str,
    event_callback: Callable[[str, str, str], None] | None = None,
    vector_callback: Callable[[str, str, int, int, str], None] | None = None,
    prompt_input_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ollama_progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    news_prompt_template: str | None = None,
    log_prompt_template: str | None = None,
) -> dict[str, Any]:

    # 질문과 가장 가까운 로그, 뉴스, 규제 문맥을 나눠서 가져옵니다.
    emit_agent_event(
        event_callback,
        "orchestrator",
        "running",
        "RAG에서 관련 로그, 뉴스, 규제를 검색 중입니다.",
    )
    similar_log_docs = search_similar_logs(user_input, k=3)
    logs = [
        (getattr(doc, "page_content", "") or "")[:500]
        for doc in similar_log_docs
    ]
    news, rules = search_news_context(user_input, k=6)
    emit_agent_event(
        event_callback,
        "orchestrator",
        "completed",
        f"문맥 검색 완료. 로그 {len(logs[:3])}건, 뉴스 {len(news[:3])}건, 규제 {len(rules[:3])}건 확보",
    )

    # 상품별로 같은 상품군끼리 묶어서 프롬프트에 넣습니다 (한도/금리/승인 여부 중심 요약).
    logs_text = build_log_context_from_similar_cases(user_input, k=1)
    news_text = trim_context(news, limit=1)
    rules_text = trim_context(rules)
    prompt_inputs = {
        "log_agent": build_agent_prompt_input(
            "log_agent",
            logs_text,
            user_input,
            "strategy_chat",
            log_prompt_template=log_prompt_template,
        ),
        "news_agent": build_agent_prompt_input(
            "news_agent",
            news_text,
            user_input,
            "strategy_chat",
            news_prompt_template=news_prompt_template,
        ),
    }
    if prompt_input_callback is not None:
        for agent_name, prompt_input in prompt_inputs.items():
            prompt_input_callback(agent_name, prompt_input)

    emit_agent_event(
        event_callback,
        "log_agent",
        "running",
        "로그 패턴과 승인 가능성을 분석 중입니다.",
    )
    log_result = log_agent(
        logs_text,
        user_input,
        prompt_template=log_prompt_template,
        progress_callback=(
            None
            if ollama_progress_callback is None
            else lambda event, payload: ollama_progress_callback(
                "log_agent", event, payload
            )
        ),
    )
    emit_agent_event(
        event_callback, "log_agent", "completed", "로그 분석 결과를 생성했습니다."
    )

    emit_agent_event(
        event_callback,
        "news_agent",
        "running",
        "시장 뉴스와 외부 리스크를 분석 중입니다.",
    )
    news_result = news_agent(
        news_text,
        user_input,
        prompt_template=news_prompt_template,
        progress_callback=(
            None
            if ollama_progress_callback is None
            else lambda event, payload: ollama_progress_callback(
                "news_agent", event, payload
            )
        ),
    )
    emit_agent_event(
        event_callback, "news_agent", "completed", "뉴스 영향 분석 결과를 생성했습니다."
    )

    rule_result = rules_text or "업로드된 규제 문서가 없거나 검색된 규제 문맥이 없습니다."

    final_result = synthesize_final_decision(log_result, news_result, rule_result)

    answer = render_report(
        user_input, log_result, news_result, rule_result, final_result
    )
    vector_update = {
        "before_count": 0,
        "after_count": 0,
        "added_count": 0,
    }

    top_log_meta = (
        getattr(similar_log_docs[0], "metadata", {}) or {}
        if similar_log_docs
        else {}
    )
    generated_log_payload = build_log_generated_payload(
        log_result,
        f"log analysis: {user_input}",
        str(top_log_meta.get("product_name") or top_log_meta.get("product") or "").strip() or None,
        top_log_meta.get("features") or {},
    )

    # 에이전트 산출물을 다시 저장해서 이후 질의에서도 재사용할 수 있게 합니다.
    try:
        emit_agent_event(
            event_callback,
            "vector_store",
            "running",
            "에이전트 결과를 벡터 DB에 추가하고 있습니다.",
        )
        before_count = get_vector_count()
        after_count = save_generated_documents(
            [
                {
                    "agent": "log",
                    "type": "generated_log",
                    "title": f"log analysis: {user_input}",
                    "content": log_result,
                    "structured_payload": generated_log_payload,
                },
                {
                    "agent": "news",
                    "type": "signal_news",
                    "title": f"news signal: {user_input}",
                    "content": news_result,
                },
            ]
        )
        vector_update = {
            "before_count": before_count,
            "after_count": after_count,
            "added_count": after_count - before_count,
        }
        emit_vector_event(
            vector_callback,
            "vector_store",
            "append",
            before_count,
            after_count,
            "에이전트 결과 2건을 벡터 DB에 추가 저장했습니다.",
        )
        emit_agent_event(
            event_callback,
            "vector_store",
            "completed",
            f"벡터 DB 적재 완료. 총 {after_count}건",
        )
    except Exception:
        emit_agent_event(
            event_callback,
            "vector_store",
            "failed",
            "에이전트 결과의 벡터 적재에 실패했습니다.",
        )

    return {
        "answer": answer,
        "question": user_input,
        "sections": {
            "log_analysis": log_result,
            "news_analysis": news_result,
            "regulation_analysis": rule_result,
            "final_decision": final_result,
        },
        "context": {
            "logs": logs[:3],
            "news": news[:3],
            "rules": rules[:3],
        },
        "prompt_inputs": prompt_inputs,
        "vector_update": vector_update,
    }


def run_periodic_news_agent(
    news_items: list[dict[str, Any]],
    should_persist: bool = True,
    event_callback: Callable[[str, str, str], None] | None = None,
    vector_callback: Callable[[str, str, int, int, str], None] | None = None,
    prompt_input_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ollama_progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    news_prompt_template: str | None = None,
    ollama_enabled: bool = True,
) -> dict[str, Any]:
    emit_agent_event(
        event_callback,
        "news_agent",
        "running",
        "백그라운드 뉴스 에이전트가 최신 뉴스를 분석 중입니다.",
    )
    news_context = trim_news_items(news_items)
    user_input = "최신 금융 뉴스를 RAG 검색용 구조화 신호로 변환하라"
    prompt_input = build_agent_prompt_input(
        "news_agent",
        news_context,
        user_input,
        "background_news_cycle",
        news_prompt_template=news_prompt_template,
    )
    if prompt_input_callback is not None:
        prompt_input_callback("news_agent", prompt_input)
    if not ollama_enabled:
        analysis = build_news_fallback_briefing(news_items)
        emit_agent_event(
            event_callback,
            "news_agent",
            "completed",
            "뉴스 에이전트 Ollama 호출이 비활성화되어 fallback 브리핑을 생성했습니다.",
        )
        return {
            "analysis": analysis,
            "prompt_input": prompt_input,
            "vector_update": {
                "before_count": 0,
                "after_count": 0,
                "added_count": 0,
            },
            "fallback": True,
            "reason": "ollama_disabled",
        }
    try:
        analysis = news_agent(
            news_context,
            user_input,
            prompt_template=news_prompt_template,
            progress_callback=(
                None
                if ollama_progress_callback is None
                else lambda event, payload: ollama_progress_callback(
                    "news_agent", event, payload
                )
            ),
        )
        emit_agent_event(
            event_callback,
            "news_agent",
            "completed",
            "백그라운드 뉴스 신호를 생성했습니다.",
        )
    except OllamaUnavailableError as error:
        analysis = build_news_fallback_briefing(news_items)
        emit_agent_event(
            event_callback,
            "news_agent",
            "failed",
            str(error),
        )
        return {
            "analysis": analysis,
            "prompt_input": prompt_input,
            "vector_update": {
                "before_count": 0,
                "after_count": 0,
                "added_count": 0,
            },
            "fallback": True,
            "reason": "ollama_unavailable",
        }

    vector_update = {
        "before_count": 0,
        "after_count": 0,
        "added_count": 0,
    }

    if should_persist and analysis.strip():
        try:
            emit_agent_event(
                event_callback,
                "vector_store",
                "running",
                "뉴스 에이전트 결과를 벡터 DB에 저장 중입니다.",
            )
            before_count = get_vector_count()
            after_count = save_generated_documents(
                [
                    {
                        "agent": "news",
                        "type": "signal_news",
                        "title": "periodic news signal",
                        "content": analysis,
                    }
                ]
            )
            vector_update = {
                "before_count": before_count,
                "after_count": after_count,
                "added_count": after_count - before_count,
            }
            emit_vector_event(
                vector_callback,
                "news_agent",
                "append",
                before_count,
                after_count,
                "주기 실행 뉴스 신호 1건을 벡터 DB에 저장했습니다.",
            )
            emit_agent_event(
                event_callback,
                "vector_store",
                "completed",
                f"뉴스 신호 저장 완료. 총 {after_count}건",
            )
        except Exception as error:
            emit_agent_event(
                event_callback,
                "vector_store",
                "failed",
                f"뉴스 신호 저장 실패: {error}",
            )

    return {
        "analysis": analysis,
        "prompt_input": prompt_input,
        "vector_update": vector_update,
    }


def run_periodic_log_agent(
    log_items: list[dict[str, Any]],
    should_persist: bool = True,
    event_callback: Callable[[str, str, str], None] | None = None,
    vector_callback: Callable[[str, str, int, int, str], None] | None = None,
    prompt_input_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ollama_progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    log_prompt_template: str | None = None,
    ollama_enabled: bool = True,
) -> dict[str, Any]:
    emit_agent_event(
        event_callback,
        "log_agent",
        "running",
        "백그라운드 로그 에이전트가 신규 로그를 분석 중입니다.",
    )
    log_context = "product_pattern_summary만 참조하는 경량 로그 에이전트"
    product_pattern_context = build_product_pattern_context_from_log_items(log_items)
    user_input = "최신 유입 로그 기준으로 승인 가능성과 위험 패턴을 요약하라"
    prompt_input = build_agent_prompt_input(
        "log_agent",
        log_context,
        user_input,
        "background_log_cycle",
        log_prompt_template=log_prompt_template,
        log_product_pattern_summary=product_pattern_context,
    )
    if prompt_input_callback is not None:
        prompt_input_callback("log_agent", prompt_input)
    if not ollama_enabled:
        analysis = build_log_fallback_briefing(log_items)
        emit_agent_event(
            event_callback,
            "log_agent",
            "completed",
            "로그 에이전트 Ollama 호출이 비활성화되어 fallback 브리핑을 생성했습니다.",
        )
        return {
            "analysis": analysis,
            "prompt_input": prompt_input,
            "vector_update": {
                "before_count": 0,
                "after_count": 0,
                "added_count": 0,
            },
            "fallback": True,
            "reason": "ollama_disabled",
        }
    analysis = log_agent(
        log_context,
        user_input,
        prompt_template=log_prompt_template,
        product_pattern_context=product_pattern_context,
        progress_callback=(
            None
            if ollama_progress_callback is None
            else lambda event, payload: ollama_progress_callback(
                "log_agent", event, payload
            )
        ),
    )
    emit_agent_event(
        event_callback,
        "log_agent",
        "completed",
        "백그라운드 로그 브리핑을 생성했습니다.",
    )

    vector_update = {
        "before_count": 0,
        "after_count": 0,
        "added_count": 0,
    }

    if should_persist and analysis.strip():
        try:
            emit_agent_event(
                event_callback,
                "vector_store",
                "running",
                "로그 에이전트 결과를 벡터 DB에 저장 중입니다.",
            )
            before_count = get_vector_count()
            top_result = log_items[0] if log_items else {}
            top_features = top_result.get("features") or {}
            generated_log_payload = build_log_generated_payload(
                analysis,
                "periodic log briefing",
                top_result.get("product") or top_result.get("product_name"),
                {
                    "인정소득": top_features.get("annual_income") or top_features.get("recognized_income") or "",
                    "금리": top_features.get("applied_rate") or "",
                    "KCB점수": top_features.get("kcb_score") or top_features.get("credit_score") or "",
                    "NICE점수": top_features.get("nice_score") or "",
                    "dti": top_features.get("dti") or "",
                    "dsr비율": top_features.get("dsr_ratio") or "",
                    "심사결과": top_result.get("out_fields", {}).get("승인 여부") or top_features.get("decision") or "",
                    "KNOCK-OUT 사유": top_result.get("reject_reason_codes") or [],
                },
            )
            after_count = save_generated_documents(
                [
                    {
                        "agent": "log",
                        "type": "generated_log",
                        "title": "periodic log briefing",
                        "content": analysis,
                        "structured_payload": generated_log_payload,
                    }
                ]
            )
            vector_update = {
                "before_count": before_count,
                "after_count": after_count,
                "added_count": after_count - before_count,
            }
            emit_vector_event(
                vector_callback,
                "log_agent",
                "append",
                before_count,
                after_count,
                "주기 실행 로그 에이전트 브리핑 1건을 벡터 DB에 저장했습니다.",
            )
            emit_agent_event(
                event_callback,
                "vector_store",
                "completed",
                f"로그 브리핑 저장 완료. 총 {after_count}건",
            )
        except Exception as error:
            emit_agent_event(
                event_callback,
                "vector_store",
                "failed",
                f"로그 브리핑 저장 실패: {error}",
            )

    return {
        "analysis": analysis,
        "prompt_input": prompt_input,
        "vector_update": vector_update,
    }
