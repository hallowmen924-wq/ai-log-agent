from __future__ import annotations

from typing import Any, Callable

import requests

from mapper.reject_code_mapper import format_reject_reason_details
from rag.vector_db import (
    get_vector_count,
    save_agent_reports,
    search_news_context,
    search_similar_logs,
)

OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaUnavailableError(RuntimeError):
    pass


def _build_ollama_unavailable_message() -> str:
    return (
        "Ollama 서버에 연결할 수 없습니다. "
        "Ollama가 실행 중인지 확인하고, 모델이 준비된 뒤 다시 시도하세요. "
        f"대상 주소: {OLLAMA_URL}"
    )


def mistral_generate(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise OllamaUnavailableError(_build_ollama_unavailable_message()) from error

    answer = str(payload.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama 응답 본문이 비어 있습니다.")
    return answer


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
        item for item in news_items if str(item.get("content", "")).strip()
    ]
    selected_items = crawled_items[:limit]

    if not selected_items:
        return "관련 데이터가 없습니다. 아직 본문 크롤링이 완료되지 않았습니다."

    snippets = []
    for item in selected_items:
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        snippets.append(f"제목: {title}\n본문: {content}")
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


def build_log_agent_prompt(log_context: str, user_input: str) -> str:
    return f"""
당신은 금융 심사 로그 분석가입니다.

[참고] 상품 코드 의미: C6=신용대출, C9=카드론, C11=개인사업자대출, C12=대환대출

[요청]
사용자 질문: {user_input}

[로그(상품별로 묶여있음)]
{log_context}

답변 시 반드시 아래 항목에 집중하세요: '한도(대출 가능 금액)', '금리(적용 금리)', '승인 여부(승인/거절/조건부)', '거절 사유 코드와 설명'

출력 형식(한국어):
1) 요약 판단 (승인/조건부 승인/거절)
2) 핵심 근거 (한도/금리/승인 여부 관련 근거를 중심으로 최대 3개)
3) 위험 패턴 및 우려 사항
4) 즉각 권장 조치(우선순위별로 3개 이내)

간결하고 실무적으로 작성하세요.
""".strip()


def build_news_agent_prompt(news_context: str, user_input: str) -> str:
    return f"""
당신은 금융 시장 분석가입니다.

[사용자 질문]
{user_input}

[뉴스]
{news_context}

반드시 아래 형식으로 답하세요.
1. 현재 금융 리스크 수준
2. 대출 시장 영향
3. 규제 강화 가능성
4. 시장 리스크 점수: LOW, MEDIUM, HIGH 중 하나

간결하지만 근거 중심으로 작성하세요.
""".strip()


def build_news_fallback_briefing(news_items: list[dict[str, Any]]) -> str:
    headlines: list[str] = []
    for item in news_items[:3]:
        title = str(item.get("title") or item.get("summary") or "").strip()
        if title:
            headlines.append(title)

    headline_text = ", ".join(headlines) if headlines else "수집된 주요 기사 제목 없음"
    return (
        "1. 현재 금융 리스크 수준\n"
        "Ollama 연결 실패로 정밀 요약은 생략했습니다. 현재 뉴스 원문은 수집되어 있으며 운영자 확인이 필요합니다.\n\n"
        "2. 대출 시장 영향\n"
        f"주요 기사: {headline_text}\n"
        "시장 변동성과 규제 이슈 가능성을 수동 검토하세요.\n\n"
        "3. 규제 강화 가능성\n"
        "자동 해석 불가 상태입니다. 규제 키워드 포함 기사부터 우선 검토가 필요합니다.\n\n"
        "4. 시장 리스크 점수\n"
        "MEDIUM"
    )


def build_agent_prompt_input(
    agent: str, context_text: str, user_input: str, source: str
) -> dict[str, str]:
    prompt = (
        build_log_agent_prompt(context_text, user_input)
        if agent == "log_agent"
        else build_news_agent_prompt(context_text, user_input)
    )
    return {
        "agent": agent,
        "source": source,
        "user_input": user_input,
        "context": context_text,
        "prompt": prompt,
    }


def log_agent(log_context: str, user_input: str) -> str:
    # 사례 기반 검색 + LLM 추론으로 원인과 조치 도출
    try:
        return case_based_log_inference(user_input, extra_context=log_context, k=5)
    except Exception:
        # 폴백: 기존 방식
        prompt = build_log_agent_prompt(log_context, user_input)
        return mistral_generate(prompt)


def case_based_log_inference(
    query: str, extra_context: str | None = None, k: int = 5
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

    prompt = f"""
당신은 금융 로그 분석 전문가입니다.

[질문]
{query}

[추가 로그 컨텍스트]
{extra_context or '(없음)'}

[유사 사례]
{cases_block}

요구사항:
1) 원인(간결하게 3개 내외)
2) 조치(우선순위가 높은 것부터 3개 내외, 실행 가능한 단계로 작성)

각 항목을 번호매겨서 한국어로 작성하세요. 근거가 되는 유사사례 번호를 괄호로 표기하세요.
""".strip()

    return mistral_generate(prompt)


def news_agent(news_context: str, user_input: str) -> str:
    prompt = build_news_agent_prompt(news_context, user_input)
    return mistral_generate(prompt)


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


def decision_agent(
    log_result: str, news_result: str, rule_result: str, user_input: str
) -> str:
    prompt = f"""
당신은 대출 심사 책임자입니다.

[사용자 질문]
{user_input}

[로그 분석]
{log_result}

[뉴스 영향]
{news_result}

[규제 분석]
{rule_result}

반드시 아래 형식으로 답하세요.
1. 최종 판단: 승인 / 조건부 승인 / 거절 중 하나
2. 이유
3. 리스크 요약
4. 대응 전략
5. 추가 확인 필요 항목
"""
    return mistral_generate(prompt)


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
) -> dict[str, Any]:

    # 질문과 가장 가까운 로그, 뉴스, 규제 문맥을 나눠서 가져옵니다.
    emit_agent_event(
        event_callback,
        "orchestrator",
        "running",
        "RAG에서 관련 로그, 뉴스, 규제를 검색 중입니다.",
    )
    logs = [
        (getattr(doc, "page_content", "") or "")[:500]
        for doc in search_similar_logs(user_input, k=3)
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
            "log_agent", logs_text, user_input, "strategy_chat"
        ),
        "news_agent": build_agent_prompt_input(
            "news_agent", news_text, user_input, "strategy_chat"
        ),
    }

    emit_agent_event(
        event_callback,
        "log_agent",
        "running",
        "로그 패턴과 승인 가능성을 분석 중입니다.",
    )
    log_result = log_agent(logs_text, user_input)
    emit_agent_event(
        event_callback, "log_agent", "completed", "로그 분석 결과를 생성했습니다."
    )

    emit_agent_event(
        event_callback,
        "news_agent",
        "running",
        "시장 뉴스와 외부 리스크를 분석 중입니다.",
    )
    news_result = news_agent(news_text, user_input)
    emit_agent_event(
        event_callback, "news_agent", "completed", "뉴스 영향 분석 결과를 생성했습니다."
    )

    emit_agent_event(
        event_callback,
        "regulation_agent",
        "running",
        "규제 위반 가능성과 보완 항목을 분석 중입니다.",
    )
    rule_result = regulation_agent(rules_text, logs_text, user_input)
    emit_agent_event(
        event_callback,
        "regulation_agent",
        "completed",
        "규제 판단 결과를 생성했습니다.",
    )

    emit_agent_event(
        event_callback,
        "decision_agent",
        "running",
        "세 결과를 종합해 최종 심사 판단을 생성 중입니다.",
    )
    final_result = decision_agent(log_result, news_result, rule_result, user_input)
    emit_agent_event(
        event_callback,
        "decision_agent",
        "completed",
        "최종 심사 결론과 대응 전략을 생성했습니다.",
    )

    answer = render_report(
        user_input, log_result, news_result, rule_result, final_result
    )
    vector_update = {
        "before_count": 0,
        "after_count": 0,
        "added_count": 0,
    }

    # 에이전트 산출물을 다시 저장해서 이후 질의에서도 재사용할 수 있게 합니다.
    try:
        emit_agent_event(
            event_callback,
            "vector_store",
            "running",
            "에이전트 결과를 벡터 DB에 추가하고 있습니다.",
        )
        before_count = get_vector_count()
        after_count = save_agent_reports(
            [
                {
                    "agent": "log",
                    "title": f"log analysis: {user_input}",
                    "content": log_result,
                },
                {
                    "agent": "news",
                    "title": f"news analysis: {user_input}",
                    "content": news_result,
                },
                {
                    "agent": "regulation",
                    "title": f"regulation analysis: {user_input}",
                    "content": rule_result,
                },
                {
                    "agent": "decision",
                    "title": f"decision: {user_input}",
                    "content": final_result,
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
            "에이전트 결과 4건을 벡터 DB에 추가 저장했습니다.",
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
) -> dict[str, Any]:
    emit_agent_event(
        event_callback,
        "news_agent",
        "running",
        "백그라운드 뉴스 에이전트가 최신 뉴스를 분석 중입니다.",
    )
    news_context = trim_news_items(news_items)
    user_input = "최신 금융 뉴스 기준으로 대출 시장 리스크 브리핑을 작성하라"
    prompt_input = build_agent_prompt_input(
        "news_agent", news_context, user_input, "background_news_cycle"
    )
    try:
        analysis = news_agent(news_context, user_input)
        emit_agent_event(
            event_callback,
            "news_agent",
            "completed",
            "백그라운드 뉴스 브리핑을 생성했습니다.",
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
            after_count = save_agent_reports(
                [
                    {
                        "agent": "news",
                        "title": "periodic news briefing",
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
                "주기 실행 뉴스 에이전트 브리핑 1건을 벡터 DB에 저장했습니다.",
            )
            emit_agent_event(
                event_callback,
                "vector_store",
                "completed",
                f"뉴스 브리핑 저장 완료. 총 {after_count}건",
            )
        except Exception as error:
            emit_agent_event(
                event_callback,
                "vector_store",
                "failed",
                f"뉴스 브리핑 저장 실패: {error}",
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
) -> dict[str, Any]:
    emit_agent_event(
        event_callback,
        "log_agent",
        "running",
        "백그라운드 로그 에이전트가 신규 로그를 분석 중입니다.",
    )
    log_context = trim_log_results(log_items)
    user_input = "최신 유입 로그 기준으로 승인 가능성과 위험 패턴을 요약하라"
    prompt_input = build_agent_prompt_input(
        "log_agent", log_context, user_input, "background_log_cycle"
    )
    analysis = log_agent(log_context, user_input)
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
            after_count = save_agent_reports(
                [
                    {
                        "agent": "log",
                        "title": "periodic log briefing",
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
