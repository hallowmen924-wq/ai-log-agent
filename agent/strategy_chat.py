from __future__ import annotations

from typing import Any, Callable

import requests

from rag.vector_db import get_vector_count, save_agent_reports, search_context

OLLAMA_URL = "http://localhost:11434/api/generate"


def mistral_generate(prompt: str) -> str:

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        },
        timeout=180,
    )
    response.raise_for_status()

    return response.json()["response"]


def emit_agent_event(event_callback: Callable[[str, str, str], None] | None, agent: str, status: str, detail: str) -> None:
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


def trim_news_items(news_items: list[dict[str, Any]], limit: int = 5) -> str:
    if not news_items:
        return "관련 데이터가 없습니다."

    snippets = []
    for item in news_items[:limit]:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        snippets.append(f"제목: {title}\n요약: {summary}")
    return "\n\n".join(snippets)


def trim_log_results(log_items: list[dict[str, Any]], limit: int = 5) -> str:
    if not log_items:
        return "관련 데이터가 없습니다."

    snippets = []
    for item in log_items[:limit]:
        product = str(item.get("product", "N/A"))
        in_fields = item.get("in_fields", {})
        out_fields = item.get("out_fields", {})
        snippets.append(
            f"상품: {product}\n"
            f"입력 필드 수: {len(in_fields)}\n"
            f"출력 필드 수: {len(out_fields)}\n"
            f"대표 입력: {str(list(in_fields.items())[:5])}\n"
            f"대표 출력: {str(list(out_fields.items())[:5])}"
        )
    return "\n\n".join(snippets)


def build_log_agent_prompt(log_context: str, user_input: str) -> str:
    return f"""
당신은 금융 심사 로그 분석가입니다.

[사용자 질문]
{user_input}

[로그]
{log_context}

반드시 아래 형식으로 답하세요.
1. 이상 거래 여부
2. 위험 패턴
3. 승인 가능성
4. 핵심 근거 3개

간결하지만 실무적으로 작성하세요.
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


def build_agent_prompt_input(agent: str, context_text: str, user_input: str, source: str) -> dict[str, str]:
    prompt = build_log_agent_prompt(context_text, user_input) if agent == "log_agent" else build_news_agent_prompt(context_text, user_input)
    return {
        "agent": agent,
        "source": source,
        "user_input": user_input,
        "context": context_text,
        "prompt": prompt,
    }


def log_agent(log_context: str, user_input: str) -> str:
    prompt = build_log_agent_prompt(log_context, user_input)
    return mistral_generate(prompt)


def news_agent(news_context: str, user_input: str) -> str:
    prompt = build_news_agent_prompt(news_context, user_input)
    return mistral_generate(prompt)


def regulation_agent(rule_context: str, log_context: str, user_input: str) -> str:
    prompt = f"""
당신은 금융 규제 전문가입니다.

[사용자 질문]
{user_input}

[규제]
{rule_context}

[로그]
{log_context}

반드시 아래 형식으로 답하세요.
1. 위반 가능 항목
2. 리스크 수준
3. 심사상 보완 필요 사항
4. 규제 대응 포인트 3개
"""
    return mistral_generate(prompt)


def decision_agent(log_result: str, news_result: str, rule_result: str, user_input: str) -> str:
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


def render_report(question: str, log_result: str, news_result: str, rule_result: str, final_result: str) -> str:
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
    emit_agent_event(event_callback, "orchestrator", "running", "RAG에서 관련 로그, 뉴스, 규제를 검색 중입니다.")
    logs, news, rules = search_context(user_input, k=6)
    emit_agent_event(
        event_callback,
        "orchestrator",
        "completed",
        f"문맥 검색 완료. 로그 {len(logs[:3])}건, 뉴스 {len(news[:3])}건, 규제 {len(rules[:3])}건 확보",
    )

    logs_text = trim_context(logs)
    news_text = trim_context(news)
    rules_text = trim_context(rules)
    prompt_inputs = {
        "log_agent": build_agent_prompt_input("log_agent", logs_text, user_input, "strategy_chat"),
        "news_agent": build_agent_prompt_input("news_agent", news_text, user_input, "strategy_chat"),
    }

    emit_agent_event(event_callback, "log_agent", "running", "로그 패턴과 승인 가능성을 분석 중입니다.")
    log_result = log_agent(logs_text, user_input)
    emit_agent_event(event_callback, "log_agent", "completed", "로그 분석 결과를 생성했습니다.")

    emit_agent_event(event_callback, "news_agent", "running", "시장 뉴스와 외부 리스크를 분석 중입니다.")
    news_result = news_agent(news_text, user_input)
    emit_agent_event(event_callback, "news_agent", "completed", "뉴스 영향 분석 결과를 생성했습니다.")

    emit_agent_event(event_callback, "regulation_agent", "running", "규제 위반 가능성과 보완 항목을 분석 중입니다.")
    rule_result = regulation_agent(rules_text, logs_text, user_input)
    emit_agent_event(event_callback, "regulation_agent", "completed", "규제 판단 결과를 생성했습니다.")

    emit_agent_event(event_callback, "decision_agent", "running", "세 결과를 종합해 최종 심사 판단을 생성 중입니다.")
    final_result = decision_agent(log_result, news_result, rule_result, user_input)
    emit_agent_event(event_callback, "decision_agent", "completed", "최종 심사 결론과 대응 전략을 생성했습니다.")

    answer = render_report(user_input, log_result, news_result, rule_result, final_result)
    vector_update = {
        "before_count": 0,
        "after_count": 0,
        "added_count": 0,
    }

    # 에이전트 산출물을 다시 저장해서 이후 질의에서도 재사용할 수 있게 합니다.
    try:
        emit_agent_event(event_callback, "vector_store", "running", "에이전트 결과를 벡터 DB에 추가하고 있습니다.")
        before_count = get_vector_count()
        after_count = save_agent_reports(
            [
                {"agent": "log", "title": f"log analysis: {user_input}", "content": log_result},
                {"agent": "news", "title": f"news analysis: {user_input}", "content": news_result},
                {"agent": "regulation", "title": f"regulation analysis: {user_input}", "content": rule_result},
                {"agent": "decision", "title": f"decision: {user_input}", "content": final_result},
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
        emit_agent_event(event_callback, "vector_store", "completed", f"벡터 DB 적재 완료. 총 {after_count}건")
    except Exception:
        emit_agent_event(event_callback, "vector_store", "failed", "에이전트 결과의 벡터 적재에 실패했습니다.")

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
    emit_agent_event(event_callback, "news_agent", "running", "백그라운드 뉴스 에이전트가 최신 뉴스를 분석 중입니다.")
    news_context = trim_news_items(news_items)
    user_input = "최신 금융 뉴스 기준으로 대출 시장 리스크 브리핑을 작성하라"
    prompt_input = build_agent_prompt_input("news_agent", news_context, user_input, "background_news_cycle")
    analysis = news_agent(news_context, user_input)
    emit_agent_event(event_callback, "news_agent", "completed", "백그라운드 뉴스 브리핑을 생성했습니다.")

    vector_update = {
        "before_count": 0,
        "after_count": 0,
        "added_count": 0,
    }

    if should_persist and analysis.strip():
        emit_agent_event(event_callback, "vector_store", "running", "뉴스 에이전트 결과를 벡터 DB에 저장 중입니다.")
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
        emit_agent_event(event_callback, "vector_store", "completed", f"뉴스 브리핑 저장 완료. 총 {after_count}건")

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
    emit_agent_event(event_callback, "log_agent", "running", "백그라운드 로그 에이전트가 신규 로그를 분석 중입니다.")
    log_context = trim_log_results(log_items)
    user_input = "최신 유입 로그 기준으로 승인 가능성과 위험 패턴을 요약하라"
    prompt_input = build_agent_prompt_input("log_agent", log_context, user_input, "background_log_cycle")
    analysis = log_agent(log_context, user_input)
    emit_agent_event(event_callback, "log_agent", "completed", "백그라운드 로그 브리핑을 생성했습니다.")

    vector_update = {
        "before_count": 0,
        "after_count": 0,
        "added_count": 0,
    }

    if should_persist and analysis.strip():
        emit_agent_event(event_callback, "vector_store", "running", "로그 에이전트 결과를 벡터 DB에 저장 중입니다.")
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
        emit_agent_event(event_callback, "vector_store", "completed", f"로그 브리핑 저장 완료. 총 {after_count}건")

    return {
        "analysis": analysis,
        "prompt_input": prompt_input,
        "vector_update": vector_update,
    }