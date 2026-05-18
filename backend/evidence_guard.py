from __future__ import annotations

from typing import Any


NO_EVIDENCE_MESSAGE = (
    "현재 학습된 심사/상품/정책/뉴스 데이터에서 관련 근거를 찾지 못했습니다. "
    "앗, 로니가 자료 보관함을 열심히 뒤져봤는데 지금 질문에는 붙일 근거가 없어요. "
    "상품명, 심사 기준, 정책명, 뉴스 키워드를 조금 더 콕 집어서 다시 물어봐 주세요."
)
GROUNDED_PREFIX = "✓ 학습된 정책 근거 기반 답변입니다."

DOMAIN_TERMS = {
    "approval",
    "approve",
    "reject",
    "rejection",
    "rate",
    "limit",
    "loan",
    "policy",
    "rule",
    "regulation",
    "news",
    "risk",
    "cluster",
    "customer",
    "product",
    "cardloan",
    "승인",
    "거절",
    "탈락",
    "금리",
    "한도",
    "대출",
    "상품",
    "고객",
    "심사",
    "정책",
    "규제",
    "뉴스",
    "연체",
    "소득",
    "군집",
    "평균",
    "코드",
    "리스크",
    "카드론",
    "이지",
    "신용",
    "근거",
}


def _text_has_domain_intent(query: str) -> bool:
    normalized = str(query or "").strip().lower()
    if not normalized:
        return False
    compact = "".join(normalized.split())
    return any(term in normalized or term in compact for term in DOMAIN_TERMS)


def _strong_retrieval_items(retrieval_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strong_items: list[dict[str, Any]] = []
    for item in retrieval_results or []:
        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        snippet = str(item.get("snippet") or "").strip()
        # In the workbench retrieval layer, a selected product alone can add a
        # small score. Require a higher score so unrelated questions do not pass.
        if score >= 5 and snippet:
            strong_items.append(item)
    return strong_items


def evaluate_runtime_evidence(
    *,
    query: str,
    answer_mode: str,
    retrieval_results: list[dict[str, Any]] | None = None,
    regulation_evidence: list[dict[str, Any]] | None = None,
    customer_clusters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieval_results = list(retrieval_results or [])
    regulation_evidence = list(regulation_evidence or [])
    customer_clusters = list(customer_clusters or [])
    strong_retrieval = _strong_retrieval_items(retrieval_results)
    has_domain_intent = _text_has_domain_intent(query)
    normalized_mode = str(answer_mode or "general").strip().lower() or "general"

    evidence_sources: list[str] = []
    if regulation_evidence:
        evidence_sources.append("regulation")
    if has_domain_intent and strong_retrieval:
        evidence_sources.append("underwriting-log")
    if has_domain_intent and customer_clusters:
        evidence_sources.append("customer-cluster")

    allowed = bool(str(query or "").strip()) and bool(evidence_sources)
    return {
        "allowed": allowed,
        "reason": "grounded" if allowed else "no_internal_evidence",
        "answer_mode": normalized_mode,
        "has_domain_intent": has_domain_intent,
        "evidence_sources": evidence_sources,
        "counts": {
            "retrieval": len(retrieval_results),
            "strong_retrieval": len(strong_retrieval),
            "regulation": len(regulation_evidence),
            "customer_cluster": len(customer_clusters),
        },
    }


def build_blocked_runtime_answer(
    *,
    query: str,
    evaluation: dict[str, Any],
    model: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    counts = dict(evaluation.get("counts") or {})
    answer_summary = {
        "headline": "근거를 찾지 못했어요",
        "explanation": NO_EVIDENCE_MESSAGE,
        "highlights": [
            {"label": "Evidence", "value": "0건"},
            {"label": "Next", "value": "키워드를 더 구체화"},
        ],
        "citations": [],
        "source": "evidence-blocked",
        "source_model": "none",
        "guardrail": "no_internal_evidence",
        "ui_motion": {
            "character": "roni",
            "mood": "clarifying",
            "animation": "ask_again",
        },
        "evidence_gate": evaluation,
    }
    ollama_runtime = {
        "enabled": False,
        "status": "blocked",
        "model": model,
        "input": {
            "query": str(query or ""),
            "answer_mode": str(evaluation.get("answer_mode") or "general"),
            "evidence_counts": counts,
        },
        "output": {
            "response_text": NO_EVIDENCE_MESSAGE,
            "response_preview": NO_EVIDENCE_MESSAGE[:280],
        },
        "error": "",
        "duration_ms": 0,
        "updated_at": "",
        "used_in_final_answer": True,
        "guardrail": "no_internal_evidence",
        "guardrail_animation": "ask_again",
    }
    return ollama_runtime, answer_summary


def apply_grounded_prompt_rules(prompt_pack: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    pack = dict(prompt_pack or {})
    guardrail_lines = [
        "[EVIDENCE GATE]",
        f"- status: {'allowed' if evaluation.get('allowed') else 'blocked'}",
        f"- evidence_sources: {', '.join(evaluation.get('evidence_sources') or []) or 'none'}",
        "- 답변은 제공된 내부 DB, 심사 로그, 정책 문서, 뉴스/규제 근거 안에서만 작성한다.",
        "- 근거에 없는 정책, 상품 조건, 뉴스 사실, 심사 기준은 추측하지 않는다.",
        f"- 근거가 있는 답변의 첫 문장은 '{GROUNDED_PREFIX}'로 시작한다.",
        "- 답변 끝에는 가능한 경우 record_id, 문서명, source, chunk_index를 표시한다.",
    ]
    pack["system_prompt"] = "\n".join(
        [
            str(pack.get("system_prompt") or "").strip(),
            "",
            *guardrail_lines,
        ]
    ).strip()
    pack["context_preview"] = [
        *list(pack.get("context_preview") or []),
        f"evidence gate: {evaluation.get('reason')}",
        f"evidence sources: {', '.join(evaluation.get('evidence_sources') or []) or 'none'}",
    ]
    pack["evidence_gate"] = evaluation
    return pack


def decorate_grounded_answer_summary(answer_summary: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    summary = dict(answer_summary or {})
    explanation = str(summary.get("explanation") or "").strip()
    if explanation and GROUNDED_PREFIX not in explanation:
        summary["explanation"] = f"{GROUNDED_PREFIX}\n\n{explanation}"
    summary["evidence_gate"] = evaluation
    highlights = [
        item for item in list(summary.get("highlights") or [])
        if str(item.get("label") or "") != "Evidence Gate"
    ]
    highlights.insert(
        0,
        {
            "label": "Evidence Gate",
            "value": ", ".join(evaluation.get("evidence_sources") or []) or "none",
        },
    )
    summary["highlights"] = highlights
    return summary
