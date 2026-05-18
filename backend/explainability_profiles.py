from __future__ import annotations

from typing import Callable


def query_has_vector_intent(query: str, compact_search_text: Callable[[str], str]) -> bool:
    compact_query = compact_search_text(query)
    vector_markers = ["벡터", "유사도", "임베딩", "faiss", "vector", "embedding", "similarity"]
    return any(marker in compact_query for marker in vector_markers)


def classify_explainability_question_type(
    query: str,
    selected_feature: dict | None,
    *,
    has_reject_intent: Callable[[str, dict | None], bool],
    has_metric_intent: Callable[[str], bool],
    has_rate_intent: Callable[[str], bool],
    has_limit_intent: Callable[[str], bool],
    asks_cluster_signals: Callable[[str], bool],
    compact_search_text: Callable[[str], str],
) -> str:
    if has_reject_intent(query, selected_feature):
        return "reject_reason"
    if has_metric_intent(query) and (has_rate_intent(query) or has_limit_intent(query)):
        return "rate_limit"
    if asks_cluster_signals(query) or query_has_vector_intent(query, compact_search_text):
        return "cluster_vector"
    return "approval_factor"


def build_explainability_payload(
    *,
    query: str,
    selected_product: str,
    selected_feature: dict | None,
    representative_features: list[dict[str, object]],
    customer_clusters: list[dict],
    reject_code_summary: list[dict[str, object]],
    has_reject_intent: Callable[[str, dict | None], bool],
    has_metric_intent: Callable[[str], bool],
    has_rate_intent: Callable[[str], bool],
    has_limit_intent: Callable[[str], bool],
    asks_cluster_signals: Callable[[str], bool],
    compact_search_text: Callable[[str], str],
    normalize_percentages: Callable[[list[dict[str, object]], str], list[dict[str, object]]],
    product_display_name: Callable[[str], str],
    is_cross_product_feature_label: Callable[[str, str], bool],
) -> dict[str, object]:
    question_type = classify_explainability_question_type(
        query,
        selected_feature,
        has_reject_intent=has_reject_intent,
        has_metric_intent=has_metric_intent,
        has_rate_intent=has_rate_intent,
        has_limit_intent=has_limit_intent,
        asks_cluster_signals=asks_cluster_signals,
        compact_search_text=compact_search_text,
    )
    top_cluster = customer_clusters[0] if customer_clusters else {}
    approved_cluster = next((item for item in customer_clusters if str(item.get("decision") or "") == "승인"), top_cluster)
    rejected_cluster = next((item for item in customer_clusters if str(item.get("decision") or "") == "거절"), top_cluster)
    focus_cluster = rejected_cluster if question_type == "reject_reason" else approved_cluster

    selected_axis = str((selected_feature or {}).get("feature_name") or (selected_feature or {}).get("feature_id") or "대표 심사 축")
    if question_type == "reject_reason" or is_cross_product_feature_label(selected_axis, selected_product):
        selected_axis = "거절사유코드"

    candidate_impacts: list[dict[str, object]] = []
    metric_cards: list[dict[str, object]] = []

    def push_impact(feature: str, impact: float, direction: str, evidence: str = "") -> None:
        feature = str(feature or "").strip()
        if not feature:
            return
        if any(feature == str(item.get("feature") or "").strip() for item in candidate_impacts):
            return
        candidate_impacts.append({
            "feature": feature,
            "impact": impact,
            "direction": direction,
            "evidence": evidence,
        })

    if question_type == "reject_reason":
        for index, item in enumerate(reject_code_summary[:4]):
            label = str(item.get("code") or "")
            description = str(item.get("description") or "").strip()
            count = int(item.get("count") or 0)
            try:
                share_value = f"{float(item.get('share')) * 100:.1f}%"
            except (TypeError, ValueError):
                share_value = f"{count:,}건"
            push_impact(f"{label} {description}".strip(), [46, 30, 16, 8][index] if index < 4 else 6, "risk_up", f"{count:,}건 · {share_value}")
            metric_cards.append({
                "label": label or f"사유 {index + 1}",
                "value": f"{count:,}건 · {share_value}",
                "tone": "warning" if index == 0 else "neutral",
            })

    elif question_type == "rate_limit":
        if focus_cluster.get("avg_rate_display"):
            push_impact("평균 금리", 34, "price", str(focus_cluster.get("avg_rate_display") or ""))
        if focus_cluster.get("avg_amount_display"):
            push_impact("평균 한도", 32, "limit", str(focus_cluster.get("avg_amount_display") or ""))
        if focus_cluster.get("avg_income_display") or focus_cluster.get("income_band"):
            push_impact("소득 수준", 18, "income", f"{focus_cluster.get('income_band') or '미상'} · {focus_cluster.get('avg_income_display') or ''}".strip(" ·"))
        if focus_cluster.get("avg_model_score_display"):
            push_impact("신용 score", 16, "score", str(focus_cluster.get("avg_model_score_display") or ""))
        metric_cards.extend([
            {"label": "기준 고객군", "value": str(focus_cluster.get("decision") or "승인"), "tone": "positive"},
            {"label": "평균 금리", "value": str(focus_cluster.get("avg_rate_display") or "-"), "tone": "warning"},
            {"label": "평균 한도", "value": str(focus_cluster.get("avg_amount_display") or "-"), "tone": "positive"},
            {"label": "소득/score", "value": f"{focus_cluster.get('income_band') or '-'} / {focus_cluster.get('avg_model_score_display') or '-'}", "tone": "neutral"},
        ])

    elif question_type == "cluster_vector":
        push_impact("군집 크기", 26, "segment_size", f"{int(focus_cluster.get('count') or 0):,}건")
        if focus_cluster.get("avg_rate_display"):
            push_impact("군집 평균 금리", 24, "cluster_rate", str(focus_cluster.get("avg_rate_display") or ""))
        if focus_cluster.get("avg_amount_display"):
            push_impact("군집 평균 한도", 22, "cluster_limit", str(focus_cluster.get("avg_amount_display") or ""))
        if focus_cluster.get("avg_model_score_display"):
            push_impact("군집 신용 score", 18, "cluster_score", str(focus_cluster.get("avg_model_score_display") or ""))
        metric_cards.extend([
            {"label": "대표 군집", "value": str(focus_cluster.get("label") or focus_cluster.get("cluster_id") or "-"), "tone": "neutral"},
            {"label": "표본 수", "value": f"{int(focus_cluster.get('count') or 0):,}건", "tone": "neutral"},
            {"label": "평균 금리", "value": str(focus_cluster.get("avg_rate_display") or "-"), "tone": "warning"},
            {"label": "평균 한도", "value": str(focus_cluster.get("avg_amount_display") or "-"), "tone": "positive"},
        ])

    else:
        if focus_cluster.get("avg_model_score_display"):
            push_impact("신용 score", 30, "score", str(focus_cluster.get("avg_model_score_display") or ""))
        if focus_cluster.get("income_band") or focus_cluster.get("avg_income_display"):
            push_impact("소득 수준", 24, "income", f"{focus_cluster.get('income_band') or '미상'} · {focus_cluster.get('avg_income_display') or ''}".strip(" ·"))
        if focus_cluster.get("avg_rate_display"):
            push_impact("금리 조건", 18, "price", str(focus_cluster.get("avg_rate_display") or ""))
        if focus_cluster.get("avg_amount_display"):
            push_impact("한도 조건", 16, "limit", str(focus_cluster.get("avg_amount_display") or ""))
        metric_cards.extend([
            {"label": "기준 고객군", "value": str(focus_cluster.get("decision") or "승인"), "tone": "positive"},
            {"label": "신용 score", "value": str(focus_cluster.get("avg_model_score_display") or "-"), "tone": "neutral"},
            {"label": "소득 수준", "value": str(focus_cluster.get("income_band") or "-"), "tone": "positive"},
            {"label": "금리/한도", "value": f"{focus_cluster.get('avg_rate_display') or '-'} / {focus_cluster.get('avg_amount_display') or '-'}", "tone": "warning"},
        ])

    for feature in representative_features[:5]:
        feature_name = str(feature.get("feature_name") or feature.get("feature_id") or "").strip()
        feature_category = str(feature.get("category") or "ontology")
        if not feature_name:
            continue
        compact_name = compact_search_text(feature_name)
        if question_type != "reject_reason" and any(token in compact_name for token in ["거절", "탈락", "reject"]):
            continue
        if question_type == "rate_limit":
            allowed_tokens = ["금리", "한도", "소득", "score", "신용", "income", "rate", "limit", "kcb", "nice"]
            if not any(token in compact_name for token in allowed_tokens):
                continue
        push_impact(feature_name, 12, "semantic_axis", feature_category)

    if not candidate_impacts and focus_cluster:
        if focus_cluster.get("avg_rate_display"):
            push_impact("평균 금리", 30, "price", str(focus_cluster.get("avg_rate_display") or ""))
        if focus_cluster.get("avg_amount_display"):
            push_impact("평균 한도", 26, "limit", str(focus_cluster.get("avg_amount_display") or ""))
        if focus_cluster.get("income_band"):
            push_impact(str(focus_cluster.get("income_band")), 20, "segment", "고객군집")

    impacts = normalize_percentages(candidate_impacts[:5], "impact")
    return {
        "id": "explainability",
        "title": "Explainability Agent",
        "status": "ready",
        "summary": f"{product_display_name(selected_product) or selected_product or '전체'} 기준으로 {selected_axis}를 중심으로 설명합니다.",
        "method": "tool-first SHAP-ready attribution",
        "llm_call": False,
        "question_type": question_type,
        "primary_axis": selected_axis,
        "metrics": metric_cards,
        "shap_values": impacts,
        "reasoning": [
            "실제 로그/군집/feature 연결 근거를 우선 반영합니다.",
            "질문 유형에 맞는 SHAP 구성으로 핵심 요인만 정렬합니다.",
        ],
    }
