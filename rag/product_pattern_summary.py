from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.faiss_logs_db import PRODUCT_DISPLAY_NAMES
from rag.decision_resolver import resolve_product_decisions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "product_pattern_summary.json"

DECISIONS = ("승인", "거절")


def _parse_number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    number = _parse_number(value)
    if number is None:
        return None
    return int(number)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _bucket_dsr(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 10:
        return "DSR 10 미만"
    if value < 20:
        return "DSR 10~20"
    if value < 40:
        return "DSR 20~40"
    return "DSR 40 이상"


def _bucket_rate(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 10:
        return "산출금리 10% 미만"
    if value < 15:
        return "산출금리 10~15%"
    if value < 18:
        return "산출금리 15~18%"
    return "산출금리 18% 이상"


def _bucket_grade(prefix: str, value: int | None) -> str | None:
    if value is None:
        return None
    if value <= 2:
        return f"{prefix} 1~2등급"
    if value <= 4:
        return f"{prefix} 3~4등급"
    if value <= 6:
        return f"{prefix} 5~6등급"
    return f"{prefix} 7등급 이상"


def _bucket_loan_count(value: int | None) -> str | None:
    if value is None:
        return None
    if value <= 0:
        return "신용대출건수 0건"
    if value == 1:
        return "신용대출건수 1건"
    if value <= 3:
        return "신용대출건수 2~3건"
    return "신용대출건수 4건 이상"


def _bucket_non_face_grade(value: int | None) -> str | None:
    if value is None:
        return None
    if value <= 2:
        return "비대면연계대출등급 1~2"
    if value <= 4:
        return "비대면연계대출등급 3~4"
    if value <= 6:
        return "비대면연계대출등급 5~6"
    return "비대면연계대출등급 7 이상"


def _prepare_log_records_for_summary(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from rag.faiss_logs_db import prepare_log_records
    from rag.vector_db import (
        apply_mapping,
        clean_faiss_text,
        find_globally_ignorable_field_keys,
        map_fields,
        sanitize_faiss_fields,
        sanitize_faiss_mapping,
        should_skip_faiss_log,
    )

    class _NullLogger:
        def info(self, *args, **kwargs) -> None:
            return None

    return prepare_log_records(
        logs,
        _NullLogger(),
        should_skip_log=should_skip_faiss_log,
        sanitize_fields=sanitize_faiss_fields,
        sanitize_mapping=sanitize_faiss_mapping,
        find_ignorable_keys=find_globally_ignorable_field_keys,
        apply_mapping=apply_mapping,
        map_fields=map_fields,
        clean_text=clean_faiss_text,
    )


def _extract_rule_candidates(record: dict[str, Any]) -> list[tuple[str, str]]:
    mapped_in = record.get("mapped_in") or {}
    mapped_out = record.get("mapped_out") or {}

    candidates: list[tuple[str, str]] = []

    dsr_bucket = _bucket_dsr(_parse_number(mapped_out.get("DSR")))
    if dsr_bucket:
        candidates.append(("DSR", dsr_bucket))

    rate_bucket = _bucket_rate(_parse_number(mapped_out.get("산출금리")))
    if rate_bucket:
        candidates.append(("산출금리", rate_bucket))

    kcb_bucket = _bucket_grade("KCB", _parse_int(mapped_in.get("KCB 등급")))
    if kcb_bucket:
        candidates.append(("KCB 등급", kcb_bucket))

    nice_bucket = _bucket_grade("NICE", _parse_int(mapped_in.get("NICE CB등급")))
    if nice_bucket:
        candidates.append(("NICE CB등급", nice_bucket))

    ml_bucket = _bucket_grade("ML스코어", _parse_int(mapped_in.get("ML스코어 등급")))
    if ml_bucket:
        candidates.append(("ML스코어 등급", ml_bucket))

    non_face_bucket = _bucket_non_face_grade(
        _parse_int(mapped_in.get("비대면연계대출등급"))
    )
    if non_face_bucket:
        candidates.append(("비대면연계대출등급", non_face_bucket))

    loan_count_bucket = _bucket_loan_count(_parse_int(mapped_in.get("신용대출건수")))
    if loan_count_bucket:
        candidates.append(("신용대출건수", loan_count_bucket))

    return candidates


def _select_patterns(
    rule_stats: dict[str, dict[str, Any]],
    decision: str,
    *,
    min_support: int,
    min_rate: float,
    top_n: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for stat in rule_stats.values():
        support = int(stat.get("support") or 0)
        decision_count = int(stat.get("counts", {}).get(decision) or 0)
        decision_rate = _safe_ratio(decision_count, support)
        if support < min_support or decision_count <= 0 or decision_rate < min_rate:
            continue
        candidates.append(
            {
                "feature": stat.get("feature") or "-",
                "rule": stat.get("rule") or "-",
                "support": support,
                "decision_count": decision_count,
                "decision_rate": round(decision_rate, 4),
                "decision_rate_percent": round(decision_rate * 100, 1),
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["decision_rate"],
            -item["decision_count"],
            -item["support"],
            str(item["rule"]),
        )
    )

    selected: list[dict[str, Any]] = []
    feature_counts: dict[str, int] = defaultdict(int)
    for item in candidates:
        feature = str(item["feature"])
        if feature_counts[feature] >= 2:
            continue
        selected.append(item)
        feature_counts[feature] += 1
        if len(selected) >= top_n:
            break
    return selected


def build_product_pattern_summary(
    logs: list[dict[str, Any]],
    *,
    min_support: int = 3,
    min_rate: float = 0.6,
    top_n: int = 4,
) -> dict[str, Any]:
    prepared_records = _prepare_log_records_for_summary(logs)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "analyzed_log_count": len(logs),
            "prepared_record_count": len(prepared_records),
            "min_support": min_support,
            "min_rate": min_rate,
            "top_n": top_n,
        },
        "products": {},
    }

    decision_results = resolve_product_decisions(prepared_records)
    product_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(prepared_records):
        product_code = str(record.get("product") or "").strip().upper()
        if not product_code:
            continue
        enriched_record = dict(record)
        enriched_record["_resolved_decision"] = str((decision_results.get(index) or {}).get("decision") or "").strip()
        enriched_record["_active_reject_codes"] = list((decision_results.get(index) or {}).get("active_reject_codes") or [])
        product_groups[product_code].append(enriched_record)

    for product_code, records in sorted(product_groups.items()):
        product_name = str(
            records[0].get("product_display")
            or PRODUCT_DISPLAY_NAMES.get(product_code)
            or product_code
        )
        decision_counter: Counter[str] = Counter()
        rule_stats: dict[str, dict[str, Any]] = {}
        reject_code_counter: Counter[str] = Counter()
        reject_code_descriptions: dict[str, str] = {}

        for record in records:
            mapped_out = record.get("mapped_out") or {}
            decision = str(record.get("_resolved_decision") or mapped_out.get("승인 여부") or "").strip()
            if decision in DECISIONS:
                decision_counter[decision] += 1

            if decision == "거절":
                active_codes = {str(code).strip().upper() for code in record.get("_active_reject_codes") or []}
                for detail in record.get("reject_reason_details") or []:
                    code = str(detail.get("code") or "").strip().upper()
                    if not code:
                        continue
                    if active_codes and code not in active_codes:
                        continue
                    reject_code_counter[code] += 1
                    description = str(detail.get("description") or "").strip()
                    if description and code not in reject_code_descriptions:
                        reject_code_descriptions[code] = description

            if decision not in DECISIONS:
                continue

            for feature_name, rule_label in _extract_rule_candidates(record):
                stat = rule_stats.setdefault(
                    f"{feature_name}:{rule_label}",
                    {
                        "feature": feature_name,
                        "rule": rule_label,
                        "support": 0,
                        "counts": {"승인": 0, "거절": 0},
                    },
                )
                stat["support"] += 1
                stat["counts"][decision] += 1

        decision_known_cases = decision_counter["승인"] + decision_counter["거절"]
        approval_rate = _safe_ratio(decision_counter["승인"], decision_known_cases)
        rejection_rate = _safe_ratio(decision_counter["거절"], decision_known_cases)

        summary["products"][product_code] = {
            "product_name": product_name,
            "totals": {
                "all_cases": len(records),
                "decision_known_cases": decision_known_cases,
                "approval_cases": decision_counter["승인"],
                "rejection_cases": decision_counter["거절"],
                "approval_rate": round(approval_rate, 4),
                "approval_rate_percent": round(approval_rate * 100, 1),
                "rejection_rate": round(rejection_rate, 4),
                "rejection_rate_percent": round(rejection_rate * 100, 1),
            },
            "approval_patterns": _select_patterns(
                rule_stats,
                "승인",
                min_support=min_support,
                min_rate=min_rate,
                top_n=top_n,
            ),
            "rejection_patterns": _select_patterns(
                rule_stats,
                "거절",
                min_support=min_support,
                min_rate=min_rate,
                top_n=top_n,
            ),
            "top_reject_reason_codes": [
                {
                    "code": code,
                    "description": reject_code_descriptions.get(code, ""),
                    "count": count,
                    "share_of_rejections": round(
                        _safe_ratio(count, decision_counter["거절"]),
                        4,
                    ),
                    "share_of_rejections_percent": round(
                        _safe_ratio(count, decision_counter["거절"]) * 100,
                        1,
                    ),
                }
                for code, count in reject_code_counter.most_common(top_n)
            ],
        }

    return summary


def write_product_pattern_summary(
    logs: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    path = Path(output_path) if output_path else DEFAULT_SUMMARY_PATH
    summary = build_product_pattern_summary(logs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_product_pattern_summary_documents(
    logs: list[dict[str, Any]],
    *,
    clean_text,
    store_name: str,
) -> list[Document]:
    summary = build_product_pattern_summary(logs)
    products = (summary.get("products") or {})
    documents: list[Document] = []

    for product_code in sorted(products.keys()):
        item = products.get(product_code) or {}
        product_name = str(item.get("product_name") or product_code).strip()
        totals = item.get("totals") or {}
        approval_patterns = item.get("approval_patterns") or []
        rejection_patterns = item.get("rejection_patterns") or []
        top_reject_codes = item.get("top_reject_reason_codes") or []

        payload = {
            "generated_at": summary.get("generated_at"),
            "source": summary.get("source") or {},
            "product_code": product_code,
            "product_name": product_name,
            "totals": totals,
            "approval_patterns": approval_patterns,
            "rejection_patterns": rejection_patterns,
            "top_reject_reason_codes": top_reject_codes,
        }

        page_content = clean_text(
            "\n".join(
                [
                    f"[상품: {product_code}] {product_name}",
                    (
                        "- 결정 분포: 승인 "
                        f"{totals.get('approval_rate_percent', 0):.1f}% "
                        f"({totals.get('approval_cases', 0)}/{totals.get('decision_known_cases', 0)}), "
                        "거절 "
                        f"{totals.get('rejection_rate_percent', 0):.1f}% "
                        f"({totals.get('rejection_cases', 0)}/{totals.get('decision_known_cases', 0)})"
                    ),
                    "- 승인 패턴: "
                    + (
                        "; ".join(
                            _format_pattern_line(pattern, "승인")
                            for pattern in approval_patterns[:3]
                        )
                        if approval_patterns
                        else "뚜렷한 패턴 없음"
                    ),
                    "- 거절 패턴: "
                    + (
                        "; ".join(
                            _format_pattern_line(pattern, "거절")
                            for pattern in rejection_patterns[:3]
                        )
                        if rejection_patterns
                        else "뚜렷한 패턴 없음"
                    ),
                    "- 주요 거절사유: "
                    + (
                        "; ".join(
                            (
                                f"{str(code_item.get('description') or code_item.get('code') or '').strip()} "
                                f"{code_item.get('share_of_rejections_percent', 0):.1f}%"
                            ).strip()
                            for code_item in top_reject_codes[:3]
                        )
                        if top_reject_codes
                        else "데이터 없음"
                    ),
                ]
            )
        )[:4000]

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "type": "product_pattern_summary",
                    "store": store_name,
                    "product": product_code,
                    "name": f"product pattern summary: {product_name}",
                    "source": "structured_log_summary",
                    "features": payload,
                },
            )
        )

    return documents


def load_product_pattern_summary(
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    if input_path is not None:
        path = Path(input_path)
        if not path.exists():
            return {"products": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"products": {}}

    try:
        from rag.vector_db import FAISS_STORE_CUSTOMER, list_vectors

        items = list_vectors(limit=1000, store_name=FAISS_STORE_CUSTOMER)
    except Exception:
        items = []

    products: dict[str, Any] = {}
    generated_at = None
    source: dict[str, Any] = {}
    for item in items:
        if str(item.get("type") or "").strip().lower() != "product_pattern_summary":
            continue
        features = item.get("features") or {}
        product_code = str(
            features.get("product_code") or item.get("product") or ""
        ).strip().upper()
        if not product_code:
            continue
        generated_at = generated_at or features.get("generated_at")
        source = source or (features.get("source") or {})
        products[product_code] = {
            "product_name": features.get("product_name") or product_code,
            "totals": features.get("totals") or {},
            "approval_patterns": features.get("approval_patterns") or [],
            "rejection_patterns": features.get("rejection_patterns") or [],
            "top_reject_reason_codes": features.get("top_reject_reason_codes") or [],
        }

    if products:
        return {
            "generated_at": generated_at,
            "source": source,
            "products": products,
        }

    path = DEFAULT_SUMMARY_PATH
    if not path.exists():
        return {"products": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"products": {}}


def _format_pattern_line(pattern: dict[str, Any], decision: str) -> str:
    return (
        f"{pattern.get('rule') or '-'} -> {decision} 비율 "
        f"{pattern.get('decision_rate_percent', 0):.1f}% "
        f"({pattern.get('decision_count', 0)}/{pattern.get('support', 0)})"
    )


def format_product_pattern_summary_for_prompt(
    summary: dict[str, Any] | None,
    *,
    product_codes: list[str] | None = None,
    max_patterns: int = 3,
) -> str:
    products = ((summary or {}).get("products") or {})
    if not products:
        return "상품별 패턴 요약이 없습니다."

    ordered_codes = product_codes or sorted(products.keys())
    lines = []
    for raw_code in ordered_codes:
        product_code = str(raw_code or "").strip().upper()
        if product_code not in products:
            continue

        item = products[product_code]
        totals = item.get("totals") or {}
        approval_patterns = (item.get("approval_patterns") or [])[:max_patterns]
        rejection_patterns = (item.get("rejection_patterns") or [])[:max_patterns]
        top_reject_codes = (item.get("top_reject_reason_codes") or [])[:max_patterns]

        lines.append(
            f"[상품: {product_code}] {item.get('product_name') or product_code}"
        )
        lines.append(
            "- 결정 분포: 승인 "
            f"{totals.get('approval_rate_percent', 0):.1f}% "
            f"({totals.get('approval_cases', 0)}/{totals.get('decision_known_cases', 0)}), "
            "거절 "
            f"{totals.get('rejection_rate_percent', 0):.1f}% "
            f"({totals.get('rejection_cases', 0)}/{totals.get('decision_known_cases', 0)})"
        )
        lines.append(
            "- 승인 패턴: "
            + (
                "; ".join(
                    _format_pattern_line(pattern, "승인")
                    for pattern in approval_patterns
                )
                if approval_patterns
                else "뚜렷한 패턴 없음"
            )
        )
        lines.append(
            "- 거절 패턴: "
            + (
                "; ".join(
                    _format_pattern_line(pattern, "거절")
                    for pattern in rejection_patterns
                )
                if rejection_patterns
                else "뚜렷한 패턴 없음"
            )
        )
        lines.append(
            "- 주요 거절사유: "
            + (
                "; ".join(
                    (
                        f"{code_item.get('code')} "
                        f"{str(code_item.get('description') or '').strip()}".strip()
                        + f" {code_item.get('share_of_rejections_percent', 0):.1f}%"
                    )
                    for code_item in top_reject_codes
                )
                if top_reject_codes
                else "데이터 없음"
            )
        )

    return "\n".join(lines).strip() or "상품별 패턴 요약이 없습니다."
