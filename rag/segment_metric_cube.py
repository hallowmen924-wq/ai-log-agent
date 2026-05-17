from __future__ import annotations

import collections
import datetime as _dt
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

from mapper.reject_code_mapper import load_reject_code_mapping
from rag.decision_resolver import resolve_product_decisions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RECORDS_PATH = DATA_DIR / "full_text_records.json"
DEFAULT_CLUSTER_PATH = DATA_DIR / "feature_customer_clusters.json"
DEFAULT_SEGMENT_CUBE_PATH = DATA_DIR / "segment_metric_cube.json"

PRODUCT_DISPLAY_NAMES = {
    "C6": "이지신용대출(C6)",
    "C9": "이지론(C9)",
    "C11": "개인사업자대출(C11)",
    "C12": "이지대환대출(C12)",
    "ALL": "전체 상품",
}

AGE_BANDS = [(29, "20대"), (39, "30대"), (49, "40대"), (59, "50대"), (999, "60대+")]
DEFAULT_INCOME_BANDS = [
    (2_800_000, "저소득"),
    (8_600_000, "중소득"),
    (33_000_000, "고소득"),
    (999_999_999_999, "초고소득"),
]
DEFAULT_AMOUNT_BANDS = [
    (3_000_000, "소액"),
    (27_000_000, "중액"),
    (40_000_000, "고액"),
    (999_999_999_999, "초대형"),
]
OPTIONAL_DIMENSIONS = ["decision", "age_band", "income_band", "amount_band", "reject_reason_code"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "nan", "None", "null"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_missing_sentinel(value: float | None) -> bool:
    if value is None:
        return True
    return value in {8888888, 8888888.8, 88888888, 99999, 999999, 9999999, 99999999}


def _parse_named_fields(*texts: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for text in texts:
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            fragments = [line]
            if "," in line:
                fragments.extend(part.strip() for part in line.split(","))
            for fragment in fragments:
                if ":" not in fragment:
                    continue
                label, value = fragment.split(":", 1)
                label = label.strip().strip("-").strip()
                value = value.strip()
                if label and value:
                    parsed[label] = value
    return parsed


def _bucketize(value: float | None, thresholds: list[tuple[float, str]], fallback: str = "미상") -> str:
    if value is None:
        return fallback
    for limit, label in thresholds:
        if value <= limit:
            return label
    return thresholds[-1][1] if thresholds else fallback


def _thresholds_from_cluster_meta(cluster_path: Path = DEFAULT_CLUSTER_PATH) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    payload = _read_json(cluster_path)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

    def parse_thresholds(raw: Any, fallback: list[tuple[float, str]]) -> list[tuple[float, str]]:
        thresholds: list[tuple[float, str]] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            limit = _to_number(item.get("max_value"))
            label = str(item.get("label") or "").strip()
            if limit is not None and label:
                thresholds.append((float(limit), label))
        return thresholds or list(fallback)

    return (
        parse_thresholds(meta.get("income_band_thresholds"), DEFAULT_INCOME_BANDS),
        parse_thresholds(meta.get("amount_band_thresholds"), DEFAULT_AMOUNT_BANDS),
    )


def _format_krw(value: float | None) -> str:
    if value is None:
        return ""
    if math.isclose(float(value), 0.0):
        return "0원"
    if abs(value) >= 100_000_000:
        return f"약 {value / 100_000_000:.1f}억원"
    if abs(value) >= 10_000:
        return f"약 {value / 10_000:,.0f}만원"
    return f"약 {value:,.0f}원"


def _format_percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}%"


def _pick_first_number(fields: dict[str, str], exact_keys: list[str], include_terms: list[str], exclude_terms: list[str] | None = None) -> tuple[float | None, str]:
    exclude_terms = exclude_terms or []
    for key in exact_keys:
        number = _to_number(fields.get(key))
        if number is not None and not _is_missing_sentinel(number):
            return number, key
    for key, value in fields.items():
        key_text = str(key)
        if not any(term in key_text for term in include_terms):
            continue
        if any(term in key_text for term in exclude_terms):
            continue
        number = _to_number(value)
        if number is not None and not _is_missing_sentinel(number):
            return number, key_text
    return None, ""


def _normalize_decision(record: dict[str, Any], fields: dict[str, str]) -> str:
    raw_decision = str(fields.get("승인 여부") or fields.get("승인여부") or "").strip()
    if "거절" in raw_decision or "탈락" in raw_decision or "부결" in raw_decision:
        return "거절"
    if "승인" in raw_decision or "통과" in raw_decision:
        return "승인"
    reject_text = str(record.get("reject_reason_text") or "").strip()
    reject_codes = [str(code).strip() for code in record.get("reject_reason_codes") or [] if str(code).strip()]
    if reject_codes and reject_text and reject_text not in {"-", "없음"}:
        return "거절"
    return "승인"


def _extract_model_score(fields: dict[str, str]) -> tuple[float | None, str]:
    priority_terms = [
        "신용대출신청평점",
        "카드론신청평점",
        "비대면신용평가점수",
        "KCB신용평점",
        "NICE신용평점",
        "통합평가점수",
        "스코어",
        "평점",
        "score",
    ]
    for term in priority_terms:
        for key, value in fields.items():
            key_text = str(key)
            if term.lower() not in key_text.lower():
                continue
            if "등급" in key_text:
                continue
            number = _to_number(value)
            if number is not None and not _is_missing_sentinel(number):
                return number, key_text
    return None, ""


def _extract_delinquency_rate(fields: dict[str, str]) -> tuple[float | None, str]:
    return _pick_first_number(fields, [], ["연체율", "부실률"], [])


def _extract_delinquency_signal(fields: dict[str, str], reject_text: str, reject_descriptions: list[str]) -> bool:
    if "연체" in reject_text or any("연체" in text for text in reject_descriptions):
        return True
    signal_terms = ["연체건수", "연체금액", "최장연체일수", "연체일수", "연체회수", "연체횟수"]
    for key, value in fields.items():
        key_text = str(key)
        if not any(term in key_text for term in signal_terms):
            continue
        number = _to_number(value)
        if number is not None and not _is_missing_sentinel(number) and number > 0:
            return True
    return False


def _build_profile(
    record: dict[str, Any],
    reject_mapping: dict[str, dict[str, str]],
    income_thresholds: list[tuple[float, str]],
    amount_thresholds: list[tuple[float, str]],
    decision_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = _parse_named_fields(record.get("in_text"), record.get("out_text"), record.get("in_text2"), record.get("out_text2"))
    normalized = record.get("normalized_features") if isinstance(record.get("normalized_features"), dict) else {}
    age = _to_number(normalized.get("age")) or _to_number(fields.get("연령"))
    income = _to_number(normalized.get("recognized_income")) or _to_number(fields.get("인정소득")) or _to_number(fields.get("소득"))
    amount = _to_number(normalized.get("available_amount")) or _to_number(fields.get("최종대출가능금액")) or _to_number(fields.get("대출금액"))
    rate, rate_source = _pick_first_number(
        fields,
        ["산출금리", "대출이율", "적용금리", "금리"],
        ["금리", "이율"],
        ["등급", "코드", "가중치"],
    )
    if rate is not None and (rate < 0 or rate > 100):
        rate = None
        rate_source = ""

    reject_codes = [
        str(code).strip().upper()
        for code in record.get("reject_reason_codes") or []
        if re.fullmatch(r"K\d{3}", str(code).strip().upper())
    ]
    reject_descriptions = [
        str((reject_mapping.get(code) or {}).get("description") or "").strip()
        for code in reject_codes
        if str((reject_mapping.get(code) or {}).get("description") or "").strip()
    ]
    decision_result = decision_result or {}
    decision = str(decision_result.get("decision") or _normalize_decision(record, fields))
    active_reject_codes = [
        str(code).strip().upper()
        for code in (decision_result.get("active_reject_codes") or (reject_codes if decision == "거절" else []))
        if str(code).strip()
    ]
    model_score, model_score_source = _extract_model_score(fields)
    delinquency_rate, delinquency_rate_source = _extract_delinquency_rate(fields)
    reject_text = str(record.get("reject_reason_text") or "").strip()
    delinquency_signal = _extract_delinquency_signal(fields, reject_text, reject_descriptions)
    return {
        "record_id": str(fields.get("접수번호") or normalized.get("case_id") or ""),
        "product": str(record.get("product") or "").strip() or "ALL",
        "product_display": str(record.get("product_display") or "").strip(),
        "decision": decision,
        "age": age,
        "age_band": _bucketize(age, AGE_BANDS),
        "income": income,
        "income_band": _bucketize(income, income_thresholds),
        "amount": amount,
        "amount_band": _bucketize(amount, amount_thresholds),
        "rate": rate,
        "rate_source": rate_source,
        "model_score": model_score,
        "model_score_source": model_score_source,
        "delinquency_rate": delinquency_rate,
        "delinquency_rate_source": delinquency_rate_source,
        "delinquency_signal": delinquency_signal,
        "reject_codes": active_reject_codes,
        "reject_descriptions": reject_descriptions if decision == "거절" else [],
        "reject_reason_text": reject_text,
        "decision_risk_score": decision_result.get("risk_score"),
        "decision_source": decision_result.get("source") or "rule",
    }


def _empty_bucket(dimensions: dict[str, str], grain: str) -> dict[str, Any]:
    return {
        "dimensions": dict(dimensions),
        "grain": grain,
        "count": 0,
        "decision_counts": collections.Counter(),
        "metric_sums": collections.defaultdict(float),
        "metric_counts": collections.Counter(),
        "delinquency_signal_count": 0,
        "reject_code_counts": collections.Counter(),
        "source_counts": collections.Counter(),
    }


def _update_bucket(bucket: dict[str, Any], profile: dict[str, Any], reject_code_for_dimension: str | None = None) -> None:
    bucket["count"] += 1
    decision = str(profile.get("decision") or "미상")
    bucket["decision_counts"][decision] += 1
    metric_map = {
        "avg_rate": profile.get("rate"),
        "avg_amount": profile.get("amount"),
        "avg_income": profile.get("income"),
        "avg_model_score": profile.get("model_score"),
        "avg_delinquency_rate": profile.get("delinquency_rate"),
    }
    for metric_key, value in metric_map.items():
        if value is None:
            continue
        bucket["metric_sums"][metric_key] += float(value)
        bucket["metric_counts"][metric_key] += 1
    if profile.get("delinquency_signal"):
        bucket["delinquency_signal_count"] += 1
    if decision == "거절":
        if reject_code_for_dimension:
            bucket["reject_code_counts"][reject_code_for_dimension] += 1
        else:
            bucket["reject_code_counts"].update(profile.get("reject_codes") or [])
    for source_key in ["rate_source", "model_score_source", "delinquency_rate_source"]:
        if profile.get(source_key):
            bucket["source_counts"][str(profile[source_key])] += 1


def _segment_key(dimensions: dict[str, str]) -> str:
    return "|".join(f"{key}={dimensions[key]}" for key in sorted(dimensions))


def _iter_dimension_sets(profile: dict[str, Any]) -> list[tuple[dict[str, str], str, str | None]]:
    rows: list[tuple[dict[str, str], str, str | None]] = []
    product_values = [str(profile.get("product") or "ALL"), "ALL"]
    for product in product_values:
        for length in range(0, len(OPTIONAL_DIMENSIONS) + 1):
            for subset in itertools.combinations(OPTIONAL_DIMENSIONS, length):
                base = {"product": product}
                reject_values: list[tuple[str, str | None]] = [("ALL", None)]
                if "reject_reason_code" in subset:
                    reject_codes = list(profile.get("reject_codes") or [])
                    reject_values = [(code, code) for code in reject_codes] if reject_codes else [("NONE", None)]
                for reject_value, active_code in reject_values:
                    dimensions = dict(base)
                    for dim in subset:
                        if dim == "reject_reason_code":
                            dimensions[dim] = reject_value
                        else:
                            dimensions[dim] = str(profile.get(dim) or "미상")
                    grain = "+".join(sorted(dimensions))
                    rows.append((dimensions, grain, active_code))
    return rows


def _bucket_to_segment(bucket: dict[str, Any], reject_mapping: dict[str, dict[str, str]]) -> dict[str, Any]:
    count = int(bucket["count"])
    decision_counts = {key: int(value) for key, value in bucket["decision_counts"].items()}
    approval_count = int(decision_counts.get("승인", 0))
    rejection_count = int(decision_counts.get("거절", 0))
    metric_counts = bucket["metric_counts"]

    def avg(metric_key: str) -> float | None:
        metric_count = int(metric_counts.get(metric_key, 0))
        if metric_count <= 0:
            return None
        return round(float(bucket["metric_sums"][metric_key]) / metric_count, 4)

    avg_rate = avg("avg_rate")
    avg_amount = avg("avg_amount")
    avg_income = avg("avg_income")
    avg_model_score = avg("avg_model_score")
    avg_delinquency_rate = avg("avg_delinquency_rate")
    delinquency_proxy_rate = round((int(bucket["delinquency_signal_count"]) / count) * 100, 2) if count else None
    top_codes = []
    for code, code_count in bucket["reject_code_counts"].most_common(5):
        description = str((reject_mapping.get(code) or {}).get("description") or "").strip()
        top_codes.append({
            "code": code,
            "description": description,
            "count": int(code_count),
            "share_of_rejections": round(int(code_count) / rejection_count, 4) if rejection_count else 0,
            "share_of_rejections_percent": round((int(code_count) / rejection_count) * 100, 1) if rejection_count else 0,
        })
    reliability = "stable" if count >= 30 else ("indicative" if count >= 10 else "sparse")
    dimensions = dict(bucket["dimensions"])
    return {
        "segment_id": _segment_key(dimensions),
        "grain": bucket["grain"],
        "dimensions": dimensions,
        "count": count,
        "decision_counts": decision_counts,
        "approval_rate": round(approval_count / count, 4) if count else 0,
        "approval_rate_percent": round((approval_count / count) * 100, 1) if count else 0,
        "rejection_rate": round(rejection_count / count, 4) if count else 0,
        "rejection_rate_percent": round((rejection_count / count) * 100, 1) if count else 0,
        "avg_rate": avg_rate,
        "avg_amount": avg_amount,
        "avg_income": avg_income,
        "avg_model_score": avg_model_score,
        "avg_delinquency_rate": avg_delinquency_rate,
        "delinquency_proxy_rate": delinquency_proxy_rate,
        "avg_rate_display": _format_percent(avg_rate, 2),
        "avg_amount_display": _format_krw(avg_amount),
        "avg_income_display": _format_krw(avg_income),
        "avg_model_score_display": f"{avg_model_score:,.0f}점" if avg_model_score is not None else "",
        "avg_delinquency_rate_display": _format_percent(avg_delinquency_rate, 2),
        "delinquency_proxy_rate_display": _format_percent(delinquency_proxy_rate, 1),
        "metric_counts": {key: int(value) for key, value in metric_counts.items()},
        "delinquency_signal_count": int(bucket["delinquency_signal_count"]),
        "top_reject_codes": top_codes,
        "reliability": reliability,
    }


def build_segment_metric_cube(
    records: list[dict[str, Any]],
    source_path: Path = DEFAULT_RECORDS_PATH,
    cluster_path: Path = DEFAULT_CLUSTER_PATH,
) -> dict[str, Any]:
    income_thresholds, amount_thresholds = _thresholds_from_cluster_meta(cluster_path)
    reject_mapping = load_reject_code_mapping(DATA_DIR)
    decision_results = resolve_product_decisions(records)
    profiles = [
        _build_profile(record, reject_mapping, income_thresholds, amount_thresholds, decision_results.get(index))
        for index, record in enumerate(records)
    ]
    buckets: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        for dimensions, grain, active_code in _iter_dimension_sets(profile):
            key = _segment_key(dimensions)
            if key not in buckets:
                buckets[key] = _empty_bucket(dimensions, grain)
            _update_bucket(buckets[key], profile, active_code)
    segments = [_bucket_to_segment(bucket, reject_mapping) for bucket in buckets.values()]
    segments.sort(key=lambda item: (str(item["grain"]), str(item["segment_id"])))
    products = sorted({str(profile.get("product") or "") for profile in profiles if str(profile.get("product") or "").strip()})
    return {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "source_path": str(source_path.relative_to(PROJECT_ROOT)) if source_path.is_relative_to(PROJECT_ROOT) else str(source_path),
            "record_count": len(records),
            "profile_count": len(profiles),
            "segment_count": len(segments),
            "products": products,
            "dimensions": ["product", *OPTIONAL_DIMENSIONS],
            "income_band_thresholds": [{"max_value": limit, "label": label} for limit, label in income_thresholds],
            "amount_band_thresholds": [{"max_value": limit, "label": label} for limit, label in amount_thresholds],
            "reliability_rules": {
                "stable": "count >= 30",
                "indicative": "10 <= count < 30",
                "sparse": "count < 10",
            },
        },
        "segments": segments,
    }


def write_segment_metric_cube(
    records_path: Path = DEFAULT_RECORDS_PATH,
    output_path: Path = DEFAULT_SEGMENT_CUBE_PATH,
) -> Path:
    payload = _read_json(records_path)
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    cube = build_segment_metric_cube(records, source_path=records_path)
    _write_json(output_path, cube)
    return output_path


def load_segment_metric_cube(path: Path = DEFAULT_SEGMENT_CUBE_PATH) -> dict[str, Any]:
    return _read_json(path)


def _compact(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def query_has_segment_metric_intent(query: str) -> bool:
    compact = _compact(query)
    strong_stat_terms = [
        "평균",
        "통계",
        "승인률",
        "거절률",
        "탈락률",
        "연체율",
        "부실률",
        "분포",
        "비율",
        "top",
        "상위",
        "average",
        "statistics",
    ]
    reason_terms = ["영향", "feature", "피처", "변수", "요인", "왜", "이유", "근거"]
    if any(term in compact for term in reason_terms) and not any(term in compact for term in strong_stat_terms):
        return False
    metric_terms = [
        "평균",
        "통계",
        "승인률",
        "거절률",
        "탈락률",
        "연체율",
        "부실률",
        "금리",
        "한도",
        "대출금액",
        "소득",
        "분포",
        "비율",
        "top",
        "상위",
        "rate",
        "limit",
        "average",
        "statistics",
        "approval",
        "rejection",
        "delinquency",
        "default",
    ]
    return any(term in compact for term in metric_terms)


def _extract_query_filters(query: str, selected_product: str = "") -> dict[str, str]:
    compact = _compact(query)
    filters: dict[str, str] = {}
    product = str(selected_product or "").strip().upper()
    if not product:
        product_aliases = {
            "C6": ["c6", "이지신용대출", "신용대출"],
            "C9": ["c9", "이지론", "카드론"],
            "C11": ["c11", "개인사업자대출", "사업자대출"],
            "C12": ["c12", "이지대환대출", "대환대출"],
        }
        for code, aliases in product_aliases.items():
            if any(alias in compact for alias in aliases):
                product = code
                break
    filters["product"] = product if product else "ALL"
    for label in ["20대", "30대", "40대", "50대", "60대+"]:
        if label.replace("+", "") in compact:
            filters["age_band"] = label
            break
    for label in ["초고소득", "고소득", "중소득", "저소득"]:
        if label in compact:
            filters["income_band"] = label
            break
    for label in ["초대형", "고액", "중액", "소액"]:
        if label in compact:
            filters["amount_band"] = label
            break
    if re.search(r"K\d{3}", str(query or ""), re.I):
        filters["reject_reason_code"] = re.search(r"K\d{3}", str(query or ""), re.I).group(0).upper()  # type: ignore[union-attr]
    asks_approval_rate = "승인률" in compact
    asks_rejection_rate = "거절률" in compact or "탈락률" in compact
    if not asks_approval_rate and any(term in compact for term in ["승인고객", "승인군", "승인된고객"]):
        filters["decision"] = "승인"
    if not asks_rejection_rate and any(term in compact for term in ["거절고객", "거절군", "탈락고객", "탈락군", "거절된고객"]):
        filters["decision"] = "거절"
    return filters


def _metric_requests(query: str) -> dict[str, bool]:
    compact = _compact(query)
    requests = {
        "approval_rate": "승인률" in compact or "승인비율" in compact,
        "rejection_rate": "거절률" in compact or "탈락률" in compact or "거절비율" in compact,
        "avg_rate": "금리" in compact,
        "avg_amount": "한도" in compact or "대출금액" in compact,
        "avg_income": "소득" in compact,
        "delinquency": "연체율" in compact or "부실률" in compact or "연체" in compact or "부실" in compact,
        "top_reject_codes": "거절사유" in compact or "탈락사유" in compact or "k코드" in compact,
    }
    if "통계" in compact or "평균" in compact:
        if not any(requests.values()):
            requests.update({"approval_rate": True, "avg_rate": True, "avg_amount": True, "delinquency": True})
    if not any(requests.values()):
        requests.update({"approval_rate": True, "avg_rate": True, "avg_amount": True})
    return requests


def _find_segment(cube: dict[str, Any], filters: dict[str, str]) -> dict[str, Any] | None:
    target = {key: value for key, value in filters.items() if value}
    segments = cube.get("segments") if isinstance(cube.get("segments"), list) else []
    exact: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for segment in segments:
        dimensions = segment.get("dimensions") if isinstance(segment.get("dimensions"), dict) else {}
        if all(str(dimensions.get(key) or "") == str(value) for key, value in target.items()):
            if set(dimensions.keys()) == set(target.keys()):
                exact.append(segment)
            else:
                partial.append(segment)
    candidates = exact or sorted(partial, key=lambda item: (len(item.get("dimensions") or {}), -int(item.get("count") or 0)))
    if candidates:
        return candidates[0]
    return None


def _find_segment_with_fallback(cube: dict[str, Any], filters: dict[str, str]) -> tuple[dict[str, Any] | None, dict[str, str], dict[str, str]]:
    applied = {key: value for key, value in filters.items() if value}
    dropped: dict[str, str] = {}
    relax_order = ["reject_reason_code", "amount_band", "income_band", "age_band", "decision"]
    while True:
        segment = _find_segment(cube, applied)
        if segment:
            return segment, applied, dropped
        removed = False
        for key in relax_order:
            if key in applied:
                dropped[key] = applied.pop(key)
                removed = True
                break
        if removed:
            continue
        if applied.get("product") != "ALL":
            dropped["product"] = applied.get("product", "")
            applied["product"] = "ALL"
            continue
        return None, applied, dropped


def _scope_label(filters: dict[str, str]) -> str:
    product_label = PRODUCT_DISPLAY_NAMES.get(filters.get("product") or "ALL", filters.get("product") or "전체 상품")
    parts = [
        filters.get("decision"),
        filters.get("age_band"),
        filters.get("income_band"),
        filters.get("amount_band"),
    ]
    if filters.get("reject_reason_code"):
        parts.append(f"거절코드 {filters['reject_reason_code']}")
    detail = " · ".join(part for part in parts if part and part != "ALL")
    return f"{product_label} {detail}".strip()


def build_metric_answer_summary_from_cube(
    query: str,
    selected_product: str = "",
    cube_path: Path = DEFAULT_SEGMENT_CUBE_PATH,
) -> dict[str, Any] | None:
    if not query_has_segment_metric_intent(query):
        return None
    cube = load_segment_metric_cube(cube_path)
    if not cube:
        return None
    filters = _extract_query_filters(query, selected_product)
    segment, applied_filters, dropped_filters = _find_segment_with_fallback(cube, filters)
    if not segment:
        return None

    requests = _metric_requests(query)
    scope = _scope_label(applied_filters)
    metric_phrases: list[str] = []
    metric_summary: list[dict[str, Any]] = []

    def add_metric(axis_key: str, label: str, display: str, value: Any, feature_id: str) -> None:
        if not display:
            return
        metric_phrases.append(f"{label} {display}")
        metric_summary.append({
            "axis_key": axis_key,
            "label": label,
            "feature_id": feature_id,
            "value": value,
            "display": display,
            "source": "segment_metric_cube",
        })

    if requests["approval_rate"]:
        add_metric("approval_rate", "승인률", f"{segment.get('approval_rate_percent')}%", segment.get("approval_rate"), "decision.approval_rate")
    if requests["rejection_rate"]:
        add_metric("rejection_rate", "거절률", f"{segment.get('rejection_rate_percent')}%", segment.get("rejection_rate"), "decision.rejection_rate")
    if requests["avg_rate"]:
        add_metric("rate", "평균 금리", str(segment.get("avg_rate_display") or ""), segment.get("avg_rate"), "decision.applied_rate")
    if requests["avg_amount"]:
        add_metric("limit", "평균 한도", str(segment.get("avg_amount_display") or ""), segment.get("avg_amount"), "decision.approved_amount")
    if requests["avg_income"]:
        add_metric("income", "평균 소득", str(segment.get("avg_income_display") or ""), segment.get("avg_income"), "income.recognized_income")
    if requests["delinquency"]:
        actual_count = int((segment.get("metric_counts") or {}).get("avg_delinquency_rate") or 0)
        if actual_count:
            add_metric("delinquency", "평균 연체율", str(segment.get("avg_delinquency_rate_display") or ""), segment.get("avg_delinquency_rate"), "risk.delinquency_rate")
        else:
            add_metric("delinquency", "연체 위험 신호율", str(segment.get("delinquency_proxy_rate_display") or ""), segment.get("delinquency_proxy_rate"), "risk.delinquency_proxy")
    if requests["top_reject_codes"] and segment.get("top_reject_codes"):
        top_code_text = ", ".join(
            f"{item.get('code')} {item.get('description')}".strip()
            for item in (segment.get("top_reject_codes") or [])[:3]
        )
        add_metric("reject_code", "상위 거절사유", top_code_text, None, "decision.reject_reason_code")

    if not metric_phrases:
        metric_phrases = [
            f"승인률 {segment.get('approval_rate_percent')}%",
            f"평균 금리 {segment.get('avg_rate_display') or '-'}",
            f"평균 한도 {segment.get('avg_amount_display') or '-'}",
        ]

    reliability = str(segment.get("reliability") or "")
    reliability_note = {
        "stable": "표본이 충분해 업무 참고용으로 보기 좋습니다.",
        "indicative": "표본이 많지는 않아 참고 지표로 보는 게 좋습니다.",
        "sparse": "표본이 적어 방향성만 참고하는 게 좋습니다.",
    }.get(reliability, "")
    count = int(segment.get("count") or 0)
    headline = f"{scope} 기준으로 {', '.join(metric_phrases[:3])}입니다."
    fallback_note = ""
    if dropped_filters:
        dropped_label = " · ".join(str(value) for value in dropped_filters.values() if value)
        fallback_note = f" 요청 조건 중 {dropped_label} 표본은 부족해서 가장 가까운 상위 기준으로 보여드립니다."
    explanation = (
        f"미리 계산한 통계 큐브에서 조건에 맞는 심사 로그 {count:,}건을 집계했습니다. "
        f"{' '.join(metric_phrases)}.{fallback_note} {reliability_note}"
    ).strip()
    highlights = [
        {"label": "집계 기준", "value": scope},
        {"label": "표본", "value": f"{count:,}건"},
        {"label": "승인률", "value": f"{segment.get('approval_rate_percent')}%"},
        {"label": "거절률", "value": f"{segment.get('rejection_rate_percent')}%"},
        {"label": "평균 금리", "value": str(segment.get("avg_rate_display") or "-")},
        {"label": "평균 한도", "value": str(segment.get("avg_amount_display") or "-")},
    ]
    if requests["delinquency"]:
        highlights.append({"label": "연체 위험", "value": str(segment.get("avg_delinquency_rate_display") or segment.get("delinquency_proxy_rate_display") or "-")})
    if dropped_filters:
        highlights.append({"label": "대체 기준", "value": " · ".join(str(value) for value in dropped_filters.values() if value)})

    return {
        "headline": headline,
        "explanation": explanation,
        "highlights": highlights,
        "metric_summary": metric_summary,
        "segment_metric": segment,
        "source": "segment-metric-cube",
        "source_model": "precomputed-statistics",
        "citations": [],
    }
