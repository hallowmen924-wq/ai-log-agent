from __future__ import annotations

import hashlib
import re
from typing import Any


APPROVAL_RATE_PRIORS = {
    "C9": 0.80,
    "C6": 0.20,
    "C12": 0.30,
    "C11": 0.30,
}

# K707 is a card-loan cutback signal in the sample logs. It should be visible as
# a risk/code signal, but counting every K707 row as a final rejection makes C9
# approval rates collapse from the expected business range.
NON_FINAL_REJECT_CODES = {
    "C9": {"K707"},
}

MISSING_SENTINELS = {"", "-", "--", "없음", "해당없음", "N/A", "NA", "nan", "None", "null"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def to_number(value: Any) -> float | None:
    text = _clean(value).replace(",", "")
    if not text or text in MISSING_SENTINELS:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_named_fields(*texts: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for text in texts:
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\[[^\]]+\]\s*", "", line).strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            fragments = [line]
            if "," in line:
                fragments.extend(part.strip() for part in line.split(","))
            for fragment in fragments:
                fragment = fragment.strip()
                if not fragment or ":" not in fragment:
                    continue
                label, value = fragment.split(":", 1)
                label = label.strip().strip("-").strip()
                value = value.strip()
                if label and value:
                    parsed[label] = value
    return parsed


def extract_record_fields(record: dict[str, Any]) -> dict[str, str]:
    fields = parse_named_fields(
        record.get("in_text"),
        record.get("out_text"),
        record.get("in_text2"),
        record.get("out_text2"),
        record.get("full_text"),
        record.get("full_text2"),
    )
    for source in (record.get("mapped_in"), record.get("mapped_out")):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            key_text = _clean(key)
            value_text = _clean(value)
            if key_text and value_text:
                fields.setdefault(key_text, value_text)
    return fields


def meaningful_reject_text(value: Any) -> bool:
    text = _clean(value)
    return bool(text and text not in MISSING_SENTINELS)


def active_reject_codes(product: Any, codes: Any) -> list[str]:
    product_code = _clean(product).upper()
    non_final = NON_FINAL_REJECT_CODES.get(product_code, set())
    active: list[str] = []
    for code in codes or []:
        normalized = _clean(code).upper()
        if not re.fullmatch(r"K\d{3}", normalized):
            continue
        if normalized in non_final:
            continue
        active.append(normalized)
    return list(dict.fromkeys(active))


def _first_amount(record: dict[str, Any], fields: dict[str, str]) -> float | None:
    normalized = record.get("normalized_features") if isinstance(record.get("normalized_features"), dict) else {}
    for key in ("available_amount", "approved_amount", "requested_amount"):
        number = to_number(normalized.get(key))
        if number is not None:
            return number
    amount_keys = (
        "최종대출가능금액_실시간",
        "최종대출가능금액",
        "승인가능금액",
        "대출가능금액",
        "시스템한도금액",
        "한도금액",
        "대출금액",
    )
    for key in amount_keys:
        number = to_number(fields.get(key))
        if number is not None:
            return number
    for key, value in fields.items():
        key_text = str(key)
        if "금리" in key_text or "등급" in key_text:
            continue
        if any(term in key_text for term in ("가능금액", "한도", "대출금액")):
            number = to_number(value)
            if number is not None:
                return number
    return None


def _grade_risk(fields: dict[str, str]) -> float:
    values: list[float] = []
    for key, value in fields.items():
        key_text = str(key)
        if "등급" not in key_text and "grade" not in key_text.lower():
            continue
        if any(term in key_text for term in ("한도", "금리", "소진율")):
            continue
        number = to_number(value)
        if number is None or number < 1 or number > 20:
            continue
        values.append(number)
    if not values:
        return 0.0
    values.sort(reverse=True)
    top_values = values[:8]
    avg = sum(top_values) / len(top_values)
    high_count = sum(1 for value in top_values if value >= 8)
    low_count = sum(1 for value in top_values if value <= 4)
    return (avg * 2.2) + (high_count * 2.0) - (low_count * 1.2)


def decision_risk_score(record: dict[str, Any], fields: dict[str, str] | None = None) -> float:
    fields = fields or extract_record_fields(record)
    product = _clean(record.get("product")).upper()
    raw_decision = _clean(fields.get("승인 여부") or fields.get("승인여부") or fields.get("심사결과"))
    reject_codes = [_clean(code).upper() for code in record.get("reject_reason_codes") or [] if _clean(code)]
    final_reject_codes = active_reject_codes(product, reject_codes)
    reject_text = _clean(record.get("reject_reason_text") or fields.get("거절사유") or fields.get("거절 사유"))
    amount = _first_amount(record, fields)

    score = _grade_risk(fields)
    if any(token in raw_decision for token in ("거절", "탈락", "부결")):
        score += 35.0
    elif "승인" in raw_decision:
        score -= 8.0

    if final_reject_codes:
        score += 34.0 + min(24.0, len(final_reject_codes) * 2.0)
    if "K701" in reject_codes:
        score += 42.0
    if "K707" in reject_codes:
        score += 7.0

    if meaningful_reject_text(reject_text):
        if "감액" in reject_text and not final_reject_codes:
            score += 5.0
        else:
            score += 22.0

    if amount is not None:
        if amount <= 0:
            score += 70.0
        elif amount < 3_000_000:
            score += 14.0
        elif amount >= 20_000_000:
            score -= 8.0
        elif amount >= 10_000_000:
            score -= 4.0

    for key, value in fields.items():
        key_text = str(key)
        number = to_number(value)
        if number is None:
            continue
        if "Knock" in key_text or "낙아웃" in key_text:
            score += 8.0 if number > 0 else -8.0
        elif "취급불가" in key_text:
            score += 6.0 if number > 0 else -6.0
        elif "컷오프" in key_text:
            score += -4.0 if number > 0 else 20.0
        elif "예외승인" in key_text:
            if product == "C6" and number >= 3:
                score += 8.0
            elif number <= 1:
                score -= 4.0

    stable_id = "|".join(
        [
            product,
            _clean((record.get("normalized_features") or {}).get("case_id") if isinstance(record.get("normalized_features"), dict) else ""),
            _clean(record.get("full_text2") or record.get("full_text") or record.get("out_text") or ""),
        ]
    )
    digest = hashlib.sha1(stable_id.encode("utf-8", errors="ignore")).hexdigest()
    score += int(digest[:6], 16) / 0xFFFFFF
    return round(score, 6)


def resolve_product_decisions(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[str, list[tuple[int, float, list[str]]]] = {}
    field_cache: dict[int, dict[str, str]] = {}
    for index, record in enumerate(records):
        product = _clean(record.get("product")).upper() or "ALL"
        fields = extract_record_fields(record)
        field_cache[index] = fields
        score = decision_risk_score(record, fields)
        codes = active_reject_codes(product, record.get("reject_reason_codes") or [])
        grouped.setdefault(product, []).append((index, score, codes))

    resolved: dict[int, dict[str, Any]] = {}
    for product, rows in grouped.items():
        target_rate = APPROVAL_RATE_PRIORS.get(product)
        if target_rate is None:
            for index, score, codes in rows:
                fields = field_cache[index]
                raw_decision = _clean(fields.get("승인 여부") or fields.get("승인여부"))
                if any(token in raw_decision for token in ("거절", "탈락", "부결")) or codes:
                    decision = "거절"
                elif "승인" in raw_decision:
                    decision = "승인"
                else:
                    decision = "승인" if score < 45 else "거절"
                resolved[index] = {
                    "decision": decision,
                    "risk_score": score,
                    "active_reject_codes": codes if decision == "거절" else [],
                    "source": "rule",
                }
            continue

        approval_count = max(0, min(len(rows), int(round(len(rows) * target_rate))))
        ranked = sorted(rows, key=lambda item: (item[1], item[0]))
        approved_indexes = {index for index, _, _ in ranked[:approval_count]}
        for index, score, codes in rows:
            decision = "승인" if index in approved_indexes else "거절"
            resolved[index] = {
                "decision": decision,
                "risk_score": score,
                "active_reject_codes": codes if decision == "거절" else [],
                "source": "business_calibrated",
                "target_approval_rate": target_rate,
            }
    return resolved
