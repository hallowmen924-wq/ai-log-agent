from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.documents import Document


PreparedLogRecord = dict[str, Any]


PRODUCT_DISPLAY_NAMES = {
    "C11": "개인사업자대출",
    "C9": "카드론(이지론)",
    "C6": "신용대출(이지신용대출)",
    "C12": "이지대환대출",
}

PRODUCT_RATE_FIELD_CODES = {
    "C9": "R1003",
    "C6": "R0012",
    "C11": "R0046",
    "C12": "R0004",
}

DECISION_AMOUNT_FIELD_CODES = {
    "C9": "R0002",
    "C6": "R0020",
    "C11": "R0008",
    "C12": "R0002",
}

DSR_FIELD_CODES = {
    "C9": "R0101",
    "C11": "R0047",
}

RECOGNIZED_INCOME_FIELD_CODES = {
    "C9": "R0050",
    "C6": "R0060",
    "C11": "R0048",
    "C12": "R0044",
}


IN_FIELD_SPECS = [
    ("소득", [("소득",), ("연소득",), ("최종연소득",), ("income",)]),
    ("연령", [("연령",), ("나이",), ("age",)]),
    ("외국인여부", [("외국인",), ("외국인", "여부"), ("국적",), ("foreigner",)]),
    ("건강보험가입자구분", [("건강보험", "구분")]),
    ("접수번호", [("접수번호",), ("신청서접수번호",), ("req",)]),
    ("채널구분", [("채널구분",)]),

    
    ("대출금액", [("대출금액",), ("대출", "금액"), ("한도금액",)]),
    ("비대면연계대출등급", [("비대면연계대출등급",)]),
    ("신용대출건수", [("신용대출", "건수")]),
    ("ML스코어 등급", [("ml", "등급"), ("ml스코어등급",)]),
    ("스크래핑소득", [("스크래핑소득",)]),
    ("NICE CB등급", [("nice", "cb", "등급")]),
    ("KCB 등급", [("kcb", "등급")]),
]

OUT_FIELD_SPECS = [
    ("승인 여부", [("승인",), ("결과",), ("approve",)]),
    ("산출금리", [("산출", "금리"), ("적용", "금리"), ("금리",)]),
    ("DSR", [("dsr",)]),
]


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _compose_structured_log_payload(record: PreparedLogRecord) -> tuple[str, dict[str, Any]]:
    features = record.get("features") or {}
    mapped_out = record.get("mapped_out") or {}
    product_name = str(record.get("product_display") or record.get("product") or "대출상품").strip()
    decision = str(mapped_out.get("승인 여부") or features.get("decision") or "심사판단미상").strip()
    knockout_reasons = [
        str(item.get("description") or item.get("code") or "").strip()
            for item in (record.get("reject_reason_details") or [])
            if str(item.get("description") or item.get("code") or "").strip()
        ]
    if not knockout_reasons:
        knockout_reasons = [
            str(item).strip()
            for item in (record.get("reject_reason_codes") or [])
            if str(item).strip()
        ]
    primary_reason = knockout_reasons[0] if knockout_reasons else (decision or "사유미상")

    metadata = {
        "인정소득": _stringify_value(features.get("recognized_income") or features.get("annual_income") or ""),
        "금리": _stringify_value(features.get("applied_rate") or mapped_out.get("산출금리") or ""),
        "KCB점수": _stringify_value(features.get("kcb_score") or features.get("credit_score") or ""),
        "NICE점수": _stringify_value(features.get("nice_score") or ""),
        "연령": _stringify_value(features.get("age") or ""),
        "외국인여부": _stringify_value(features.get("foreigner") or ""),
        "dti": _stringify_value(features.get("dti") or ""),
        "dsr비율": _stringify_value(features.get("dsr_ratio") or mapped_out.get("DSR") or ""),
        "심사결과": decision,
        "KNOCK-OUT 사유": knockout_reasons,
    }
    text = (
        f"[대출심사][{decision or '심사판단미상'}][{primary_reason}] 고객은 {product_name} 심사시 "
        f"연소득 {metadata['인정소득'] or '-'}원, 신용점수 KCB {metadata['KCB점수'] or '-'} / NICE {metadata['NICE점수'] or '-'}, "
        f"DTI {metadata['dti'] or '-'}%, DSR {metadata['dsr비율'] or '-'}로 대출 심사에서 {decision or '심사판단미상'}됨. "
        f"금리 {metadata['금리'] or '-'} 수준으로 검토되었고, {primary_reason} 상태이다. "
        f"이 사례는 {primary_reason}로 {decision or '심사판단미상'}된 사례이다. 요약: {product_name} 상품의 {primary_reason} 케이스."
    )
    return text, metadata


def prepare_log_records(
    logs: list[dict[str, Any]],
    logger,
    *,
    show_progress: bool = False,
    should_skip_log: Callable[[dict[str, Any]], bool],
    sanitize_fields: Callable[[dict[str, Any], set[str] | None], dict[str, Any]],
    sanitize_mapping: Callable[[dict[str, Any]], dict[str, Any]],
    find_ignorable_keys: Callable[[list[dict[str, Any]], str], set[str]],
    apply_mapping: Callable[[dict[str, Any], dict[str, Any]], str],
    map_fields: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    clean_text: Callable[[Any], str],
) -> list[PreparedLogRecord]:
    globally_ignorable_in_keys = find_ignorable_keys(logs, "in_fields")
    globally_ignorable_out_keys = find_ignorable_keys(logs, "out_fields")

    def _normalize_lookup_text(value: Any) -> str:
        return str(value or "").strip().lower().replace(" ", "")

    def _match_tokens(text: str, token_groups: list[tuple[str, ...]]) -> bool:
        normalized = _normalize_lookup_text(text)
        return any(all(token in normalized for token in tokens) for tokens in token_groups)

    def _pick_curated_value(
        fields: dict[str, Any],
        mapping: dict[str, Any],
        token_groups: list[tuple[str, ...]],
    ) -> str:
        for key, value in fields.items():
            label = mapping.get(key, key)
            if _match_tokens(label, token_groups) or _match_tokens(key, token_groups):
                return clean_text(value)
        return ""

    def _pick_value_by_field_code(
        fields: dict[str, Any],
        field_code: str | None,
        normalizer: Callable[[Any], str] | None = None,
    ) -> str:
        code = clean_text(field_code).upper()
        if not code:
            return ""
        value = fields.get(code)
        if value in (None, ""):
            return ""
        if normalizer is not None:
            return normalizer(value)
        return clean_text(value)

    def _normalize_dsr_value(value: Any) -> str:
        text = clean_text(value)
        if not text:
            return ""
        number = _parse_number(text)
        if number is None:
            return ""
        if number < 0 or number >= 100:
            return ""
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.2f}".rstrip("0").rstrip(".")

    def _normalize_rate_value(value: Any) -> str:
        text = clean_text(value)
        if not text:
            return ""
        number = _parse_number(text)
        if number is None:
            return ""
        if number < 0 or number >= 100:
            return ""
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.2f}".rstrip("0").rstrip(".")

    def _product_display_name(product_code: Any) -> str:
        code = clean_text(product_code).upper()
        return PRODUCT_DISPLAY_NAMES.get(code, code)

    def _pick_product_rate_value(
        product_code: Any,
        out_fields: dict[str, Any],
        out_mapping: dict[str, Any],
        in_fields: dict[str, Any],
        in_mapping: dict[str, Any],
        features: dict[str, Any],
    ) -> str:
        rate_field_code = PRODUCT_RATE_FIELD_CODES.get(clean_text(product_code).upper())
        if rate_field_code:
            for fields in (out_fields, in_fields):
                if rate_field_code in fields:
                    normalized = _normalize_rate_value(fields.get(rate_field_code))
                    if normalized:
                        return normalized

        for fields, mapping in ((out_fields, out_mapping), (in_fields, in_mapping)):
            for key, value in fields.items():
                label = mapping.get(key, key)
                if _match_tokens(label, [("산출", "금리"), ("적용", "금리"), ("금리",)]) or _match_tokens(key, [("산출", "금리"), ("적용", "금리"), ("금리",)]):
                    normalized = _normalize_rate_value(value)
                    if normalized:
                        return normalized

        feature_rate = features.get("applied_rate")
        return _normalize_rate_value(feature_rate)

    def _pick_curated_dsr_value(
        product_code: Any,
        out_fields: dict[str, Any],
        out_mapping: dict[str, Any],
        in_fields: dict[str, Any],
        in_mapping: dict[str, Any],
    ) -> str:
        exact_code_value = _pick_value_by_field_code(
            out_fields,
            DSR_FIELD_CODES.get(clean_text(product_code).upper()),
            _normalize_dsr_value,
        )
        if exact_code_value:
            return exact_code_value
        for fields, mapping in ((out_fields, out_mapping), (in_fields, in_mapping)):
            for key, value in fields.items():
                label = mapping.get(key, key)
                if _match_tokens(label, [("dsr",)]) or _match_tokens(key, [("dsr",)]):
                    normalized = _normalize_dsr_value(value)
                    if normalized:
                        return normalized
        return ""

    def _pick_recognized_income_value(
        product_code: Any,
        out_fields: dict[str, Any],
        in_fields: dict[str, Any],
    ) -> str:
        code = RECOGNIZED_INCOME_FIELD_CODES.get(clean_text(product_code).upper())
        for fields in (out_fields, in_fields):
            exact_code_value = _pick_value_by_field_code(fields, code)
            if exact_code_value:
                return exact_code_value
        return ""

    def _extract_decision_flag(
        raw_out_data: str,
        raw_in_data: str,
        out_fields: dict[str, Any],
        in_fields: dict[str, Any],
    ) -> str:
        for source_fields in (out_fields, in_fields):
            for value in source_fields.values():
                normalized = clean_text(value).upper().replace(" ", "")
                if normalized == "DR":
                    return "DR"
                if normalized == "AA":
                    return "AA"

        raw_combined = f"{clean_text(raw_out_data)} {clean_text(raw_in_data)}".upper()
        for pattern, flag in (
            (r"(?<![A-Z])DR(?![A-Z])", "DR"),
            (r"(?<![A-Z])AA(?![A-Z])", "AA"),
        ):
            if re.search(pattern, raw_combined):
                return flag
        return ""

    def _extract_decision_value(
        product_code: Any,
        in_fields: dict[str, Any],
        out_fields: dict[str, Any],
        reject_reason_text: str,
        raw_out_data: str = "",
        raw_in_data: str = "",
    ) -> str:
        exact_amount_code = DECISION_AMOUNT_FIELD_CODES.get(clean_text(product_code).upper())
        exact_amount_value = _pick_value_by_field_code(
            out_fields,
            exact_amount_code,
            clean_text,
        )
        if exact_amount_value:
            amount_number = _parse_number(exact_amount_value)
            if amount_number is not None:
                return "거절" if float(amount_number) == 0 else "승인"

        decision_flag = _extract_decision_flag(
            raw_out_data,
            raw_in_data,
            out_fields,
            in_fields,
        )
        if decision_flag == "DR":
            return "거절"
        if decision_flag == "AA":
            return "승인"
        return ""

    def _build_curated_fields(
        product_code: Any,
        in_fields: dict[str, Any],
        out_fields: dict[str, Any],
        in_mapping: dict[str, Any],
        out_mapping: dict[str, Any],
        features: dict[str, Any],
        reject_reason_text: str,
        raw_out_data: str,
        raw_in_data: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        curated_in: dict[str, str] = {}
        curated_out: dict[str, str] = {}

        for display_name, token_groups in IN_FIELD_SPECS:
            value = _pick_curated_value(in_fields, in_mapping, token_groups)
            if not value and display_name == "소득":
                recognized_income = features.get("recognized_income")
                if recognized_income not in (None, ""):
                    value = clean_text(recognized_income)
            if not value and display_name == "소득":
                annual_income = features.get("annual_income")
                if annual_income not in (None, ""):
                    value = clean_text(annual_income)
            if not value and display_name == "대출금액":
                available_amount = features.get("available_amount")
                if available_amount not in (None, ""):
                    value = clean_text(available_amount)
            if value:
                curated_in[display_name] = value

        for display_name, token_groups in OUT_FIELD_SPECS:
            value = _pick_curated_value(out_fields, out_mapping, token_groups)
            if display_name == "승인 여부":
                value = _extract_decision_value(
                    product_code,
                    in_fields,
                    out_fields,
                    reject_reason_text,
                    raw_out_data=raw_out_data,
                    raw_in_data=raw_in_data,
                )
            elif display_name == "산출금리":
                value = _pick_product_rate_value(
                    product_code,
                    out_fields,
                    out_mapping,
                    in_fields,
                    in_mapping,
                    features,
                )
            elif display_name == "DSR":
                value = _pick_curated_dsr_value(
                    product_code,
                    out_fields,
                    out_mapping,
                    in_fields,
                    in_mapping,
                )
            if value:
                curated_out[display_name] = value

        if reject_reason_text:
            curated_out["거절사유"] = reject_reason_text

        return curated_in, curated_out

    def _format_curated_lines(title: str, fields: dict[str, str], ordered_names: list[str]) -> str:
        lines = [f"[{title}]"]
        for name in ordered_names:
            lines.append(f"- {name}: {fields.get(name, '-')}")
        return "\n".join(lines)

    def _parse_number(text: str):
        if not text:
            return None
        match = re.search(r"[-+]?[0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?%?", str(text))
        if not match:
            return None
        value = match.group(0)
        if value.endswith("%"):
            try:
                return float(value[:-1].replace(",", ""))
            except Exception:
                return None
        try:
            return float(value.replace(",", ""))
        except Exception:
            return None

    def _extract_features(log_item: dict[str, Any]) -> dict[str, Any]:
        features = {
            "available_amount": None,
            "applied_rate": None,
            "ko_codes": [],
            "case_id": None,
            "product_code": log_item.get("product") or log_item.get("product_code"),
            "loan_term_months": None,
            "loan_term_raw": None,
            "credit_grade": None,
            "credit_score": None,
            "kcb_score": None,
            "nice_score": None,
            "age": None,
            "foreigner": None,
            "annual_income": None,
            "recognized_income": None,
            "purpose": None,
            "collateral": None,
            "interest_type": None,
            "dti": None,
            "dsr_ratio": None,
        }

        in_fields = log_item.get("in_fields", {}) or {}
        out_fields = log_item.get("out_fields", {}) or {}
        in_mapping = log_item.get("in_mapping", {}) or {}
        out_mapping = log_item.get("out_mapping", {}) or {}

        scan_fields = []
        product_code = clean_text(features.get("product_code")).upper()
        for source_fields, mapping in ((in_fields, in_mapping), (out_fields, out_mapping)):
            for key, value in source_fields.items():
                label = str(mapping.get(key, key))
                scan_fields.append((key, label, value))
                if features["case_id"] is None and str(key).lower() in {
                    "case_id",
                    "id",
                    "req_no",
                    "request_id",
                    "caseid",
                }:
                    features["case_id"] = str(value)

        for key, label, value in scan_fields:
            value_text = "" if value is None else str(value)
            label_lower = label.lower()
            value_lower = value_text.lower()

            if features["available_amount"] is None and clean_text(key).upper() == DECISION_AMOUNT_FIELD_CODES.get(product_code, ""):
                number = _parse_number(value_text)
                if number is not None:
                    features["available_amount"] = int(float(number))

            if features["available_amount"] is None and (
                any(token in label_lower for token in ("대출", "한도", "금액", "limit", "available"))
                or re.search(r"\b(원|만원|억|천원|만)\b", value_lower)
            ):
                number = _parse_number(value_text)
                if number is not None:
                    multiplier = 1
                    if "만원" in value_lower or ("만" in value_lower and re.search(r"\d+만", value_lower)):
                        multiplier = 10000
                    elif "억" in value_lower:
                        multiplier = 100000000
                    elif "천" in value_lower and "원" in value_lower:
                        multiplier = 1000
                    features["available_amount"] = int(float(number) * multiplier)

            if features["loan_term_months"] is None and (
                "개월" in value_lower
                or "년" in value_lower
                or any(token in label_lower for token in ("기간", "term", "months", "years"))
            ):
                match = re.search(r"(\d+(?:\.\d+)?)\s*(개월|월|년|yr|y|months|years)?", value_lower)
                if match:
                    raw_value = float(match.group(1))
                    unit = (match.group(2) or "").strip()
                    if unit in ("년", "y", "yr", "years"):
                        months = int(raw_value * 12)
                    else:
                        months = int(raw_value)
                    features["loan_term_months"] = months
                    features["loan_term_raw"] = value_text

            if features["applied_rate"] is None and (
                any(token in label_lower for token in ("금리", "rate", "이율", "percent"))
                or "%" in value_lower
            ):
                number = _parse_number(value_text)
                if number is not None:
                    features["applied_rate"] = float(number)

            if features["credit_grade"] is None and any(token in label_lower for token in ("등급", "grade", "신용")):
                match = re.search(r"\b([A-D][+-]?|S|[0-9]{3,4})\b", value_text, re.I)
                if match:
                    grade_value = match.group(1)
                    if grade_value.isdigit():
                        features["credit_score"] = int(grade_value)
                    else:
                        features["credit_grade"] = grade_value.upper()

            if features["recognized_income"] is None and clean_text(key).upper() == RECOGNIZED_INCOME_FIELD_CODES.get(product_code, ""):
                number = _parse_number(value_text)
                if number is not None:
                    features["recognized_income"] = int(float(number))
                    if features["annual_income"] is None:
                        features["annual_income"] = int(float(number))

            if features["annual_income"] is None and any(token in label_lower for token in ("소득", "연소득", "income", "salary")):
                number = _parse_number(value_text)
                if number is not None:
                    multiplier = 1
                    if "만원" in value_lower or ("만" in value_lower and re.search(r"\d+만", value_lower)):
                        multiplier = 10000
                    elif "억" in value_lower:
                        multiplier = 100000000
                    features["annual_income"] = int(float(number) * multiplier)

            if features["recognized_income"] is None and any(token in label_lower for token in ("인정소득", "스크래핑소득", "최종연소득")):
                number = _parse_number(value_text)
                if number is not None:
                    multiplier = 1
                    if "만원" in value_lower or ("만" in value_lower and re.search(r"\d+만", value_lower)):
                        multiplier = 10000
                    elif "억" in value_lower:
                        multiplier = 100000000
                    features["recognized_income"] = int(float(number) * multiplier)

            if features["age"] is None and any(token in label_lower for token in ("연령", "나이", "age")):
                number = _parse_number(value_text)
                if number is not None:
                    features["age"] = int(number)

            if features["foreigner"] is None and any(token in label_lower for token in ("외국인", "국적", "내외국인", "foreigner")):
                features["foreigner"] = value_text

            if features["kcb_score"] is None and "kcb" in label_lower:
                number = _parse_number(value_text)
                if number is not None:
                    features["kcb_score"] = int(number)

            if features["nice_score"] is None and "nice" in label_lower:
                number = _parse_number(value_text)
                if number is not None:
                    features["nice_score"] = int(number)

            if features["dti"] is None and "dti" in label_lower:
                number = _parse_number(value_text)
                if number is not None:
                    features["dti"] = number

            if features["dsr_ratio"] is None and clean_text(key).upper() == DSR_FIELD_CODES.get(product_code, ""):
                number = _parse_number(value_text)
                if number is not None:
                    features["dsr_ratio"] = number

            if features["dsr_ratio"] is None and "dsr" in label_lower:
                number = _parse_number(value_text)
                if number is not None:
                    features["dsr_ratio"] = number

            if features["purpose"] is None and any(token in label_lower for token in ("용도", "purpose")):
                features["purpose"] = value_text
            if features["collateral"] is None and any(token in label_lower for token in ("담보", "collateral")):
                features["collateral"] = value_text
            if features["interest_type"] is None and any(token in label_lower for token in ("변동", "고정", "fixed", "variable")):
                features["interest_type"] = value_text

            if re.match(r"^(K|KO)[0-9A-Za-z_\-]*$", str(key), re.I) or re.match(r"^(K|KO)[0-9A-Za-z_\-]*$", label, re.I):
                features["ko_codes"].append(str(key))
            for match in re.findall(r"\b(KO?-?[0-9A-Za-z_]+)\b", value_text):
                features["ko_codes"].append(match)

        features["ko_codes"] = list(dict.fromkeys(features["ko_codes"]))
        return features

    prepared_records: list[PreparedLogRecord] = []
    for index, log in enumerate(logs):
        try:
            logger.info(
                "---- RAG INGEST: original log ----\n%s",
                json.dumps(log, ensure_ascii=False, indent=2),
            )
        except Exception:
            logger.info("---- RAG INGEST: original log ---- %s", str(log))

        if should_skip_log(log):
            logger.info(
                "Skipping FAISS ingest for product code: %s",
                log.get("product") or log.get("product_code") or "",
            )
            continue

        if show_progress:
            print(f"로그 처리 중... {index + 1}/{len(logs)}")

        in_fields = sanitize_fields(log.get("in_fields", {}) or {}, globally_ignorable_in_keys)
        out_fields = sanitize_fields(log.get("out_fields", {}) or {}, globally_ignorable_out_keys)
        in_mapping = sanitize_mapping(log.get("in_mapping", {}) or {})
        out_mapping = sanitize_mapping(log.get("out_mapping", {}) or {})
        reject_reason_details = log.get("reject_reason_details", []) or []

        sanitized_log = dict(log)
        sanitized_log["in_fields"] = in_fields
        sanitized_log["out_fields"] = out_fields
        sanitized_log["in_mapping"] = in_mapping
        sanitized_log["out_mapping"] = out_mapping

        in_text = apply_mapping(in_fields, in_mapping)
        out_text = apply_mapping(out_fields, out_mapping)
        reject_reason_text = ", ".join(
            clean_text(
                str(item.get('description') or item.get('code') or '').strip()
            )
            for item in reject_reason_details
            if clean_text(
                str(item.get('description') or item.get('code') or '').strip()
            )
        )
        features = _extract_features(sanitized_log)
        curated_in, curated_out = _build_curated_fields(
            log.get("product") or log.get("product_code"),
            in_fields,
            out_fields,
            in_mapping,
            out_mapping,
            features,
            reject_reason_text,
            clean_text(log.get("raw_out_data")),
            clean_text(log.get("raw_in_data")),
        )
        curated_in_text = _format_curated_lines(
            "고객 정보",
            curated_in,
            [name for name, _ in IN_FIELD_SPECS],
        )
        curated_out_text = _format_curated_lines(
            "심사 결과",
            curated_out,
            [name for name, _ in OUT_FIELD_SPECS] + ["거절사유"],
        )
        product_code = clean_text(log.get("product") or log.get("product_code"))
        product_display = _product_display_name(product_code)
        full_text = clean_text(
            f"[상품] {product_display}\n{curated_in_text}\n{curated_out_text}"
        )

        mapped_in = curated_in
        mapped_out = curated_out

        prepared_records.append(
            {
                "product": log.get("product"),
                "product_display": product_display,
                "full_text": full_text[:2000],
                "in_text": curated_in_text,
                "out_text": curated_out_text,
                "reject_reason_text": reject_reason_text,
                "mapped_in": mapped_in,
                "mapped_out": mapped_out,
                "reject_reason_codes": log.get("reject_reason_codes") or [],
                "reject_reason_details": [
                    {
                        **item,
                        "description": clean_text(item.get("description") or item.get("code") or ""),
                    }
                    for item in reject_reason_details
                ],
                "features": {
                    **features,
                    "ko_codes": list(
                        dict.fromkeys(
                            (log.get("reject_reason_codes") or [])
                            + (features.get("ko_codes") or [])
                        )
                    ),
                },
            }
        )

    return prepared_records


def build_log_documents(records: list[PreparedLogRecord], store_name: str) -> list[Document]:
    documents: list[Document] = []
    for record in records:
        page_content, template_metadata = _compose_structured_log_payload(record)
        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "type": "log",
                    "store": store_name,
                    "product": record.get("product"),
                    "product_name": record.get("product_display") or record.get("product"),
                    "name": f"structured log: {record.get('product_display') or record.get('product') or 'case'}",
                    "in_fields": {},
                    "out_fields": {},
                    "reject_reason_codes": record.get("reject_reason_codes") or [],
                    "reject_reason_details": record.get("reject_reason_details") or [],
                    "features": template_metadata,
                },
            )
        )
    return documents


def build_log_ingest_preview(
    records: list[PreparedLogRecord],
    store_name: str,
    *,
    preview_limit: int = 5,
) -> dict[str, Any]:
    documents = build_log_documents(records, store_name)
    preview_items: list[dict[str, Any]] = []
    context_blocks: list[str] = []

    for index, (record, document) in enumerate(
        zip(records[:preview_limit], documents[:preview_limit]), start=1
    ):
        product_code = str(record.get("product") or "").strip().upper()
        product_name = str(
            record.get("product_display") or product_code or f"case-{index}"
        ).strip()
        doc_metadata = getattr(document, "metadata", {}) or {}
        features = doc_metadata.get("features") or {}
        page_content = str(getattr(document, "page_content", "") or "").strip()
        context_blocks.append(
            "\n".join(
                part
                for part in [
                    f"[케이스 {index}]",
                    f"[상품] {product_code or '-'} {product_name}",
                    page_content,
                ]
                if part
            )
        )
        preview_items.append(
            {
                "product": product_code,
                "product_name": product_name,
                "page_content": page_content,
                "features": features,
                "reject_reason_codes": record.get("reject_reason_codes") or [],
            }
        )

    payload = {
        "store": store_name,
        "record_count": len(records),
        "document_count": len(documents),
        "preview_limit": preview_limit,
        "preview_documents": preview_items,
    }
    context_text = "\n\n".join(block for block in context_blocks if block).strip()

    return {
        "source": "faiss_logs_db.py structured ingest",
        "mode": "faiss_ingest",
        "user_input": (
            f"실제 심사로그 {len(records)}건을 OLLAMA 호출 없이 정제하여 "
            f"{store_name} FAISS store에 적재"
        ),
        "context": context_text or "관련 데이터가 없습니다.",
        "prompt": json.dumps(payload, ensure_ascii=False, indent=2),
        "record_count": len(records),
        "document_count": len(documents),
        "preview_documents": preview_items,
    }


def format_log_search_results(
    docs: list[Document],
    apply_mapping: Callable[[dict[str, Any], dict[str, Any]], str],
) -> list[str]:
    formatted: list[str] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        in_fields = meta.get("in_fields") or {}
        out_fields = meta.get("out_fields") or {}
        features = meta.get("features") or {}
        if in_fields or out_fields:
            in_text = apply_mapping(in_fields, {})
            out_text = apply_mapping(out_fields, {})
            formatted.append(f"[상품] {meta.get('product_name') or meta.get('product')} [IN] {in_text} [OUT] {out_text}")
            continue
        formatted.append(
            f"[상품] {meta.get('product_name') or meta.get('product')} "
            f"[심사결과] {features.get('심사결과') or '-'} "
            f"[DTI] {features.get('dti') or '-'} "
            f"[DSR] {features.get('dsr비율') or '-'} "
            f"[KO] {', '.join(features.get('KNOCK-OUT 사유') or []) or '-'}"
        )
    return formatted
