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


IN_FIELD_SPECS = [
    ("소득", [("소득",), ("연소득",), ("최종연소득",), ("income",)]),
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


def prepare_log_records(
    logs: list[dict[str, Any]],
    logger,
    *,
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
        out_fields: dict[str, Any],
        out_mapping: dict[str, Any],
        in_fields: dict[str, Any],
        in_mapping: dict[str, Any],
    ) -> str:
        for fields, mapping in ((out_fields, out_mapping), (in_fields, in_mapping)):
            for key, value in fields.items():
                label = mapping.get(key, key)
                if _match_tokens(label, [("dsr",)]) or _match_tokens(key, [("dsr",)]):
                    normalized = _normalize_dsr_value(value)
                    if normalized:
                        return normalized
        return ""

    def _extract_decision_value(
        in_fields: dict[str, Any],
        out_fields: dict[str, Any],
        reject_reason_text: str,
        raw_out_data: str = "",
        raw_in_data: str = "",
    ) -> str:
        raw_combined = f"{clean_text(raw_out_data)} {clean_text(raw_in_data)}".strip()
        raw_upper = raw_combined.upper()
        raw_lower = raw_combined.lower()

        if "decline" in raw_lower or "reject" in raw_lower or "denied" in raw_lower:
            return "거절"
        if "accept" in raw_lower or "approved" in raw_lower or "approve" in raw_lower:
            return "승인"
        if re.search(r"R\d{4}DR(?=\s|[A-Z]\d{4}|$)", raw_upper):
            return "거절"
        if re.search(r"R\d{4}AA(?=\s|[A-Z]\d{4}|$)", raw_upper):
            return "승인"

        for source_fields in (out_fields, in_fields):
            for value in source_fields.values():
                cleaned = clean_text(value)
                text = cleaned.lower()
                normalized_code = re.sub(r"\s+", "", cleaned).upper()
                if not text:
                    continue
                if normalized_code == "DR":
                    return "거절"
                if normalized_code == "AA":
                    return "승인"
                if any(token in text for token in ("거절", "불가", "reject", "denied", "불허")):
                    return "거절"
                if any(token in text for token in ("승인", "approve", "approved", "ok")):
                    return "승인"
        if reject_reason_text:
            return "거절"
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
            "annual_income": None,
            "purpose": None,
            "collateral": None,
            "interest_type": None,
        }

        in_fields = log_item.get("in_fields", {}) or {}
        out_fields = log_item.get("out_fields", {}) or {}
        in_mapping = log_item.get("in_mapping", {}) or {}
        out_mapping = log_item.get("out_mapping", {}) or {}

        scan_fields = []
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

            if features["annual_income"] is None and any(token in label_lower for token in ("소득", "연소득", "income", "salary")):
                number = _parse_number(value_text)
                if number is not None:
                    multiplier = 1
                    if "만원" in value_lower or ("만" in value_lower and re.search(r"\d+만", value_lower)):
                        multiplier = 10000
                    elif "억" in value_lower:
                        multiplier = 100000000
                    features["annual_income"] = int(float(number) * multiplier)

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
                f"{item.get('code') or ''} {item.get('description') or ''}".strip()
            )
            for item in reject_reason_details
            if clean_text(
                f"{item.get('code') or ''} {item.get('description') or ''}".strip()
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
        print(f"변환된 로그:\n{full_text[:300]}")

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
                "reject_reason_details": reject_reason_details,
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
    return [
        Document(
            page_content=record["full_text"],
            metadata={
                "type": "log",
                "store": store_name,
                "product": record.get("product"),
                "product_name": record.get("product_display") or record.get("product"),
                "in_fields": record.get("mapped_in") or {},
                "out_fields": record.get("mapped_out") or {},
                "reject_reason_codes": record.get("reject_reason_codes") or [],
                "reject_reason_details": record.get("reject_reason_details") or [],
                "features": record.get("features") or {},
            },
        )
        for record in records
    ]


def format_log_search_results(
    docs: list[Document],
    apply_mapping: Callable[[dict[str, Any], dict[str, Any]], str],
) -> list[str]:
    formatted: list[str] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        in_fields = meta.get("in_fields") or {}
        out_fields = meta.get("out_fields") or {}
        in_text = apply_mapping(in_fields, {})
        out_text = apply_mapping(out_fields, {})
        formatted.append(f"[상품] {meta.get('product_name') or meta.get('product')} [IN] {in_text} [OUT] {out_text}")
    return formatted
