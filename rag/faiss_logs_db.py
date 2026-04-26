from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.documents import Document


PreparedLogRecord = dict[str, Any]


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
            clean_text(item.get("description") or item.get("code") or "")
            for item in reject_reason_details
            if clean_text(item.get("description") or item.get("code") or "")
        )
        full_text = clean_text(
            f"[상품] {clean_text(log.get('product'))} [IN] {in_text} [OUT] {out_text} [거절사유] {reject_reason_text or '-'}"
        )
        print(f"변환된 로그:\n{full_text[:300]}")

        features = _extract_features(sanitized_log)
        mapped_in = map_fields(in_fields, in_mapping)
        mapped_out = map_fields(out_fields, out_mapping)

        prepared_records.append(
            {
                "product": log.get("product"),
                "full_text": full_text[:2000],
                "in_text": in_text,
                "out_text": out_text,
                "reject_reason_text": reject_reason_text,
                "mapped_in": mapped_in,
                "mapped_out": mapped_out,
                "reject_reason_codes": log.get("reject_reason_codes") or [],
                "reject_reason_details": reject_reason_details,
                "features": features,
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
        formatted.append(f"[상품] {meta.get('product')} [IN] {in_text} [OUT] {out_text}")
    return formatted
