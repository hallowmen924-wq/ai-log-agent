from __future__ import annotations

import datetime
import pathlib
import re
from typing import Any

from analyzer.log_parser import parse_logs_fast

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_LOG_CANDIDATES = [
    PROJECT_ROOT / "logs" / "stdout.log.20260407.txt",
    PROJECT_ROOT / "data" / "logs" / "stdout.log.20260407.txt",
]
TARGET_LOG = PROJECT_ROOT / "logs" / "generated_live.log"
SUPPORTED_PRODUCTS = {"C6", "C9", "C11", "C12"}
APPROVAL_PATCH_BY_PRODUCT = {
    "C11": {
        "amount_code": "R0008",
        "amount_value": "5000000",
        "decision_flag_code": "R0050",
    },
    "C12": {
        "amount_code": "R0002",
        "amount_value": "3000000",
        "decision_flag_code": "R0015",
    },
}


def _resolve_source_log() -> pathlib.Path:
    for candidate in SOURCE_LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    return SOURCE_LOG_CANDIDATES[0]


def _load_seed_pairs() -> list[dict[str, str]]:
    source_log = _resolve_source_log()
    if not source_log.exists():
        return []

    raw_logs = source_log.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_logs_fast(raw_logs)
    pairs: list[dict[str, str]] = []
    current_in_line: str | None = None

    for line in raw_logs.splitlines():
        if "in_data = [" in line:
            current_in_line = line
            continue

        if "out_data = [" in line and current_in_line:
            in_payload = current_in_line.split("in_data = [", 1)[-1].rstrip("]")
            product = "UNKNOWN"
            for row in parsed:
                if row.get("in_data") == in_payload:
                    product = row.get("product", "UNKNOWN")
                    break

            if product in SUPPORTED_PRODUCTS:
                pairs.append(
                    {
                        "product": product,
                        "in_line": current_in_line,
                        "out_line": line,
                    }
                )
            current_in_line = None

    return pairs


_SEED_PAIRS: list[dict[str, str]] = []
_SEED_INDEX_BY_PRODUCT: dict[str, int] = {}


def _get_seed_pairs(force_reload: bool = False) -> list[dict[str, str]]:
    global _SEED_PAIRS
    if force_reload or not _SEED_PAIRS:
        _SEED_PAIRS = _load_seed_pairs()
    return _SEED_PAIRS


def _set_fixed_field_value(raw_payload: str, field_code: str, value: str, width: int = 15) -> str:
    normalized_value = str(value or "")[:width].ljust(width)
    pattern = re.compile(rf"({re.escape(field_code)})(.{{{width}}})")
    return pattern.sub(rf"\1{normalized_value}", raw_payload, count=1)


def _strip_reject_codes(raw_payload: str) -> str:
    cleaned = re.sub(r"KORLT", "     ", raw_payload, flags=re.I)
    cleaned = re.sub(r"K\d{3}", "    ", cleaned, flags=re.I)
    return cleaned


def _apply_forced_decision(raw_payload: str, product: str, force_decision: str | None) -> str:
    normalized_decision = str(force_decision or "").strip()
    if normalized_decision != "승인":
        return raw_payload

    patch = APPROVAL_PATCH_BY_PRODUCT.get(product)
    if not patch:
        return raw_payload

    updated_payload = _set_fixed_field_value(
        raw_payload,
        str(patch.get("amount_code") or ""),
        str(patch.get("amount_value") or ""),
    )
    decision_flag_code = str(patch.get("decision_flag_code") or "").strip()
    if decision_flag_code:
        updated_payload = _set_fixed_field_value(updated_payload, decision_flag_code, "AA")
    updated_payload = _strip_reject_codes(updated_payload)
    return updated_payload


def append_synthetic_log(product: str | None = None, force_decision: str | None = None) -> dict[str, Any]:
    seed_pairs = _get_seed_pairs(force_reload=True)

    normalized_product = str(product or "").strip().upper() or None
    if normalized_product and normalized_product not in SUPPORTED_PRODUCTS:
        raise ValueError(f"지원하지 않는 상품코드입니다: {product}")

    if not seed_pairs:
        raise RuntimeError("지원 가능한 테스트 로그 시드가 없습니다.")

    if normalized_product:
        filtered_pairs = [
            pair for pair in seed_pairs if pair.get("product") == normalized_product
        ]
        if not filtered_pairs:
            raise RuntimeError(
                f"상품 {normalized_product} 에 해당하는 테스트 로그 시드가 없습니다."
            )
        pair_index = _SEED_INDEX_BY_PRODUCT.get(normalized_product, 0)
        pair = filtered_pairs[pair_index % len(filtered_pairs)]
        _SEED_INDEX_BY_PRODUCT[normalized_product] = pair_index + 1
    else:
        pair_index = _SEED_INDEX_BY_PRODUCT.get("__all__", 0)
        pair = seed_pairs[pair_index % len(seed_pairs)]
        _SEED_INDEX_BY_PRODUCT["__all__"] = pair_index + 1

    generated_at = datetime.datetime.now()
    prefix = generated_at.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    thread_no = ((_SEED_INDEX_BY_PRODUCT.get("__all__", 0) or 1) % 8) + 1
    line_prefix = (
        f"{prefix} [[ACTIVE] ExecuteThread: '{thread_no}' for queue: 'weblogic.kernel.Default (self-tuning)'] "
        f"INFO  [com.nice.rclips.server.online.main.RclipsOnlineServlet] "
    )

    in_payload = pair["in_line"].split("in_data = [", 1)[-1]
    out_payload = pair["out_line"].split("out_data = [", 1)[-1]
    out_payload = _apply_forced_decision(out_payload, pair["product"], force_decision)

    TARGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_LOG.open("a", encoding="utf-8") as file:
        file.write(f"{line_prefix}in_data = [{in_payload}\n")
        file.write(f"{line_prefix}out_data = [{out_payload}\n")
        file.write(f"{line_prefix}process time[WAS]: 0.0{thread_no}\n")

    return {
        "product": pair["product"],
        "forced_decision": force_decision,
        "file_path": str(TARGET_LOG),
        "generated_at": generated_at.isoformat(),
    }
