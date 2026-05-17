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
FIELD_WIDTH = 15


def _profile(decision: str, k_codes: list[str], in_fields: dict[str, str], out_fields: dict[str, str]) -> dict[str, Any]:
    return {"decision": decision, "k_codes": k_codes, "in": in_fields, "out": out_fields}


# Rotating profiles keep synthetic logs useful for statistics. The target mix is
# close to the business expectation: C9 80%, C6 20%, C11/C12 30% approval.
PRODUCT_SYNTHETIC_PROFILES: dict[str, list[dict[str, Any]]] = {
    "C9": [
        _profile("승인", [], {"A1004": "29"}, {"R0002": "40000000", "R0003": "13.1", "R1003": "12.7", "R0065": "0", "R0067": "4", "R0070": "3"}),
        _profile("승인", ["K707"], {"A1004": "41"}, {"R0002": "30000000", "R0003": "14.8", "R1003": "14.2", "R0065": "0", "R0067": "5", "R0070": "4"}),
        _profile("승인", [], {"A1004": "36"}, {"R0002": "36500000", "R0003": "12.9", "R1003": "12.3", "R0065": "0", "R0067": "3", "R0070": "2"}),
        _profile("승인", ["K707"], {"A1004": "52"}, {"R0002": "20000000", "R0003": "15.6", "R1003": "15.0", "R0065": "0", "R0067": "6", "R0070": "5"}),
        _profile("승인", [], {"A1004": "44"}, {"R0002": "18000000", "R0003": "14.4", "R1003": "13.8", "R0065": "0", "R0067": "5", "R0070": "4"}),
        _profile("승인", ["K707"], {"A1004": "33"}, {"R0002": "9900000", "R0003": "16.9", "R1003": "16.1", "R0065": "0", "R0067": "7", "R0070": "6"}),
        _profile("승인", [], {"A1004": "48"}, {"R0002": "25000000", "R0003": "15.2", "R1003": "14.7", "R0065": "0", "R0067": "5", "R0070": "5"}),
        _profile("승인", [], {"A1004": "27"}, {"R0002": "12000000", "R0003": "17.3", "R1003": "16.8", "R0065": "0", "R0067": "8", "R0070": "7"}),
        _profile("거절", ["K701"], {"A1004": "46"}, {"R0002": "3000000", "R0003": "18.9", "R1003": "18.5", "R0065": "1", "R0067": "10", "R0070": "9"}),
        _profile("거절", ["K701", "K707"], {"A1004": "58"}, {"R0002": "3000000", "R0003": "19.4", "R1003": "18.8", "R0065": "1", "R0067": "10", "R0070": "10"}),
    ],
    "C6": [
        _profile("승인", [], {"A1002": "20000000", "A1004": "34", "A2046": "0"}, {"R0007": "0", "R0012": "12.8", "R0020": "27000000", "R0023": "1", "R0077": "1", "R0039": "5", "R0047": "2"}),
        _profile("승인", [], {"A1002": "10000000", "A1004": "42", "A2046": "1"}, {"R0007": "0", "R0012": "13.7", "R0020": "15000000", "R0023": "1", "R0077": "1", "R0039": "6", "R0047": "3"}),
        _profile("거절", ["K373", "K374"], {"A1002": "5000000", "A1004": "31", "A2046": "0"}, {"R0007": "1", "R0012": "17.4", "R0020": "0", "R0023": "2", "R0077": "3", "R0039": "12", "R0047": "4"}),
        _profile("거절", ["K364"], {"A1002": "12000000", "A1004": "55", "A2046": "0"}, {"R0007": "1", "R0012": "18.1", "R0020": "1200000", "R0023": "2", "R0077": "3", "R0039": "13", "R0047": "4"}),
        _profile("거절", ["K361", "K354"], {"A1002": "30000000", "A1004": "47", "A2046": "1"}, {"R0007": "1", "R0012": "18.9", "R0020": "0", "R0023": "2", "R0077": "3", "R0039": "14", "R0047": "5"}),
        _profile("거절", ["K366"], {"A1002": "7000000", "A1004": "29", "A2046": "0"}, {"R0007": "1", "R0012": "16.8", "R0020": "2500000", "R0023": "2", "R0077": "3", "R0039": "10", "R0047": "4"}),
        _profile("거절", ["K363", "K357"], {"A1002": "18000000", "A1004": "39", "A2046": "0"}, {"R0007": "1", "R0012": "19.2", "R0020": "0", "R0023": "2", "R0077": "3", "R0039": "15", "R0047": "5"}),
        _profile("거절", ["K365"], {"A1002": "15000000", "A1004": "62", "A2046": "1"}, {"R0007": "1", "R0012": "18.4", "R0020": "1500000", "R0023": "2", "R0077": "3", "R0039": "12", "R0047": "4"}),
        _profile("거절", ["K351"], {"A1002": "8000000", "A1004": "25", "A2046": "0"}, {"R0007": "1", "R0012": "17.7", "R0020": "0", "R0023": "2", "R0077": "3", "R0039": "11", "R0047": "4"}),
        _profile("거절", ["K355", "K362"], {"A1002": "22000000", "A1004": "50", "A2046": "1"}, {"R0007": "1", "R0012": "19.6", "R0020": "3000000", "R0023": "2", "R0077": "3", "R0039": "15", "R0047": "5"}),
    ],
    "C11": [
        _profile("승인", [], {"A1004": "38", "A1008": "56101", "A2043": "85000000"}, {"R0008": "28000000", "R0021": "1", "R0022": "0", "R0046": "9.8", "R0050": "AA", "R0028": "2"}),
        _profile("승인", [], {"A1004": "45", "A1008": "47121", "A2043": "62000000"}, {"R0008": "18000000", "R0021": "1", "R0022": "0", "R0046": "11.4", "R0050": "AA", "R0028": "3"}),
        _profile("승인", [], {"A1004": "33", "A1008": "62010", "A2043": "120000000"}, {"R0008": "35000000", "R0021": "1", "R0022": "0", "R0046": "8.9", "R0050": "AA", "R0028": "2"}),
        _profile("거절", ["K351"], {"A1004": "52", "A1008": "56101", "A2043": "18000000"}, {"R0008": "0", "R0021": "0", "R0022": "1", "R0046": "16.9", "R0050": "DR", "R0028": "9"}),
        _profile("거절", ["K354"], {"A1004": "60", "A1008": "47121", "A2043": "24000000"}, {"R0008": "0", "R0021": "0", "R0022": "1", "R0046": "17.8", "R0050": "DR", "R0028": "10"}),
        _profile("거절", ["K361"], {"A1004": "49", "A1008": "96010", "A2043": "30000000"}, {"R0008": "3000000", "R0021": "0", "R0022": "1", "R0046": "18.1", "R0050": "DR", "R0028": "8"}),
        _profile("거절", ["K364"], {"A1004": "41", "A1008": "55101", "A2043": "15000000"}, {"R0008": "0", "R0021": "0", "R0022": "1", "R0046": "18.4", "R0050": "DR", "R0028": "10"}),
        _profile("거절", ["K365"], {"A1004": "57", "A1008": "74210", "A2043": "22000000"}, {"R0008": "2000000", "R0021": "0", "R0022": "1", "R0046": "17.2", "R0050": "DR", "R0028": "9"}),
        _profile("거절", ["K373"], {"A1004": "36", "A1008": "56199", "A2043": "28000000"}, {"R0008": "0", "R0021": "0", "R0022": "1", "R0046": "18.7", "R0050": "DR", "R0028": "11"}),
        _profile("거절", ["K374"], {"A1004": "44", "A1008": "47611", "A2043": "26000000"}, {"R0008": "0", "R0021": "0", "R0022": "1", "R0046": "19.1", "R0050": "DR", "R0028": "11"}),
    ],
    "C12": [
        _profile("승인", [], {"A1004": "37", "A2056": "68000000", "A2046": "0"}, {"R0002": "30000000", "R0004": "11.2", "R0009": "0", "R0011": "1", "R0012": "0", "R0015": "AA", "R0016": "3", "R0072": "1"}),
        _profile("승인", [], {"A1004": "46", "A2056": "54000000", "A2046": "1"}, {"R0002": "18000000", "R0004": "12.9", "R0009": "0", "R0011": "1", "R0012": "0", "R0015": "AA", "R0016": "4", "R0072": "2"}),
        _profile("승인", [], {"A1004": "31", "A2056": "82000000", "A2046": "0"}, {"R0002": "42000000", "R0004": "10.8", "R0009": "0", "R0011": "1", "R0012": "0", "R0015": "AA", "R0016": "2", "R0072": "1"}),
        _profile("거절", ["K354"], {"A1004": "55", "A2056": "32000000", "A2046": "0"}, {"R0002": "0", "R0004": "18.9", "R0009": "1", "R0011": "0", "R0012": "1", "R0015": "DR", "R0016": "10", "R0072": "2"}),
        _profile("거절", ["K145"], {"A1004": "42", "A2056": "28000000", "A2046": "1"}, {"R0002": "0", "R0004": "17.6", "R0009": "1", "R0011": "0", "R0012": "1", "R0015": "DR", "R0016": "9", "R0072": "2"}),
        _profile("거절", ["K370"], {"A1004": "50", "A2056": "36000000", "A2046": "0"}, {"R0002": "3000000", "R0004": "18.2", "R0009": "1", "R0011": "0", "R0012": "1", "R0015": "DR", "R0016": "8", "R0072": "2"}),
        _profile("거절", ["K361"], {"A1004": "59", "A2056": "24000000", "A2046": "1"}, {"R0002": "0", "R0004": "19.1", "R0009": "1", "R0011": "0", "R0012": "1", "R0015": "DR", "R0016": "10", "R0072": "2"}),
        _profile("거절", ["K364"], {"A1004": "48", "A2056": "30000000", "A2046": "0"}, {"R0002": "2000000", "R0004": "17.9", "R0009": "1", "R0011": "0", "R0012": "1", "R0015": "DR", "R0016": "9", "R0072": "2"}),
        _profile("거절", ["K373"], {"A1004": "35", "A2056": "26000000", "A2046": "0"}, {"R0002": "0", "R0004": "18.5", "R0009": "1", "R0011": "0", "R0012": "1", "R0015": "DR", "R0016": "10", "R0072": "2"}),
        _profile("거절", ["K374"], {"A1004": "63", "A2056": "22000000", "A2046": "1"}, {"R0002": "0", "R0004": "19.4", "R0009": "1", "R0011": "0", "R0012": "1", "R0015": "DR", "R0016": "11", "R0072": "2"}),
    ],
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
                pairs.append({"product": product, "in_line": current_in_line, "out_line": line})
            current_in_line = None
    return pairs


_SEED_PAIRS: list[dict[str, str]] = []
_SEED_INDEX_BY_PRODUCT: dict[str, int] = {}


def _get_seed_pairs(force_reload: bool = False) -> list[dict[str, str]]:
    global _SEED_PAIRS
    if force_reload or not _SEED_PAIRS:
        _SEED_PAIRS = _load_seed_pairs()
    return _SEED_PAIRS


def _set_fixed_field_value(raw_payload: str, field_code: str, value: str, width: int = FIELD_WIDTH) -> str:
    normalized_value = str(value or "")[:width].ljust(width)
    pattern = re.compile(rf"({re.escape(field_code)})(.{{{width}}})")
    return pattern.sub(lambda match: f"{match.group(1)}{normalized_value}", raw_payload, count=1)


def _set_or_append_field_value(raw_payload: str, field_code: str, value: str, width: int = FIELD_WIDTH) -> str:
    if not field_code:
        return raw_payload
    updated_payload = _set_fixed_field_value(raw_payload, field_code, value, width=width)
    if updated_payload != raw_payload:
        return updated_payload
    return f"{raw_payload}{field_code}{str(value or '')[:width].ljust(width)}"


def _strip_reject_codes(raw_payload: str) -> str:
    cleaned = re.sub(r"KORLT", "     ", raw_payload, flags=re.I)
    cleaned = re.sub(r"K\d{3}", "    ", cleaned, flags=re.I)
    return cleaned


def _apply_reject_codes(raw_payload: str, codes: list[str]) -> str:
    updated_payload = _strip_reject_codes(raw_payload)
    for code in codes:
        normalized_code = str(code or "").strip().upper()
        if re.fullmatch(r"K\d{3}", normalized_code):
            updated_payload = f"{updated_payload}KORLT{normalized_code}{'':15}"
    return updated_payload


def _pick_profile(product: str, profile_index: int, force_decision: str | None = None) -> dict[str, Any]:
    profiles = PRODUCT_SYNTHETIC_PROFILES.get(product) or []
    if not profiles:
        return {}
    normalized_decision = str(force_decision or "").strip()
    if "승인" in normalized_decision or "½ÂÀÎ" in normalized_decision:
        return next((profile for profile in profiles if profile.get("decision") == "승인"), profiles[0])
    if any(token in normalized_decision for token in ("거절", "탈락", "부결", "°ÅÀý")):
        return next((profile for profile in profiles if profile.get("decision") == "거절"), profiles[0])
    return profiles[profile_index % len(profiles)]


def _apply_profile_fields(raw_payload: str, fields: dict[str, str]) -> str:
    updated_payload = raw_payload
    for field_code, value in fields.items():
        updated_payload = _set_or_append_field_value(updated_payload, str(field_code), str(value))
    return updated_payload


def _select_seed_pair(seed_pairs: list[dict[str, str]], product: str | None) -> tuple[dict[str, str], int]:
    if product:
        filtered_pairs = [pair for pair in seed_pairs if pair.get("product") == product]
        if not filtered_pairs:
            raise RuntimeError(f"상품 {product}에 해당하는 테스트 로그 시드가 없습니다.")
        pair_index = _SEED_INDEX_BY_PRODUCT.get(product, 0)
        _SEED_INDEX_BY_PRODUCT[product] = pair_index + 1
        return filtered_pairs[pair_index % len(filtered_pairs)], pair_index

    pair_index = _SEED_INDEX_BY_PRODUCT.get("__all__", 0)
    _SEED_INDEX_BY_PRODUCT["__all__"] = pair_index + 1
    return seed_pairs[pair_index % len(seed_pairs)], pair_index


def append_synthetic_log(product: str | None = None, force_decision: str | None = None) -> dict[str, Any]:
    seed_pairs = _get_seed_pairs(force_reload=True)
    normalized_product = str(product or "").strip().upper() or None
    if normalized_product and normalized_product not in SUPPORTED_PRODUCTS:
        raise ValueError(f"지원하지 않는 상품코드입니다: {product}")
    if not seed_pairs:
        raise RuntimeError("지원 가능한 테스트 로그 시드가 없습니다.")

    pair, profile_index = _select_seed_pair(seed_pairs, normalized_product)
    product_code = str(pair.get("product") or normalized_product or "").strip().upper()
    profile = _pick_profile(product_code, profile_index, force_decision)

    generated_at = datetime.datetime.now()
    prefix = generated_at.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    thread_no = ((sum(_SEED_INDEX_BY_PRODUCT.values()) or 1) % 8) + 1
    line_prefix = (
        f"{prefix} [[ACTIVE] ExecuteThread: '{thread_no}' for queue: 'weblogic.kernel.Default (self-tuning)'] "
        f"INFO  [com.nice.rclips.server.online.main.RclipsOnlineServlet] "
    )

    in_payload = pair["in_line"].split("in_data = [", 1)[-1]
    out_payload = pair["out_line"].split("out_data = [", 1)[-1]
    if profile:
        in_payload = _apply_profile_fields(in_payload, profile.get("in") or {})
        out_payload = _apply_profile_fields(out_payload, profile.get("out") or {})
        out_payload = _apply_reject_codes(out_payload, list(profile.get("k_codes") or []))

    TARGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_LOG.open("a", encoding="utf-8") as file:
        file.write(f"{line_prefix}in_data = [{in_payload}\n")
        file.write(f"{line_prefix}out_data = [{out_payload}\n")
        file.write(f"{line_prefix}process time[WAS]: 0.0{thread_no}\n")

    return {
        "product": product_code,
        "profile_decision": profile.get("decision") if profile else "",
        "profile_k_codes": profile.get("k_codes") if profile else [],
        "forced_decision": force_decision,
        "file_path": str(TARGET_LOG),
        "generated_at": generated_at.isoformat(),
    }
