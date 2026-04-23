from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

_reject_code_mapping_cache: dict[
    str, tuple[tuple[tuple[str, int, int], ...], dict[str, dict[str, str]]]
] = {}


def _normalize_column_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.replace(" ", "")


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        matched = normalized.get(_normalize_column_name(candidate))
        if matched:
            return matched
    for col in columns:
        norm = _normalize_column_name(col)
        if any(candidate in norm for candidate in candidates):
            return col
    return None


def _load_mapping_from_frame(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    if df.empty:
        return {}

    frame = df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    columns = list(frame.columns)

    code_col = _pick_column(columns, ["코드", "rclips코드", "심사코드"])
    desc_col = _pick_column(columns, ["설명", "항목명", "사유"])
    risk_col = _pick_column(columns, ["리스크레벨", "리스크", "등급"])

    if not code_col or not desc_col:
        return {}

    mapping: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        code = str(row.get(code_col, "") or "").strip().upper()
        if not re.fullmatch(r"K\d{3}", code):
            continue

        description = str(row.get(desc_col, "") or "").strip()
        risk_level = str(row.get(risk_col, "") or "").strip() if risk_col else ""
        if not description:
            continue

        mapping[code] = {
            "description": description,
            "risk_level": risk_level,
        }

    return mapping


def load_reject_code_mapping(data_dir: str | Path) -> dict[str, dict[str, str]]:
    base_dir = str(Path(data_dir).resolve())
    search_dir = Path(base_dir)
    if not search_dir.exists():
        _reject_code_mapping_cache[base_dir] = ((), {})
        return {}

    candidates = sorted(
        [path for path in search_dir.iterdir() if path.suffix.lower() in {".xlsx", ".xls", ".csv"}],
        key=lambda path: (
            0 if "ko_full" in path.name.lower() else 1,
            0 if path.suffix.lower() == ".csv" else 1,
            path.name.lower(),
        ),
    )
    signature = tuple(
        (path.name.lower(), int(path.stat().st_mtime_ns), int(path.stat().st_size))
        for path in candidates
    )

    cached = _reject_code_mapping_cache.get(base_dir)
    if cached is not None and cached[0] == signature:
        return cached[1]

    mapping: dict[str, dict[str, str]] = {}

    for path in candidates:
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                mapping.update(_load_mapping_from_frame(frame))
                continue

            excel_file = pd.ExcelFile(path)
            preferred_sheets = sorted(
                excel_file.sheet_names,
                key=lambda name: (
                    0 if "ko" in str(name).lower() and "full" in str(name).lower() else 1,
                    0 if "심사" in str(name) else 1,
                    str(name),
                ),
            )
            for sheet_name in preferred_sheets:
                frame = pd.read_excel(path, sheet_name=sheet_name)
                sheet_mapping = _load_mapping_from_frame(frame)
                if sheet_mapping:
                    mapping.update(sheet_mapping)
        except Exception:
            continue

    _reject_code_mapping_cache[base_dir] = (signature, mapping)
    return mapping


def extract_reject_reason_codes(out_data: str) -> list[str]:
    if not out_data:
        return []
    seen: set[str] = set()
    codes: list[str] = []
    for code in re.findall(r"KORLT(K\d{3})", out_data.upper()):
        if code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def map_reject_reason_codes(
    codes: list[str], mapping: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for code in codes:
        mapped = mapping.get(code, {})
        results.append(
            {
                "code": code,
                "description": str(mapped.get("description", "") or "").strip(),
                "risk_level": str(mapped.get("risk_level", "") or "").strip(),
            }
        )
    return results


def format_reject_reason_details(
    details: list[dict[str, str]], limit: int = 5
) -> list[str]:
    formatted: list[str] = []
    for item in details[:limit]:
        code = str(item.get("code", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        risk_level = str(item.get("risk_level", "") or "").strip()

        if description and risk_level:
            formatted.append(f"{code}={description} ({risk_level})")
        elif description:
            formatted.append(f"{code}={description}")
        elif code:
            formatted.append(code)
    return formatted