import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer.log_analyzer import analyze_logs
from rag.faiss_logs_db import prepare_log_records
from rag.vector_db import (
    apply_mapping,
    clean_faiss_text,
    find_globally_ignorable_field_keys,
    ingest_logger,
    map_fields,
    sanitize_faiss_fields,
    sanitize_faiss_mapping,
    should_skip_faiss_log,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "logs"
FALLBACK_LOG_DIR = PROJECT_ROOT / "logs"
ANALYZER_RESULTS_PATH = PROJECT_ROOT / "data" / "log_analyzer_results.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "full_text_records.json"

INCOME_MEASURE_SPECS = [
    ("최종연소득금액", 100_000, 10_000),
    ("최종연소득금액(4.0)", 100_000, 10_000),
    ("KCB추정소득", 100_000, 1_000),
    ("NICE추정소득", 100_000, 1_000),
    ("소득", 100_000, 1_000),
]

AMOUNT_MEASURE_SPECS = [
    ("최종대출가능금액_실시간", None, 1),
    ("최종대출가능금액", None, 1),
    ("초기한도", None, 1),
    ("대출금액", 100_000, 1_000),
]


def _parse_named_lines(*chunks: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in chunks:
        text = str(chunk or "")
        if not text:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("["):
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            if ":" in line:
                key, value = line.split(":", 1)
                parsed[key.strip()] = value.strip()
                continue
            for segment in [part.strip() for part in line.split(",") if part.strip()]:
                if ":" not in segment:
                    continue
                key, value = segment.split(":", 1)
                parsed[key.strip()] = value.strip()
    return parsed


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "--", "nan", "None"}:
        return None
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _extract_scaled_measure(fields: dict[str, str], source_specs: list[tuple[str, int | None, int]]) -> tuple[float | None, str]:
    for field_name, small_value_threshold, multiplier in source_specs:
        raw_value = _to_number(fields.get(field_name))
        if raw_value is None:
            continue
        normalized_value = raw_value
        if small_value_threshold is not None and abs(normalized_value) < small_value_threshold:
            normalized_value *= multiplier
        return normalized_value, field_name
    return None, ""


def _build_normalized_features(record: dict[str, Any]) -> dict[str, Any]:
    features = dict(record.get("features") or {})
    fields = _parse_named_lines(
        record.get("in_text"),
        record.get("out_text"),
        record.get("in_text2"),
        record.get("out_text2"),
    )
    recognized_income, recognized_income_source = _extract_scaled_measure(fields, INCOME_MEASURE_SPECS)
    available_amount, available_amount_source = _extract_scaled_measure(fields, AMOUNT_MEASURE_SPECS)
    age = _to_number(fields.get("연령"))
    return {
        "case_id": features.get("case_id"),
        "age": int(age) if age is not None else features.get("age"),
        "annual_income": features.get("annual_income"),
        "recognized_income": recognized_income,
        "recognized_income_source": recognized_income_source,
        "available_amount": available_amount,
        "available_amount_source": available_amount_source,
    }


def load_analyzer_results() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if ANALYZER_RESULTS_PATH.exists():
        with ANALYZER_RESULTS_PATH.open(encoding="utf-8") as file:
            payload = json.load(file)
        return list(payload.get("results") or []), dict(payload.get("source") or {})

    chunks: list[str] = []
    loaded_files: list[str] = []
    seen_paths: set[str] = set()
    file_count = 0

    for candidate_dir in [LOG_DIR, FALLBACK_LOG_DIR]:
        if not candidate_dir.exists():
            continue

        for path in sorted(candidate_dir.iterdir()):
            if path.suffix.lower() not in {".txt", ".log"}:
                continue

            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue

            text = ""
            for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
                try:
                    text = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as error:
                    print(f"skip_file={path.name} error={error}")
                    text = ""
                    break

            if not text:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception as error:
                    print(f"skip_file={path.name} error={error}")
                    continue

            if not text.strip():
                continue

            chunks.append(text)
            loaded_files.append(str(path.relative_to(PROJECT_ROOT)))
            seen_paths.add(resolved)
            file_count += 1

    results = analyze_logs("".join(chunks))
    return results, {
        "log_dir": str(LOG_DIR.relative_to(PROJECT_ROOT)),
        "fallback_log_dir": str(FALLBACK_LOG_DIR.relative_to(PROJECT_ROOT)),
        "file_count": file_count,
        "loaded_files": loaded_files,
        "result_count": len(results),
        "source": "analyze_logs",
    }


def main() -> None:
    results, source_info = load_analyzer_results()
    prepared_records = prepare_log_records(
        results,
        ingest_logger,
        show_progress=False,
        should_skip_log=should_skip_faiss_log,
        sanitize_fields=sanitize_faiss_fields,
        sanitize_mapping=sanitize_faiss_mapping,
        find_ignorable_keys=find_globally_ignorable_field_keys,
        apply_mapping=apply_mapping,
        map_fields=map_fields,
        clean_text=clean_faiss_text,
    )

    export_records = [
        {
            "product": record.get("product"),
            "product_display": record.get("product_display"),
            "full_text": record.get("full_text"),
            "in_text": record.get("in_text"),
            "out_text": record.get("out_text"),
            "full_text2": record.get("full_text2"),
            "in_text2": record.get("in_text2"),
            "out_text2": record.get("out_text2"),
            "reject_reason_text": record.get("reject_reason_text"),
            "reject_reason_codes": record.get("reject_reason_codes") or [],
            "normalized_features": _build_normalized_features(record),
        }
        for record in prepared_records
    ]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            **source_info,
            "prepared_record_count": len(prepared_records),
            "export_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        },
        "records": export_records,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote={OUTPUT_PATH}")
    print(
        json.dumps(
            {
                "result_count": len(results),
                "prepared_record_count": len(prepared_records),
                "export_record_count": len(export_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()