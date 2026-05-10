import json
import os
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