import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer.log_analyzer import analyze_logs
from mapper.reject_code_mapper import load_reject_code_mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "logs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "ko_mapping_inventory.json"


def load_raw_logs(log_dir: Path) -> tuple[str, int]:
    chunks: list[str] = []
    file_count = 0
    for path in sorted(log_dir.iterdir()):
        if path.suffix.lower() not in {".txt", ".log"}:
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8"))
            file_count += 1
        except Exception as error:
            print(f"skip_file={path.name} error={error}")
    return "".join(chunks), file_count


def main() -> None:
    raw_logs, file_count = load_raw_logs(LOG_DIR)
    analyzed = analyze_logs(raw_logs)
    mapping = load_reject_code_mapping(DATA_DIR)

    code_counter: Counter[str] = Counter()
    fallback_examples: dict[str, str] = {}
    products_by_code: dict[str, set[str]] = {}

    for item in analyzed:
        product = str(item.get("product") or "")
        for detail in item.get("reject_reason_details", []) or []:
            code = str(detail.get("code") or "").strip().upper()
            if not code:
                continue
            code_counter[code] += 1
            if product:
                products_by_code.setdefault(code, set()).add(product)
            description = str(detail.get("description") or "").strip()
            if description and code not in mapping and code not in fallback_examples:
                fallback_examples[code] = description

    mapped_codes = []
    unmapped_codes = []
    for code, count in sorted(code_counter.items()):
        mapped = mapping.get(code)
        entry = {
            "code": code,
            "count": count,
            "products": sorted(products_by_code.get(code, set())),
        }
        if mapped:
            entry["description"] = str(mapped.get("description") or "")
            entry["risk_level"] = str(mapped.get("risk_level") or "")
            mapped_codes.append(entry)
        else:
            entry["fallback_description"] = fallback_examples.get(code, "")
            unmapped_codes.append(entry)

    inventory = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "log_dir": str(LOG_DIR.relative_to(PROJECT_ROOT)),
            "file_count": file_count,
            "analyzed_log_count": len(analyzed),
            "mapping_file": "data/KO_full.xlsx",
        },
        "summary": {
            "observed_unique_codes": len(code_counter),
            "mapped_unique_codes": len(mapped_codes),
            "unmapped_unique_codes": len(unmapped_codes),
            "mapping_catalog_size": len(mapping),
        },
        "mapped_codes": mapped_codes,
        "unmapped_codes": unmapped_codes,
    }

    OUTPUT_PATH.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote={OUTPUT_PATH}")
    print(json.dumps(inventory["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()