import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer.log_analyzer import analyze_logs
from mapper.excel_mapper import get_excel_sheet, load_excel_mapping
from mapper.reject_code_mapper import load_reject_code_mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "logs"
FALLBACK_LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
EXCEL_PATH = str((DATA_DIR / "R-CLIPS code def.xlsx").resolve())
OUTPUT_PATH = DATA_DIR / "product_mapping_snapshot.json"
PRODUCTS = ["C9", "C6", "C11", "C12"]
MAX_SAMPLE_VALUES = 5


def load_raw_logs(log_dir: Path) -> tuple[str, int]:
    chunks: list[str] = []
    file_count = 0

    candidate_dirs = [log_dir, FALLBACK_LOG_DIR]
    seen_paths: set[str] = set()

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue

        for path in sorted(candidate_dir.iterdir()):
            if path.suffix.lower() not in {".txt", ".log"}:
                continue
            if str(path.resolve()) in seen_paths:
                continue

            loaded = False
            for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
                try:
                    chunks.append(path.read_text(encoding=encoding))
                    file_count += 1
                    seen_paths.add(str(path.resolve()))
                    loaded = True
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as error:
                    print(f"skip_file={path.name} error={error}")
                    loaded = True
                    break

            if not loaded:
                try:
                    chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
                    file_count += 1
                    seen_paths.add(str(path.resolve()))
                except Exception as error:
                    print(f"skip_file={path.name} error={error}")

    return "".join(chunks), file_count


def build_field_inventory(
    mapping: dict[str, str],
    observed_values: dict[str, Counter[str]],
) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}

    normalized_mapping: dict[str, str] = {}
    for raw_code, label in (mapping or {}).items():
        code = str(raw_code or "").strip().upper()
        if not code or code == "NAN":
            continue
        normalized_mapping[code] = str(label or "").strip()

    normalized_observed_values: dict[str, Counter[str]] = {}
    for raw_code, samples in (observed_values or {}).items():
        code = str(raw_code or "").strip().upper()
        if not code or code == "NAN":
            continue
        normalized_observed_values[code] = samples

    all_codes = sorted(set(normalized_mapping) | set(normalized_observed_values))
    for code in all_codes:
        samples = normalized_observed_values.get(code, Counter())
        inventory[code] = {
            "label": normalized_mapping.get(code, ""),
            "observed_count": int(sum(samples.values())),
            "sample_values": [
                {"value": value, "count": count}
                for value, count in samples.most_common(MAX_SAMPLE_VALUES)
            ],
        }

    return inventory


def build_reject_inventory(
    reject_mapping: dict[str, dict[str, str]],
    observed_details: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}

    for code in sorted(observed_details):
        observed = observed_details[code]
        mapped = reject_mapping.get(code, {})
        fallback_descriptions = Counter(observed.get("fallback_descriptions") or {})
        inventory[code] = {
            "description": str(mapped.get("description") or observed.get("description") or "").strip(),
            "risk_level": str(mapped.get("risk_level") or observed.get("risk_level") or "").strip(),
            "observed_count": int(observed.get("observed_count") or 0),
            "fallback_descriptions": [
                {"value": value, "count": count}
                for value, count in fallback_descriptions.most_common(MAX_SAMPLE_VALUES)
            ],
        }

    return inventory


def main() -> None:
    raw_logs, file_count = load_raw_logs(LOG_DIR)
    analyzed = analyze_logs(raw_logs)
    reject_mapping = load_reject_code_mapping(DATA_DIR)

    products_payload: dict[str, dict[str, object]] = {}

    for product in PRODUCTS:
        in_mapping = load_excel_mapping(EXCEL_PATH, get_excel_sheet(product, "in"))
        out_mapping = load_excel_mapping(EXCEL_PATH, get_excel_sheet(product, "out"))

        in_observed_values: dict[str, Counter[str]] = defaultdict(Counter)
        out_observed_values: dict[str, Counter[str]] = defaultdict(Counter)
        reject_observed_details: dict[str, dict[str, object]] = {}
        analyzed_count = 0

        for item in analyzed:
            if str(item.get("product") or "").strip() != product:
                continue

            analyzed_count += 1

            for code, value in (item.get("in_fields") or {}).items():
                normalized = str(value or "").strip()
                if normalized:
                    in_observed_values[str(code)].update([normalized])

            for code, value in (item.get("out_fields") or {}).items():
                normalized = str(value or "").strip()
                if normalized:
                    out_observed_values[str(code)].update([normalized])

            for detail in item.get("reject_reason_details") or []:
                code = str(detail.get("code") or "").strip().upper()
                if not code:
                    continue
                current = reject_observed_details.setdefault(
                    code,
                    {
                        "description": "",
                        "risk_level": "",
                        "observed_count": 0,
                        "fallback_descriptions": Counter(),
                    },
                )
                current["observed_count"] = int(current["observed_count"] or 0) + 1
                description = str(detail.get("description") or "").strip()
                risk_level = str(detail.get("risk_level") or "").strip()
                if description and not current["description"]:
                    current["description"] = description
                if risk_level and not current["risk_level"]:
                    current["risk_level"] = risk_level
                if description and description != reject_mapping.get(code, {}).get("description", ""):
                    current["fallback_descriptions"].update([description])

        products_payload[product] = {
            "in_sheet": get_excel_sheet(product, "in"),
            "out_sheet": get_excel_sheet(product, "out"),
            "analyzed_log_count": analyzed_count,
            "in_mapping": build_field_inventory(in_mapping, in_observed_values),
            "out_mapping": build_field_inventory(out_mapping, out_observed_values),
            "reject_reason_codes": build_reject_inventory(reject_mapping, reject_observed_details),
        }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "log_dir": str(LOG_DIR.relative_to(PROJECT_ROOT)),
            "fallback_log_dir": str(FALLBACK_LOG_DIR.relative_to(PROJECT_ROOT)),
            "file_count": file_count,
            "analyzed_log_count": len(analyzed),
            "excel_path": str(Path(EXCEL_PATH).resolve().relative_to(PROJECT_ROOT)),
            "reject_mapping_sources": sorted(
                path.name
                for path in DATA_DIR.iterdir()
                if path.suffix.lower() in {".xlsx", ".xls", ".csv"}
            ),
        },
        "products": products_payload,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote={OUTPUT_PATH}")
    print(
        json.dumps(
            {
                product: {
                    "in_mapping_codes": len(item["in_mapping"]),
                    "out_mapping_codes": len(item["out_mapping"]),
                    "reject_codes": len(item["reject_reason_codes"]),
                    "analyzed_log_count": item["analyzed_log_count"],
                }
                for product, item in products_payload.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()