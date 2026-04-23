import time
from pathlib import Path

from analyzer.log_field_parser import parse_fields
from analyzer.log_parser import parse_logs_fast
from mapper.excel_mapper import get_excel_sheet, load_excel_mapping
from mapper.reject_code_mapper import (
    extract_reject_reason_codes,
    load_reject_code_mapping,
    map_reject_reason_codes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = str((PROJECT_ROOT / "data" / "R-CLIPS code def.xlsx").resolve())


def analyze_logs(raw_logs):

    t0 = time.time()
    parsed_logs = parse_logs_fast(raw_logs)
    print(f"parse_logs_fast 실행 시간: {time.time() - t0:.2f}s")

    results = []

    # 🔥 1번만 로드
    sheet_cache = {}
    reject_code_mapping = load_reject_code_mapping(PROJECT_ROOT / "data")

    for product in ["C9", "C6", "C11", "C12"]:

        in_sheet = get_excel_sheet(product, "in")
        out_sheet = get_excel_sheet(product, "out")

        sheet_cache[(product, "in")] = load_excel_mapping(EXCEL_PATH, in_sheet)
        sheet_cache[(product, "out")] = load_excel_mapping(EXCEL_PATH, out_sheet)

    print("[log_analyzer] excel mapping preload completed")

    # 🔥 로그 처리
    for log in parsed_logs:

        if log["product"] not in ["C9", "C6", "C11", "C12"]:
            continue

        t1 = time.time()

        in_fields = parse_fields(log["in_data"])
        out_fields = parse_fields(log["out_data"])

        print(
            f"처리 중인 로그: {log['product']} - parse_fields 실행 시간: {time.time() - t1:.2f}s"
        )

        # 🔥 캐시에서 꺼냄
        in_mapping = sheet_cache[(log["product"], "in")]
        out_mapping = sheet_cache[(log["product"], "out")]
        reject_reason_codes = extract_reject_reason_codes(log["out_data"])

        results.append(
            {
                "product": log["product"],
                "in_fields": in_fields,
                "out_fields": out_fields,
                "in_mapping": in_mapping,
                "out_mapping": out_mapping,
                "reject_reason_codes": reject_reason_codes,
                "reject_reason_details": map_reject_reason_codes(
                    reject_reason_codes,
                    reject_code_mapping,
                ),
            }
        )

        print(
            f"로그 {log['product']} 처리 완료 - 총 필드: {len(in_fields) + len(out_fields)}, "
            f"인풋 매핑: {len(in_mapping)}, 아웃풋 매핑: {len(out_mapping)}"
        )
    return results
