from __future__ import annotations

import json
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer.log_analyzer import analyze_logs
from rag.product_pattern_summary import write_product_pattern_summary


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "logs"


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
    output_path = write_product_pattern_summary(analyzed)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    print(f"wrote={output_path}")
    print(
        json.dumps(
            {
                "file_count": file_count,
                "analyzed_log_count": len(analyzed),
                "product_count": len((payload.get("products") or {})),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()