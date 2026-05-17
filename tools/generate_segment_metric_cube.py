from __future__ import annotations

import json
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.segment_metric_cube import DEFAULT_RECORDS_PATH, DEFAULT_SEGMENT_CUBE_PATH, write_segment_metric_cube


def main() -> None:
    output_path = write_segment_metric_cube(DEFAULT_RECORDS_PATH, DEFAULT_SEGMENT_CUBE_PATH)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    print(f"wrote={output_path}")
    print(
        json.dumps(
            {
                "record_count": meta.get("record_count"),
                "segment_count": meta.get("segment_count"),
                "products": meta.get("products"),
                "output_path": str(Path(output_path).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
