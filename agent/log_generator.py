from __future__ import annotations

import datetime
import pathlib
from typing import Any

from analyzer.log_parser import parse_logs_fast

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_LOG = PROJECT_ROOT / "data" / "logs" / "stdout.log.20260407.txt"
TARGET_LOG = PROJECT_ROOT / "data" / "logs" / "generated_live.log"
SUPPORTED_PRODUCTS = {"C6", "C9", "C11", "C12"}


def _load_seed_pairs() -> list[dict[str, str]]:
    if not SOURCE_LOG.exists():
        return []

    raw_logs = SOURCE_LOG.read_text(encoding="utf-8", errors="ignore")
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


_SEED_PAIRS = _load_seed_pairs()
_SEED_INDEX = 0


def append_synthetic_log() -> dict[str, Any]:
    global _SEED_INDEX

    if not _SEED_PAIRS:
        raise RuntimeError("지원 가능한 테스트 로그 시드가 없습니다.")

    pair = _SEED_PAIRS[_SEED_INDEX % len(_SEED_PAIRS)]
    _SEED_INDEX += 1

    generated_at = datetime.datetime.now()
    prefix = generated_at.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    thread_no = (_SEED_INDEX % 8) + 1
    line_prefix = (
        f"{prefix} [[ACTIVE] ExecuteThread: '{thread_no}' for queue: 'weblogic.kernel.Default (self-tuning)'] "
        f"INFO  [com.nice.rclips.server.online.main.RclipsOnlineServlet] "
    )

    in_payload = pair["in_line"].split("in_data = [", 1)[-1]
    out_payload = pair["out_line"].split("out_data = [", 1)[-1]

    TARGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_LOG.open("a", encoding="utf-8") as file:
        file.write(f"{line_prefix}in_data = [{in_payload}\n")
        file.write(f"{line_prefix}out_data = [{out_payload}\n")
        file.write(f"{line_prefix}process time[WAS]: 0.0{thread_no}\n")

    return {
        "product": pair["product"],
        "file_path": str(TARGET_LOG),
        "generated_at": generated_at.isoformat(),
    }
