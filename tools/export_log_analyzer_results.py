import json
import os
import sys
from datetime import datetime
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer.log_analyzer import analyze_logs


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "logs"
FALLBACK_LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "data" / "log_analyzer_results.json"


def load_raw_logs() -> tuple[str, int, list[str]]:
    chunks: list[str] = []
    file_count = 0
    loaded_files: list[str] = []
    seen_paths: set[str] = set()

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
            file_count += 1
            loaded_files.append(str(path.relative_to(PROJECT_ROOT)))
            seen_paths.add(resolved)

    return "".join(chunks), file_count, loaded_files


def main() -> None:
    raw_logs, file_count, loaded_files = load_raw_logs()
    results = analyze_logs(raw_logs)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "log_dir": str(LOG_DIR.relative_to(PROJECT_ROOT)),
            "fallback_log_dir": str(FALLBACK_LOG_DIR.relative_to(PROJECT_ROOT)),
            "file_count": file_count,
            "loaded_files": loaded_files,
            "result_count": len(results),
        },
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote={OUTPUT_PATH}")
    print(
        json.dumps(
            {
                "file_count": file_count,
                "result_count": len(results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()