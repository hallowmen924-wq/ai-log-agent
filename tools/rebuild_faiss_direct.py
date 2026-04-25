import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer.log_analyzer import analyze_logs
from rag.vector_db import build_vector_db, get_vector_count


def main() -> None:
    log_dir = "data/logs"
    chunks = []
    file_count = 0

    for name in os.listdir(log_dir):
        if name.endswith(".txt") or name.endswith(".log"):
            path = os.path.join(log_dir, name)
            try:
                with open(path, encoding="utf-8") as file:
                    chunks.append(file.read())
                    file_count += 1
            except Exception as error:
                print(f"skip_file={name} error={error}")

    raw_text = "".join(chunks)
    print(f"loaded_files={file_count}")

    results = analyze_logs(raw_text or "")
    print(f"parsed_logs={len(results)}")

    count = build_vector_db(results, [])
    print(f"vector_count={count}")
    print(f"current_vector_count={get_vector_count()}")


if __name__ == "__main__":
    main()