from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    py = sys.executable
    print("[rebuild] start")
    _run([py, "tools/export_full_text_records.py"])
    _run([py, "tools/build_common_feature_ontology.py"])
    _run([py, "tools/generate_segment_metric_cube.py"])
    _run([py, "tools/rebuild_faiss.py"])

    use_graph = str(os.getenv("USE_NEO4J_GRAPH", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if use_graph:
        print("[rebuild] USE_NEO4J_GRAPH=1 -> neo4j rebuild endpoint should be called from running backend")
        print("         POST /graph/neo4j/rebuild")
    else:
        print("[rebuild] neo4j disabled (set USE_NEO4J_GRAPH=1 to enable graph rebuild path)")
    print("[rebuild] done")


if __name__ == "__main__":
    main()
