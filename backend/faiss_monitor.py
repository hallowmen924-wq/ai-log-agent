import os
import sys
import json

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def to_serializable(doc):
    # Handle common shapes (dict, LangChain Document-like)
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "metadata") or hasattr(doc, "page_content"):
        return {
            "id": getattr(doc, "id", None),
            "page_content": getattr(doc, "page_content", None),
            "metadata": getattr(doc, "metadata", None),
        }
    try:
        return dict(doc)
    except Exception:
        return str(doc)


def main():
    try:
        from rag import vector_db as vdb
    except Exception as e:
        print("ERROR: failed importing rag.vector_db:", e)
        sys.exit(2)

    get_count = getattr(vdb, "get_vector_count", None)
    list_vectors = getattr(vdb, "list_vectors", None)

    if not callable(get_count) or not callable(list_vectors):
        print("ERROR: required functions not found in rag.vector_db")
        sys.exit(3)

    count = get_count()
    # fetch a sane sample size
    sample = list_vectors(limit=200)
    sample_ser = [to_serializable(d) for d in sample]

    # compute simple product-level stats from sample
    products = {}
    for it in sample_ser:
        md = it.get("metadata") if isinstance(it, dict) else None
        prod = None
        if isinstance(md, dict):
            prod = md.get("product") or md.get("상품") or md.get("product_name")
        prod = prod or "unknown"
        stats = products.setdefault(prod, {"count": 0})
        stats["count"] += 1

    out_dir = os.path.dirname(__file__)
    out_mon = os.path.join(out_dir, "faiss_monitor.json")
    out_stats = os.path.join(out_dir, "faiss_monitor_stats.json")

    with open(out_mon, "w", encoding="utf-8") as f:
        json.dump({"count": count, "sample": sample_ser}, f, ensure_ascii=False, indent=2, default=str)

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump({"products": products}, f, ensure_ascii=False, indent=2)

    print("WROTE:", out_mon)
    print("WROTE:", out_stats)


if __name__ == "__main__":
    main()
