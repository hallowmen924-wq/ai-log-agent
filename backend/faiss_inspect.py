import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    try:
        from rag.vector_db import get_embeddings
        from langchain_community.vectorstores import FAISS
    except Exception as e:
        print(json.dumps({'error': f'import failed: {e}'}))
        return

    try:
        db = FAISS.load_local('faiss_db', get_embeddings(), allow_dangerous_deserialization=True)
    except Exception as e:
        print(json.dumps({'error': f'load_local failed: {e}'}))
        return

    idx_to_id = getattr(db, 'index_to_docstore_id', None)
    try:
        idx_list = list(idx_to_id) if idx_to_id is not None else []
    except Exception:
        idx_list = []

    docmap = {}
    try:
        docmap = getattr(getattr(db, 'docstore', None), '_dict', {}) or {}
    except Exception:
        docmap = {}

    out = {
        'index_len': len(idx_list),
        'index_sample': idx_list[:50],
        'docmap_len': len(docmap),
        'docmap_keys_sample': list(docmap.keys())[:50]
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == '__main__':
    main()
