import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.vector_db import get_embeddings
from langchain_community.vectorstores import FAISS

def main():
    db = FAISS.load_local('faiss_db', get_embeddings(), allow_dangerous_deserialization=True)
    idx = getattr(db, 'index_to_docstore_id', None)
    print('TYPE index_to_docstore_id:', type(idx))
    try:
        print('len index:', len(idx))
    except Exception as e:
        print('len error', e)
    try:
        print('sample index repr:', repr(idx[:10]))
    except Exception as e:
        print('sample repr error', e)

    docmap = getattr(getattr(db,'docstore',None), '_dict', None)
    print('TYPE docmap:', type(docmap))
    try:
        print('docmap len:', len(docmap))
    except Exception as e:
        print('docmap len error', e)
    try:
        keys = list(docmap.keys())[:10]
        print('docmap keys sample types:', [type(k) for k in keys])
        print('docmap keys sample repr:', [repr(k) for k in keys])
    except Exception as e:
        print('docmap keys error', e)

if __name__ == '__main__':
    main()
