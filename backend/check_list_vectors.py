import json
import sys
print('DEBUG: start script')
sys.path.insert(0, r"c:\work")
print('DEBUG: path adjusted')
from rag.vector_db import list_vectors, get_vector_count
print('DEBUG: imported rag.vector_db')

if __name__ == '__main__':
    try:
        vc = get_vector_count()
        print('vector_count:', vc)
        entries = list_vectors(5)
        print('entries_count:', len(entries))
        print(json.dumps(entries, ensure_ascii=False, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('ERROR:', str(e))
