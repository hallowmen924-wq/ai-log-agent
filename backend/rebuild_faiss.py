import sys, time
sys.path.insert(0, r"c:\work")
from backend.services import get_vector_count, build_faiss_bundle

print('PY: start rebuild')
try:
    before = get_vector_count()
    print('before_count:', before)
    t0 = time.time()
    count = build_faiss_bundle()
    t1 = time.time()
    print('after_count:', count)
    print('elapsed_sec:', round(t1-t0,2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('ERROR:', e)
