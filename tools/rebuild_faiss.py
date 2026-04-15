import json
import traceback
import sys
import os

# Ensure project root is on sys.path so imports like `backend` and `rag` resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services import run_full_analysis
from rag.vector_db import get_vector_count

if __name__ == '__main__':
    try:
        print('Running full analysis (this may take a while)...')
        snapshot = run_full_analysis()
        print('\nFull analysis completed.')
        try:
            vc = get_vector_count()
            print(f'Vector count after rebuild: {vc}')
        except Exception as e:
            print('Could not get vector count:', e)
        # Print brief summary
        keys = list(snapshot.keys())
        print('Snapshot keys:', keys)
    except Exception:
        traceback.print_exc()
        raise
