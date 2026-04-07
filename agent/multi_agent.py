from rag.vector_db import search_knowledge

def run_multi_agent(log):
    data = search_knowledge(log)
    return data