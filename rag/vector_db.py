from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.excel_loader import load_excel_knowledge

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = None


def load_knowledge():
    global db

    # 1. 기존 txt
    with open("data/knowledge.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()

    # 2. 엑셀 추가 ⭐
    excel_texts = load_excel_knowledge()

    # 3. 합치기
    all_texts = texts + excel_texts

    db = FAISS.from_texts(all_texts, embeddings)


def search_knowledge(query):
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])