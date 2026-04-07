from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.excel_loader import load_excel_knowledge

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = None


def load_knowledge():
    global db

    with open("data/knowledge.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()

    excel_texts = load_excel_knowledge()

    # ⭐ 여기 추가
    print("엑셀 데이터 개수:", len(excel_texts))
    print("엑셀 샘플:", excel_texts[:3])

    all_texts = texts + excel_texts

    print("전체 데이터 개수:", len(all_texts))

    db = FAISS.from_texts(all_texts, embeddings)


def search_knowledge(query):
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])