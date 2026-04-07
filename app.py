import streamlit as st
from rag.vector_db import search_knowledge
from rag.vector_db import load_knowledge

load_knowledge()  # ⭐ 이거 꼭 있어야 함


st.title("📊 RAG 테스트")

query = st.text_input("코드 입력 (예: A6001)")


if st.button("검색"):
    result = search_knowledge(query)
    st.write(result)


