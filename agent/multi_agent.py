from langchain_community.llms import Ollama
from rag.vector_db import load_knowledge, search_knowledge
from utils.parser import parse_log

llm = Ollama(model="mistral")

load_knowledge()


def run_multi_agent(log: str) -> str:
    # 1. 로그 파싱
    parsed = parse_log(log)

    # 2. RAG 검색
    query = " ".join(parsed["codes"])
    knowledge = search_knowledge(query)

    # 3. 프롬프트
    prompt = f"""
    너는 카드론 심사 시스템 전문가다.

    [로그 요약]
    시간: {parsed['time']}
    API: {parsed['api']}
    처리시간: {parsed['process_time']}
    코드들: {parsed['codes']}

    [참고 지식]
    {knowledge}

    아래 형식으로 분석:
    - 이상 여부
    - 원인
    - 영향
    - 조치
    """

    result = llm.invoke(prompt)

    return result