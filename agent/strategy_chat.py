import requests
from rag.vector_db import search_context

OLLAMA_URL = "http://localhost:11434/api/generate"


def mistral_generate(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


def strategy_chat(user_input):

    # 🔥 RAG 통합 검색
    logs, news, rules = search_context(user_input, k=6)

    # 🔥 너무 길면 잘라서 성능 최적화
    logs_text = "\n\n".join(logs[:3])
    news_text = "\n\n".join(news[:3])
    rules_text = "\n\n".join(rules[:3])

    # 🔥 핵심: 전략형 프롬프트
    prompt = f"""
당신은 금융 심사 리스크 및 규제 대응 전문가입니다.

아래 정보를 기반으로 분석하시오.

========================
[관련 심사 로그]
{logs_text}

========================
[관련 뉴스]
{news_text}

========================
[관련 규제]
{rules_text}

========================
[사용자 질문]
{user_input}

========================

다음 형식으로 반드시 구체적으로 답변하시오:

0. 사용자 질문에 대한 답변
- 질문에 대한 명확한 답변
   
1. 🔍 현재 리스크 상황  
- 해당 로그의 위험 수준
- 외부 환경(뉴스) 영향

2. ⚠️ 문제점 분석  
- 승인/거절에 영향을 준 핵심 요소
- 규제 위반 가능성

3. 🛠 대응 전략  
- 심사 기준 보완
- 리스크 감소 방안
- 승인율 개선 방법

4. 📜 규제 대응 전략  
- 관련 규제 요약
- 위반 가능 시 해결 방법

5. 🚀 정책 개선 방향  
- 향후 심사 정책 개선 제안
"""

    return mistral_generate(prompt)