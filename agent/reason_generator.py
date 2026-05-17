import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_reason(log, risk, news_score, decision):

    prompt = f"""
    당신은 금융 심사 전문가입니다.

    아래 정보를 기반으로 대출 심사 결과의 "이유"를 설명하세요.

    [심사 결과]
    {decision}

    [리스크 점수]
    {risk}

    [뉴스 영향]
    {news_score}

    [로그 정보]
    {log}

    요구사항:
    1. 핵심 이유 3가지
    2. 위험 요인 설명
    3. 간결하게 (5줄 이내)
    """

    response = requests.post(
        OLLAMA_URL, json={"model": "mistral", "prompt": prompt, "stream": False}
    )

    return response.json()["response"]
