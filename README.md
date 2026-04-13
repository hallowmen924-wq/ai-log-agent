# AI Log Agent

현재 구조는 기존 분석 코드 위에 FastAPI + Worker 레이어를 추가한 형태입니다. Streamlit은 직접 모듈 호출 대신 HTTP API로 바로 붙일 수 있습니다.

구성
- backend/app_main.py: FastAPI 엔트리포인트
- backend/services.py: 뉴스/로그/FAISS/전략챗 서비스 레이어
- backend/worker.py: 10초 주기 뉴스 수집 + 벡터 빌드 워커
- backend/streamlit_client.py: Streamlit 연동용 API 클라이언트
- app.py: 기존 Streamlit 대시보드

빠른 시작

1. 공통 의존성 설치

```powershell
pip install -r requirements.txt
```

2. 백엔드 실행

```powershell
cd backend
C:/Python314/python.exe -m uvicorn main:app --host 127.0.0.1 --port 18000
```

3. Streamlit 실행

```powershell
cd ..
streamlit run app.py
```

지원 API
- GET /health
- POST /news/collect
- POST /logs/analyze
- POST /faiss/build
- POST /faiss/search
- POST /chat/strategy
- POST /analysis/run
- GET /analysis/status
- POST /worker/start
- POST /worker/stop

Streamlit 연동 예시

```python
from backend.streamlit_client import BackendClient

client = BackendClient("http://127.0.0.1:18000")
status = client.get_status()
result = client.run_full_analysis()
chat = client.strategy_chat("승인율을 높이려면?")
```

현재 활성 파일
- app.py: Streamlit 메인 화면
- backend/main.py: FastAPI 엔트리포인트
- backend/worker.py: 백그라운드 뉴스/FAISS 워커

보관 파일
- 파일명 앞에 `~`가 붙은 파이썬 파일은 예제, 구버전, 실험용 코드입니다.