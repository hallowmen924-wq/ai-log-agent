# FastAPI Backend + Worker

이 백엔드는 기존 분석 코드(`agent`, `analyzer`, `rag`)를 감싸서 Streamlit이나 다른 프론트엔드가 바로 호출할 수 있는 API를 제공합니다.

## 엔드포인트
- `GET /health` : 서버/워커 상태 확인
- `POST /news/collect` : 뉴스 수집 + 이슈 추출
- `POST /logs/analyze` : 로그 분석
- `POST /faiss/build` : 현재 결과/뉴스로 FAISS 재생성
- `POST /faiss/search` : 벡터 검색
- `POST /chat/strategy` : 전략 챗 응답
- `POST /analysis/run` : 전체 분석 동기 실행
- `GET /analysis/status` : 현재 상태/결과 조회
- `POST /worker/start` : 10초 주기 뉴스 수집 워커 시작
- `POST /worker/stop` : 워커 정지

## 실행
```powershell
cd backend
pip install -r requirements.txt
uvicorn app_main:app --reload --host 0.0.0.0 --port 8000
```

## Streamlit 연동
`backend/streamlit_client.py`의 `BackendClient`를 import해서 바로 호출하면 됩니다.

예시:
```python
from backend.streamlit_client import BackendClient

client = BackendClient("http://127.0.0.1:8000")
status = client.get_status()
```
