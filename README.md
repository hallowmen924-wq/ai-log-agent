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

**초급 개발자용 빠른 안내**

- **환경 준비**: Python 3.10+ 권장. 가상환경을 만들고 의존성을 설치하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **환경 변수**: 민감한 키는 환경변수로 관리하세요. 예시:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
setx FAISS_INDEX_PATH "./faiss_db/index.faiss"
```

- **백엔드 실행**:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 18000
```

- **Streamlit 실행 (로컬 UI)**:

```powershell
cd ..
streamlit run app.py
```

**개발자가 알아둘 것들**

- **로그 파일 관리**: `backend/logs/rag_ingest.log` 같은 대형 로그는 Git에 커밋하면 안 됩니다. 이미 문제로 리포지토리 히스토리를 정리했습니다 — 모든 협업자는 아래 지침을 따라 로컬을 동기화해야 합니다.

- **히스토리 재작성 후 동기화 방법 (협업자용)**
	- 안전한 방법: 리포지토리를 재클론하세요.
		```powershell
		git clone https://github.com/hallowmen924-wq/ai-log-agent.git
		```
	- 간편 재설정(로컬 변경이 없는 경우):
		```powershell
		git fetch origin
		git reset --hard origin/main
		```

- **비밀(키) 노출 대응**: 저장소에서 노출된 키는 이미 제거했지만, 즉시 해당 키(예: OpenAI API 키)를 폐기하고 재발급하세요.

**코드 스타일 & 정적분석**

- 개발자가 로컬에서 실행할 수 있는 도구:
	- 포매터: `black`, `isort`
	- 린터/정적분석: `ruff`, `mypy`(선택)

설치 및 실행 예시:

```powershell
pip install --user black isort ruff mypy
python -m isort .
python -m black .
python -m ruff check . --fix
# 선택: 타입 검사
python -m mypy .
```

**FAISS 재구성(개발자용)**

- 대용량 인덱스가 손상되었거나 재생성해야 할 때는 `tools/rebuild_faiss.py`를 사용하세요. 실행 전에 `faiss_db/` 디렉터리와 관련 환경 변수를 확인하세요.

```powershell
python tools/rebuild_faiss.py --input data/knowledge --output faiss_db/index.faiss
```

**프로젝트 구조(중요 파일)**
- `backend/app_main.py`: FastAPI 엔트리포인트
- `backend/services.py`: 핵심 서비스(로그 분석, 뉴스 수집, FAISS 빌드 등)
- `backend/worker.py`: 주기작업(뉴스/벡터 빌드) 워커
- `backend/streamlit_client.py`: Streamlit에서 호출하는 클라이언트
- `app.py`: Streamlit UI 진입점 (최신)

---

필요하면 제가 다음 작업을 진행하겠습니다:

- 코드베이스 전체에서 사용되지 않는 import/변수 정리 및 자동 수정(이미 일부 적용됨)
- 성능 병목 프로파일링 및 소규모 최적화
- `requirements.txt` 정리 및 `pyproject.toml` 제안

어떤 작업을 먼저 진행할까요? (예: 린트 완전 정리, 프로파일링, LFS 설정 등)