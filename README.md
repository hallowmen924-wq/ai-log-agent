# AI Log Agent — React + FastAPI Dashboard

이 리포지토리는 기존 Streamlit 앱을 React (create-react-app) + FastAPI로 분해한 예시입니다.

구성
- backend/: FastAPI 앱
- frontend/: React 앱 (create-react-app 스타일)

빠른 시작 (Windows)

1) 백엔드 설치 및 실행

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2) 프론트엔드 실행

```powershell
cd frontend
npm install
npm start
```

프론트엔드는 기본적으로 `http://localhost:3000`에서, 백엔드는 `http://localhost:8000`에서 실행됩니다.

노트
- 실제 배포 전에는 `agent`, `analyzer`, `rag` 모듈들의 의존성(오픈AI 등)과 환경변수를 점검하세요.
- 프론트엔드는 `REACT_APP_API` 환경변수로 백엔드 주소를 오버라이드할 수 있습니다.
