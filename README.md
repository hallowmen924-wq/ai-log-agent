# AI Log Agent

로컬(집 데스크탑/노트북)에서 빠르게 개발하고 실행하기 위한 가이드입니다.

## 1. 권장 환경

- OS: Windows (PowerShell)
- Python: **3.13**
- Node.js: 20+
- Ollama: 로컬 실행 (`http://127.0.0.1:11434`)

## 2. 최초 1회 세팅

```powershell
cd C:\work\ai-log-agent
powershell -ExecutionPolicy Bypass -File .\scripts\dev_setup.ps1
```

이 스크립트가 다음을 자동 수행합니다.
- `.venv` 생성 (Python 3.13 기준)
- 백엔드 의존성 설치 (`requirements.txt`)
- 프론트 의존성 설치 (`frontend\package-lock.json` 기반)

## 3. 실행 방법

### 백엔드만 실행
```powershell
cd C:\work\ai-log-agent
powershell -ExecutionPolicy Bypass -File .\scripts\run_backend.ps1
```

### 프론트만 실행
```powershell
cd C:\work\ai-log-agent
powershell -ExecutionPolicy Bypass -File .\scripts\run_frontend.ps1
```

### 둘 다 실행(권장)
```powershell
cd C:\work\ai-log-agent
powershell -ExecutionPolicy Bypass -File .\scripts\run_dev.ps1
```

- Frontend: `http://127.0.0.1:3001`
- Backend: `http://127.0.0.1:18000`

## 4. 로그 추가 후 재구성

심사 로그를 `C:\work\ai-log-agent\data\logs`에 추가한 뒤 아래를 실행하세요.

```powershell
cd C:\work\ai-log-agent
.venv\Scripts\python.exe .\tools\rebuild_from_logs_with_graph.py
```

주요 산출물:
- `C:\work\ai-log-agent\data\full_text_records.json`
- `C:\work\ai-log-agent\data\commonfeature.json`
- `C:\work\ai-log-agent\data\segment_metric_cube.json`
- `C:\work\ai-log-agent\faiss_*` 인덱스

## 5. Neo4j(선택)

기본 동작은 기존 JSON/FAISS 기반이며, Neo4j는 선택적으로 켤 수 있습니다.

환경변수 예:
- `USE_NEO4J_GRAPH=1`
- `NEO4J_URI=bolt://127.0.0.1:7687`
- `NEO4J_USERNAME=neo4j`
- `NEO4J_PASSWORD=...`
- `NEO4J_DATABASE=neo4j`

확인 API:
- `GET /graph/neo4j/health`
- `POST /graph/neo4j/rebuild`

## 6. 자주 쓰는 문제 해결

- `ModuleNotFoundError` 발생 시: `.venv` 활성화 여부 확인
- Python 3.14 경고 시: Python 3.13 가상환경으로 재생성
- `Could not import rank_bm25` 시:
  ```powershell
  .venv\Scripts\pip.exe install rank_bm25
  ```

## 7. 현재 기본 운영 원칙

- Docker 없이 로컬 중심 개발
- 버전 고정: `requirements.txt`, `package-lock.json`
- 백엔드/프론트 실행 커맨드 스크립트화
- 필요할 때만 Neo4j 활성화

