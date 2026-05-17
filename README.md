# AI Log Agent

금융 심사 로그와 뉴스, 규제 문서를 JSON + FAISS 기반으로 분석하는 프로젝트입니다. 현재 대표 UI는 React 기반 Ontology 콘솔이며, FastAPI 백엔드와 연결되어 질문 해석, feature 선택, 군집 검색, retrieval trace, answer summary 를 한 화면에서 확인할 수 있습니다.

PostgreSQL 없이 로컬 JSON 파일과 FAISS 인덱스를 사용합니다.

## 현재 구조

- backend/app_main.py: FastAPI 메인 엔트리포인트
- backend/services.py: 뉴스 수집, 상태 집계, runtime 서비스
- backend/worker.py: 주기 뉴스 수집 및 백그라운드 작업
- frontend/: React + Vite Ontology 콘솔
- app.py: Streamlit 기반 레거시 대시보드
- data/: 공통 feature, ontology, 제품 매핑, 로그 파생 JSON
- rag/: FAISS 및 retrieval 유틸리티
- agent/, analyzer/, mapper/, utils/: 분석 및 에이전트 로직

## 주요 기능

- Ontology 탭 중심 semantic runtime console
- 질문별 runtime stage 추적
- JSON + FAISS 기반 feature / cluster / retrieval 검색
- 카드론 심사 연관 뉴스 수집 및 중복 제거
- 규제 문서 업로드 후 Ontology 화면에서 즉시 학습 반영
- Streamlit 레거시 화면과 FastAPI API 동시 사용 가능

## 온톨로지 기준으로 현재 작업이 흐르는 방식

이 프로젝트에서 온톨로지는 단순한 용어 사전이 아니라, 서로 다른 상품과 로그 필드 이름을 하나의 공통 의미로 묶는 기준표 역할을 합니다.

초보자 관점에서는 아래처럼 이해하면 됩니다.

- 실제 심사 로그에는 상품마다 필드 이름이 다를 수 있습니다.
- 하지만 분석 화면에서는 이런 필드들을 공통 feature 로 묶어서 봅니다.
- 사용자가 질문을 입력하면, 시스템은 먼저 질문이 어떤 공통 feature 를 뜻하는지 찾습니다.
- 그 다음 그 feature 와 연결된 고객군집, 유사 로그, reject code, 요약 답변을 이어서 만듭니다.

예를 들어 상품마다 아래처럼 같은 뜻을 다른 이름으로 저장할 수 있습니다.

- C9 카드론: IN_연령
- C12 이지대환대출: IN_연령
- 다른 화면 질문: 나이, 연령, 고객 나이

온톨로지에서는 이것들을 applicant.age 라는 하나의 공통 feature 로 묶습니다. 그래서 사용자가 나이 관련 질문을 해도, 시스템은 내부적으로 applicant.age 를 중심으로 데이터를 모읍니다.

### 핵심 파일이 맡는 역할

- data/ontology.json: 상품별 입력 필드와 공통 feature 가 어떻게 연결되는지 보여주는 온톨로지 결과물입니다.
- data/commonfeature.json: 화면에서 직접 검색하는 공통 feature 사전입니다. feature_id, feature_name, alias, 상품 범위, 샘플값이 들어 있습니다.
- data/full_text_records.json: 실제 로그를 사람이 읽기 쉬운 형태로 펼쳐 놓은 레코드 집합입니다.
- backend/app_main.py: 온톨로지 워크벤치 API 와 런타임 단계를 조립하는 핵심 엔트리입니다.
- frontend/src/api.js: 프런트엔드가 온톨로지 워크벤치 API 를 호출하는 경로입니다.

### 큰 흐름 한 번에 보기

1. 로그 원본과 매핑 정보에서 공통 feature 사전을 만듭니다.
2. 상품별로 제각각인 필드 이름을 공통 feature 기준으로 묶습니다.
3. 사용자가 온톨로지 화면에서 질문을 입력합니다.
4. 백엔드는 Query Intent Parsing 으로 질문 토큰을 연령, 직업, 상품, 심사 맥락으로 분류합니다.
5. 선택 상품 범위에서 Domain-scoped FAISS Retrieval 을 수행하고 Top-K Feature Selection 으로 후보를 압축합니다.
6. 대표 feature 를 중심으로 Ontology Graph Expansion (1-hop) 을 수행해 관련 feature 와 relation 을 확장합니다.
7. Semantic Context Compression 으로 hit token, top-k, relation, retrieval evidence 만 남긴 뒤 Prompt Context Generation 으로 LLM 입력을 만듭니다.
8. 마지막으로 answer summary 를 만들어 화면 상단에 보여줍니다.

### 현재 LLM grounding 구조

워크벤치의 LLM 입력은 자유 문장형 컨텍스트가 아니라 아래 블록을 갖는 구조화된 semantic prompt 로 만들어집니다.

1. [SYSTEM ROLE]
2. [USER QUERY]
3. [SEMANTIC PIPELINE]
4. [SEMANTIC RETRIEVAL RESULT]
5. [ONTOLOGY EXPANSION]
6. [BUSINESS CONSTRAINTS]
7. [ANSWER INSTRUCTION]

이 구조의 목적은 hallucination 을 줄이고, answer generation 이 실제 retrieval / ontology expansion 결과만 참조하도록 강제하는 것입니다.

### 초보자용 비유

엑셀 파일마다 열 이름이 제각각인데, 우리가 분석할 때는 모두 같은 표준 컬럼명으로 보고 싶다고 생각하면 됩니다.

- 실제 로그 필드명: IN_접수번호, 신청서접수번호, 요청번호
- 공통 의미: application.case_id
- 화면에서 보는 기준: 접수번호

즉, 온톨로지는 여러 현업 용어를 하나의 표준 용어로 번역해 주는 통역 레이어입니다.

## 온톨로지 런타임 상세 흐름

현재 Ontology 콘솔에서 질문을 실행하면, 백엔드는 대략 아래 순서로 움직입니다.

### 1. Runtime Data Load

먼저 data/commonfeature.json 과 data/full_text_records.json 을 읽습니다.

- commonfeature.json: 어떤 공통 feature 가 있는지 확인
- full_text_records.json: 실제 로그 레코드가 어떤 값과 reject code 를 가졌는지 확인

쉽게 말하면, 사전과 원본 예문을 동시에 펼쳐 놓는 단계입니다.

### 2. Product Scope Filter

사용자가 특정 상품을 선택했다면, 그 상품에 실제로 등장하는 feature 만 남깁니다.

예시:

- 사용자가 C9 카드론을 선택
- applicant.age 는 유지
- C12 에만 있는 전용 필드는 제외 가능

이 단계가 필요한 이유는 상품마다 쓰는 필드와 심사 규칙이 조금씩 다르기 때문입니다.

### 3. Semantic Feature Rank

질문 문장과 가장 가까운 feature 후보를 점수화합니다.

예시 질문:

```text
카드론에서 나이가 심사에 얼마나 영향 있어?
```

이 질문이 들어오면 아래 같은 후보를 비교합니다.

- applicant.age
- customer.valid_customer_flag
- loan.requested_limit

여기서 나이, 연령, age 같은 alias 와 feature 이름이 질문과 얼마나 가까운지 계산해서 상위 후보를 정렬합니다.

### 4. Primary Feature Select

점수가 가장 높은 feature 하나를 대표 feature 로 정하고, 그 주변 feature 를 related feature 로 모읍니다.

예시:

- 대표 feature: applicant.age
- related feature: customer.valid_member_flag, loan.requested_limit, application.product_code

초보자 입장에서는 이 단계를 질문의 중심 주제 하나를 뽑는 과정으로 보면 됩니다.

### 5. Cluster Cache Build

대표 feature 와 질문을 기준으로 비슷한 고객 묶음, 즉 고객군집 후보를 계산합니다.

예를 들어 applicant.age 를 중심으로 보면 아래처럼 묶일 수 있습니다.

- 20대 후반 신청자 그룹
- 30대 중반 신청자 그룹
- 고연령 신청자 그룹

여기서 FAISS 와 캐시 데이터는 비슷한 패턴을 빠르게 찾기 위한 보조 장치입니다. 초보자는 비슷한 사례를 빨리 모아 주는 검색 가속기라고 이해하면 충분합니다.

### 6. Retrieval Result Build

이제 실제 로그 레코드 중에서 질문과 관련 있는 사례를 뽑고, 어떤 reject code 가 자주 같이 나오는지도 정리합니다.

예시:

- 질문: 카드론에서 나이가 영향 있어?
- 대표 feature: applicant.age
- 검색 결과: 연령이 높은 구간에서 특정 거절 사유가 자주 나온 사례 몇 건 추출
- 함께 표시: 관련 reject_reason_codes, 유사 레코드, 검색 trace

즉, 추상적인 feature 설명에서 끝나지 않고, 실제 로그 증거까지 연결해 주는 단계입니다.

### 7. Answer Summary Build

마지막으로 화면 상단에 보여 줄 요약 답변을 만듭니다.

이 답변은 아래 재료를 조합해서 만들어집니다.

- 사용자가 입력한 질문
- 선택 상품
- 대표 feature
- related feature
- 고객군집 후보
- retrieval 결과

Ollama 가 켜져 있으면 더 자연스러운 문장으로 요약하고, 꺼져 있어도 기본 요약은 생성됩니다.

## 질문 하나가 실제로 처리되는 예시

예를 들어 온톨로지 화면에서 아래처럼 질문한다고 가정하겠습니다.

```text
카드론에서 연령이 높으면 어떤 거절 사유가 자주 보여?
```

그러면 내부 흐름은 아래처럼 이어집니다.

1. 상품 C9 카드론 범위의 feature 만 우선 남깁니다.
2. 연령, 나이, age 와 연결된 applicant.age 가 대표 feature 후보로 올라옵니다.
3. 관련 feature 로 requested_limit, valid_customer_flag 같은 주변 feature 를 같이 묶습니다.
4. applicant.age 와 가까운 사례를 고객군집 단위로 모읍니다.
5. 실제 레코드에서 자주 등장한 reject_reason_codes 를 확인합니다.
6. 최종적으로 화면에는 이런 형태의 설명이 나옵니다.

예시 요약:

```text
카드론 상품에서 연령 관련 질문은 applicant.age feature 로 해석되었습니다.
관련 사례를 보면 특정 연령대 구간에서 일부 거절 코드가 반복적으로 관찰되며,
함께 조회된 feature 는 대출금액과 고객 유효성 여부입니다.
```

중요한 점은, 시스템이 처음부터 연령 컬럼만 보는 것이 아니라 온톨로지로 연령의 공통 의미를 찾고, 그 의미를 기준으로 유사 사례와 거절 사유를 연결한다는 것입니다.

## 왜 이 구조가 중요한가

- 상품마다 필드명이 달라도 하나의 공통 의미로 검색할 수 있습니다.
- 사람이 질문한 자연어를 바로 로그 필드와 연결할 수 있습니다.
- 단순 키워드 검색이 아니라 feature 중심으로 군집과 사례를 이어 볼 수 있습니다.
- 나중에 규제 문서나 신규 상품이 추가되어도 같은 공통 feature 체계로 확장하기 쉽습니다.

## 처음 보는 사람이 기억하면 좋은 한 줄 정리

이 프로젝트의 온톨로지는 상품별 로그 필드를 공통 feature 로 번역하고, 사용자의 질문을 그 feature 에 연결한 뒤, 관련 군집과 실제 사례를 찾아 최종 답변으로 보여주는 중심 축입니다.

## 권장 개발 환경

- Windows 10/11
- Python 3.11 또는 3.12 권장
- Node.js 20 LTS 권장
- Git

참고:

- 코드 안에 Python 3.14 와 호환되지 않는 경고 우회 로직이 있어, 노트북에서도 Python 3.11 또는 3.12 로 맞추는 편이 안전합니다.

## 노트북에서 처음 실행하기

### 1. 저장소 클론

```powershell
git clone https://github.com/hallowmen924-wq/ai-log-agent.git
cd ai-log-agent
```

### 2. Python 가상환경 생성 및 활성화

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 프런트엔드 의존성 설치

```powershell
cd frontend
npm install
cd ..
```

### 4. 백엔드 실행

```powershell
cd backend
python -m uvicorn app_main:app --host 127.0.0.1 --port 18000
```

### 5. 프런트엔드 실행

새 터미널에서:

```powershell
cd frontend
npm run dev
```

기본 접속 주소:

- 프런트엔드: http://127.0.0.1:3000
- 백엔드: http://127.0.0.1:18000

Vite 개발 서버는 /api 와 /ws 요청을 백엔드로 프록시합니다.

### 6. Streamlit 레거시 화면이 필요할 때만 실행

```powershell
streamlit run app.py
```

## 프로덕션 빌드

```powershell
cd frontend
npm run build
```

빌드 결과물은 frontend/dist 에 생성됩니다.

## 주요 API

- GET /health
- POST /news/collect
- POST /logs/analyze
- POST /faiss/build
- POST /faiss/search
- POST /chat/cardloan-debate
- POST /analysis/run
- GET /analysis/status
- POST /regulation/upload
- POST /worker/start
- POST /worker/stop
- GET /feature-ontology/workbench
- POST /feature-ontology/runtime-jobs
- GET /feature-ontology/runtime-jobs/{job_id}
- GET /feature-ontology/clusters
- POST /feature-ontology/clusters/rebuild
- POST /feature-ontology/ollama
- WS /ws/faiss

## 노트북으로 옮길 때 같이 알아둘 점

- .venv, node_modules, frontend/dist, logs, FAISS 생성물은 커밋 대상이 아닙니다.
- 루트의 faiss_customer, faiss_document, faiss_logs, faiss_news 같은 폴더는 런타임 생성물이므로 노트북에서 재생성될 수 있습니다.
- data 폴더의 JSON 원본은 프로젝트 실행에 필요하므로 유지합니다.
- 환경변수나 API 키가 있다면 노트북에 별도로 다시 설정해야 합니다.

예시:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

## Git에 올리기 전 확인

아래 항목은 커밋하지 않는 것이 안전합니다.

- .venv/
- frontend/node_modules/
- frontend/dist/
- logs/
- faiss_customer/
- faiss_document/
- faiss_logs/
- faiss_news/

이미 Git이 추적 중인 생성물이 있다면 한 번만 캐시에서 제거하세요.

```powershell
git rm -r --cached .venv frontend/node_modules frontend/dist logs faiss_customer faiss_document faiss_logs faiss_news
```

존재하지 않는 경로가 있으면 해당 이름만 빼고 실행하면 됩니다.

## Git push 절차

### 변경 확인

```powershell
git status
```

### 필요한 파일만 스테이징

소스와 문서만 올리려면:

```powershell
git add README.md requirements.txt .gitignore app.py backend frontend agent analyzer mapper rag utils data
```

만약 data 아래 생성 JSON 중 일부를 제외하고 싶으면 git status 를 보고 개별 파일만 추가하세요.

### 커밋

```powershell
git commit -m "Update ontology runtime UI and repo setup docs"
```

### 원격으로 푸시

브랜치명을 확인한 뒤:

```powershell
git branch --show-current
git push origin 브랜치명
```

예를 들어 현재 브랜치가 main 이면:

```powershell
git push origin main
```

## 추천 점검 순서

노트북에서 바로 이어서 작업하려면, 푸시 전에 아래 두 가지만 확인하는 편이 좋습니다.

1. backend 가 18000 포트에서 뜨는지 확인
2. frontend 에서 npm run build 가 성공하는지 확인

## 트러블슈팅

### npm build 가 루트에서 실패할 때

frontend 폴더에서 실행해야 합니다.

```powershell
cd frontend
npm run build
```

### uvicorn 실행이 안 될 때

가상환경이 활성화되어 있는지 확인하고, backend 폴더에서 app_main:app 으로 실행하세요.

```powershell
cd backend
python -m uvicorn app_main:app --host 127.0.0.1 --port 18000
```

### 규제 업로드가 실패할 때

루트 requirements.txt 에 python-multipart 가 포함되어 있어야 하며, 백엔드를 재시작해야 합니다.

## 참고 파일

- backend/app_main.py
- backend/services.py
- backend/worker.py
- frontend/src/App.jsx
- frontend/src/components/OntologyWorkbench.jsx
- frontend/src/components/ontologyRuntimeStore.js