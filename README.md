# AI Log Agent

금융 심사 로그, 온톨로지, 고객 군집, LLM 답변 생성을 하나의 흐름으로 묶어 보는 분석형 워크벤치입니다.  
이 문서는 개발환경 설치보다 **프로젝트가 질문을 어떻게 이해하고, 어떤 근거를 만들고, 최종 답변과 상품개발모드로 어떻게 이어지는지**를 설명하는 데 초점을 둡니다.

## 한 줄 요약

사용자가 자연어로 질문하면 시스템은 질문 안의 상품, 심사 결과, 금리, 한도, 거절 사유, 고객 속성 같은 표현을 온톨로지 feature로 번역합니다.  
그 feature를 기준으로 실제 심사 로그, 고객 군집, 거절 코드, 통계 큐브, 관련 feature를 찾아 최종 답변 컨텍스트를 만들고, 필요하면 Ollama가 그 근거 안에서만 자연어 답변을 다듬습니다.

예를 들어:

```text
이지신용대출 신청자의 승인 한도에 영향을 주는 feature는?
```

이 질문은 대략 이렇게 해석됩니다.

- `이지신용대출` -> 상품 코드 `C6`, 대표 feature `application.product_code`
- `승인`, `승인 한도` -> `decision.approved_amount`
- `한도`, `대출금액` -> `loan.requested_limit`
- `영향 주는 feature` -> 승인가능금액/대출금액에 영향을 줄 수 있는 related feature 탐색
- 최종 컨텍스트 -> 신용대출잔액, KCB 등급, NICE 등급, 연소득, 인정소득, 비대면연계대출등급 등 영향 feature 중심으로 구성

여기서 중요한 점은 군집 결과가 항상 답변의 주인공이 아니라는 것입니다.  
질문이 “평균 금리와 한도”라면 군집/통계가 중요하고, 질문이 “영향 feature”라면 related feature와 온톨로지 관계가 더 중요합니다.

## 주요 모듈

- `backend/app_main.py`  
  FastAPI 메인 서버입니다. 온톨로지 워크벤치, runtime job, 고객 군집, semantic refresh, 상품개발모드 API가 여기에서 연결됩니다.

- `backend/services.py`  
  뉴스/로그/FAISS/상태 스냅샷/활동 로그 같은 공통 서비스 레이어입니다.

- `backend/worker.py`  
  뉴스 수집, 로그 분석, 벡터 갱신 같은 백그라운드 worker입니다. 현재는 서버 startup 때 기본 자동 실행하지 않습니다.

- `frontend/src/components/OntologyWorkbench.jsx`  
  온톨로지 워크벤치 화면입니다. 질문 입력, 진행 애니메이션, VECTOR 관계형 화면, 상품개발모드, Step 2 회의실 UI가 이 컴포넌트에 있습니다.

- `frontend/src/components/ontologyRuntimeStore.js`  
  프론트에서 runtime stage, 진행상황, 로그, 결과 요약을 관리하는 Zustand store입니다.

- `data/ontology.json`  
  상품별 필드, 공통 feature, 거절 코드 taxonomy, feature 관계를 담는 온톨로지 원천입니다.

- `data/commonfeature.json`  
  화면과 검색에서 직접 쓰는 공통 feature 사전입니다. feature id, feature name, alias, 상품 범위, 샘플 값 등이 들어갑니다.

- `data/full_text_records.json`  
  실제 심사 로그를 검색과 통계에 쓰기 좋은 레코드 형태로 펼친 데이터입니다.

- `data/feature_customer_clusters.json`  
  고객 군집 캐시입니다. 상품, 승인/거절 decision, 연령대, 소득대, 금액대, 거절코드, 평균 금리/한도 같은 군집 요약이 들어갑니다.

- `data/segment_metric_cube.json`  
  평균 금리, 평균 한도, 승인/거절 건수 같은 세그먼트 통계 질의에 쓰는 큐브 데이터입니다.

## 전체 프로세스

### 1. 데이터 준비

프로젝트는 PostgreSQL 같은 외부 DB 대신 로컬 JSON과 FAISS 산출물을 중심으로 동작합니다.

주요 입력은 다음과 같습니다.

- 상품/심사 로그
- 상품별 필드 매핑
- 공통 feature 사전
- 거절 사유 코드 매핑
- 고객 군집 캐시
- 통계 큐브
- 선택적으로 규제 문서 벡터

온톨로지는 이 데이터들을 “같은 의미의 다른 필드명”으로 묶는 번역 레이어입니다.

예를 들어 상품마다 실제 로그 필드명이 다를 수 있습니다.

```text
C6 이지신용대출: 승인가능금액
C9 카드론: 대출가능금액
C12 이지론: 한도금액
```

하지만 화면과 질의에서는 이들을 공통 feature인 `decision.approved_amount` 또는 `loan.requested_limit`로 바라봅니다.

### 2. 질문 입력

사용자는 온톨로지 워크벤치에서 자연어 질문을 입력합니다.

예시:

```text
이지신용대출 평균 금리와 한도는?
40대 카드론 신청자들의 평균 탈락 사유는?
이지신용대출 신청자의 승인 한도에 영향을 주는 feature는?
40대 직장인의 카드론 승인에 중요한 요소는?
```

프론트는 질문을 백엔드의 runtime job으로 넘기고, 백엔드는 단계별 상태를 업데이트합니다.

주요 API:

- `POST /feature-ontology/runtime-jobs`
- `GET /feature-ontology/runtime-jobs/{job_id}`
- `GET /feature-ontology/workbench`

### 3. Runtime Stage

질문 하나가 처리될 때 백엔드는 다음 stage를 순서대로 통과합니다.

#### 3.1 Load Runtime Data

`commonfeature.json`, `ontology.json`, `full_text_records.json` 등을 읽습니다.

이 단계는 질문을 해석하기 위한 사전과 실제 로그 원본을 준비하는 단계입니다.

#### 3.2 Product Scope Filter

질문 또는 화면 선택값에서 상품 범위를 잡습니다.

예:

- `이지신용대출` -> `C6`
- `카드론` -> `C9`
- `개인사업자대출` -> `C11`
- `이지론` 또는 유사 표현 -> `C12`

현재 중요한 보정:

- `이지신용대출`은 `C6`로 처리합니다.
- 상품명이 질문에 있으면 `application.product_code`를 대표 feature 후보에 넣습니다.
- 특정 상품 질문에서는 해당 상품과 무관한 feature가 과하게 올라오지 않도록 상품 범위를 먼저 좁힙니다.

#### 3.3 Semantic Feature Rank

질문 토큰과 feature alias/name을 비교해 관련 feature 후보를 정렬합니다.

예:

```text
이지신용대출 신청자의 승인 한도에 영향을 주는 feature는?
```

후보 예시:

- `application.product_code` / 이지신용대출
- `decision.approved_amount` / 승인가능금액
- `loan.requested_limit` / 대출금액
- `credit.kcb_grade` / KCB 등급
- `credit.nice_grade` / NICE 등급
- `income.annual_income` / 연소득

직접 연결 feature가 없더라도 질문의 보조 토큰은 버리지 않습니다.  
예를 들어 “탈락 사유”처럼 feature에 바로 매핑되지 않는 표현은 거절 코드, decision, reject reason 쪽으로 의미를 보강하는 힌트로 사용합니다.

#### 3.4 Primary Axis Select

상위 후보를 하나만 무조건 고르는 방식이 아니라, 질문 의도에 따라 여러 대표 feature를 함께 잡습니다.

예:

```text
이지신용대출 평균 금리와 한도는?
```

대표 feature는 하나가 아니라 다음처럼 여러 개가 될 수 있습니다.

- `application.product_code` / 이지신용대출
- `pricing.approved_rate` 또는 금리 관련 feature
- `decision.approved_amount`
- `loan.requested_limit`

질문이 복합 질문이면 primary feature도 복수로 잡는 것이 자연스럽습니다.  
“금리와 한도”는 두 개의 지표를 묻는 질문이므로 하나의 feature로 압축하면 답변이 잘립니다.

#### 3.5 Cluster Cache Build

고객 군집 캐시를 읽거나 필요하면 재생성합니다.

군집은 승인 고객만 보지 않습니다.  
현재 구조에서는 승인/거절 decision을 함께 보고, 질문이 탈락/거절/부결/사유를 묻는 경우 거절군의 reject code와 사유 요약을 더 중요하게 봅니다.

군집 예시:

```text
C6 / approved / 30대 / 고소득 / 중액
C6 / rejected / 40대 / 중소득 / 고액 / K코드 반복
C9 / rejected / 40대 / 직장인 / DSR 부담
```

군집은 다음 질문에 특히 유용합니다.

- 평균 금리와 한도
- 승인/거절 고객군 비교
- 40대, 직장인, 고소득 같은 세그먼트 조건
- 탈락 사유와 reject code 반복 패턴
- 특정 상품의 대표 고객군

반대로 “승인 한도에 영향을 주는 feature는?”처럼 feature 영향도를 묻는 질문에서는 군집은 보조 근거일 뿐이고, 최종 답변의 주인공은 related feature입니다.

#### 3.6 Retrieval Result Build

대표 feature와 질문 의도에 맞는 실제 레코드, 군집, 거절 코드, 통계 큐브 결과를 모읍니다.

예:

```text
40대 카드론 신청자들의 평균 탈락 사유는?
```

이 경우 retrieval은 다음을 우선합니다.

- 상품: 카드론 `C9`
- 세그먼트: 40대
- decision: rejected
- reject code 분포
- reject reason label
- 유사 로그 레코드

“탈락”과 “사유”가 직접 feature로 hit되지 않아도, 이 질문은 거절 고객군과 reject taxonomy를 봐야 하는 질문입니다.

#### 3.7 Answer Summary Build

백엔드는 화면 상단에 보여줄 answer summary를 만듭니다.

이 summary는 Ollama가 없어도 생성됩니다.  
Ollama가 켜져 있으면 같은 근거를 바탕으로 문장을 더 자연스럽게 다듬습니다.

중요한 guardrail:

- 내부 처리 용어인 `대표 축`, `ontology expansion`, `retrieval evidence` 같은 말은 최종 답변에 직접 노출하지 않도록 제한합니다.
- 질문이 influence feature 질문이면 군집 결론으로 시작하지 않도록 제한합니다.
- 질문이 평균 금리/한도 질문이면 통계 큐브와 군집 평균을 우선합니다.
- 질문이 탈락 사유 질문이면 거절군과 reject code를 우선합니다.

## 온톨로지의 역할

온톨로지는 단순 키워드 검색 사전이 아닙니다.  
상품별 필드명, 업무 용어, 질문 표현, 거절 코드, related feature를 하나의 의미망으로 연결하는 레이어입니다.

### 온톨로지가 하는 일

1. 상품명/별칭을 상품 코드로 연결합니다.
2. 질문 표현을 feature로 연결합니다.
3. 상품별 다른 필드명을 공통 feature로 묶습니다.
4. 대표 feature 주변의 related feature를 확장합니다.
5. reject code taxonomy와 연결해 탈락 사유 질문을 해석합니다.
6. LLM에게 넘길 컨텍스트를 제한합니다.

### 예시: 이지신용대출 승인 한도 영향 feature

질문:

```text
이지신용대출 신청자의 승인 한도에 영향을 주는 feature는?
```

온톨로지 해석:

- `이지신용대출` -> `C6`
- `이지신용대출` -> 대표 feature `application.product_code`
- `승인 한도` -> `decision.approved_amount`
- `대출금액`, `한도` -> `loan.requested_limit`
- `영향 feature` -> 위 feature들에 연결된 related feature 탐색

답변에 들어갈 수 있는 영향 feature 예시:

- 신용대출잔액
- KCB 등급
- NICE 등급
- 연소득
- 인정소득
- 비대면연계대출등급
- DSR 또는 상환부담 관련 feature
- 기존 대출/연체/한도 사용 관련 feature

이 질문에서 “C6 기준 상위 군집은 고소득/30대/중액입니다” 같은 답변이 먼저 나오면 흐름이 어긋난 것입니다.  
군집은 보조 설명으로만 쓰고, Ollama에는 승인가능금액과 대출금액에 영향을 주는 feature 목록과 근거가 먼저 들어가야 합니다.

### 예시: 평균 금리와 한도

질문:

```text
이지신용대출 평균 금리와 한도는?
```

온톨로지 해석:

- 상품: `C6`
- 지표 1: 금리
- 지표 2: 승인 한도 또는 대출금액
- 군집/통계 큐브: C6 기준 평균 금리와 평균 한도 계산

이 질문은 feature 영향도보다 통계 질의에 가깝습니다.  
따라서 `segment_metric_cube.json`과 `feature_customer_clusters.json`의 평균값이 중요한 근거가 됩니다.

### 예시: 평균 탈락 사유

질문:

```text
40대 카드론 신청자들의 평균 탈락 사유는?
```

온톨로지 해석:

- 상품: 카드론 `C9`
- 세그먼트: 40대
- decision: rejected
- 질의 의도: reject reason summary
- 근거: reject code 빈도, reject taxonomy label, 거절군 고객 군집

여기서 “탈락”, “사유”가 직접 feature hit가 없더라도 괜찮습니다.  
직접 feature가 아니라 decision/reject taxonomy를 여는 의도 토큰으로 봐야 합니다.

## 군집검색

군집검색은 `data/feature_customer_clusters.json`을 기준으로 질문과 가까운 고객군을 찾는 과정입니다.

군집의 목적은 “비슷한 신청자 묶음에서 어떤 금리, 한도, 승인/거절, reject code 패턴이 반복되는가”를 보여주는 것입니다.

### 군집이 구성되는 축

대표적으로 다음 축이 쓰입니다.

- 상품 코드
- 승인/거절 decision
- 연령대
- 소득대
- 신청/승인 금액대
- 금리 구간
- 거절 코드
- 위험 proxy
- 주요 feature 값

### 승인군과 거절군

초기에는 승인된 고객 중심으로 군집이 보일 수 있었지만, 현재 방향은 승인군과 거절군을 모두 봅니다.

질문별 우선순위:

- 승인 요인 질문 -> 승인군 + 승인금액/금리/한도 feature
- 탈락 사유 질문 -> 거절군 + reject code
- 평균 금리/한도 질문 -> 해당 상품/세그먼트의 승인/전체 통계
- 리스크 질문 -> 거절군, 연체 proxy, 등급, DSR, 기존대출 feature

### VECTOR 버튼 화면

왼쪽 `VECTOR` 버튼은 `feature_customer_clusters.json`를 가장 잘 설명하는 관계형 화면으로 연결됩니다.

화면에서 기대하는 구조:

- 상품별 군집 수
- 승인/거절 decision 분포
- 대표 군집 카드
- 평균 금리/한도
- 주요 reject code
- 군집 간 관계
- feature와 cluster의 연결

이 화면은 단순 테이블이 아니라 “상품 -> 고객군 -> feature/reject code -> 평균 지표”의 관계를 읽는 도구입니다.

## 상품개발모드

상품개발모드는 온톨로지와 군집 결과를 바탕으로 신규 상품 아이디어를 만드는 모드입니다.  
단순히 LLM에게 “상품 만들어줘”라고 하는 것이 아니라, 현재 데이터에서 발견된 고객군/거절군/기회 영역을 바탕으로 부서별 관점의 토론을 구성합니다.

### Step 1. 상품개발 후보 만들기

시스템은 온톨로지, 군집, 통계 큐브에서 상품개발에 쓸 수 있는 힌트를 모읍니다.

예:

- C6 이지신용대출에서 승인 경계에 있는 고객군
- 카드론 거절군 중 특정 reject code가 반복되는 세그먼트
- 40대 직장인 중 소득은 충분하지만 DSR 또는 등급 때문에 막히는 그룹
- 소액 한도에서는 리스크가 낮아 보이는 후보군

이 결과를 바탕으로 agenda를 만듭니다.

주요 API:

- `POST /feature-ontology/product-development/agendas`

### Step 2. 4개 부서 토론

선택한 agenda에 대해 네 명의 역할이 회의실에서 토론합니다.

등장 인물:

- 금융솔루션부 금프로  
  상품 컨셉, 고객 가치, 실험 범위를 봅니다.

- 신용기획부 신프로  
  리스크, 심사 룰, 거절 코드, 한도/금리 제한을 봅니다.

- 금융영업부 영프로  
  영업 가능성, 고객 반응, 승인 전환 가능성을 봅니다.

- IT개발자 아프로  
  기존 시스템에 어떻게 얹을 수 있는지, 배포 난이도와 데이터 연동 가능성을 봅니다.

Ollama 응답이 오래 걸리는 동안 화면은 결과만 기다리지 않고 회의실에 사람들이 들어와 인사하고, 스몰토크하고, 실무자들이 논점을 주고받는 것처럼 보여줍니다.

주요 API:

- `POST /feature-ontology/product-development/debate`

Step 2의 목적은 “정답 생성”보다 “상품화 가능한 논점 정리”입니다.

예시 agenda:

```text
씬파일 승인 전환형 소액 신용 상품
```

토론에서 다루는 질문:

- 어떤 고객군을 대상으로 할 것인가?
- 기존 룰을 얼마나 흔들 수 있는가?
- 거절군 중 어떤 세그먼트를 재심사 후보로 볼 수 있는가?
- 한도는 작게 시작할 것인가?
- 금리는 어떤 위험 보상 구조로 둘 것인가?
- IT 구현은 기존 feature와 decision pipeline 안에서 가능한가?

### Step 3. 실행안 정리

토론 결과는 다음처럼 정리됩니다.

- 제안 상품명
- 대상 고객군
- 핵심 feature 조건
- 승인/거절 기준 후보
- 한도/금리 가이드
- 리스크 제한
- 실험 방식
- 필요한 데이터/개발 작업

상품개발모드는 분석 결과를 보고 끝내지 않고, “그래서 어떤 상품 실험을 할 수 있는가”까지 이어가는 모드입니다.

## 애매한 온톨로지 결과 처리

질문 해석이 애매하면 시스템은 무리하게 단정하지 않고 사용자에게 재질문하는 방향을 갖습니다.

예:

```text
한도에 영향 주는 요소 알려줘
```

애매한 점:

- 어떤 상품의 한도인가?
- 승인한도인가, 신청금액인가?
- 승인 고객 기준인가, 거절 고객 포함인가?

이 경우 이상적인 재질문:

```text
한도 기준을 승인가능금액으로 볼까요, 신청 대출금액으로 볼까요?
상품은 이지신용대출(C6) 기준이면 될까요?
```

재질문이 필요한 대표 상황:

- 상품명이 없고 상품별 feature 차이가 큰 경우
- 승인/거절 기준이 불명확한 경우
- 금리/한도/대출금액처럼 유사 지표가 섞인 경우
- “사유”, “원인”, “영향”처럼 해석 경로가 여러 개인 경우

## Ollama 사용 방식

Ollama는 전체 시스템의 필수 조건이 아니라 최종 답변을 자연어로 다듬는 선택 레이어입니다.

백엔드는 먼저 자체적으로 다음을 만듭니다.

- representative features
- related features
- customer clusters
- retrieval results
- reject code summary
- answer summary
- Ollama prompt pack

그 다음 Ollama가 가능하면 prompt pack을 받아 문장을 정리합니다.

중요한 원칙:

- Ollama에는 원천 데이터 전체를 던지지 않습니다.
- 질문 의도에 맞게 압축된 컨텍스트만 줍니다.
- influence feature 질문에는 영향 feature 중심 컨텍스트를 줍니다.
- 평균/통계 질문에는 통계 큐브와 군집 평균을 줍니다.
- 거절 사유 질문에는 reject code와 거절군 근거를 줍니다.

## 백그라운드 서버 동작

현재 서버는 가볍게 뜨도록 기본값을 조정했습니다.

기본 OFF:

- background worker 자동 시작
- semantic refresh 자동 스케줄러
- `/health`에서 Ollama live probe

필요할 때만 켜는 환경변수:

```powershell
$env:AUTO_START_WORKER='1'
$env:AUTO_START_SEMANTIC_REFRESH='1'
$env:HEALTH_CHECK_OLLAMA='1'
```

각 옵션의 의미:

- `AUTO_START_WORKER=1`  
  서버 시작과 동시에 뉴스/로그/벡터 갱신 worker를 돌립니다. 장시간 데모나 운영 모드에 가깝습니다.

- `AUTO_START_SEMANTIC_REFRESH=1`  
  일정 주기로 `full_text_records`, 통계 큐브, 군집 캐시를 자동 갱신합니다. 무거운 작업이므로 개발 중에는 보통 끕니다.

- `HEALTH_CHECK_OLLAMA=1`  
  `/health` 호출 때마다 Ollama 서버를 실제로 확인합니다. 프론트가 자주 health를 부르면 느려질 수 있으므로 기본 OFF입니다.

수동 실행 API:

- `POST /worker/start`
- `POST /worker/stop`
- `POST /feature-ontology/semantic-refresh`
- `GET /health/ollama`

## 주요 API

온톨로지/워크벤치:

- `GET /ontology/state`
- `POST /ontology/save`
- `GET /feature-ontology/workbench`
- `POST /feature-ontology/runtime-jobs`
- `GET /feature-ontology/runtime-jobs/{job_id}`
- `POST /feature-ontology/ollama`

군집/통계:

- `GET /feature-ontology/clusters`
- `POST /feature-ontology/clusters/rebuild`
- `GET /feature-ontology/segment-metric-cube`
- `GET /feature-ontology/semantic-refresh-status`
- `POST /feature-ontology/semantic-refresh`

상품개발모드:

- `POST /feature-ontology/product-development/agendas`
- `POST /feature-ontology/product-development/debate`

백그라운드/상태:

- `GET /health`
- `GET /health/ollama`
- `POST /worker/start`
- `POST /worker/stop`
- `WS /ws/faiss`

기존 분석 기능:

- `POST /news/collect`
- `POST /logs/analyze`
- `POST /faiss/build`
- `POST /faiss/search`
- `POST /analysis/run`
- `GET /analysis/status`
- `POST /regulation/upload`

## 화면 흐름

### 온톨로지 워크벤치

1. 사용자가 질문을 입력합니다.
2. 프론트가 runtime job을 생성합니다.
3. 진행 애니메이션이 stage별 상태를 보여줍니다.
4. 백엔드가 feature 후보, 대표 feature, 군집, retrieval 결과를 계산합니다.
5. answer summary가 상단에 표시됩니다.
6. VECTOR 화면에서 군집 관계를 볼 수 있습니다.
7. 필요하면 상품개발모드로 넘어갑니다.

### 옵션 기본값

현재 의도한 기본값:

- Ollama GPU 모드: 켜기
- 온톨로지 질의 최우선: 켜기
- 로그 에이전트 호출: 끄기
- 뉴스 에이전트 호출: 끄기

이 기본값은 온톨로지 질의 실험을 빠르게 하기 위한 설정입니다.  
로그/뉴스 에이전트는 유용하지만 무거우므로 사용자가 필요할 때만 켜는 것이 좋습니다.

## 자주 쓰는 질문 예시

평균/통계:

```text
이지신용대출 평균 금리와 한도는?
카드론 40대 신청자의 평균 승인 한도는?
C6 승인 고객의 평균 금리와 대출금액은?
```

영향 feature:

```text
이지신용대출 신청자의 승인 한도에 영향을 주는 feature는?
카드론 승인에 중요한 feature는?
거절 가능성을 높이는 feature는 뭐야?
```

거절/탈락 사유:

```text
40대 카드론 신청자들의 평균 탈락 사유는?
이지신용대출 거절 고객의 주요 reject code는?
고소득인데 거절된 고객군은 왜 떨어졌어?
```

군집:

```text
C6 기준 상위 고객군을 설명해줘
feature_customer_clusters.json 기준으로 가장 반복되는 관계를 설명해줘
카드론 거절군 중 위험도가 높은 군집은?
```

상품개발:

```text
거절 고객군에서 승인 전환 가능한 신규 상품 아이디어를 만들어줘
40대 직장인 카드론 거절군을 대상으로 소액 상품을 설계해줘
씬파일 고객을 위한 승인 전환형 상품을 논의해줘
```

## 답변 품질을 볼 때 확인할 것

좋은 답변의 조건:

- 질문에 나온 상품이 맞게 잡혔는가?
- `C6` 이지신용대출처럼 상품 코드가 정확한가?
- 질문이 복합 지표이면 representative feature가 여러 개 잡혔는가?
- 평균 질문이면 통계/군집 평균이 먼저 나오는가?
- 영향 feature 질문이면 related feature가 먼저 나오는가?
- 탈락 사유 질문이면 거절군과 reject code를 보는가?
- Ollama 답변이 내부 용어를 그대로 말하지 않는가?

나쁜 신호:

- 영향 feature 질문인데 군집 결과가 첫 문장으로 나옴
- “대표 축”, “ontology expansion” 같은 내부 용어가 답변에 노출됨
- 탈락 사유 질문인데 승인 고객군만 봄
- 상품명이 C6/C9 등으로 잘못 매핑됨
- “탈락”, “사유”를 feature hit 없음으로만 처리하고 reject code를 보지 않음

## 개발자가 기억할 핵심

이 프로젝트는 단순 검색 UI가 아닙니다.  
핵심은 다음 순서입니다.

```text
질문
-> 상품/의도/지표 해석
-> 온톨로지 feature 매핑
-> 대표 feature 복수 선택
-> related feature 확장
-> 고객 군집/거절 코드/통계 큐브 조회
-> 질문 의도별 컨텍스트 압축
-> answer summary
-> 선택적으로 Ollama 자연어 생성
```

온톨로지는 feature를 찾는 도구이고, 군집은 실제 고객 패턴을 보여주는 도구이며, 상품개발모드는 그 둘을 이용해 실험 가능한 금융상품 논의를 만드는 도구입니다.
