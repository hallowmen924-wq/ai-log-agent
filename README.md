# AI Log Agent

대출 심사 질의를 대상으로,  
`의도 분류(Intent)` → `feature/cluster/retrieval` → `요약 카드`까지 연결하는 분석 워크스페이스입니다.

---

## 1) 현재 구성(작업 반영 기준)

- **Backend**: `backend/app_main.py` (FastAPI 메인)
- **Frontend**: `frontend/src/components/OntologyWorkbench.jsx`, `frontend/src/styles.css`
- **데이터/인덱스**
  - `data/commonfeature.json` (feature 사전)
  - `data/full_text_records.json` (로그 레코드)
  - `data/feature_customer_clusters.json` (군집 캐시)
  - `data/segment_metric_cube.json` (지표 큐브)
  - `data/regulation_uploads/*` (규제문서 PDF)
  - `faiss_*` 계열 인덱스(로그/뉴스/규제 검색)

---

## 2) 실행 방법

## Backend
```powershell
cd C:\work\ai-log-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 18000
```

## Frontend
```powershell
cd C:\work\ai-log-agent\frontend
npm install
npm run dev -- --host 127.0.0.1 --port 3001
```

브라우저: `http://127.0.0.1:3001`

---

## 3) 질의 처리 파이프라인

1. 질문 입력  
2. Intent 분류 (`intent_classification`)
3. Output 카테고리 분류 (`output_classification`)
4. feature 선택 / related feature 확장
5. cluster/retrieval/regulation evidence 조합
6. answer summary 생성 + TOOL AGENT 카드 렌더링

---

## 4) Intent 분류(최근 수정사항 포함)

핵심 함수:
- `_classify_query_intent(...)`
- `_rule_based_query_intent(...)`
- `_score_intents_with_embeddings(...)`
- `_query_requires_strategy_simulation(...)`

의도 프로토타입:
- `approval_factor`
- `reject_reason`
- `rate_limit`
- `cluster_vector`
- `regulation_policy`
- `strategy_simulation`
- `general_fallback`

### 최근 보정(중요)

`"승인 한도에 영향을 주는 feature는?"` 같은 질문이  
`strategy_simulation`으로 잘못 분류되던 이슈를 수정했습니다.

- rule intent에서 영향 feature 질의는 `cluster_vector`로 라우팅
- 전략 시뮬레이션 판정에서 설명형 feature 질의를 제외
- 프론트 임시 우회 대신 **백엔드 분류 결과 자체**가 바뀌도록 수정

---

## 5) Graph RAG는 언제 쓰이나?

현재 코드 기준으로 Graph RAG 성격은 아래 단계에서 사용됩니다.

- 질문 토큰 ↔ feature 매핑
- selected feature ↔ related feature 확장
- representative feature 선정 시 `base + intent + graph` 하이브리드 점수
- 최종 답변/카드에 들어갈 근거 후보 정리

즉, 단순 벡터 검색만이 아니라 **ontology relation(그래프 관계)** 를 함께 써서  
질문 의도와 연관 feature를 더 안정적으로 선택합니다.

---

## 6) 요약탭/카드 동작(최근 UI 반영)

- 요약 상단에 **Intent 해석 문구** 표시
- Intent 문구 클릭 시 팝업:
  - `rule_intent / embedding_intent / final_intent`
  - 분류 사유
  - 카테고리/카드 우선순위
  - 질문 처리 흐름 다이어그램
- Explainability 카드가 존재하면 요약설명 아래 고정 노출
- 질문/카드/테마 UI 개선(부서원 아바타, 강조 스타일, 레이아웃 정리 등)

---

## 7) 파일 포인트

- Backend intent/라우팅:
  - `C:\work\ai-log-agent\backend\app_main.py`
- Frontend 요약/탭/인텐트 팝업:
  - `C:\work\ai-log-agent\frontend\src\components\OntologyWorkbench.jsx`
- Frontend 스타일:
  - `C:\work\ai-log-agent\frontend\src\styles.css`

---

## 8) 점검 체크리스트

- [ ] `uvicorn backend.main:app` 정상 기동
- [ ] 프론트 `npm run dev` 접속 가능
- [ ] `"승인 한도 영향 feature"` 질의 시 intent가 strategy가 아닌 cluster 계열로 분류
- [ ] 요약 Intent 팝업에서 rule/embedding/final intent 표시 확인
- [ ] Explainability 카드 고정 노출 동작 확인

---

## 9) 참고

- Python 3.14에서 일부 라이브러리 경고가 있을 수 있어, 실무 실행은 3.13 가상환경 권장
- 규제문서/뉴스/군집 데이터는 로컬 데이터 상태에 따라 결과가 달라질 수 있음
