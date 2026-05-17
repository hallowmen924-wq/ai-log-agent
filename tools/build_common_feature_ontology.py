import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mapper.reject_code_mapper import load_reject_code_mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOT_PATH = DATA_DIR / "product_mapping_snapshot.json"
FULL_TEXT_RECORDS_PATH = DATA_DIR / "full_text_records.json"
COMMON_FEATURE_PATH = DATA_DIR / "commonfeature.json"
ONTOLOGY_PATH = DATA_DIR / "ontology.json"

AUTO_CLUSTER_DROP_TOKENS = {
    "사전",
    "실시간",
    "적용배제",
    "확인용",
    "당일포함",
    "마케팅용",
    "버전",
    "버젼",
    "기준",
    "기준값",
    "ver",
}

AUTO_CLUSTER_SUFFIX_PATTERNS = (
    r"(?:_)+(?:사전|실시간|적용배제|확인용|당일포함|마케팅용|기준|ver\d+(?:\.\d+)?)$",
    r"(?:_)+(?:n|ln|v)\d+$",
)

PRODUCT_NAMES = {
    "C11": "개인사업자대출",
    "C9": "카드론(이지론)",
    "C6": "신용대출(이지신용대출)",
    "C12": "이지대환대출",
}

FEATURE_RULES: list[dict[str, Any]] = [
    {
        "feature_id": "application.case_id",
        "name": "접수번호",
        "category": "application",
        "directions": ["input", "output"],
        "description": "신청 건을 식별하는 접수 또는 요청 번호",
        "aliases": ["접수번호", "신청서접수번호", "요청번호", "요청관리번호"],
        "token_groups": [("접수번호",), ("신청서접수번호",), ("req",), ("요청", "번호")],
    },
    {
        "feature_id": "application.product_code",
        "name": "상품코드",
        "category": "application",
        "directions": ["input", "output"],
        "description": "대출 상품 식별 코드",
        "aliases": ["상품코드", "상품코드값"],
        "token_groups": [("상품코드",), ("상품", "코드")],
    },
    {
        "feature_id": "loan.requested_amount",
        "name": "신청금액",
        "category": "loan",
        "directions": ["input"],
        "description": "고객이 신청한 대출 금액",
        "aliases": ["대출신청금액", "신청금액", "요청금액"],
        "token_groups": [("신청", "금액"), ("대출", "신청", "금액"), ("요청", "금액")],
    },
    {
        "feature_id": "loan.requested_term",
        "name": "신청기간",
        "category": "loan",
        "directions": ["input"],
        "description": "고객이 요청한 대출 기간",
        "aliases": ["신청상환기간", "신청대출기간", "신청기간"],
        "token_groups": [("신청", "기간"), ("신청", "상환", "기간"), ("신청", "대출", "기간")],
    },
    {
        "feature_id": "applicant.age",
        "name": "연령",
        "category": "applicant",
        "directions": ["input"],
        "description": "신청자 연령",
        "aliases": ["연령", "나이", "age"],
        "token_groups": [("연령",), ("나이",), ("age",)],
    },
    {
        "feature_id": "applicant.foreigner_flag",
        "name": "외국인여부",
        "category": "applicant",
        "directions": ["input"],
        "description": "신청자의 외국인 여부 또는 국적 관련 구분",
        "aliases": ["외국인여부", "내외국인구분", "국적"],
        "token_groups": [("외국인",), ("국적",), ("내외국인",)],
    },
    {
        "feature_id": "income.recognized_income",
        "name": "인정소득",
        "category": "income",
        "directions": ["input", "output"],
        "description": "심사에 반영된 인정소득 또는 최종소득",
        "aliases": ["인정소득", "최종연소득", "스크래핑소득"],
        "token_groups": [("인정소득",), ("최종", "연소득"), ("스크래핑", "소득")],
    },
    {
        "feature_id": "income.annual_income",
        "name": "연소득",
        "category": "income",
        "directions": ["input", "output"],
        "description": "신청자 연소득 또는 소득 금액",
        "aliases": ["연소득", "소득금액", "income", "salary"],
        "token_groups": [("연소득",), ("소득", "금액"), ("income",), ("salary",)],
    },
    {
        "feature_id": "income.health_insurance_type",
        "name": "건강보험가입자구분",
        "category": "income",
        "directions": ["input"],
        "description": "건강보험 가입자 구분",
        "aliases": ["건강보험가입자구분", "건강보험가입자구분코드"],
        "token_groups": [("건강보험", "가입자", "구분"), ("건강보험", "구분")],
    },
    {
        "feature_id": "income.health_insurance_paid_3m_flag",
        "name": "건강보험3개월납부여부",
        "category": "income",
        "directions": ["input"],
        "description": "최근 3개월 건강보험료 납부 여부",
        "aliases": ["건강보험3개월납부여부"],
        "token_groups": [("건강보험", "3개월", "납부")],
    },
    {
        "feature_id": "channel.channel_code",
        "name": "채널구분",
        "category": "channel",
        "directions": ["input"],
        "description": "유입 채널 또는 제휴 채널 구분",
        "aliases": ["채널구분", "채널구분코드", "제휴채널"],
        "token_groups": [("채널", "구분"), ("제휴", "채널")],
    },
    {
        "feature_id": "loan.requested_limit",
        "name": "대출금액",
        "category": "loan",
        "directions": ["input"],
        "description": "입력 측 대출 금액 또는 한도 금액",
        "aliases": ["대출금액", "한도금액", "카드론한도금액"],
        "token_groups": [("대출", "금액"), ("한도", "금액")],
    },
    {
        "feature_id": "credit.non_face_to_face_loan_grade",
        "name": "비대면연계대출등급",
        "category": "credit",
        "directions": ["input"],
        "description": "비대면 연계 대출 등급",
        "aliases": ["비대면연계대출등급"],
        "token_groups": [("비대면", "연계", "대출", "등급")],
    },
    {
        "feature_id": "credit.loan_count",
        "name": "신용대출건수",
        "category": "credit",
        "directions": ["input"],
        "description": "보유한 신용대출 건수",
        "aliases": ["신용대출건수"],
        "token_groups": [("신용대출", "건수")],
    },
    {
        "feature_id": "credit.ml_score_grade",
        "name": "ML스코어 등급",
        "category": "credit_model",
        "directions": ["input", "output"],
        "description": "ML 기반 신용평가 등급 또는 점수",
        "aliases": ["ML스코어 등급", "ML등급", "ML점수"],
        "token_groups": [("ml", "등급"), ("ml", "스코어"), ("ml", "점수")],
    },
    {
        "feature_id": "credit.kcb_grade",
        "name": "KCB 등급",
        "category": "credit_bureau",
        "directions": ["input", "output"],
        "description": "KCB 등급 또는 평가값",
        "aliases": ["KCB 등급", "KCB평점", "KCB스코어"],
        "token_groups": [("kcb", "등급"), ("kcb", "평점"), ("kcb", "스코어")],
    },
    {
        "feature_id": "credit.nice_grade",
        "name": "NICE 등급",
        "category": "credit_bureau",
        "directions": ["input", "output"],
        "description": "NICE 등급 또는 평가값",
        "aliases": ["NICE 등급", "NICE CB등급", "NICE평점", "NICE스코어"],
        "token_groups": [("nice", "등급"), ("nice", "cb", "등급"), ("nice", "평점"), ("nice", "스코어")],
    },
    {
        "feature_id": "customer.valid_member_flag",
        "name": "유효회원여부",
        "category": "customer_relationship",
        "directions": ["input"],
        "description": "회원 유효성 여부",
        "aliases": ["유효회원여부"],
        "token_groups": [("유효", "회원")],
    },
    {
        "feature_id": "customer.valid_customer_flag",
        "name": "유효고객여부",
        "category": "customer_relationship",
        "directions": ["input"],
        "description": "고객 유효성 여부",
        "aliases": ["유효고객여부"],
        "token_groups": [("유효", "고객")],
    },
    {
        "feature_id": "customer.credit_card_holding_flag",
        "name": "신용카드소지여부",
        "category": "customer_relationship",
        "directions": ["input"],
        "description": "신용카드 보유 여부",
        "aliases": ["신용카드소지여부"],
        "token_groups": [("신용카드", "소지")],
    },
    {
        "feature_id": "customer.check_card_holding_flag",
        "name": "체크카드보유여부",
        "category": "customer_relationship",
        "directions": ["input"],
        "description": "체크카드 보유 여부",
        "aliases": ["체크카드보유여부"],
        "token_groups": [("체크카드", "보유")],
    },
    {
        "feature_id": "exposure.card_loan_balance",
        "name": "카드론대출잔액",
        "category": "exposure",
        "directions": ["input"],
        "description": "카드론 잔액",
        "aliases": ["카드론대출잔액"],
        "token_groups": [("카드론", "잔액")],
    },
    {
        "feature_id": "exposure.auto_loan_balance",
        "name": "오토론대출잔액",
        "category": "exposure",
        "directions": ["input"],
        "description": "오토론 잔액",
        "aliases": ["오토론대출잔액"],
        "token_groups": [("오토론", "잔액")],
    },
    {
        "feature_id": "exposure.credit_loan_balance",
        "name": "신용대출잔액",
        "category": "exposure",
        "directions": ["input"],
        "description": "신용대출 잔액",
        "aliases": ["신용대출잔액"],
        "token_groups": [("신용대출", "잔액")],
    },
    {
        "feature_id": "exposure.refinance_loan_balance",
        "name": "대환대출잔액",
        "category": "exposure",
        "directions": ["input"],
        "description": "대환성 대출 잔액",
        "aliases": ["대환론대출잔액", "대환대출잔액"],
        "token_groups": [("대환론", "잔액"), ("대환", "대출", "잔액")],
    },
    {
        "feature_id": "exposure.total_credit",
        "name": "TOTAL CREDIT",
        "category": "exposure",
        "directions": ["input"],
        "description": "총 신용한도 또는 TOTAL CREDIT",
        "aliases": ["TOTAL CREDIT", "TOTAL CREDIT잔여한도"],
        "token_groups": [("total", "credit")],
    },
    {
        "feature_id": "decision.approved_amount",
        "name": "승인가능금액",
        "category": "decision",
        "directions": ["output"],
        "description": "심사 결과 산출된 승인 가능 금액 또는 한도",
        "aliases": ["승인가능금액", "한도금액", "대출가능금액", "한도"],
        "token_groups": [("승인", "가능", "금액"), ("가능", "금액"), ("한도", "금액"), ("대출", "가능", "금액")],
    },
    {
        "feature_id": "decision.applied_rate",
        "name": "산출금리",
        "category": "decision",
        "directions": ["output"],
        "description": "산출 또는 적용 금리",
        "aliases": ["산출금리", "적용금리", "금리"],
        "token_groups": [("산출", "금리"), ("적용", "금리"), ("금리",)],
    },
    {
        "feature_id": "decision.loan_term",
        "name": "대출기간",
        "category": "decision",
        "directions": ["output"],
        "description": "산출된 대출 기간",
        "aliases": ["대출기간", "상환기간"],
        "token_groups": [("대출", "기간"), ("상환", "기간"), ("기간",)],
    },
    {
        "feature_id": "decision.grace_period",
        "name": "거치기간",
        "category": "decision",
        "directions": ["output"],
        "description": "산출된 거치 기간",
        "aliases": ["거치기간"],
        "token_groups": [("거치", "기간")],
    },
    {
        "feature_id": "decision.dti",
        "name": "DTI",
        "category": "decision",
        "directions": ["output"],
        "description": "부채상환비율 DTI",
        "aliases": ["DTI"],
        "token_groups": [("dti",)],
    },
    {
        "feature_id": "decision.dsr",
        "name": "DSR",
        "category": "decision",
        "directions": ["output"],
        "description": "총부채원리금상환비율 DSR",
        "aliases": ["DSR"],
        "token_groups": [("dsr",)],
    },
    {
        "feature_id": "decision.result_flag",
        "name": "승인여부",
        "category": "decision",
        "directions": ["output"],
        "description": "승인 또는 거절 여부",
        "aliases": ["승인여부", "심사결과", "결과"],
        "token_groups": [("승인",), ("심사", "결과"), ("결과",)],
    },
    {
        "feature_id": "decision.risk_grade",
        "name": "RISK등급",
        "category": "decision",
        "directions": ["output"],
        "description": "리스크 등급 또는 위험도 그룹",
        "aliases": ["RISK등급", "리스크등급"],
        "token_groups": [("risk", "등급"), ("리스크", "등급")],
    },
    {
        "feature_id": "decision.limit_group",
        "name": "한도그룹",
        "category": "decision",
        "directions": ["output"],
        "description": "한도 그룹 또는 한도 정책 분류",
        "aliases": ["한도그룹", "한도정책그룹"],
        "token_groups": [("한도", "그룹")],
    },
    {
        "feature_id": "score.k_score",
        "name": "K-Score",
        "category": "credit_model",
        "directions": ["output"],
        "description": "내부 K-Score 점수 또는 등급",
        "aliases": ["K-Score Score", "K-Score"],
        "token_groups": [("kscore",), ("k-score",), ("kscore", "score")],
    },
    {
        "feature_id": "decision.reject_reason_text",
        "name": "거절사유",
        "category": "decision",
        "directions": ["reject"],
        "description": "거절사유 설명 텍스트",
        "aliases": ["거절사유", "심사사유"],
        "token_groups": [("거절", "사유"), ("심사", "사유")],
    },
    {
        "feature_id": "decision.reject_reason_code",
        "name": "거절사유코드",
        "category": "decision",
        "directions": ["reject"],
        "description": "거절사유 코드체계(K코드)",
        "aliases": ["K코드", "거절사유코드"],
        "token_groups": [("거절", "코드"), ("k", "code")],
    },
]


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[()\[\]{}\"'`]+", " ", text)
    text = re.sub(r"\s+", "", text)
    return text


def canonicalize_label(label: str) -> tuple[str, str]:
    cleaned = str(label or "").strip()
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = cleaned.replace("||", " ")
    cleaned = cleaned.replace("·", " ")
    cleaned = re.sub(r"[_|/]+", " ", cleaned)
    cleaned = re.sub(r"\((?:ver|VER)?\s*\d+(?:\.\d+)?\)", " ", cleaned)
    cleaned = re.sub(r"\b(?:사전|실시간|적용배제|확인용|당일포함|마케팅용|기준값?)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for pattern in AUTO_CLUSTER_SUFFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    if re.search(r"(?:^|\s)(?:사전|실시간|적용배제)(?:\s|$)", cleaned):
        cleaned = re.sub(r"(?:^|\s)(?:사전|실시간|적용배제)(?:\s|$)", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    kept_tokens: list[str] = []
    seen_normalized_tokens: set[str] = set()
    for token in cleaned.split():
        normalized = normalize_text(token)
        if not normalized:
            continue
        if normalized in AUTO_CLUSTER_DROP_TOKENS:
            continue
        if re.fullmatch(r"(?:n|ln|v)\d+", normalized):
            continue
        if re.fullmatch(r"(?:ver)?\d+(?:\.\d+)?", normalized):
            continue
        if normalized in seen_normalized_tokens:
            continue
        seen_normalized_tokens.add(normalized)
        kept_tokens.append(token)

    display = " ".join(kept_tokens).strip() or str(label or "").strip()
    return display, normalize_text(display)


def make_slug(value: str) -> str:
    compact = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", str(value or "").strip().lower())
    compact = compact.strip("_")
    if not compact:
        compact = "unknown"
    if len(compact) > 48:
        digest = hashlib.sha1(compact.encode("utf-8")).hexdigest()[:8]
        compact = f"{compact[:48]}_{digest}"
    return compact


def build_rule_index() -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for priority, rule in enumerate(FEATURE_RULES):
        indexed.append(
            {
                **rule,
                "priority": priority,
                "normalized_aliases": [normalize_text(alias) for alias in rule.get("aliases") or []],
                "normalized_token_groups": [
                    tuple(normalize_text(token) for token in tokens)
                    for tokens in (rule.get("token_groups") or [])
                ],
            }
        )
    return indexed


RULE_INDEX = build_rule_index()


def classify_label(label: str, direction: str) -> tuple[dict[str, Any], str, str]:
    normalized_label = normalize_text(label)
    canonical_label, normalized_canonical = canonicalize_label(label)
    best_match: tuple[int, int, int, dict[str, Any], str] | None = None

    for rule in RULE_INDEX:
        if direction not in rule["directions"]:
            continue

        exact_match = normalized_label in rule["normalized_aliases"] or normalized_canonical in rule["normalized_aliases"]
        if exact_match:
            candidate = (1000, 0, -int(rule["priority"]), rule, "high")
            if best_match is None or candidate[:3] > best_match[:3]:
                best_match = candidate
        for tokens in rule["normalized_token_groups"]:
            if not tokens:
                continue
            if len(tokens) == 1:
                token_matched = normalized_label == tokens[0] or normalized_canonical == tokens[0]
            else:
                token_matched = all(token in normalized_label or token in normalized_canonical for token in tokens)
            if token_matched:
                token_score = len(tokens)
                exact_bonus = 100 if exact_match else 0
                priority_score = -int(rule["priority"])
                candidate = (exact_bonus + token_score, token_score, priority_score, rule, "high" if exact_match or token_score >= 2 else "medium")
                if best_match is None or candidate[:3] > best_match[:3]:
                    best_match = candidate

    if best_match is not None:
        return best_match[3], best_match[4], "rule"

    cluster_label = canonical_label or label or "unknown"
    cluster_key = normalized_canonical or normalized_label or make_slug(cluster_label)
    cluster_rule = {
        "feature_id": f"{direction}.auto.{make_slug(cluster_key)}",
        "name": cluster_label,
        "category": "auto_cluster",
        "directions": [direction],
        "description": f"{direction} 영역의 실제 로그 라벨을 자동 군집화한 공통 feature",
        "aliases": [cluster_label] if cluster_label else ([label] if label else []),
        "cluster_key": cluster_key,
        "cluster_label": cluster_label,
    }
    return cluster_rule, "auto", "cluster"


def load_snapshot() -> dict[str, Any]:
    with SNAPSHOT_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_full_text_records() -> dict[str, Any] | None:
    if not FULL_TEXT_RECORDS_PATH.exists():
        return None
    with FULL_TEXT_RECORDS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def parse_labeled_text_block(text: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        payload = line[1:].strip()
        if ":" not in payload:
            continue
        label, value = payload.split(":", 1)
        clean_label = str(label or "").strip()
        if not clean_label or clean_label.lower() == "nan":
            continue
        rows.append((clean_label, str(value or "").strip()))
    return rows


def make_synthetic_field_code(prefix: str, label: str, existing: dict[str, Any]) -> str:
    base = f"{prefix}_{make_slug(label)}"
    candidate = base
    if candidate not in existing:
        return candidate
    if str((existing.get(candidate) or {}).get("label") or "").strip() == str(label or "").strip():
        return candidate
    digest = hashlib.sha1(str(label or "").encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def append_sample_value(counter: dict[str, int], value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    counter[text] = int(counter.get(text) or 0) + 1


def finalize_sample_values(counter: dict[str, int], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))[:limit]
    ]


def upsert_section_mapping(section: dict[str, Any], prefix: str, label: str, value: Any) -> None:
    field_code = make_synthetic_field_code(prefix, label, section)
    entry = section.setdefault(
        field_code,
        {
            "label": label,
            "observed_count": 0,
            "sample_values": {},
        },
    )
    entry["observed_count"] = int(entry.get("observed_count") or 0) + 1
    append_sample_value(entry["sample_values"], value)


def finalize_section_mapping(section: dict[str, Any]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for field_code, metadata in section.items():
        finalized[field_code] = {
            "label": metadata.get("label"),
            "observed_count": int(metadata.get("observed_count") or 0),
            "sample_values": finalize_sample_values(dict(metadata.get("sample_values") or {})),
        }
    return finalized


def build_products_from_full_text_records(payload: dict[str, Any]) -> dict[str, Any]:
    products: dict[str, Any] = {}

    for record in payload.get("records") or []:
        product_code = str(record.get("product") or "").strip()
        if not product_code:
            continue
        product_name = str(record.get("product_display") or PRODUCT_NAMES.get(product_code, product_code)).strip()
        product_payload = products.setdefault(
            product_code,
            {
                "product_name": product_name,
                "observed_record_count": 0,
                "in_mapping": {},
                "out_mapping": {},
                "reject_reason_codes": defaultdict(int),
                "reject_reason_texts": defaultdict(int),
            },
        )
        product_payload["observed_record_count"] = int(product_payload.get("observed_record_count") or 0) + 1

        in_text = record.get("in_text2") or record.get("in_text") or ""
        out_text = record.get("out_text2") or record.get("out_text") or ""

        for label, value in parse_labeled_text_block(in_text):
            upsert_section_mapping(product_payload["in_mapping"], "IN", label, value)

        for label, value in parse_labeled_text_block(out_text):
            upsert_section_mapping(product_payload["out_mapping"], "OUT", label, value)

        reject_reason_text = str(record.get("reject_reason_text") or "").strip()
        if reject_reason_text:
            product_payload["reject_reason_texts"][reject_reason_text] += 1
            upsert_section_mapping(product_payload["out_mapping"], "OUT", "거절사유", reject_reason_text)

        for reject_code in record.get("reject_reason_codes") or []:
            code = str(reject_code or "").strip()
            if code:
                product_payload["reject_reason_codes"][code] += 1

    finalized: dict[str, Any] = {}
    for product_code, product_payload in products.items():
        finalized[product_code] = {
            "product_name": product_payload.get("product_name") or PRODUCT_NAMES.get(product_code, product_code),
            "observed_record_count": int(product_payload.get("observed_record_count") or 0),
            "in_mapping": finalize_section_mapping(dict(product_payload.get("in_mapping") or {})),
            "out_mapping": finalize_section_mapping(dict(product_payload.get("out_mapping") or {})),
            "reject_reason_codes": dict(product_payload.get("reject_reason_codes") or {}),
            "reject_reason_texts": finalize_sample_values(dict(product_payload.get("reject_reason_texts") or {})),
        }
    return finalized


def append_sample_values(target: list[dict[str, Any]], samples: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    seen = {str(item.get("value")): item for item in target}
    for sample in samples or []:
        value = str(sample.get("value") or "").strip()
        if not value:
            continue
        existing = seen.get(value)
        if existing is None:
            entry = {"value": value, "count": int(sample.get("count") or 0)}
            target.append(entry)
            seen[value] = entry
        else:
            existing["count"] = max(int(existing.get("count") or 0), int(sample.get("count") or 0))
    target.sort(key=lambda item: (-int(item.get("count") or 0), str(item.get("value") or "")))
    return target[:limit]


def ensure_common_feature(index: dict[str, dict[str, Any]], rule: dict[str, Any], direction: str) -> dict[str, Any]:
    feature = index.get(rule["feature_id"])
    if feature is None:
        feature = {
            "feature_id": rule["feature_id"],
            "feature_name": rule["name"],
            "category": rule.get("category", "unclassified"),
            "description": rule.get("description", ""),
            "directions": sorted(set(rule.get("directions") or [direction])),
            "aliases": sorted(set(rule.get("aliases") or [])),
            "cluster_key": rule.get("cluster_key"),
            "cluster_label": rule.get("cluster_label"),
            "products": set(),
            "field_mappings": [],
            "sample_values": [],
        }
        index[rule["feature_id"]] = feature
    else:
        if rule.get("cluster_key") and not feature.get("cluster_key"):
            feature["cluster_key"] = rule.get("cluster_key")
        if rule.get("cluster_label") and not feature.get("cluster_label"):
            feature["cluster_label"] = rule.get("cluster_label")
    return feature


def build_common_feature_and_ontology() -> tuple[dict[str, Any], dict[str, Any]]:
    reject_mapping = load_reject_code_mapping(DATA_DIR)
    full_text_payload = load_full_text_records()
    snapshot = load_snapshot() if SNAPSHOT_PATH.exists() else {}
    if full_text_payload and (full_text_payload.get("records") or []):
        products = build_products_from_full_text_records(full_text_payload)
        source_meta = {
            "source_mode": "full_text_records",
            "full_text_records_path": str(FULL_TEXT_RECORDS_PATH.relative_to(PROJECT_ROOT)),
            "full_text_records_generated_at": full_text_payload.get("generated_at"),
            "full_text_record_count": len(full_text_payload.get("records") or []),
            "mapping_snapshot_path": str(SNAPSHOT_PATH.relative_to(PROJECT_ROOT)) if SNAPSHOT_PATH.exists() else None,
            "snapshot_generated_at": snapshot.get("generated_at"),
            "snapshot_analyzed_log_count": int(((snapshot.get("source") or {}).get("analyzed_log_count") or 0)),
            "reject_taxonomy_code_count": len(reject_mapping),
        }
    else:
        products = snapshot.get("products") or {}
        source_meta = {
            "source_mode": "mapping_snapshot",
            "mapping_snapshot_path": str(SNAPSHOT_PATH.relative_to(PROJECT_ROOT)),
            "snapshot_generated_at": snapshot.get("generated_at"),
            "snapshot_analyzed_log_count": int(((snapshot.get("source") or {}).get("analyzed_log_count") or 0)),
            "reject_taxonomy_code_count": len(reject_mapping),
        }

    common_feature_index: dict[str, dict[str, Any]] = {}
    ontology_products: dict[str, Any] = {}

    for product_code, product_payload in products.items():
        product_name = str(product_payload.get("product_name") or PRODUCT_NAMES.get(product_code, product_code))
        ontology_product = {
            "product_code": product_code,
            "product_name": product_name,
            "observed_record_count": int(product_payload.get("observed_record_count") or 0),
            "input_fields": {},
            "output_fields": {},
            "reject_reason": {
                "feature_id": "decision.reject_reason_code",
                "feature_name": "거절사유코드",
                "description": "K 코드 기반 거절사유 taxonomy 참조",
                "observed_codes": sorted((product_payload.get("reject_reason_codes") or {}).keys()),
                "observed_texts": list(product_payload.get("reject_reason_texts") or []),
            },
        }

        for section_key, direction, ontology_key in (
            ("in_mapping", "input", "input_fields"),
            ("out_mapping", "output", "output_fields"),
        ):
            section_mapping = product_payload.get(section_key) or {}
            for field_code, metadata in section_mapping.items():
                label = str((metadata or {}).get("label") or "").strip()
                if not label or label.lower() == "nan":
                    continue

                rule, confidence, match_basis = classify_label(label, direction)
                common_feature = ensure_common_feature(common_feature_index, rule, direction)
                common_feature["products"].add(product_code)
                common_feature["field_mappings"].append(
                    {
                        "product": product_code,
                        "product_name": product_name,
                        "direction": direction,
                        "field_code": field_code,
                        "label": label,
                        "observed_count": int((metadata or {}).get("observed_count") or 0),
                    }
                )
                common_feature["sample_values"] = append_sample_values(
                    common_feature["sample_values"],
                    list((metadata or {}).get("sample_values") or []),
                )

                ontology_product[ontology_key][field_code] = {
                    "label": label,
                    "feature_id": common_feature["feature_id"],
                    "feature_name": common_feature["feature_name"],
                    "category": common_feature["category"],
                    "cluster_key": common_feature.get("cluster_key"),
                    "cluster_label": common_feature.get("cluster_label"),
                    "confidence": confidence,
                    "match_basis": match_basis,
                    "observed_count": int((metadata or {}).get("observed_count") or 0),
                    "sample_values": list((metadata or {}).get("sample_values") or []),
                }

        ontology_product["summary"] = {
            "observed_record_count": int(product_payload.get("observed_record_count") or 0),
            "input_field_count": len(ontology_product["input_fields"]),
            "output_field_count": len(ontology_product["output_fields"]),
            "observed_reject_code_count": len(ontology_product["reject_reason"]["observed_codes"]),
        }
        ontology_products[product_code] = ontology_product

    reject_feature = ensure_common_feature(
        common_feature_index,
        {
            "feature_id": "decision.reject_reason_code",
            "name": "거절사유코드",
            "category": "decision",
            "directions": ["reject"],
            "description": "K코드 기반 거절사유 taxonomy",
            "aliases": ["거절사유코드", "K코드"],
        },
        "reject",
    )
    reject_feature["taxonomy_code_count"] = len(reject_mapping)
    for product_code, product_payload in ontology_products.items():
        reject_feature["products"].add(product_code)
        observed_codes = list(((products.get(product_code) or {}).get("reject_reason_codes") or {}).keys())
        reject_feature["field_mappings"].append(
            {
                "product": product_code,
                "product_name": str(product_payload.get("product_name") or product_code),
                "direction": "reject",
                "field_code": "REJECT_REASON_CODE",
                "label": "거절사유코드",
                "observed_count": len(observed_codes),
            }
        )
        reject_feature["sample_values"] = append_sample_values(
            reject_feature["sample_values"],
            [{"value": code, "count": 1} for code in observed_codes],
        )

    common_features = []
    fallback_count = 0
    for feature in common_feature_index.values():
        directions = sorted(set(feature.pop("directions", [])))
        products_set = feature.pop("products", set())
        field_mappings = feature.get("field_mappings") or []
        if feature.get("category") == "unclassified":
            fallback_count += 1
        feature["directions"] = directions
        feature["products"] = sorted(products_set)
        feature["coverage"] = {
            "product_count": len(products_set),
            "mapping_count": len(field_mappings),
        }
        if not feature.get("cluster_key"):
            feature.pop("cluster_key", None)
        if not feature.get("cluster_label"):
            feature.pop("cluster_label", None)
        feature["field_mappings"] = sorted(
            field_mappings,
            key=lambda item: (
                str(item.get("product") or ""),
                str(item.get("direction") or ""),
                str(item.get("field_code") or ""),
            ),
        )
        common_features.append(feature)

    common_features.sort(key=lambda item: (item.get("category") == "unclassified", item.get("feature_id") or ""))

    commonfeature_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_meta,
        "statistics": {
            "product_count": len(ontology_products),
            "common_feature_count": len(common_features),
            "fallback_feature_count": fallback_count,
            "classified_feature_count": len(common_features) - fallback_count,
        },
        "common_features": common_features,
    }

    ontology_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            **source_meta,
            "commonfeature_path": str(COMMON_FEATURE_PATH.relative_to(PROJECT_ROOT)),
        },
        "products": ontology_products,
        "common_feature_index": {
            feature["feature_id"]: {
                "feature_name": feature["feature_name"],
                "category": feature["category"],
                "directions": feature["directions"],
            }
            for feature in common_features
        },
        "reject_reason_taxonomy": {
            "feature_id": "decision.reject_reason_code",
            "feature_name": "거절사유코드",
            "codes": reject_mapping,
        },
    }

    return commonfeature_payload, ontology_payload


def main() -> None:
    commonfeature_payload, ontology_payload = build_common_feature_and_ontology()
    COMMON_FEATURE_PATH.write_text(
        json.dumps(commonfeature_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ONTOLOGY_PATH.write_text(
        json.dumps(ontology_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote={COMMON_FEATURE_PATH}")
    print(f"wrote={ONTOLOGY_PATH}")
    print(
        json.dumps(
            {
                "common_feature_count": commonfeature_payload["statistics"]["common_feature_count"],
                "fallback_feature_count": commonfeature_payload["statistics"]["fallback_feature_count"],
                "product_count": commonfeature_payload["statistics"]["product_count"],
                "reject_taxonomy_code_count": commonfeature_payload["source"]["reject_taxonomy_code_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()