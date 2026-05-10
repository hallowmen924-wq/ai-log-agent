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
COMMON_FEATURE_PATH = DATA_DIR / "commonfeature.json"
ONTOLOGY_PATH = DATA_DIR / "ontology.json"

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
    best_match: tuple[int, int, int, dict[str, Any], str] | None = None

    for rule in RULE_INDEX:
        if direction not in rule["directions"]:
            continue

        exact_match = normalized_label in rule["normalized_aliases"]
        for tokens in rule["normalized_token_groups"]:
            if not tokens:
                continue
            if all(token in normalized_label for token in tokens):
                token_score = len(tokens)
                exact_bonus = 100 if exact_match else 0
                priority_score = -int(rule["priority"])
                candidate = (exact_bonus + token_score, token_score, priority_score, rule, "high" if exact_match or token_score >= 2 else "medium")
                if best_match is None or candidate[:3] > best_match[:3]:
                    best_match = candidate

    if best_match is not None:
        return best_match[3], best_match[4], "rule"

    slug = make_slug(label)
    fallback_rule = {
        "feature_id": f"{direction}.unclassified.{slug}",
        "name": label or slug,
        "category": "unclassified",
        "directions": [direction],
        "description": f"{direction} 영역에서 자동 규칙에 매핑되지 않은 원본 필드",
        "aliases": [label] if label else [],
    }
    return fallback_rule, "fallback", "fallback"


def load_snapshot() -> dict[str, Any]:
    with SNAPSHOT_PATH.open(encoding="utf-8") as file:
        return json.load(file)


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
            "products": set(),
            "field_mappings": [],
            "sample_values": [],
        }
        index[rule["feature_id"]] = feature
    return feature


def build_common_feature_and_ontology() -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = load_snapshot()
    reject_mapping = load_reject_code_mapping(DATA_DIR)
    products = snapshot.get("products") or {}

    common_feature_index: dict[str, dict[str, Any]] = {}
    ontology_products: dict[str, Any] = {}

    for product_code, product_payload in products.items():
        product_name = PRODUCT_NAMES.get(product_code, product_code)
        ontology_product = {
            "product_code": product_code,
            "product_name": product_name,
            "input_fields": {},
            "output_fields": {},
            "reject_reason": {
                "feature_id": "decision.reject_reason_code",
                "feature_name": "거절사유코드",
                "description": "K 코드 기반 거절사유 taxonomy 참조",
                "observed_codes": sorted((product_payload.get("reject_reason_codes") or {}).keys()),
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
                    "confidence": confidence,
                    "match_basis": match_basis,
                    "observed_count": int((metadata or {}).get("observed_count") or 0),
                    "sample_values": list((metadata or {}).get("sample_values") or []),
                }

        ontology_product["summary"] = {
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
        "source": {
            "mapping_snapshot_path": str(SNAPSHOT_PATH.relative_to(PROJECT_ROOT)),
            "snapshot_generated_at": snapshot.get("generated_at"),
            "snapshot_analyzed_log_count": int(((snapshot.get("source") or {}).get("analyzed_log_count") or 0)),
            "reject_taxonomy_code_count": len(reject_mapping),
        },
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
            "mapping_snapshot_path": str(SNAPSHOT_PATH.relative_to(PROJECT_ROOT)),
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