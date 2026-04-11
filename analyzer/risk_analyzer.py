"""
Risk analyzer adapted to R-CLIPS mappings provided in attachments.

calculate_risk examines `in_fields` and `out_fields` using the provided
`in_mapping`/`out_mapping` (excel code -> human-readable name). It
searches the mapping names for keywords (예: '대출', 'DSR', '연체', '소득',
'신용') to find which field codes contain the values we need.

This makes the function resilient to product-specific code names (A2001,
R0003 등) as long as the mapping contains meaningful Korean labels.
"""

from typing import Any, Dict, Optional


def calculate_risk(in_fields: Dict[str, Any], out_fields: Dict[str, Any], in_mapping: Optional[Dict[str, str]] = None, out_mapping: Optional[Dict[str, str]] = None, product: Optional[str] = None) -> Dict[str, Any]:

    in_mapping = in_mapping or {}
    out_mapping = out_mapping or {}

    result = {
        "score": 0,
        "grade": "",
        "reasons": [],
        "details": {
            "financial": 0,
            "credit": 0,
            "behavior": 0,
            "regulation": 0,
        },
    }

    # 제품별 규칙: 카테고리별 가중치와 키워드 기반 특화 체크
    product_rules = {
        "C9": {
            "multipliers": {"financial": 0.7, "credit": 1.0, "behavior": 1.1, "regulation": 1.0},
            "keywords": ["중고차", "자동차", "차량"],
        },
        "C6": {
            "multipliers": {"financial": 1.2, "credit": 1.2, "behavior": 1.0, "regulation": 1.0},
            "keywords": ["신용", "무담보", "일시"],
        },
        "C11": {
            "multipliers": {"financial": 1.1, "credit": 0.9, "behavior": 1.0, "regulation": 0.9},
            "keywords": ["사업", "자영업", "법인"],
        },
        "C12": {
            "multipliers": {"financial": 1.0, "credit": 1.0, "behavior": 1.1, "regulation": 1.2},
            "keywords": ["대환", "재대출", "재융자"],
        },
    }

    # 기본 가중치
    category_multipliers = {"financial": 1.0, "credit": 1.0, "behavior": 1.0, "regulation": 1.0}
    if product and product in product_rules:
        category_multipliers.update(product_rules[product].get("multipliers", {}))

    def apply_multiplier(category: str, base_score: float) -> float:
        return base_score * category_multipliers.get(category, 1.0)

    def add_risk(category: str, score: float, reason: str):
        # 제품 가중치 적용
        adj = apply_multiplier(category, score)
        result["score"] += adj
        result["details"][category] += adj
        result["reasons"].append(reason)

    def find_code_by_keywords(mapping: Dict[str, str], keywords):
        # return first code whose human-readable name contains any keyword
        for code, name in mapping.items():
            lname = str(name)
            for kw in keywords:
                if kw in lname:
                    return code
        return None

    def safe_int(v):
        try:
            return int(float(str(v).strip()))
        except Exception:
            return None

    def safe_float(v):
        try:
            return float(str(v).strip())
        except Exception:
            return None

    # --- identify likely fields via mapping names ---
    amount_code = find_code_by_keywords(in_mapping, ["대출금액", "대출잔액", "신청금액", "한도", "총대출", "한도금액", "대출신청금액"]) or \
                  find_code_by_keywords(in_mapping, ["A2001", "A2035", "A2025"])  # fallback common codes

    income_code = find_code_by_keywords(in_mapping, ["소득", "연봉", "연간소득", "INCOME"]) or find_code_by_keywords(in_mapping, ["A2027", "A2028"]) 

    credit_code = find_code_by_keywords(in_mapping, ["신용등급", "신용점수", "CREDIT", "신용점수"]) or find_code_by_keywords(in_mapping, ["CREDIT_SCORE"]) 

    loancnt_code = find_code_by_keywords(in_mapping, ["대출건수", "다중", "LOAN_CNT"]) or find_code_by_keywords(in_mapping, ["LOAN_CNT"]) 

    job_code = find_code_by_keywords(in_mapping, ["직업", "JOB", "직업유형"]) 

    overdue_code = find_code_by_keywords(out_mapping, ["연체", "연체여부", "OVERDUE"]) 

    dsr_code = find_code_by_keywords(out_mapping, ["DSR", "DSR가이드", "R0003"]) or find_code_by_keywords(out_mapping, ["R0003", "R0020"]) 

    # --- extract values ---
    amount = safe_int(in_fields.get(amount_code)) if amount_code else None
    income = safe_int(in_fields.get(income_code)) if income_code else None
    credit = safe_int(in_fields.get(credit_code)) if credit_code else None
    loan_cnt = safe_int(in_fields.get(loancnt_code)) if loancnt_code else None
    job = in_fields.get(job_code) if job_code else None
    overdue_val = out_fields.get(overdue_code) if overdue_code else None
    dsr = safe_float(out_fields.get(dsr_code)) if dsr_code else None

    # --- scoring rules (product-agnostic heuristics) ---
    # 1) Amount-related risk
    if amount:
        if amount > 100_000_000:
            add_risk("financial", 50, "초고액 대출")
        elif amount > 50_000_000:
            add_risk("financial", 30, "고액 대출")
        elif amount > 30_000_000:
            add_risk("financial", 15, "중금액 대출")

    # 2) DSR / 규제
    if dsr is not None:
        if dsr > 70:
            add_risk("regulation", 60, "DSR 심각 초과")
        elif dsr > 40:
            add_risk("regulation", 40, "DSR 초과")
        elif dsr > 30:
            add_risk("regulation", 15, "DSR 높음")

    # 3) Credit score / grade
    if credit is not None:
        if credit < 500:
            add_risk("credit", 50, "저신용자")
        elif credit < 700:
            add_risk("credit", 30, "중신용자")

    # 4) Overdue / 연체
    if overdue_val is not None:
        sval = str(overdue_val).strip()
        if sval in ("Y", "y", "1") or "연체" in sval or "미납" in sval:
            add_risk("behavior", 50, "연체 이력 존재")

    # 5) Job stability
    if job:
        sjob = str(job)
        if any(x in sjob for x in ["무직", "프리랜서", "자영업"]):
            add_risk("financial", 25, "소득 불안정(직업) ")

    # 6) Income vs amount ratio
    if income and amount:
        try:
            ratio = amount / max(1, income)
            if ratio > 10:
                add_risk("financial", 40, "소득 대비 과도한 대출")
            elif ratio > 5:
                add_risk("financial", 20, "대출 비율 높음")
        except Exception:
            pass

    # 7) Multiple loans
    if loan_cnt is not None:
        if loan_cnt >= 5:
            add_risk("credit", 40, "다중 채무자")
        elif loan_cnt >= 3:
            add_risk("credit", 20, "다중 대출 보유")

    # 8) Out-fields semantic checks (거절/부결/초과 등)
    for code, val in out_fields.items():
        try:
            name = out_mapping.get(code, "")
            sval = str(val)
            if any(k in sval for k in ["거절", "부결", "거부"]):
                add_risk("regulation", 50, f"심사결과 거절: {name}")
            if any(k in name for k in ["초과", "한도초과", "제한"]):
                add_risk("regulation", 20, f"규제/한도 초과 항목: {name}")
        except Exception:
            continue

    # --- 제품별 특화 추가 규칙 ---
    # 매핑명에 제품 키워드가 포함되는지 탐지하여 추가 규칙 적용
    mapping_text = " ".join([str(v) for v in list(in_mapping.values()) + list(out_mapping.values())])
    prod_spec = product_rules.get(product) if product else None
    # 차량대출 특화 (C9 또는 매핑에 차량 관련 키워드가 있으면 적용)
    vehicle_kw = ["중고차", "자동차", "차량"]
    is_vehicle = False
    if product == "C9":
        is_vehicle = True
    else:
        if any(kw in mapping_text for kw in vehicle_kw):
            is_vehicle = True

    if is_vehicle:
        # 차량 대출의 경우 소득 정보가 부족하면 위험 가중
        if not income:
            add_risk("financial", 20, "차량대출인데 소득정보 부족")

    # C6 (무담보/신용성향 상품) 특화
    if product == "C6":
        if loan_cnt is not None and loan_cnt >= 3:
            add_risk("credit", 15, "C6: 다중대출(제품 특화)")

    # C11 (사업자/법인 관련상품) 특화
    if product == "C11":
        sjob = str(job) if job else ""
        if any(x in sjob for x in ["사업", "자영업", "법인"]):
            if not income:
                add_risk("behavior", 20, "C11: 사업자 소득 정보 부족(제품 특화)")

    # C12 (대환/재대출) 특화
    if product == "C12":
        # out_fields/매핑에 '대환' 키워드가 있으면 규제 가중
        if any(kw in mapping_text for kw in ["대환", "재대출", "재융자"]):
            add_risk("regulation", 25, "C12: 대환 관련 항목 존재")


    # Final grade
    result["grade"] = _score_to_grade(result["score"])
    return result


def _score_to_grade(score: float) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"
