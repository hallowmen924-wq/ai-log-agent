import re


def parse_log(log: str) -> dict:
    result = {}

    # 1. 시간
    time_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", log)
    result["time"] = time_match.group() if time_match else ""

    # 2. API 이름
    api_match = re.search(r"Online_[A-Z0-9_]+", log)
    result["api"] = api_match.group() if api_match else ""

    # 3. 처리시간
    time_taken = re.search(r"process time\[WAS\]: ([0-9.]+)", log)
    result["process_time"] = time_taken.group(1) if time_taken else ""

    # 4. 주요 코드 일부 추출 (앞부분만)
    codes = re.findall(r"[AR]\d{4}", log)
    result["codes"] = list(set(codes[:20]))  # 너무 많아서 제한

    return result
