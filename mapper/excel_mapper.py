import pandas as pd

mapping_cache = {}


def load_excel_mapping(path, sheet_name):

    key = f"{path}_{sheet_name}"

    if key not in mapping_cache:
        print(f"🔥 엑셀 최초 로드: {sheet_name}")
        mapping_cache[key] = pd.read_excel(path, sheet_name=sheet_name)

    return mapping_cache[key]


def get_excel_sheet(product, io_type):

    mapping = {
        "C9": {"in": "RCLIPS송신(이지론)_최종", "out": "RCLIPS수신(이지론)_최종"},
        "C6": {
            "in": "RCLIPS송신(일사천리론)_최종",
            "out": "RCLIPS수신(일사천리론)_최종",
        },
        "C11": {"in": "RCLIPS송신(개사)", "out": "RCLIPS수신(개사)"},
        "C12": {"in": "RCLIPS송신(대환)", "out": "RCLIPS수신(대환)"},
    }

    return mapping.get(product, {}).get(io_type, "UNKNOWN")


def load_excel_mapping(file_path, sheet_name):

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 🔥 컬럼명 맞춰주기
        df.columns = df.columns.str.strip()

        return dict(
            zip(
                df["RCLIPS코드"].astype(str).str.strip(),
                df["항목명"].astype(str).str.strip(),
            )
        )

    except Exception as e:
        print("엑셀 로딩 오류:", e)
        return {}
