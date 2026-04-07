import pandas as pd

def load_excel_knowledge(file_path="data/R-CLIPS code def.xlsx"):
    xls = pd.ExcelFile(file_path)

    all_texts = []

    print("시트 목록:", xls.sheet_names)  # ⭐ 확인용

    for sheet_name in xls.sheet_names:
        print(f"현재 시트: {sheet_name}")

        df = pd.read_excel(xls, sheet_name=sheet_name)

        print("컬럼명:", df.columns)  # ⭐ 중요

        for _, row in df.iterrows():
            try:
                code = str(row.get("RCLIPS코드", "")).strip()
                name = str(row.get("R클립스_항목명", "")).strip()
                desc = str(row.get("원천정보그룹", "")).strip()
                io_type = str(row.get("길이", "")).strip()

                if code == "" or code == "nan":
                    continue

                # ⭐ 핵심: 시트명 포함
                text = f"[{sheet_name}] {io_type} 코드 {code}는 {name}이며, {desc}"

                all_texts.append(text)

            except Exception as e:
                print("에러:", e)
                continue

    print("총 엑셀 데이터 개수:", len(all_texts))  # ⭐ 핵심

    return all_texts