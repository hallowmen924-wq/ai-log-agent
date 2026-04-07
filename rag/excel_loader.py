import pandas as pd

def load_excel_knowledge(file_path="data/R-CLIPS code def.xlsx"):
    xls = pd.ExcelFile(file_path)

    all_texts = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        for _, row in df.iterrows():
            try:
                code = str(row.get("RCLIPS코드", ""))
                name = str(row.get("R클립스_항목명", ""))
                desc = str(row.get("원천정보그룹", ""))
                io_type = str(row.get("길이", ""))
                
                # ⭐ 핵심: "맥락 포함"
                text = f"[{sheet_name}] {io_type} 코드 {code}는 {name}이며, {desc}"

                if code != "nan":
                    all_texts.append(text)

            except:
                continue

    return all_texts