import os

def load_logs():
    log_dir = "C:/work/data/logs"
    logs = []

    print("📂 파일 리스트:", os.listdir(log_dir))

    for file in os.listdir(log_dir):
        print("👉 발견 파일:", file)

        if file.lower().endswith(".txt"):
            print("✅ 로그 파일 인식됨:", file)

            with open(os.path.join(log_dir, file), "r", encoding="utf-8") as f:
                content = f.read()
                print("📄 내용 길이:", len(content))

                if content.strip():
                    logs.append(content)

    print("🔥 최종 로그 개수:", len(logs))

    return logs