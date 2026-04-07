import os

def load_logs():
    log_dir = "data/logs"
    logs = []

    for file in os.listdir(log_dir):
        if file.endswith(".log"):
            with open(os.path.join(log_dir, file), "r", encoding="utf-8") as f:
                logs.append(f.read())

    return logs