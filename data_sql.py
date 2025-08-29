import os, json, pandas as pd

# 使用相对路径，兼容不同环境
BASE_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(BASE_DIR, "logs.ndjson")

if not os.path.exists(LOG_FILE):
    # 首次运行自动创建空文件并提示
    with open(LOG_FILE, "a", encoding="utf-8"):
        pass
    print(f"已创建空日志文件: {LOG_FILE}，当前无数据可读。")
    raise SystemExit(0)

rows = []
with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

if not rows:
    print("日志为空。")
    raise SystemExit(0)

df = pd.json_normalize(rows)
print(df.head())