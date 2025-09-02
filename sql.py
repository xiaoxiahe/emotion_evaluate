import sqlite3, json
import pandas as pd

conn = sqlite3.connect("logs.db")
sql = """
SELECT id, timestamp, user_mood, user_note, override_text, result_json, image_path, audio_path
FROM logs
ORDER BY id DESC
LIMIT ?;
"""
df = pd.read_sql_query(sql, conn, params=[200])
conn.close()

# 展开 result_json 为列
if not df.empty:
    res = df["result_json"].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
    res_df = pd.json_normalize(res)
    df = pd.concat([df.drop(columns=["result_json"]), res_df], axis=1)

print(df.head())