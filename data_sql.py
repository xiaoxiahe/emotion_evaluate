import os
import json
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime

class EmotionDatabase:
    """情绪识别数据库操作类"""
    
    def __init__(self, db_path: str = "logs.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库表结构"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    user_mood TEXT,
                    user_note TEXT,
                    override_text TEXT,
                    result_json TEXT,
                    image_path TEXT,
                    audio_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引以提高查询性能
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_mood ON logs(user_mood)")
            
            conn.commit()
            conn.close()
            print(f"✅ 数据库 {self.db_path} 初始化成功")
            
        except Exception as e:
            print(f"❌ 数据库初始化失败: {e}")
    
    def insert_record(self, record: Dict[str, Any]) -> bool:
        """插入新记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO logs (timestamp, user_mood, user_note, override_text, result_json, image_path, audio_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get('timestamp'),
                record.get('user_mood'),
                record.get('user_note'),
                record.get('override_text'),
                json.dumps(record.get('result'), ensure_ascii=False),
                record.get('image_path'),
                record.get('audio_path')
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ 插入记录失败: {e}")
            return False
    
    def query_records(self, 
                     limit: int = 100, 
                     user_mood: Optional[str] = None,
                     fused_pred: Optional[str] = None,
                     start_time: Optional[int] = None,
                     end_time: Optional[int] = None) -> Optional[pd.DataFrame]:
        """查询记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 构建查询条件
            conditions = []
            params = []
            
            if user_mood:
                conditions.append("user_mood = ?")
                params.append(user_mood)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            # 基础查询
            base_sql = "SELECT * FROM logs"
            if conditions:
                base_sql += " WHERE " + " AND ".join(conditions)
            
            base_sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # 执行查询
            df = pd.read_sql_query(base_sql, conn, params=params)
            conn.close()
            
            if df.empty:
                return None
            
            # 展开 result_json 字段
            if 'result_json' in df.columns:
                try:
                    result_data = df['result_json'].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and x else {}
                    )
                    result_df = pd.json_normalize(result_data)
                    df = pd.concat([df.drop(columns=['result_json']), result_df], axis=1)
                except Exception as e:
                    print(f"⚠️ 解析 result_json 失败: {e}")
            
            # 按融合结果过滤（如果指定了）
            if fused_pred and 'fused_pred' in df.columns:
                df = df[df['fused_pred'] == fused_pred]
            
            # 添加时间格式化列
            if 'timestamp' in df.columns:
                df['datetime'] = df['timestamp'].apply(
                    lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') if x else None
                )
            
            return df
            
        except Exception as e:
            print(f"❌ 查询记录失败: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM logs")
            total_records = cursor.fetchone()[0]
            
            # 按心情分组统计
            cursor.execute("""
                SELECT user_mood, COUNT(*) as count 
                FROM logs 
                WHERE user_mood IS NOT NULL 
                GROUP BY user_mood
            """)
            mood_stats = dict(cursor.fetchall())
            
            # 按融合结果分组统计
            cursor.execute("""
                SELECT json_extract(result_json, '$.fused_pred') as fused_pred, COUNT(*) as count
                FROM logs 
                WHERE result_json IS NOT NULL
                GROUP BY json_extract(result_json, '$.fused_pred')
            """)
            fused_stats = dict(cursor.fetchall())
            
            # 时间范围
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM logs")
            time_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_records': total_records,
                'mood_distribution': mood_stats,
                'fused_prediction_distribution': fused_stats,
                'time_range': {
                    'start': datetime.fromtimestamp(time_range[0]).strftime('%Y-%m-%d %H:%M:%S') if time_range[0] else None,
                    'end': datetime.fromtimestamp(time_range[1]).strftime('%Y-%m-%d %H:%M:%S') if time_range[1] else None
                }
            }
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def export_to_csv(self, filename: str = "emotion_records.csv", **query_kwargs) -> bool:
        """导出记录到CSV文件"""
        try:
            df = self.query_records(**query_kwargs)
            if df is not None and not df.empty:
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"✅ 数据已导出到 {filename}")
                return True
            else:
                print("⚠️ 没有数据可导出")
                return False
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 创建数据库实例
    db = EmotionDatabase()
    
    # 获取统计信息
    print("\n📊 数据库统计信息:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 查询最近10条记录
    print("\n📋 最近10条记录:")
    df = db.query_records(limit=10)
    if df is not None:
        print(df.head())
        
        # 导出到CSV
        db.export_to_csv("recent_records.csv", limit=10)
    else:
        print("暂无数据")
    
    # 按心情查询
    print("\n😊 查询HAPPY心情的记录:")
    happy_df = db.query_records(limit=20, user_mood="HAPPY")
    if happy_df is not None:
        print(f"找到 {len(happy_df)} 条HAPPY记录")
        print(happy_df[['datetime', 'user_mood', 'fused_pred']].head())
    else:
        print("未找到HAPPY记录")