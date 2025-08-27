import pandas as pd
import os
import time
try:
    from volcenginesdkarkruntime import Ark  # type: ignore
except Exception:
    Ark = None  # 在云端或未安装时允许脚本被安全导入
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

# 设置豆包API Key
os.environ["ARK_API_KEY"] = "be62df6f-1828-47a4-84f1-2932c111bc64"

# 初始化豆包客户端（若 Ark SDK 不可用则友好退出）
if Ark is None:
    raise SystemExit("volcenginesdkarkruntime 未安装，跳过 emotion_based_text 脚本运行")
client = Ark(api_key=os.environ.get("ARK_API_KEY"), timeout=1800)

# 标签映射
label_map = {
    "中性": 0,
    "伤心": 1,
    "生气": 2,
    "高兴": 3
}

# 英文结果→中文标签（用于与数据集标签对齐）
en_to_cn = {
    "NEUTRAL": "中性",
    "SAD": "伤心",
    "ANGRY": "生气",
    "HAPPY": "高兴",
}

# 读取数据
df = pd.read_excel("D:/shixi/fulai/emotion_data/音频/对话音频/elderly_emotion_dataset.xlsx", usecols=["text", "label"])
df = df.dropna(subset=["label"])
texts = df["text"].tolist()
true_labels = df["label"].tolist()  # 真实标签（英文）
pred_labels = []

# 输出文件准备
ts = time.strftime("%Y%m%d_%H%M%S")
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
results = []
response_times = []

# 中/英互转辅助
cn_to_en = {"中性": "NEUTRAL", "伤心": "SAD", "生气": "ANGRY", "高兴": "HAPPY"}

print("=== 开始处理文本情感分类 ===")
print(f"总文本数量: {len(texts)}")
print("=" * 50)

for i, text in enumerate(texts, 1):
    t0 = time.perf_counter()
    
    print(f"\n处理第 {i}/{len(texts)} 条文本:")
    print(f"文本内容: {text}")
    print(f"真实标签: {true_labels[i-1]}")
    
    response = client.chat.completions.create(
        model="doubao-seed-1.6-250615",
        messages=[
            {
                "role": "system",
                "content": """你是一个严格的文本情感分类器。必须只输出一个严格的 JSON 对象，不要输出多余文本、解释或反引号。
                ## 你的任务
                根据用户输入的文本，判断其表达的情感。
                ## 输出要求
                1. 输出必须为严格 JSON，且仅包含以下字段：
                {"emotion": string, "reason": string}
                2. `emotion` 从 ["NEUTRAL", "SAD", "ANGRY", "HAPPY"] 中选择。
                3. `reason` 字段提供简要判断依据（20字内）。
                4. 若难以判断，统一判为"NEUTRAL"。
                """
            },
            {
                "role": "user",
                "content": f"待分类文本：「{text}」"
            }
        ],
        thinking={"type": "disabled"},
        response_format={"type": "json_object"}
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)  # 毫秒
    response_times.append(elapsed_ms / 1000.0)

    # 解析模型返回的JSON（兼容 SDK 的 parsed 字段或 message.content）
    msg = response.choices[0].message
    try:
        obj = getattr(msg, "parsed", None) or json.loads(getattr(msg, "content", msg))
    except Exception as e:
        print(f"JSON 解析失败: {e}; 原始内容: {getattr(msg, 'content', None)!r}")
        obj = {"emotion": "NEUTRAL", "reason": "无法解析"}

    # 兼容字段名：优先 emotion，其次 result
    emotion_field = obj.get("emotion") if "emotion" in obj else obj.get("result", "NEUTRAL")
    em_str = str(emotion_field).strip().upper()  # 直接转为大写英文

    # 直接使用英文标签，不需要转换
    pred_labels.append(em_str)

    # 输出预测结果
    print(f"预测标签: {em_str}")
    print(f"判断依据: {obj.get('reason', '无')}")
    print(f"响应时间: {elapsed_ms}ms")
    
    # 添加详细的调试信息
    print(f"模型原始返回: {obj}")
    print(f"提取的emotion字段: {emotion_field}")
    print(f"最终英文标签: {em_str}")
    
    # 判断预测是否正确（忽略大小写）
    is_correct = "✓" if em_str.upper() == true_labels[i-1].upper() else "✗"
    print(f"预测结果: {is_correct}")
    if is_correct == "✗":
        print(f"❌ 预测错误原因: 预测标签 '{em_str}' 与真实标签 '{true_labels[i-1]}' 不匹配")
        print(f"   注意: 已忽略大小写差异，如果单词相同但大小写不同，请检查标签标准化")
    print("-" * 40)

    # 记录结果（不保存到文件）
    record = {
        "text": text, 
        "true_label": true_labels[i-1],
        "predicted_label": em_str,
        "prediction": obj, 
        "response_time_s": elapsed_ms / 1000.0,
        "is_correct": em_str.upper() == true_labels[i-1].upper()
    }
    results.append(record)

# 计算准确率和召回率（使用大小写不敏感的比较）
# 为了计算指标，需要将真实标签也标准化为大写
true_labels_normalized = [label.upper() for label in true_labels]
pred_labels_normalized = [label.upper() for label in pred_labels]

accuracy = accuracy_score(true_labels_normalized, pred_labels_normalized)
recall = recall_score(true_labels_normalized, pred_labels_normalized, average='macro', labels=["NEUTRAL", "SAD", "ANGRY", "HAPPY"])

# 计算响应时长分位数
response_times_np = np.array(response_times)
percentile_50 = np.percentile(response_times_np, 50)  # 50分位（中位数）
percentile_95 = np.percentile(response_times_np, 95)  # 95分位

# 输出评估指标
avg_time = np.mean(response_times)  # 平均响应时长

print("\n" + "=" * 50)
print("=== 评估指标汇总 ===")
print(f"准确率: {accuracy:.4f}")
print(f"召回率: {recall:.4f}")
print(f"每条文本的平均响应时长: {avg_time:.3f}秒")
print(f"响应时长50分位: {percentile_50:.3f}秒")
print(f"响应时长95分位: {percentile_95:.3f}秒")

# 统计各类别的预测情况（使用大小写不敏感的比较）
print("\n=== 各类别预测统计 ===")
unique_labels = list(set(true_labels_normalized))
for label in unique_labels:
    true_count = true_labels_normalized.count(label)
    correct_count = sum(1 for t, p in zip(true_labels_normalized, pred_labels_normalized) if t == label and p == label)
    accuracy_per_class = correct_count / true_count if true_count > 0 else 0
    print(f"{label}: 总数={true_count}, 正确预测={correct_count}, 准确率={accuracy_per_class:.4f}")

# ===== 详细分类统计 =====
print("\n=== 详细分类统计 ===")

# 统计各种错误类型
neutral_misclassified_files = []      # 实际为neutral但被识别为有情绪的文件
emotion_misclassified_neutral_files = []  # 实际为有情绪但被识别为neutral的文件
emotion_misclassified_other_files = []    # 实际有情绪但被识别为别的情绪的文件

for i, (truth, pred) in enumerate(zip(true_labels_normalized, pred_labels_normalized)):
    if not truth:  # 跳过没有真实标签的文件
        continue
        
    if truth == "NEUTRAL" and pred != "NEUTRAL":
        # 实际为neutral但被识别为有情绪
        neutral_misclassified_files.append(f"文本{i} (误识别为: {pred})")
    elif truth != "NEUTRAL" and pred == "NEUTRAL":
        # 实际为有情绪但被识别为neutral
        emotion_misclassified_neutral_files.append(f"文本{i}")
    elif truth != "NEUTRAL" and pred != "NEUTRAL" and truth != pred:
        # 实际有情绪但被识别为别的情绪
        emotion_misclassified_other_files.append(f"文本{i} (真实: {truth}, 预测: {pred})")

# 计算各种错误比例
total_texts = len([t for t in true_labels_normalized if t])
neutral_texts = len([t for t in true_labels_normalized if t == "NEUTRAL"])
emotion_texts = len([t for t in true_labels_normalized if t and t != "NEUTRAL"])

if neutral_texts > 0:
    neutral_misclassified_ratio = len(neutral_misclassified_files) / neutral_texts
    print(f"实际为neutral但被识别为有情绪的比例: {neutral_misclassified_ratio:.2%}")
    print("实际为neutral但被识别为有情绪的文本及误识别结果:")
    for text_info in neutral_misclassified_files:
        print(f"  - {text_info}")
else:
    print("实际为neutral但被识别为有情绪的比例: 0.00% (无neutral标签文本)")

if emotion_texts > 0:
    emotion_misclassified_neutral_ratio = len(emotion_misclassified_neutral_files) / emotion_texts
    print(f"\n实际为有情绪但被识别为neutral的比例: {emotion_misclassified_neutral_ratio:.2%}")
    print("实际为有情绪但被识别为neutral的文本:")
    for text_info in emotion_misclassified_neutral_files:
        print(f"  - {text_info}")
else:
    print("实际为有情绪但被识别为neutral的比例: 0.00% (无情绪标签文本)")

if emotion_texts > 0:
    emotion_misclassified_other_ratio = len(emotion_misclassified_other_files) / emotion_texts
    print(f"\n实际有情绪但被识别为别的情绪（neutral除外）的比例: {emotion_misclassified_other_ratio:.2%}")
    print("实际有情绪但被识别为别的情绪的文本:")
    for text_info in emotion_misclassified_other_files:
        print(f"  - {text_info}")
else:
    print("实际有情绪但被识别为别的情绪（neutral除外）的比例: 0.00% (无情绪标签文本)")

# 统计各类别的具体错误情况
print("\n=== 各类别详细错误统计 ===")
for label in unique_labels:
    if label == "NEUTRAL":
        continue
        
    label_texts = [i for i, t in enumerate(true_labels_normalized, 1) if t == label]
    label_error_texts = []
    
    for i, (truth, pred) in enumerate(zip(true_labels_normalized, pred_labels_normalized)):
        if truth == label and pred != label:
            label_error_texts.append(f"文本{i} (误识别为: {pred})")
    
    if label_texts:
        error_ratio = len(label_error_texts) / len(label_texts)
        print(f"{label} 类别:")
        print(f"  总数: {len(label_texts)}, 错误数: {len(label_error_texts)}, 错误率: {error_ratio:.2%}")
        if label_error_texts:
            print("  错误文本详情:")
            for text_info in label_error_texts:
                print(f"    - {text_info}")
    else:
        print(f"{label} 类别: 无此类文本")

print("\n" + "=" * 50)
print("处理完成！")
