import os
import json
import time
import pandas as pd
from typing import Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

# 本次更新：
# - 去除对远端 audio_url 的依赖，支持从本地音频路径转写为文本（ASR）后做文本情绪识别
# - 仅融合 视觉 与 文本 两模态；文本权重大，但在文本为 NEUTRAL 且视觉给出非 NEUTRAL 时采用视觉结果，避免“全靠文本”

# 本脚本执行端到端评估：
# - 读取 multimodal_emotion_data.xlsx
# - 分别调用视觉、语音、文本模型获取预测
# - 按规则融合得到最终预测
# - 与 Final_Label 对比，输出统计

EXCEL_PATH = r"D:\shixi\fulai\emotion_code\multimodal_emotion_data.xlsx"

# 标签标准化：统一为大写英文
VALID_LABELS = {"ANGRY", "SAD", "NEUTRAL", "HAPPY"}
CN2EN = {"生气": "ANGRY", "伤心": "SAD", "中性": "NEUTRAL", "高兴": "HAPPY"}
EN_NORMALIZE = {"angry": "ANGRY", "sad": "SAD", "neutral": "NEUTRAL", "happy": "HAPPY"}


def normalize_label(label: str) -> str:
    if label is None:
        return "NEUTRAL"
    s = str(label).strip()
    if not s:
        return "NEUTRAL"
    # 兼容中文
    if s in CN2EN:
        return CN2EN[s]
    # 兼容大小写
    low = s.lower()
    if low in EN_NORMALIZE:
        return EN_NORMALIZE[low]
    up = s.upper()
    return up if up in VALID_LABELS else "NEUTRAL"


def predict_visual(image_path: str) -> str:
    # 改为调用 emotion_visual_integrated 的单图处理逻辑
    try:
        from emotion_visual_integrated import process_single_image
        # 使用与集成评估相同的默认设置
        result = process_single_image(
            image_path=image_path,
            model_name='doubao_1_6_flash',
            prompt="""你是一个专业的人类情绪判断专家。请严格按以下规则判断此刻用户表情中传达的情感：
## 要求
1. 根据上传的图片，判断用户此刻的情绪，图片可能模糊，请你尽可能地分析用户表情中表达的情感；
2. 输出必须为严格JSON格式，且仅包含：{"result": string, "reason": string}
## 判断标准
1. 从 [ANGRY, HAPPY, SAD, NEUTRAL] 中选择；
2. 无法判断一律 NEUTRAL；
## 注意
仅输出 JSON，不要额外解释。
""",
            index=1,
            total=1,
        )
        pred = result.get('predicted_label', 'NEUTRAL')
        return normalize_label(pred)
    except Exception:
        return "NEUTRAL"


def predict_visual_detail(image_path: str) -> Tuple[str, str, float]:
    """返回 (label, reason, duration_s)。失败时返回 (NEUTRAL, '', 0.0)。"""
    try:
        import base64
        from volcenginesdkarkruntime import Ark

        # 与 emotion_based_visual 中一致的配置
        model_name = 'doubao_1_6_flash'
        model_endpoint = 'doubao-seed-1-6-flash-250715'

        # 读取图片并编码
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        # 创建客户端（这里直接从环境变量读取 key，避免硬编码）
        api_key = os.environ.get("ARK_API_KEY") or "be62df6f-1828-47a4-84f1-2932c111bc64"
        client = Ark(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=1800)

        prompt = """你是一个专业的人类情绪判断专家。请严格按以下规则判断此刻用户表情中传达的情感：
## 要求
1. 根据上传的图片，判断用户此刻的情绪，图片可能模糊，请你尽可能地分析用户表情中表达的情感；
2. 输出必须为严格JSON格式，且仅包含：{"result": string, "reason": string}
## 判断标准
1. 从 [ANGRY, HAPPY, SAD, NEUTRAL] 中选择；
2. 无法判断一律 NEUTRAL；
## 注意
仅输出 JSON，不要额外解释。"""

        t0 = time.time()
        response = client.chat.completions.create(
            model=model_endpoint,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            temperature=0.1,
            top_p=0.7,
            max_tokens=4096,
            thinking={"type": "disabled"},
        )
        duration = time.time() - t0

        content = response.choices[0].message.content
        try:
            obj = json.loads(content)
        except Exception:
            obj = {"result": "NEUTRAL", "reason": ""}
        label = normalize_label(obj.get("result", "NEUTRAL"))
        reason = str(obj.get("reason", "") or "")
        return label, reason, duration
    except Exception:
        return "NEUTRAL", "", 0.0


# ========= 本地 ASR → 文本 =========
# 使用优化的快速语音转文本模块
try:
    from stt_integration import transcribe_single_audio
    print("使用快速语音转文本模块")
except ImportError:
    print("快速模块导入失败，使用备用方案")
    # 备用方案
    _WHISPER_MODEL = None
    
    def _get_whisper_model():
        global _WHISPER_MODEL
        if _WHISPER_MODEL is None:
            try:
                from faster_whisper import WhisperModel
                model_name = "tiny"  # 最快，39MB
                device = "cpu"
                compute_type = "int8"
                
                print(f"加载快速模型: {model_name}")
                _WHISPER_MODEL = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=False,
                )
            except Exception as e:
                print(f"加载模型失败: {e}")
                _WHISPER_MODEL = False
        return _WHISPER_MODEL
    
    def transcribe_single_audio(audio_path: str) -> Optional[str]:
        if not audio_path or not os.path.exists(audio_path):
            return None
        model = _get_whisper_model()
        if not model:
            return None
        try:
            segments, _info = model.transcribe(
                audio_path,
                beam_size=1,
                language="zh",
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
                word_timestamps=False
            )
            texts = [seg.text for seg in segments]
            merged = " ".join(t.strip() for t in texts if t and t.strip())
            return merged or None
        except Exception as e:
            print(f"转写音频失败 {audio_path}: {e}")
            return None


def transcribe_audio_to_text(audio_path: str) -> Optional[str]:
    """调用快速语音转文本功能"""
    return transcribe_single_audio(audio_path)


def predict_text(text: str) -> str:
    # 复用 emotion_based_text 的客户端与提示，拿到 emotion 字段
    # 为降低耗时，这里使用模块级复用的 Ark 客户端
    from volcenginesdkarkruntime import Ark
    import os as _os

    global _ARK_CLIENT
    try:
        _ARK_CLIENT
    except NameError:
        _ARK_CLIENT = None

    if _ARK_CLIENT is None:
        api_key = _os.environ.get("ARK_API_KEY") or "be62df6f-1828-47a4-84f1-2932c111bc64"
        _ARK_CLIENT = Ark(api_key=api_key, timeout=1800)
    client = _ARK_CLIENT
    try:
        response = client.chat.completions.create(
            model="doubao-seed-1.6-250615",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严格的文本情感分类器。必须仅输出严格 JSON：{\"emotion\": string, \"reason\": string}；emotion ∈ [NEUTRAL,SAD,ANGRY,HAPPY]；不确定判 NEUTRAL。"
                },
                {"role": "user", "content": f"待分类文本：「{text}」"}
            ],
            thinking={"type": "disabled"},
            response_format={"type": "json_object"}
        )
        msg = response.choices[0].message
        try:
            obj = getattr(msg, "parsed", None) or json.loads(getattr(msg, "content", msg))
        except Exception:
            obj = {"emotion": "NEUTRAL"}
        emotion_field = obj.get("emotion") if "emotion" in obj else obj.get("result", "NEUTRAL")
        return normalize_label(emotion_field)
    except Exception:
        return "NEUTRAL"


def predict_text_detail(text: str) -> Tuple[str, str, float]:
    """返回 (label, reason, duration_s)。失败时返回 (NEUTRAL, '', 0.0)。"""
    from volcenginesdkarkruntime import Ark
    import os as _os

    global _ARK_CLIENT
    try:
        _ARK_CLIENT
    except NameError:
        _ARK_CLIENT = None

    if _ARK_CLIENT is None:
        api_key = _os.environ.get("ARK_API_KEY") or "be62df6f-1828-47a4-84f1-2932c111bc64"
        _ARK_CLIENT = Ark(api_key=api_key, timeout=1800)
    client = _ARK_CLIENT
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="doubao-seed-1.6-250615",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严格的文本情感分类器。必须仅输出严格 JSON：{\"emotion\": string, \"reason\": string}；emotion ∈ [NEUTRAL,SAD,ANGRY,HAPPY]；不确定判 NEUTRAL。",
                },
                {"role": "user", "content": f"待分类文本：「{text}」"},
            ],
            thinking={"type": "disabled"},
            response_format={"type": "json_object"},
        )
        duration = time.time() - t0
        msg = response.choices[0].message
        try:
            obj = getattr(msg, "parsed", None) or json.loads(getattr(msg, "content", msg))
        except Exception:
            obj = {"emotion": "NEUTRAL", "reason": ""}
        emotion_field = obj.get("emotion") if "emotion" in obj else obj.get("result", "NEUTRAL")
        label = normalize_label(emotion_field)
        reason = str(obj.get("reason", "") or "")
        return label, reason, duration
    except Exception:
        return "NEUTRAL", "", 0.0


def fuse_visual_and_text(vision_pred: str, text_pred: str) -> str:
    """文本权重更高但不完全依赖文本的融合：
    1) 若一致：返回该标签
    2) 若 text!=NEUTRAL 且 vision==NEUTRAL：返回 text（文本优先）
    3) 若 text==NEUTRAL 且 vision!=NEUTRAL：返回 vision（保留视觉意义）
    4) 若二者均为非 NEUTRAL 且不一致：返回 text（文本权重大于视觉）
    """
    v, t = vision_pred, text_pred
    if v == t:
        return t
    if t != "NEUTRAL" and v == "NEUTRAL":
        return t
    if t == "NEUTRAL" and v != "NEUTRAL":
        return v
    return t


def evaluate_row(row: pd.Series) -> Tuple[Dict, Dict]:
    picture_path = str(row.get("Picture", "") or "").strip()
    # 兼容多种本地音频列名
    audio_path = str(
        row.get("Audio_Path")
        or row.get("Audio")
        or row.get("Audio_File")
        or row.get("Audio Local")
        or ""
    ).strip()
    text_content = str(row.get("Text_Content", "") or "").strip()
    final_label_true = normalize_label(row.get("Final_Label", ""))

    # 若无文本但有本地音频，则先做转写并计时
    asr_time = 0.0
    if not text_content and audio_path:
        _t0_asr = time.time()
        transcribed = transcribe_audio_to_text(audio_path)
        asr_time = time.time() - _t0_asr
        if transcribed:
            text_content = transcribed

    def timed(fn, arg, default_label="NEUTRAL"):
        if arg is None:
            return default_label, 0.0
        if isinstance(arg, str) and not arg.strip():
            return default_label, 0.0
        t0 = time.time()
        try:
            pred = fn(arg)
        except Exception:
            pred = default_label
        return pred, time.time() - t0

    # 并行跑 视觉 与 文本 两个分支
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_v = ex.submit(timed, predict_visual, picture_path, "NEUTRAL")
        fut_t = ex.submit(timed, predict_text,   text_content,  "NEUTRAL")
        wait([fut_v, fut_t], return_when=ALL_COMPLETED)
        v_pred, v_time = fut_v.result()
        t_pred, t_time = fut_t.result()

    row_elapsed = max(v_time, t_time, asr_time)

    fused = fuse_visual_and_text(v_pred, t_pred)
    ok = fused == final_label_true

    return (
        {
            "vision_pred": v_pred,
            "text_pred": t_pred,
            "fused_pred": fused,
            "true_label": final_label_true,
            "correct": ok,
            "vision_time_s": v_time,
            "text_time_s": t_time,
            "asr_time_s": asr_time,
            "row_time_s": row_elapsed,
        },
        {
            "Picture": picture_path,
            "Audio_Path": audio_path,
            "Text_Content": text_content,
        }
    )

def evaluate_row(row: pd.Series) -> Tuple[Dict, Dict]:
    picture_path = str(row.get("Picture", "") or "").strip()
    audio_path = str(
        row.get("Audio_Path")
        or row.get("Audio")
        or row.get("Audio_File")
        or row.get("Audio Local")
        or ""
    ).strip()
    text_content = str(row.get("Text_Content", "") or "").strip()
    final_label_true = normalize_label(row.get("Final_Label", ""))

    # 若无文本但有本地音频，则先做转写并计时
    asr_time = 0.0
    if not text_content and audio_path:
        _t0_asr = time.time()
        transcribed = transcribe_audio_to_text(audio_path)
        asr_time = time.time() - _t0_asr
        if transcribed:
            text_content = transcribed

    # 并行执行 视觉 与 文本
    def timed(fn, arg, default_label="NEUTRAL"):
        if arg is None:
            return default_label, 0.0
        if isinstance(arg, str) and not arg.strip():
            return default_label, 0.0
        _t0 = time.time()
        try:
            pred = fn(arg)
        except Exception:
            pred = default_label
        return pred, time.time() - _t0

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_v = ex.submit(timed, predict_visual, picture_path, "NEUTRAL")
        fut_t = ex.submit(timed, predict_text,   text_content,  "NEUTRAL")
        wait([fut_v, fut_t], return_when=ALL_COMPLETED)
        v_pred, v_time = fut_v.result()
        t_pred, t_time = fut_t.result()

    row_elapsed = max(v_time, t_time, asr_time)

    fused = fuse_visual_and_text(v_pred, t_pred)
    ok = fused == final_label_true
    return (
        {
            "vision_pred": v_pred,
            "text_pred": t_pred,
            "fused_pred": fused,
            "true_label": final_label_true,
            "correct": ok,
            "vision_time_s": v_time,
            "text_time_s": t_time,
            "asr_time_s": asr_time,
            "row_time_s": row_elapsed,
        },
        {
            "Picture": picture_path,
            "Audio_Path": audio_path,
            "Text_Content": text_content,
        }
    )


def main():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel 文件不存在: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    # 仅要求 Picture 与 Final_Label；文本与音频（本地）二选一提供即可
    required_cols = {"Picture", "Final_Label"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少必要列: {missing}; 现有列: {list(df.columns)}")

    records = []
    t0 = time.time()
    for idx, row in df.iterrows():
        result, refs = evaluate_row(row)
        data_id = row.get("ID", idx)
        records.append({"ID": data_id, **refs, **result})
        print(
            f"[{idx+1}/{len(df)}] ID={data_id} "
            f"vision={result['vision_pred']}({result['vision_time_s']:.2f}s), "
            f"text={result['text_pred']}({result['text_time_s']:.2f}s) | "
            f"asr={result.get('asr_time_s', 0.0):.2f}s | "
            f"fused={result['fused_pred']} | true={result['true_label']} | ok={result['correct']} | "
            f"row_time={result['row_time_s']:.2f}s"
        )

    elapsed = time.time() - t0
    total = len(records)
    correct = sum(1 for r in records if r["correct"]) if total else 0
    acc = correct / total if total else 0.0

    # 每类召回
    labels = sorted(list(VALID_LABELS))
    per_class = {}
    for lbl in labels:
        tp = sum(1 for r in records if r["true_label"] == lbl and r["fused_pred"] == lbl)
        actual = sum(1 for r in records if r["true_label"] == lbl)
        per_class[lbl] = (tp / actual) if actual > 0 else float("nan")

    # ========== 新增统计 ==========
    # 用时统计
    row_times = [r["row_time_s"] for r in records]
    avg_time = sum(row_times) / len(row_times) if row_times else float("nan")
    p50_time = pd.Series(row_times).quantile(0.5) if row_times else float("nan")
    p95_time = pd.Series(row_times).quantile(0.95) if row_times else float("nan")

    # 误分类统计
    neutral_as_emotion = sum(1 for r in records if r["true_label"]=="NEUTRAL" and r["fused_pred"]!="NEUTRAL")
    emotion_as_neutral = sum(1 for r in records if r["true_label"]!="NEUTRAL" and r["fused_pred"]=="NEUTRAL")
    emotion_as_other   = sum(1 for r in records if r["true_label"]!="NEUTRAL" and r["fused_pred"]!="NEUTRAL" and r["fused_pred"]!=r["true_label"])

    neutral_total = sum(1 for r in records if r["true_label"]=="NEUTRAL")
    emotion_total = sum(1 for r in records if r["true_label"]!="NEUTRAL")

    neutral_as_emotion_ratio = neutral_as_emotion/neutral_total if neutral_total>0 else float("nan")
    emotion_as_neutral_ratio = emotion_as_neutral/emotion_total if emotion_total>0 else float("nan")
    emotion_as_other_ratio   = emotion_as_other/emotion_total if emotion_total>0 else float("nan")

    # 输出汇总
    print("\n=== 多模态端到端评估 ===")
    print(f"样本数: {total}")
    print(f"准确率: {acc:.4f}")
    print("每类召回率:")
    for lbl in labels:
        v = per_class[lbl]
        if isinstance(v, float) and v != v:
            print(f"  {lbl}: nan")
        else:
            print(f"  {lbl}: {v:.4f}")
    print(f"总耗时: {elapsed:.2f}s, 平均每条: {elapsed/total:.2f}s" if total else "无样本")
    print(f"平均单条用时: {avg_time:.3f}s, 50分位: {p50_time:.3f}s, 95分位: {p95_time:.3f}s")
    print("\n=== 误分类分析 ===")
    print(f"实际为 NEUTRAL 但识别为情绪的比例: {neutral_as_emotion_ratio:.4f}")
    print(f"实际为情绪但识别为 NEUTRAL 的比例: {emotion_as_neutral_ratio:.4f}")
    print(f"实际为情绪但识别为其它情绪的比例: {emotion_as_other_ratio:.4f}")


if __name__ == "__main__":
    main()
