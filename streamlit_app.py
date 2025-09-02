import os
import io
import time
import json
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import streamlit as st
import base64
import requests
import sqlite3
import shutil
import uuid
try:
    from streamlit_mic_recorder import mic_recorder  # optional mic widget
except Exception:
    mic_recorder = None

# 必须在任何 Streamlit 调用前设置页面配置，避免 Cloud 上布局/滚动异常
st.set_page_config(page_title="多模态情绪识别演示", layout="wide")

# 将 Cloud Secrets 注入到环境变量，供 REST 客户端读取
try:
    if "ARK_API_KEY" in st.secrets:
        os.environ["ARK_API_KEY"] = st.secrets["ARK_API_KEY"]
except Exception:
    pass

# 复用已有逻辑
from p2p_evaluate import (
    EXCEL_PATH as DEFAULT_EXCEL_PATH,
    evaluate_row,
    predict_visual,
    predict_text,
    predict_visual_detail,
    predict_text_detail,
    fuse_visual_and_text,
    normalize_label,
)

# 预加载火山引擎WebSocket语音转文本模型
@st.cache_resource
def load_streaming_asr_model():
    """预加载火山引擎WebSocket语音转文本模型"""
    try:
        from streaming_asr_demo import AsrWsClient, execute_one
        # st.success("✅ 火山引擎WebSocket语音转文本模型加载成功")
        return True
    except Exception as e:
        st.warning(f"⚠️ 火山引擎WebSocket语音转文本模型加载失败: {e}，将使用备用方案")
        return None

# 火山引擎WebSocket语音转文本函数
def streaming_asr_transcribe_audio(audio_path: str) -> Optional[str]:
    """使用火山引擎WebSocket进行语音转文本"""
    if not audio_path or not os.path.exists(audio_path):
        return None
    
    # 检查火山引擎WebSocket模型是否可用
    if not st.session_state.get('streaming_asr_available'):
        # 备用方案：使用p2p_evaluate中的函数
        from p2p_evaluate import transcribe_audio_to_text
        return transcribe_audio_to_text(audio_path)
    
    try:
        start_time = time.time()
        
        # 使用火山引擎WebSocket API进行语音识别
        from streaming_asr_demo import execute_one
        
        # 配置参数
        kwargs = {
            'appid': "5851744862",
            'token': "HdMaaKvnrzQ4vuLGJ0tP2u_v5Xd97_Ho",
            'cluster': "volcengine_input_common",
            'format': "wav",
            'language': "zh-CN",
            'rate': 16000,
            'bits': 16,
            'channel': 1
        }
        
        # 执行语音识别
        result = execute_one(
            {'id': 'streamlit_audio', 'path': audio_path},
            **kwargs
        )
        
        # 解析响应结果
        recognized_text = ""
        if 'result' in result and isinstance(result['result'], dict):
            result_data = result['result']
            if 'payload_msg' in result_data:
                payload_msg = result_data['payload_msg']
                if 'data' in payload_msg:
                    data = payload_msg['data']
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        if isinstance(first_item, dict) and 'text' in first_item:
                            recognized_text = first_item['text']
                        elif isinstance(first_item, str):
                            recognized_text = first_item
                    elif isinstance(data, str):
                        recognized_text = data
                elif 'text' in payload_msg:
                    recognized_text = payload_msg['text']
                elif 'result' in payload_msg:
                    result_data = payload_msg['result']
                    if isinstance(result_data, list) and len(result_data) > 0:
                        if isinstance(result_data[0], dict) and 'text' in result_data[0]:
                            recognized_text = result_data[0]['text']
                        elif isinstance(result_data[0], str):
                            recognized_text = result_data[0]
                    elif isinstance(result_data, str):
                        recognized_text = result_data
        
        if not recognized_text:
            # 如果无法解析，尝试备用方案
            from p2p_evaluate import transcribe_audio_to_text
            return transcribe_audio_to_text(audio_path)
        
        processing_time = time.time() - start_time
        st.info(f"🎵 火山引擎WebSocket语音转写完成 ({processing_time:.2f}秒)")
        
        return recognized_text if recognized_text.strip() else None
        
    except Exception as e:
        st.warning(f"火山引擎WebSocket语音转写失败: {e}，使用备用方案")
        # 备用方案：使用p2p_evaluate中的函数
        from p2p_evaluate import transcribe_audio_to_text
        return transcribe_audio_to_text(audio_path)


def try_init_ark_sdk() -> None:
    """根据 REST 方案检查 Ark 可用性：只要存在 ARK_API_KEY 即视为可用。"""
    if st.session_state.get('ark_initialized'):
        return
    api_key = os.environ.get("ARK_API_KEY")
    st.session_state['ark_available'] = bool(api_key)
    st.session_state['ark_initialized'] = True


def run_batch_auto_test(excel_path: str) -> Dict[str, Any]:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")
    df = pd.read_excel(excel_path)
    required_cols = {"Picture", "Final_Label"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少必要列: {missing}; 现有列: {list(df.columns)}")

    records = []
    t0 = time.time()

    progress = st.progress(0.0, text="正在评估...")
    for idx, row in df.iterrows():
        result, refs = evaluate_row(row)
        data_id = row.get("ID", idx)
        records.append({"ID": data_id, **refs, **result})
        progress.progress((idx + 1) / len(df), text=f"{idx+1}/{len(df)}")

    elapsed = time.time() - t0
    total = len(records)
    correct = sum(1 for r in records if r["correct"]) if total else 0
    acc = correct / total if total else 0.0

    labels = sorted(["ANGRY", "HAPPY", "NEUTRAL", "SAD"])  # 与 VALID_LABELS 一致次序
    per_class = {}
    for lbl in labels:
        tp = sum(1 for r in records if r["true_label"] == lbl and r["fused_pred"] == lbl)
        actual = sum(1 for r in records if r["true_label"] == lbl)
        per_class[lbl] = (tp / actual) if actual > 0 else float("nan")

    row_times = [r["row_time_s"] for r in records]
    avg_time = sum(row_times) / len(row_times) if row_times else float("nan")
    p50_time = pd.Series(row_times).quantile(0.5) if row_times else float("nan")
    p95_time = pd.Series(row_times).quantile(0.95) if row_times else float("nan")

    return {
        "elapsed": elapsed,
        "total": total,
        "accuracy": acc,
        "per_class": per_class,
        "avg_time": avg_time,
        "p50_time": p50_time,
        "p95_time": p95_time,
        "records": records,
    }


def save_uploaded_file(uploaded_file, suffix: Optional[str] = None) -> Optional[str]:
    if uploaded_file is None:
        return None
    # 依据原文件名保留扩展名，除非显式传入 suffix
    if suffix is None:
        try:
            _, ext = os.path.splitext(getattr(uploaded_file, "name", ""))
            suffix = ext if ext else ""
        except Exception:
            suffix = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def _file_to_base64(path: Optional[str], default_mime: str) -> Optional[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return None
    try:
        _, ext = os.path.splitext(path)
        ext = (ext or "").lower()
        mime = default_mime
        if default_mime.startswith("image/"):
            if ext in [".png"]:
                mime = "image/png"
            elif ext in [".jpg", ".jpeg"]:
                mime = "image/jpeg"
        elif default_mime.startswith("audio/"):
            if ext in [".mp3"]:
                mime = "audio/mpeg"
            elif ext in [".m4a"]:
                mime = "audio/mp4"
            elif ext in [".flac"]:
                mime = "audio/flac"
            elif ext in [".wav"]:
                mime = "audio/wav"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"mime": mime, "data": b64}
    except Exception:
        return None


def upload_data_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    """根据环境变量 LOG_TARGETS 写入多种落盘目标。
    支持：
      - http: 使用 DATA_LOG_ENDPOINT POST
      - file: 写入 logs.ndjson，并将媒体复制到 logs_media/
      - sqlite: 写入 logs.db，并将媒体复制到 logs_media/
    默认：file,sqlite
    """
    targets = (os.environ.get("LOG_TARGETS") or "file,sqlite").split(",")
    targets = [t.strip().lower() for t in targets if t.strip()]
    results = {t: False for t in targets}

    # 复制媒体到本地目录，返回新路径，便于 file/sqlite 存储
    def _save_media_to_dir(tmp_path: Optional[str]) -> Optional[str]:
        if not tmp_path or not os.path.exists(tmp_path):
            return None
        try:
            os.makedirs("logs_media", exist_ok=True)
            _, ext = os.path.splitext(tmp_path)
            filename = f"{int(time.time())}_{uuid.uuid4().hex}{ext or ''}"
            dst = os.path.join("logs_media", filename)
            shutil.copyfile(tmp_path, dst)
            return dst
        except Exception:
            return None

    image_path = payload.get("__tmp_image_path")
    audio_path = payload.get("__tmp_audio_path")
    saved_image = _save_media_to_dir(image_path)
    saved_audio = _save_media_to_dir(audio_path)
    # 若复制失败，回退为内联 base64，确保数据不丢失
    image_b64 = None if saved_image else _file_to_base64(image_path, "image/jpeg")
    audio_b64 = None if saved_audio else _file_to_base64(audio_path, "audio/wav")

    # 构建简化记录（媒体改为本地路径）
    record = dict(payload)
    record.pop("media", None)
    record.pop("__tmp_image_path", None)
    record.pop("__tmp_audio_path", None)
    record["media_paths"] = {"image": saved_image, "audio": saved_audio}
    if image_b64 or audio_b64:
        record["media_b64"] = {"image": image_b64, "audio": audio_b64}

    if "http" in targets:
        endpoint = os.environ.get("DATA_LOG_ENDPOINT")
        if endpoint:
            for _ in range(2):
                try:
                    resp = requests.post(endpoint, json=payload, timeout=20)
                    if 200 <= resp.status_code < 300:
                        results["http"] = True
                        break
                except Exception:
                    time.sleep(0.5)
        else:
            results["http"] = False

    if "file" in targets:
        try:
            # 确保文件存在
            if not os.path.exists("logs.ndjson"):
                with open("logs.ndjson", "a", encoding="utf-8") as _:
                    pass
            with open("logs.ndjson", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            results["file"] = True
        except Exception:
            results["file"] = False

    if "sqlite" in targets:
        try:
            conn = sqlite3.connect("logs.db")
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    user_mood TEXT,
                    user_note TEXT,
                    override_text TEXT,
                    result_json TEXT,
                    image_path TEXT,
                    audio_path TEXT
                )
                """
            )
            cur.execute(
                "INSERT INTO logs (timestamp, user_mood, user_note, override_text, result_json, image_path, audio_path) VALUES (?,?,?,?,?,?,?)",
                (
                    record.get("timestamp"),
                    record.get("user_mood"),
                    record.get("user_note"),
                    record.get("override_text"),
                    json.dumps(record.get("result"), ensure_ascii=False),
                    record["media_paths"].get("image"),
                    record["media_paths"].get("audio"),
                ),
            )
            conn.commit()
            conn.close()
            results["sqlite"] = True
        except Exception:
            results["sqlite"] = False

    # 如果没有指定任何目标，视为失败
    overall = bool(targets) and all(results.get(t, False) for t in targets)
    return {"ok": overall, "results": results, "record": record}


def init_local_logging_storage() -> None:
    """初始化本地日志存储，避免首次读取/写入时报文件不存在。"""
    try:
        # 确保媒体目录存在
        os.makedirs("logs_media", exist_ok=True)
        # 确保 NDJSON 文件存在
        if not os.path.exists("logs.ndjson"):
            with open("logs.ndjson", "a", encoding="utf-8") as _:
                pass
        # 确保 SQLite 表存在
        conn = sqlite3.connect("logs.db")
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                user_mood TEXT,
                user_note TEXT,
                override_text TEXT,
                result_json TEXT,
                image_path TEXT,
                audio_path TEXT
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _load_logs_from_sqlite(limit: int = 200, user_mood: Optional[str] = None, fused_pred: Optional[str] = None):
    if not os.path.exists("logs.db"):
        return None
    try:
        conn = sqlite3.connect("logs.db")
        base_sql = "SELECT id, timestamp, user_mood, user_note, override_text, result_json, image_path, audio_path FROM logs"
        conds = []
        params = []
        if user_mood:
            conds.append("user_mood = ?")
            params.append(user_mood)
        if fused_pred:
            conds.append("json_extract(result_json, '$.fused_pred') = ?")
            params.append(fused_pred)
        if conds:
            base_sql += " WHERE " + " AND ".join(conds)
        base_sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        df = pd.read_sql_query(base_sql, conn, params=params)
        conn.close()
        # 展开 result_json
        if not df.empty:
            res = df["result_json"].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
            res_df = pd.json_normalize(res)
            df = pd.concat([df.drop(columns=["result_json"]), res_df], axis=1)
        return df
    except Exception:
        return None


def _load_logs_from_ndjson(limit: int = 200, user_mood: Optional[str] = None, fused_pred: Optional[str] = None):
    if not os.path.exists("logs.ndjson"):
        return None
    rows = []
    try:
        with open("logs.ndjson", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    continue
        if not rows:
            return None
        rows = rows[-limit:][::-1]
        df = pd.json_normalize(rows)
        # 过滤
        if user_mood:
            df = df[df.get("user_mood").fillna("") == user_mood]
        if fused_pred:
            df = df[df.get("result.fused_pred").fillna("") == fused_pred]
        return df
    except Exception:
        return None


def run_single_test(image_path: Optional[str], audio_path: Optional[str], override_text: str = "") -> Dict[str, Any]:
    # 1) 视觉（使用详细函数，拿到原因与精确耗时）
    v_pred, v_reason, v_time = "NEUTRAL", "", 0.0
    if image_path and os.path.exists(image_path):
        v_pred, v_reason, v_time = predict_visual_detail(image_path)

            # 2) 文本：若 override_text 提供则优先使用，否则尝试音频转写
    asr_time = 0.0
    text_content = override_text.strip()
    if not text_content and audio_path and os.path.exists(audio_path):
        t0 = time.time()
        # 使用火山引擎WebSocket语音转文本功能
        text_from_asr = streaming_asr_transcribe_audio(audio_path)
        asr_time = time.time() - t0
        if text_from_asr:
            text_content = text_from_asr

    t_pred, t_reason, t_time = "NEUTRAL", "", 0.0
    if text_content:
        t_pred, t_reason, t_time = predict_text_detail(text_content)

    fused = fuse_visual_and_text(v_pred, t_pred)
    row_time = max(v_time, t_time, asr_time)
    return {
        "vision_pred": v_pred,
        "vision_reason": v_reason,
        "vision_time_s": v_time,
        "text_pred": t_pred,
        "text_reason": t_reason,
        "text_time_s": t_time,
        "asr_time_s": asr_time,
        "fused_pred": fused,
        "text_content": text_content,
        "row_time_s": row_time,
    }


def main():
    st.title("多模态情绪识别演示")
    
    # 在界面启动时预加载 Ark SDK 与语音转文本模型
    with st.spinner("正在加载依赖与模型..."):
        init_local_logging_storage()
        try_init_ark_sdk()
        streaming_asr_available = load_streaming_asr_model()
        # 将火山引擎WebSocket状态存储在session_state中，供后续使用
        st.session_state['streaming_asr_available'] = streaming_asr_available
    
    st.success("🚀 系统初始化完成！")
    
    # 显示当前系统时间信息
    current_time = datetime.now()
    st.info(f"🕐 当前系统时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 移除 UI 内部设置 ARK_API_KEY 的入口，统一使用环境变量/Secrets

    tab1, tab2, tab3 = st.tabs(["自动测试", "单条测试（上传图片与音频）", "历史记录"]) 

    with tab1:
        st.subheader("自动测试（批量评估）")
        choice = st.radio("选择测试规模", ["简单（50条）", "完整（196条）", "自定义路径"], horizontal=True)
        c1, c2, _ = st.columns([4, 1, 6])
        with c1:
            if choice == "简单（50条）":
                excel_path = os.path.join(".", "multimodal_emotion_data_50.xlsx")
                st.text_input("Excel 路径", value=excel_path, disabled=True)
            elif choice == "完整（196条）":
                excel_path = os.path.join(".", "multimodal_emotion_data_196.xlsx")
                st.text_input("Excel 路径", value=excel_path, disabled=True)
            else:
                excel_path = st.text_input("Excel 路径", value=DEFAULT_EXCEL_PATH)
        with c2:
            run_btn = st.button("开始自动测试", type="primary")
        if run_btn:
            try:
                res = run_batch_auto_test(excel_path)
                st.success(f"样本数: {res['total']} | 准确率: {res['accuracy']:.4f} | 总耗时: {res['elapsed']:.2f}s | 平均单条: {res['avg_time']:.3f}s | 50分位: {res['p50_time']:.3f}s | 95分位: {res['p95_time']:.3f}s")

                st.write("每类召回率：")
                pc_df = pd.DataFrame({"label": list(res["per_class"].keys()), "recall": list(res["per_class"].values())})
                st.dataframe(pc_df, use_container_width=True, height=140)

                out_df = pd.DataFrame(res["records"]).sort_values("ID")
                st.dataframe(out_df, use_container_width=True, height=300)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载结果 CSV", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")
            except Exception as e:
                st.error(f"批量评估失败: {e}")

    with tab2:
        st.subheader("单条测试（上传图片与音频）")
        
        # 显示模型状态
        model_status = "✅ 火山引擎WebSocket模型已加载" if st.session_state.get('streaming_asr_available') else "⚠️ 使用备用模型"
        ark_status = "✅ Ark 可用" if st.session_state.get('ark_available') else "⚠️ Ark 不可用（使用回退）"
        st.info(f"语音转文本模型状态: {model_status} | 大模型: {ark_status}")
        
        # 仅文本模式可在弱网/移动端时跳过视觉模型
        only_text = st.checkbox("仅文本模式（跳过视觉识别）", value=False)

        col_left, col_right = st.columns([1,1])
        with col_left:
            img_file = st.file_uploader("📷 上传图片", type=["jpg", "jpeg", "png"])
            if img_file is not None:
                st.image(img_file, caption="上传的图片", width=280)
        with col_right:
            # 将“上传音频”和“或者录音”并排放置
            a1, a2 = st.columns([1,1])
            with a1:
                wav_file = st.file_uploader("🎵 上传音频", type=["wav", "mp3", "m4a", "flac"]) 
            # 可选：使用麦克风直接录音（与上传音频并排）
            rec_tmp_path = None
            with a2:
                if mic_recorder is not None:
                    st.caption("或者录音")
                    rec = mic_recorder(start_prompt="开始录音", stop_prompt="停止录音", format="wav", key="mic_recorder")
                    if rec and isinstance(rec, dict) and rec.get("bytes"):
                        st.audio(rec["bytes"])
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmprec:
                                tmprec.write(rec["bytes"])
                                rec_tmp_path = tmprec.name
                        except Exception:
                            rec_tmp_path = None

            override_text = st.text_area("📝 可选：直接输入文本（将跳过语音转写）", 
                                       placeholder="在这里输入文本内容...", 
                                       height=120)

        # rec_tmp_path 已在右侧列内设置（若使用录音）

        b1, b2, _ = st.columns([1,1,6])
        with b1:
            run_single_btn = st.button("🚀 开始单条测试", type="primary")
        if run_single_btn:
            with st.spinner("处理中..."):
                # 保留上传图片原始扩展名
                tmp_img = save_uploaded_file(img_file) if img_file else None
                # 录音优先，其次是上传文件
                if rec_tmp_path:
                    tmp_wav = rec_tmp_path
                else:
                    # 保留上传音频原始扩展名，避免仅限 .wav
                    tmp_wav = save_uploaded_file(wav_file) if wav_file else None

                try:
                    # 调试：展示服务器端保存的临时文件路径与大小，定位手机端路径相关问题
                    if tmp_img and os.path.exists(tmp_img):
                        st.caption(f"图片已保存: {tmp_img} ({os.path.getsize(tmp_img)} bytes)")
                    if tmp_wav and os.path.exists(tmp_wav):
                        st.caption(f"音频已保存: {tmp_wav} ({os.path.getsize(tmp_wav)} bytes)")
                    # 若勾选仅文本模式，则不传图片路径
                    img_arg = None if only_text else tmp_img
                    res = run_single_test(img_arg, tmp_wav, override_text=override_text)
                    # 将结果与临时路径保存到会话，供结果页下方二次确认后再入库
                    st.session_state['last_result'] = res
                    st.session_state['last_tmp_image'] = tmp_img
                    st.session_state['last_tmp_audio'] = tmp_wav
                    st.session_state['last_override_text'] = override_text
                finally:
                    # 清理临时文件
                    # 不立即删除，以便用户在结果页选择后保存记录。
                    # 实际清理发生在“保存记录”动作之后。
                    pass

        # 无论是否点击按钮，只要 session_state 有结果，就展示并允许保存
        res = st.session_state.get('last_result')
        if res:
            st.markdown("### 📊 预测结果")
            m1, m2, m3 = st.columns(3)
            m1.metric("👁️ 视觉预测", res["vision_pred"]) 
            m2.metric("📝 文本预测", res["text_pred"]) 
            m3.metric("🎯 融合结果", res["fused_pred"]) 

            st.markdown("### 💡 预测原因")
            r1, r2 = st.columns(2)
            with r1:
                st.info(f"视觉原因：{res.get('vision_reason','') or '无'}")
            with r2:
                st.info(f"文本原因：{res.get('text_reason','') or '无'}")

            st.markdown("### ⏱️ 耗时情况")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("👁️ 视觉用时", f"{res['vision_time_s']:.3f}s")
            t2.metric("🎵 ASR转写", f"{res['asr_time_s']:.3f}s")
            t3.metric("📝 文本用时", f"{res['text_time_s']:.3f}s")
            t4.metric("⚡ 整体(并行)", f"{res['row_time_s']:.3f}s")

            st.markdown("### 🎵 语音转写文本")
            if res.get("text_content"):
                st.success(f"转写结果: {res['text_content']}")
            else:
                st.info("无转写文本")

            # 结果之后再让用户确认当下心情并保存（加持久 key，避免选择后刷新丢失）
            st.markdown("### ✅ 保存本次记录")
            c1, c2 = st.columns([1,1])
            with c1:
                user_mood = st.selectbox("当下心情（自报）", ["", "ANGRY", "HAPPY", "SAD", "NEUTRAL"], index=0, key="user_mood_select")
            with c2:
                user_note = st.text_input("备注（可选）", placeholder="补充说明…", key="user_note_input")
            save_btn = st.button("保存记录")
            if save_btn:
                try:
                    # 获取准确的系统时间
                    current_timestamp = int(time.time())
                    current_datetime = datetime.now()
                    
                    payload = {
                        "timestamp": current_timestamp,
                        "datetime_str": current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        "user_mood": (st.session_state.get("user_mood_select") or None),
                        "user_note": (st.session_state.get("user_note_input") or None),
                        "override_text": st.session_state.get('last_override_text'),
                        "result": {
                            "vision_pred": res.get("vision_pred"),
                            "text_pred": res.get("text_pred"),
                            "fused_pred": res.get("fused_pred"),
                            "vision_time_s": res.get("vision_time_s"),
                            "text_time_s": res.get("text_time_s"),
                            "asr_time_s": res.get("asr_time_s"),
                            "row_time_s": res.get("row_time_s"),
                        },
                        "__tmp_image_path": st.session_state.get('last_tmp_image'),
                        "__tmp_audio_path": st.session_state.get('last_tmp_audio'),
                    }
                    result_info = upload_data_log(payload)
                    # 根据结果判断是否全部目标成功
                    if not result_info.get("ok"):
                        st.warning(f"部分写入失败: {result_info.get('results')}")
                    else:
                        st.success("已保存")
                    # 保存后清理临时文件并清空会话状态
                    for k in ['last_tmp_image', 'last_tmp_audio']:
                        p = st.session_state.get(k)
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    for k in ['last_result','last_tmp_image','last_tmp_audio','last_override_text','user_mood_select','user_note_input']:
                        if k in st.session_state:
                            del st.session_state[k]
                except Exception as e:
                    st.warning(f"保存失败：{e}")

    with tab3:
        st.subheader("历史记录（数据库/文件）")
        src_col1, src_col2, src_col3 = st.columns([1,1,2])
        with src_col1:
            source = st.radio("数据来源", ["sqlite", "ndjson"], horizontal=True)
        with src_col2:
            limit = st.number_input("读取条数", min_value=10, max_value=1000, value=200, step=10)
        with src_col3:
            refresh = st.button("刷新", type="secondary")

        f1, f2 = st.columns([1,1])
        with f1:
            mood_filter = st.selectbox("按自报心情过滤", ["", "ANGRY", "HAPPY", "SAD", "NEUTRAL"], index=0)
        with f2:
            fused_filter = st.selectbox("按融合结果过滤", ["", "ANGRY", "HAPPY", "SAD", "NEUTRAL"], index=0)

        if refresh or True:
            if source == "sqlite":
                df = _load_logs_from_sqlite(limit=int(limit), 
                                            user_mood=(mood_filter or None), 
                                            fused_pred=(fused_filter or None))
            else:
                df = _load_logs_from_ndjson(limit=int(limit), 
                                            user_mood=(mood_filter or None), 
                                            fused_pred=(fused_filter or None))

            if df is None or df.empty:
                st.info("当前无数据或未匹配到记录。")
            else:
                st.dataframe(df, use_container_width=True, height=360)
                csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载当前结果 CSV", data=csv_bytes, file_name="history_records.csv", mime="text/csv")
                
                # 媒体文件下载功能
                st.markdown("### 📁 媒体文件下载")
                
                # 统计可下载的媒体文件
                image_files = []
                audio_files = []
                for _, row in df.iterrows():
                    img_path = row.get("image_path")
                    audio_path = row.get("audio_path")
                    
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        image_files.append({
                            'path': img_path,
                            'id': row.get('id', 'unknown'),
                            'timestamp': row.get('timestamp', 0),
                            'mood': row.get('user_mood', 'unknown')
                        })
                    
                    if isinstance(audio_path, str) and os.path.exists(audio_path):
                        audio_files.append({
                            'path': audio_path,
                            'id': row.get('id', 'unknown'),
                            'timestamp': row.get('timestamp', 0),
                            'mood': row.get('user_mood', 'unknown')
                        })
                
                # 显示媒体文件统计
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📷 可下载图片: {len(image_files)} 个")
                with col2:
                    st.info(f"🎵 可下载音频: {len(audio_files)} 个")
                
                # 批量下载选项
                if image_files or audio_files:
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        if image_files:
                            # 创建图片压缩包
                            import zipfile
                            import io
                            
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for img_info in image_files:
                                    try:
                                        # 生成文件名：ID_心情_时间戳.扩展名
                                        ext = os.path.splitext(img_info['path'])[1]
                                        filename = f"{img_info['id']}_{img_info['mood']}_{img_info['timestamp']}{ext}"
                                        zip_file.write(img_info['path'], filename)
                                    except Exception as e:
                                        st.warning(f"添加图片到压缩包失败: {e}")
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                "📦 下载所有图片 (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="emotion_images.zip",
                                mime="application/zip"
                            )
                        else:
                            st.info("暂无图片可下载")
                    
                    with download_col2:
                        if audio_files:
                            # 创建音频压缩包
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for audio_info in audio_files:
                                    try:
                                        ext = os.path.splitext(audio_info['path'])[1]
                                        filename = f"{audio_info['id']}_{audio_info['mood']}_{audio_info['timestamp']}{ext}"
                                        zip_file.write(audio_info['path'], filename)
                                    except Exception as e:
                                        st.warning(f"添加音频到压缩包失败: {e}")
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                "📦 下载所有音频 (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="emotion_audios.zip",
                                mime="application/zip"
                            )
                        else:
                            st.info("暂无音频可下载")
                

                
                # 媒体文件管理页面
                st.markdown("### 🗂️ 媒体文件管理")
                
                # 显示所有媒体文件的详细信息
                if image_files or audio_files:
                    # 创建媒体文件列表
                    media_df_data = []
                    
                    for img_info in image_files:
                        media_df_data.append({
                            '类型': '图片',
                            'ID': img_info['id'],
                            '心情': img_info['mood'],
                            '时间': datetime.fromtimestamp(img_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S') if img_info['timestamp'] else 'Unknown',
                            '文件名': os.path.basename(img_info['path']),
                            '路径': img_info['path'],
                            '大小(KB)': round(os.path.getsize(img_info['path']) / 1024, 2)
                        })
                    
                    for audio_info in audio_files:
                        media_df_data.append({
                            '类型': '音频',
                            'ID': audio_info['id'],
                            '心情': audio_info['mood'],
                            '时间': datetime.fromtimestamp(audio_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S') if audio_info['timestamp'] else 'Unknown',
                            '文件名': os.path.basename(audio_info['path']),
                            '路径': audio_info['path'],
                            '大小(KB)': round(os.path.getsize(audio_info['path']) / 1024, 2)
                        })
                    
                    if media_df_data:
                        media_df = pd.DataFrame(media_df_data)
                        st.dataframe(media_df, use_container_width=True, height=300)
                        
                        # 按类型筛选下载
                        st.markdown("#### 📥 分类下载")
                        type_col1, type_col2 = st.columns(2)
                        
                        with type_col1:
                            if image_files:
                                # 只下载图片
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for img_info in image_files:
                                        try:
                                            ext = os.path.splitext(img_info['path'])[1]
                                            filename = f"图片_{img_info['id']}_{img_info['mood']}_{img_info['timestamp']}{ext}"
                                            zip_file.write(img_info['path'], filename)
                                        except Exception as e:
                                            st.warning(f"添加图片到压缩包失败: {e}")
                                
                                zip_buffer.seek(0)
                                st.download_button(
                                    "📷 下载所有图片",
                                    data=zip_buffer.getvalue(),
                                    file_name="emotion_images_only.zip",
                                    mime="application/zip"
                                )
                        
                        with type_col2:
                            if audio_files:
                                # 只下载音频
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for audio_info in audio_files:
                                        try:
                                            ext = os.path.splitext(audio_info['path'])[1]
                                            filename = f"音频_{audio_info['id']}_{audio_info['mood']}_{audio_info['timestamp']}{ext}"
                                            zip_file.write(audio_info['path'], filename)
                                        except Exception as e:
                                            st.warning(f"添加音频到压缩包失败: {e}")
                                
                                zip_buffer.seek(0)
                                st.download_button(
                                    "🎵 下载所有音频",
                                    data=zip_buffer.getvalue(),
                                    file_name="emotion_audios_only.zip",
                                    mime="application/zip"
                                )
                        
                        # 按心情筛选下载
                        st.markdown("#### 😊 按心情下载")
                        mood_options = list(set([item['mood'] for item in image_files + audio_files if item['mood'] != 'unknown']))
                        if mood_options:
                            selected_mood = st.selectbox("选择心情进行下载", mood_options)
                            
                            mood_files = [item for item in image_files + audio_files if item['mood'] == selected_mood]
                            if mood_files:
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for file_info in mood_files:
                                        try:
                                            ext = os.path.splitext(file_info['path'])[1]
                                            file_type = "图片" if file_info in image_files else "音频"
                                            filename = f"{file_type}_{file_info['id']}_{file_info['mood']}_{file_info['timestamp']}{ext}"
                                            zip_file.write(file_info['path'], filename)
                                        except Exception as e:
                                            st.warning(f"添加文件到压缩包失败: {e}")
                                
                                zip_buffer.seek(0)
                                st.download_button(
                                    f"📦 下载 {selected_mood} 心情的所有文件",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"emotion_{selected_mood.lower()}_files.zip",
                                    mime="application/zip"
                                )
                else:
                    st.info("暂无媒体文件可管理")

if __name__ == "__main__":
    main()


