import os
import io
import time
import json
import tempfile
from typing import Dict, Any, Optional

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

# 预加载快速语音转文本模型
@st.cache_resource
def load_fast_stt_model():
    """预加载快速语音转文本模型"""
    try:
        from stt_integration import get_fast_whisper_model
        model = get_fast_whisper_model()
        if model:
            st.success("✅ 快速语音转文本模型加载成功")
            return model
        else:
            st.warning("⚠️ 快速语音转文本模型加载失败，将使用备用方案")
            return None
    except Exception as e:
        st.warning(f"⚠️ 快速语音转文本模型加载失败: {e}，将使用备用方案")
        return None

# 快速语音转文本函数
def fast_transcribe_audio(audio_path: str) -> Optional[str]:
    """使用预加载的快速模型进行语音转文本"""
    if not audio_path or not os.path.exists(audio_path):
        return None
    
    # 获取预加载的模型
    model = st.session_state.get('fast_stt_model')
    if not model:
        # 备用方案：使用p2p_evaluate中的函数
        from p2p_evaluate import transcribe_audio_to_text
        return transcribe_audio_to_text(audio_path)
    
    try:
        start_time = time.time()
        
        # 使用优化的参数配置
        segments, info = model.transcribe(
            audio_path,
            beam_size=1,                    # 最小beam size，最快速度
            language="zh",                  # 指定语言，避免检测
            vad_filter=False,               # 关闭VAD，提高速度
            condition_on_previous_text=False, # 不依赖前文
            temperature=0.0,                # 确定性输出
            word_timestamps=False           # 关闭时间戳，提高速度
        )
        
        # 拼接所有分段为一条完整句子
        final_text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", None))
        
        processing_time = time.time() - start_time
        st.info(f"🎵 语音转写完成 ({processing_time:.2f}秒)")
        
        return final_text if final_text.strip() else None
        
    except Exception as e:
        hint = ""
        try:
            ext = os.path.splitext(audio_path)[1].lower()
            if ext in [".mp3", ".m4a", ".flac"]:
                hint = "（可能缺少 ffmpeg，建议安装后重试，或先转为 wav）"
        except Exception:
            pass
        st.error(f"❌ 语音转写失败: {e} {hint}")
        return None


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


def upload_data_log(payload: Dict[str, Any]) -> None:
    """根据环境变量 LOG_TARGETS 写入多种落盘目标。
    支持：
      - http: 使用 DATA_LOG_ENDPOINT POST
      - file: 写入 logs.ndjson，并将媒体复制到 logs_media/
      - sqlite: 写入 logs.db，并将媒体复制到 logs_media/
    默认：file,sqlite
    """
    targets = (os.environ.get("LOG_TARGETS") or "file,sqlite").split(",")
    targets = [t.strip().lower() for t in targets if t.strip()]

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

    # 构建简化记录（媒体改为本地路径）
    record = dict(payload)
    record.pop("media", None)
    record.pop("__tmp_image_path", None)
    record.pop("__tmp_audio_path", None)
    record["media_paths"] = {"image": saved_image, "audio": saved_audio}

    if "http" in targets:
        endpoint = os.environ.get("DATA_LOG_ENDPOINT")
        if endpoint:
            for _ in range(2):
                try:
                    resp = requests.post(endpoint, json=payload, timeout=20)
                    if 200 <= resp.status_code < 300:
                        break
                except Exception:
                    time.sleep(0.5)

    if "file" in targets:
        try:
            with open("logs.ndjson", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

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
        except Exception:
            pass


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
        # 使用新的快速语音转文本功能
        text_from_asr = fast_transcribe_audio(audio_path)
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
        fast_stt_model = load_fast_stt_model()
        # 将模型存储在session_state中，供后续使用
        st.session_state['fast_stt_model'] = fast_stt_model
    
    st.success("🚀 系统初始化完成！")

    # 移除 UI 内部设置 ARK_API_KEY 的入口，统一使用环境变量/Secrets

    tab1, tab2 = st.tabs(["自动测试", "单条测试（上传图片与音频）"]) 

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
        model_status = "✅ 快速模型已加载" if st.session_state.get('fast_stt_model') else "⚠️ 使用备用模型"
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
                    payload = {
                        "timestamp": int(time.time()),
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
                    upload_data_log(payload)
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
                    st.success("已保存")
                except Exception as e:
                    st.warning(f"保存失败：{e}")

    

if __name__ == "__main__":
    main()


