import os
import io
import time
import json
import tempfile
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

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
        st.error(f"❌ 语音转写失败: {e}")
        return None


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


def save_uploaded_file(uploaded_file, suffix: str) -> Optional[str]:
    if uploaded_file is None:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


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


APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_abs_path(path: str) -> str:
    """Return absolute path; if relative, resolve relative to this file directory."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(APP_DIR, path)


def main():
    st.set_page_config(page_title="多模态情绪识别演示", layout="wide")
    # 强制启用页面滚动（Cloud 有时出现 overflow 限制）
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { overflow: auto !important; }
        [data-testid="stVerticalBlock"] { overflow: visible !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("多模态情绪识别演示")
    
    # 在界面启动时预加载语音转文本模型
    with st.spinner("正在加载语音转文本模型..."):
        fast_stt_model = load_fast_stt_model()
        # 将模型存储在session_state中，供后续使用
        st.session_state['fast_stt_model'] = fast_stt_model
    
    st.success("🚀 系统初始化完成！")

    tab1, tab2 = st.tabs(["自动测试", "单条测试（上传图片与音频）"]) 

    with tab1:
        st.subheader("自动测试（批量评估）")
        choice = st.radio("选择测试规模", ["简单（50条）", "完整（196条）", "自定义路径"], horizontal=True)
        c1, c2, _ = st.columns([4, 1, 6])
        with c1:
            if choice == "简单（50条）":
                excel_path = os.path.join(APP_DIR, "multimodal_emotion_data_50.xlsx")
                st.text_input("Excel 路径", value=excel_path, disabled=True)
            elif choice == "完整（196条）":
                excel_path = os.path.join(APP_DIR, "multimodal_emotion_data_196.xlsx")
                st.text_input("Excel 路径", value=excel_path, disabled=True)
            else:
                default_path = _ensure_abs_path(DEFAULT_EXCEL_PATH)
                excel_path = st.text_input("Excel 路径", value=default_path)
        with c2:
            run_btn = st.button("开始自动测试", type="primary")
        if run_btn:
            try:
                res = run_batch_auto_test(_ensure_abs_path(excel_path))
                st.success(f"样本数: {res['total']} | 准确率: {res['accuracy']:.4f} | 总耗时: {res['elapsed']:.2f}s | 平均单条: {res['avg_time']:.3f}s | 50分位: {res['p50_time']:.3f}s | 95分位: {res['p95_time']:.3f}s")

                st.write("每类召回率：")
                pc_df = pd.DataFrame({"label": list(res["per_class"].keys()), "recall": list(res["per_class"].values())})
                st.dataframe(pc_df, width='stretch', height=140)

                out_df = pd.DataFrame(res["records"]).sort_values("ID")
                st.dataframe(out_df, width='stretch', height=300)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载结果 CSV", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")
            except Exception as e:
                st.error(f"批量评估失败: {e}")

    with tab2:
        st.subheader("单条测试（上传图片与音频）")
        
        # 显示模型状态
        model_status = "✅ 快速模型已加载" if st.session_state.get('fast_stt_model') else "⚠️ 使用备用模型"
        st.info(f"语音转文本模型状态: {model_status}")
        
        col_left, col_right = st.columns([1,1])
        with col_left:
            img_file = st.file_uploader("📷 上传图片", type=["jpg", "jpeg", "png"])
            if img_file is not None:
                st.image(img_file, caption="上传的图片", width=280)
        with col_right:
            wav_file = st.file_uploader("🎵 上传音频", type=["wav", "mp3", "m4a", "flac"]) 
            override_text = st.text_area("📝 可选：直接输入文本（将跳过语音转写）", 
                                       placeholder="在这里输入文本内容...", 
                                       height=120)

        b1, b2, _ = st.columns([1,1,6])
        with b1:
            run_single_btn = st.button("🚀 开始单条测试", type="primary")
        if run_single_btn:
            with st.spinner("处理中..."):
                tmp_img = save_uploaded_file(img_file, suffix=".png") if img_file else None
                tmp_wav = save_uploaded_file(wav_file, suffix=".wav") if wav_file else None

                try:
                    res = run_single_test(tmp_img, tmp_wav, override_text=override_text)
                finally:
                    # 清理临时文件
                    if tmp_img and os.path.exists(tmp_img):
                        try:
                            os.remove(tmp_img)
                        except Exception:
                            pass
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.remove(tmp_wav)
                        except Exception:
                            pass

            # 展示结果
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


if __name__ == "__main__":
    main()


