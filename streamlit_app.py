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
    transcribe_audio_to_text,
    fuse_visual_and_text,
    normalize_label,
)


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
    # 1) 视觉
    v_pred, v_time = "NEUTRAL", 0.0
    if image_path and os.path.exists(image_path):
        t0 = time.time()
        v_pred = predict_visual(image_path)
        v_time = time.time() - t0

    # 2) 文本：若 override_text 提供则优先使用，否则尝试音频转写
    asr_time = 0.0
    text_content = override_text.strip()
    if not text_content and audio_path and os.path.exists(audio_path):
        t0 = time.time()
        text_from_asr = transcribe_audio_to_text(audio_path)
        asr_time = time.time() - t0
        if text_from_asr:
            text_content = text_from_asr

    t_pred, t_time = "NEUTRAL", 0.0
    if text_content:
        t0 = time.time()
        t_pred = predict_text(text_content)
        t_time = time.time() - t0

    fused = fuse_visual_and_text(v_pred, t_pred)
    row_time = max(v_time, t_time, asr_time)
    return {
        "vision_pred": v_pred,
        "vision_time_s": v_time,
        "text_pred": t_pred,
        "text_time_s": t_time,
        "asr_time_s": asr_time,
        "fused_pred": fused,
        "text_content": text_content,
        "row_time_s": row_time,
    }


def main():
    st.set_page_config(page_title="多模态情绪识别演示", layout="wide")
    st.title("多模态情绪识别演示")

    tab1, tab2 = st.tabs(["自动测试", "单条测试（上传图片与音频）"]) 

    with tab1:
        st.subheader("自动测试（批量评估）")
        default_path = DEFAULT_EXCEL_PATH
        c1, c2, _ = st.columns([4, 1, 6])
        with c1:
            excel_path = st.text_input("Excel 路径", value=default_path)
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
        col_left, col_right = st.columns([1,1])
        with col_left:
            img_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
            if img_file is not None:
                st.image(img_file, caption="上传的图片", width=280)
        with col_right:
            wav_file = st.file_uploader("上传音频（.wav）", type=["wav"]) 
            override_text = st.text_area("可选：直接输入文本（将跳过语音转写）", height=120)

        b1, b2, _ = st.columns([1,1,6])
        with b1:
            run_single_btn = st.button("开始单条测试", type="primary")
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
            st.markdown("**预测结果**")
            m1, m2, m3 = st.columns(3)
            m1.metric("视觉预测", res["vision_pred"]) 
            m2.metric("文本预测", res["text_pred"]) 
            m3.metric("融合结果", res["fused_pred"]) 

            st.markdown("**耗时情况（秒）**")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("视觉用时", f"{res['vision_time_s']:.3f}")
            t2.metric("ASR转写", f"{res['asr_time_s']:.3f}")
            t3.metric("文本用时", f"{res['text_time_s']:.3f}")
            t4.metric("整体(并行)", f"{res['row_time_s']:.3f}")

            st.markdown("**语音转写文本**")
            st.write(res.get("text_content") or "无")


if __name__ == "__main__":
    main()


