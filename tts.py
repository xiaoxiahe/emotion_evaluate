import os
import time
from faster_whisper import WhisperModel

path = r"faster-whisper-large-v3"
# path = r"faster-whisper-base"
# path = r"faster-whisper-medium"
model = WhisperModel(path, device="cpu", compute_type="int8", local_files_only=True)

audio_folder = [r"D:\shixi\fulai\emotion_code\emotion_data\audio\angry",
                r"D:\shixi\fulai\emotion_code\emotion_data\audio\happy",
                r"D:\shixi\fulai\emotion_code\emotion_data\audio\neutral",
                r"D:\shixi\fulai\emotion_code\emotion_data\audio\sad",]

for folder in audio_folder:
    if not os.path.isdir(folder):
        continue
    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            audio_path = os.path.join(folder, filename)
            start_time = time.time()
            segments, info = model.transcribe(audio_path, beam_size=5, language="zh", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
            end_time = time.time()
            print(f"文件: {filename}")
            print(f"识别耗时: {end_time - start_time:.2f} 秒")
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            # 拼接所有分段为一条完整句子
            final_text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", None))
            print(f"识别文本: {final_text}")
            print("-" * 40)
