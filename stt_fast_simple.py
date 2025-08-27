import os
import time
from faster_whisper import WhisperModel

# 简化但高效的faster-whisper配置
# 使用最小的模型和基本优化参数

# 模型选择（按速度排序）
#model_name = "tiny"  # 改为tiny以获得最快速度
#model_name = "base"   # 平衡速度和精度
model_name = "small"  # 更好精度

# 设备配置
device = "cpu"  # 如果有GPU，改为"cuda"
compute_type = "int8"  # 最快的计算类型

print(f"加载模型: {model_name}")
model = WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=False)

# 音频文件夹路径 - 支持多种可能的路径
possible_audio_folders = [
    r"D:\shixi\fulai\emotion_code\emotion_data\audio\angry",
    r"D:\shixi\fulai\emotion_code\emotion_data\audio\happy", 
    r"D:\shixi\fulai\emotion_code\emotion_data\audio\neutral",
    r"D:\shixi\fulai\emotion_code\emotion_data\audio\sad",
    # 备用路径
    r"emotion_data\audio\angry",
    r"emotion_data\audio\happy",
    r"emotion_data\audio\neutral", 
    r"emotion_data\audio\sad",
    # 当前目录下的音频文件夹
    r"audio\angry",
    r"audio\happy",
    r"audio\neutral",
    r"audio\sad"
]

# 过滤出存在的文件夹
audio_folders = [folder for folder in possible_audio_folders if os.path.isdir(folder)]

if not audio_folders:
    print("错误：未找到音频文件夹！")
    print("请检查以下路径是否存在：")
    for folder in possible_audio_folders:
        print(f"  - {folder}")
    exit(1)

print(f"找到 {len(audio_folders)} 个音频文件夹")
for folder in audio_folders:
    print(f"  - {folder}")

total_time = 0
file_count = 0

for folder in audio_folders:
    print(f"\n处理文件夹: {folder}")
    
    for filename in os.listdir(folder):
        if filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
            audio_path = os.path.join(folder, filename)
            start_time = time.time()
            
            try:
                # 简化的优化参数配置
                segments, info = model.transcribe(
                    audio_path, 
                    beam_size=1,                    # 最小beam size，最快速度
                    language="zh",                  # 指定语言，避免检测
                    vad_filter=False,               # 关闭VAD，提高速度
                    condition_on_previous_text=False, # 不依赖前文
                    temperature=0.0,                # 确定性输出
                    word_timestamps=False           # 关闭时间戳，提高速度
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                total_time += processing_time
                file_count += 1
                
                print(f"文件: {filename}")
                print(f"识别耗时: {processing_time:.2f} 秒")
                print(f"语言: {info.language} (概率: {info.language_probability:.2f})")
                
                # 拼接所有分段为一条完整句子
                final_text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", None))
                print(f"识别文本: {final_text}")
                print("-" * 40)
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                print("-" * 40)

if file_count > 0:
    avg_time = total_time / file_count
    print(f"\n=== 性能统计 ===")
    print(f"总文件数: {file_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均耗时: {avg_time:.2f} 秒/文件")
    print(f"处理速度: {file_count/total_time:.2f} 文件/秒")
else:
    print("\n未找到任何音频文件进行处理")
