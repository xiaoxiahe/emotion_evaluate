import os
from typing import Optional, Tuple, Iterable

# 统一在此模块管理 faster-whisper 的加载与转写，供 p2p_evaluate 与 streamlit_app 复用

_FAST_MODEL = None


def get_fast_whisper_model():
	"""懒加载并返回 faster-whisper 模型实例。
	优先选择体积较小的模型以确保速度；若环境有 GPU，可将 device 调整为 "cuda"。
	"""
	global _FAST_MODEL
	if _FAST_MODEL is not None:
		return _FAST_MODEL
	try:
		from faster_whisper import WhisperModel
		model_name = os.environ.get("FAST_STT_MODEL", "tiny")  # 优先最快
		device = os.environ.get("FAST_STT_DEVICE", "cpu")
		compute_type = os.environ.get("FAST_STT_COMPUTE", "int8")
		_FAST_MODEL = WhisperModel(
			model_name,
			device=device,
			compute_type=compute_type,
			local_files_only=False,
		)
		return _FAST_MODEL
	except Exception:
		# 返回 None 以让上层走备用方案
		return None


def _merge_segments_text(segments: Iterable) -> str:
	texts = []
	for seg in segments or []:
		text = getattr(seg, "text", None)
		if text:
			texts.append(text.strip())
	return " ".join(t for t in texts if t)


def transcribe_single_audio(audio_path: str) -> Optional[str]:
	"""使用预加载的快速模型将本地音频转写为文本。
	返回转写文本；失败或无结果时返回 None。
	"""
	if not audio_path or not os.path.exists(audio_path):
		return None
	model = get_fast_whisper_model()
	if model is None:
		return None
	try:
		segments, _info = model.transcribe(
			audio_path,
			beam_size=1,
			language="zh",
			vad_filter=False,
			condition_on_previous_text=False,
			temperature=0.0,
			word_timestamps=False,
		)
		merged = _merge_segments_text(segments)
		return merged or None
	except Exception:
		return None
