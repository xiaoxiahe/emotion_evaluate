import os
import json
import base64
from typing import List, Dict, Any

import requests

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def _auth_headers() -> Dict[str, str]:
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("ARK_API_KEY 未设置")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def ark_chat_json(model: str, messages: List[Dict[str, Any]], temperature: float = 0.1, top_p: float = 0.7, timeout: int = 60) -> Dict[str, Any]:
    url = f"{ARK_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    resp = requests.post(url, headers=_auth_headers(), data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def image_to_data_url(path: str) -> str:
    """将图片压缩为较小尺寸再以 data URL 返回，减少大模型入参体积。
    - 最长边压到 768px
    - 统一转 JPEG，质量 80
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            max_side = max(w, h)
            if max_side > 768:
                scale = 768 / float(max_side)
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size)
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        # 退化为原图直传
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 简化：若失败统一按 jpeg 处理
        return f"data:image/jpeg;base64,{b64}"


