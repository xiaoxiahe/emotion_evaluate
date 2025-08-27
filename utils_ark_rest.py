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
    # 根据扩展名推断 MIME，默认 jpeg
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg"
    if ext in [".png"]:
        mime = "image/png"
    elif ext in [".gif"]:
        mime = "image/gif"
    elif ext in [".webp"]:
        mime = "image/webp"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


