import base64
import json
import logging
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
try:
    from volcenginesdkarkruntime import Ark  # type: ignore
except Exception:
    Ark = None

model_key_dict = {
    'doubao_1_6':'be62df6f-1828-47a4-84f1-2932c111bc64',
    'doubao_1_6_flash':'be62df6f-1828-47a4-84f1-2932c111bc64',
}

model_endpoint_dict = {
    'doubao_1_6':'doubao-seed-1-6-250615',
    'doubao_1_6_flash':'doubao-seed-1-6-flash-250715',
}


# logging.basicConfig(level=logging.INFO)
def create_client(model_name):
    if Ark is None:
        raise RuntimeError("volcenginesdkarkruntime 未安装，无法创建 Ark 客户端")
    return Ark(api_key=model_key_dict[model_name], base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=1800)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def request_doubao_vision_api(client, model_ep, prompt, base64_image, temperature, top_p):
    start_time = time.time()
    response = client.chat.completions.create(
        model=model_ep,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=4096,
        thinking={"type":"disabled"},
    )
    end_time = time.time()
    
    # 计算耗时
    duration = end_time - start_time
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    print(response)
    return response.choices[0].message.content, prompt_tokens, completion_tokens, duration

def retry_request(client, model_ep, prompt, base64_image, temperature, top_p,retries=5, delay=30):
    for _ in range(retries):
        try:
            return request_doubao_vision_api(client, model_ep, prompt, base64_image, temperature, top_p)
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(delay)
    return '', -1,-1,-1

def request_doubao_vision(client, image_path, model_name, prompt, index, output_file):
    temperature = 0.1
    top_p = 0.7
    model_ep = model_endpoint_dict[model_name]
    base64_image = encode_image(image_path)

    content, prompt_tokens, completion_tokens, duration = retry_request(client, model_ep, prompt, base64_image, temperature, top_p)
    result = {
        'file_name': image_path.split('/')[-1],
        'answer': content,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'duration': duration,
        'prompt':prompt
    }
    # print(result)
    with open(output_file, 'a', encoding='utf-8') as f:
        json_line = json.dumps(result, ensure_ascii=False)
        f.write(json_line + '\n')

def write_results_to_jsonl(file_path, results):
    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')

def calculate_average_length(items):
    total_length = sum(len(item) for item in items if item)
    return total_length / len(items)

def get_image_paths_from_folder(folder_path):
    image_paths = []

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # 构建完整路径
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)

    return image_paths

def process_queries_concurrently(file_paths, prompt, model_name, output_file):
    client = create_client(model_name)
    for index, file_path in enumerate(file_paths):
        request_doubao_vision(client, file_path, model_name, prompt, index, output_file)
        
def main():
    # 情绪识别
    folder_path = './data/emotion_judge' # 图片文件夹
    output_file = './data/emotion_judge.jsonl' # 输出结果
    prompt = open('./prompt/emotion_judge').read()
    model_name = 'doubao_1_6_flash'
    file_paths = get_image_paths_from_folder(folder_path)

    process_queries_concurrently(file_paths, prompt, model_name, output_file)

if __name__ == "__main__":
    main()