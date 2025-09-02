#coding=utf-8
'''
火山一句话识别
'''

"""
requires Python 3.6 or later

pip install asyncio
pip install websockets
"""

import asyncio
import base64
import gzip
import hmac
import json
import logging
import os
import uuid
import wave
from enum import Enum
from hashlib import sha256
from io import BytesIO
from typing import List
from urllib.parse import urlparse
import time
import websockets
import glob
import re
from collections import defaultdict

appid = "5851744862"    # 项目的 appid
token = "HdMaaKvnrzQ4vuLGJ0tP2u_v5Xd97_Ho"    # 项目的 token
cluster = "volcengine_input_common"  # 请求的集群
base_audio_path = r"D:\shixi\emotion_evaluate\emotion_data\audio"  # 音频根目录
audio_format = "wav"   # wav 或者 mp3，根据实际音频格式设置

# 情感文件夹列表
emotion_folders = ["angry", "happy", "sad", "neutral"]

PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

PROTOCOL_VERSION_BITS = 4
HEADER_BITS = 4
MESSAGE_TYPE_BITS = 4
MESSAGE_TYPE_SPECIFIC_FLAGS_BITS = 4
MESSAGE_SERIALIZATION_BITS = 4
MESSAGE_COMPRESSION_BITS = 4
RESERVED_BITS = 8

# Message Type:
CLIENT_FULL_REQUEST = 0b0001
CLIENT_AUDIO_ONLY_REQUEST = 0b0010
SERVER_FULL_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE = 0b0000  # no check sequence
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_SEQUENCE_1 = 0b0011

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001
THRIFT = 0b0011
CUSTOM_TYPE = 0b1111

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001
CUSTOM_COMPRESSION = 0b1111


def generate_header(
    version=PROTOCOL_VERSION,
    message_type=CLIENT_FULL_REQUEST,
    message_type_specific_flags=NO_SEQUENCE,
    serial_method=JSON,
    compression_type=GZIP,
    reserved_data=0x00,
    extension_header=bytes()
):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    """
    header = bytearray()
    header_size = int(len(extension_header) / 4) + 1
    header.append((version << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    header.extend(extension_header)
    return header


def generate_full_default_header():
    return generate_header()


def generate_audio_default_header():
    return generate_header(
        message_type=CLIENT_AUDIO_ONLY_REQUEST
    )


def generate_last_audio_default_header():
    return generate_header(
        message_type=CLIENT_AUDIO_ONLY_REQUEST,
        message_type_specific_flags=NEG_SEQUENCE
    )

def parse_response(res):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    payload 类似与http 请求体
    """
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0f
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0f
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0f
    reserved = res[3]
    header_extensions = res[4:header_size * 4]
    payload = res[header_size * 4:]
    result = {}
    payload_msg = None
    payload_size = 0
    if message_type == SERVER_FULL_RESPONSE:
        payload_size = int.from_bytes(payload[:4], "big", signed=True)
        payload_msg = payload[4:]
    elif message_type == SERVER_ACK:
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result['seq'] = seq
        if len(payload) >= 8:
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
    elif message_type == SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        result['code'] = code
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]
    if payload_msg is None:
        return result
    if message_compression == GZIP:
        payload_msg = gzip.decompress(payload_msg)
    if serialization_method == JSON:
        payload_msg = json.loads(str(payload_msg, "utf-8"))
    elif serialization_method != NO_SERIALIZATION:
        payload_msg = str(payload_msg, "utf-8")
    result['payload_msg'] = payload_msg
    result['payload_size'] = payload_size
    return result


def read_wav_info(data: bytes = None) -> (int, int, int, int, int):
    with BytesIO(data) as _f:
        wave_fp = wave.open(_f, 'rb')
        nchannels, sampwidth, framerate, nframes = wave_fp.getparams()[:4]
        wave_bytes = wave_fp.readframes(nframes)
    return nchannels, sampwidth, framerate, nframes, len(wave_bytes)

def calculate_accuracy(predicted_text, actual_text):
    """
    计算字符级准确率（忽略标点符号）
    """
    # 移除标点符号和空格
    def clean_text(text):
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    
    pred_clean = clean_text(predicted_text)
    actual_clean = clean_text(actual_text)
    
    if not actual_clean:
        return 0.0
    
    # 计算编辑距离
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = levenshtein_distance(pred_clean, actual_clean)
    accuracy = max(0, (len(actual_clean) - distance) / len(actual_clean))
    return accuracy

def load_ground_truth_texts():
    """
    加载所有情感文件夹下的真实文本
    """
    ground_truth = {}
    
    for emotion in emotion_folders:
        txt_path = os.path.join(base_audio_path, emotion, f"{emotion}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            audio_id = parts[0]
                            text = parts[1]
                            ground_truth[audio_id] = text
    
    return ground_truth

class AudioType(Enum):
    LOCAL = 1  # 使用本地音频文件

class AsrWsClient:
    def __init__(self, audio_path, cluster, **kwargs):
        """
        :param config: config
        """
        self.audio_path = audio_path
        self.cluster = cluster
        self.success_code = 1000  # success code, default is 1000
        self.seg_duration = int(kwargs.get("seg_duration", 15000))
        self.nbest = int(kwargs.get("nbest", 1))
        self.appid = kwargs.get("appid", "")
        self.token = kwargs.get("token", "")
        self.ws_url = kwargs.get("ws_url", "wss://openspeech.bytedance.com/api/v2/asr")
        self.uid = kwargs.get("uid", "streaming_asr_demo")
        self.workflow = kwargs.get("workflow", "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate")
        self.show_language = kwargs.get("show_language", False)
        self.show_utterances = kwargs.get("show_utterances", False)
        self.result_type = kwargs.get("result_type", "full")
        self.format = kwargs.get("format", "wav")
        self.rate = kwargs.get("sample_rate", 16000)
        self.language = kwargs.get("language", "zh-CN")
        self.bits = kwargs.get("bits", 16)
        self.channel = kwargs.get("channel", 1)
        self.codec = kwargs.get("codec", "raw")
        self.audio_type = kwargs.get("audio_type", AudioType.LOCAL)
        self.secret = kwargs.get("secret", "access_secret")
        self.auth_method = kwargs.get("auth_method", "token")
        self.mp3_seg_size = int(kwargs.get("mp3_seg_size", 10000))

    def construct_request(self, reqid):
        req = {
            'app': {
                'appid': self.appid,
                'cluster': self.cluster,
                'token': self.token,
            },
            'user': {
                'uid': self.uid
            },
            'request': {
                'reqid': reqid,
                'nbest': self.nbest,
                'workflow': self.workflow,
                'show_language': self.show_language,
                'show_utterances': self.show_utterances,
                'result_type': self.result_type,
                "sequence": 1
            },
            'audio': {
                'format': self.format,
                'rate': self.rate,
                'language': self.language,
                'bits': self.bits,
                'channel': self.channel,
                'codec': self.codec
            }
        }
        return req

    @staticmethod
    def slice_data(data: bytes, chunk_size: int) -> (list, bool):
        """
        slice data
        :param data: wav data
        :param chunk_size: the segment size in one request
        :return: segment data, last flag
        """
        data_len = len(data)
        offset = 0
        while offset + chunk_size < data_len:
            yield data[offset: offset + chunk_size], False
            offset += chunk_size
        else:
            yield data[offset: data_len], True

    def _real_processor(self, request_params: dict) -> dict:
        pass

    def token_auth(self):
        return {'Authorization': 'Bearer; {}'.format(self.token)}

    def signature_auth(self, data):
        header_dicts = {
            'Custom': 'auth_custom',
        }

        url_parse = urlparse(self.ws_url)
        input_str = 'GET {} HTTP/1.1\n'.format(url_parse.path)
        auth_headers = 'Custom'
        for header in auth_headers.split(','):
            input_str += '{}\n'.format(header_dicts[header])
        input_data = bytearray(input_str, 'utf-8')
        input_data += data
        mac = base64.urlsafe_b64encode(
            hmac.new(self.secret.encode('utf-8'), input_data, digestmod=sha256).digest())
        header_dicts['Authorization'] = 'HMAC256; access_token="{}"; mac="{}"; h="{}"'.format(self.token,
                                                                                              str(mac, 'utf-8'), auth_headers)
        return header_dicts

    async def segment_data_processor(self, wav_data: bytes, segment_size: int):
        reqid = str(uuid.uuid4())
        # 构建 full client request，并序列化压缩
        request_params = self.construct_request(reqid)
        payload_bytes = str.encode(json.dumps(request_params))
        payload_bytes = gzip.compress(payload_bytes)
        full_client_request = bytearray(generate_full_default_header())
        full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
        full_client_request.extend(payload_bytes)  # payload
        header = None
        if self.auth_method == "token":
            header = self.token_auth()
        elif self.auth_method == "signature":
            header = self.signature_auth(full_client_request)
        # 兼容不同版本的websockets库
        try:
            # 新版本websockets
            async with websockets.connect(self.ws_url, extra_headers=header, max_size=1000000000) as ws:
                # 发送 full client request
                await ws.send(full_client_request)
                res = await ws.recv()
                result = parse_response(res)
                if 'payload_msg' in result and result['payload_msg']['code'] != self.success_code:
                    return result
                for seq, (chunk, last) in enumerate(AsrWsClient.slice_data(wav_data, segment_size), 1):
                    # if no compression, comment this line
                    payload_bytes = gzip.compress(chunk)
                    audio_only_request = bytearray(generate_audio_default_header())
                    if last:
                        audio_only_request = bytearray(generate_last_audio_default_header())
                    audio_only_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
                    audio_only_request.extend(payload_bytes)  # payload
                    # 发送 audio-only client request
                    await ws.send(audio_only_request)
                    res = await ws.recv()
                    result = parse_response(res)
                    if 'payload_msg' in result and result['payload_msg']['code'] != self.success_code:
                        return result
        except TypeError:
            try:
                # 旧版本websockets
                async with websockets.connect(self.ws_url, additional_headers=header, max_size=1000000000) as ws:
                    # 发送 full client request
                    await ws.send(full_client_request)
                    res = await ws.recv()
                    result = parse_response(res)
                    if 'payload_msg' in result and result['payload_msg']['code'] != self.success_code:
                        return result
                    for seq, (chunk, last) in enumerate(AsrWsClient.slice_data(wav_data, segment_size), 1):
                        # if no compression, comment this line
                        payload_bytes = gzip.compress(chunk)
                        audio_only_request = bytearray(generate_audio_default_header())
                        if last:
                            audio_only_request = bytearray(generate_last_audio_default_header())
                        audio_only_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
                        audio_only_request.extend(payload_bytes)  # payload
                        # 发送 audio-only client request
                        await ws.send(audio_only_request)
                        res = await ws.recv()
                        result = parse_response(res)
                        if 'payload_msg' in result and result['payload_msg']['code'] != self.success_code:
                            return result
            except TypeError:
                # 如果都不支持，尝试不使用headers
                async with websockets.connect(self.ws_url, max_size=1000000000) as ws:
                    # 发送 full client request
                    await ws.send(full_client_request)
                    res = await ws.recv()
                    result = parse_response(res)
                    if 'payload_msg' in result and result['payload_msg']['code'] != self.success_code:
                        return result
                    for seq, (chunk, last) in enumerate(AsrWsClient.slice_data(wav_data, segment_size), 1):
                        # if no compression, comment this line
                        payload_bytes = gzip.compress(chunk)
                        audio_only_request = bytearray(generate_audio_default_header())
                        if last:
                            audio_only_request = bytearray(generate_last_audio_default_header())
                        audio_only_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
                        audio_only_request.extend(payload_bytes)  # payload
                        # 发送 audio-only client request
                        await ws.send(audio_only_request)
                        res = await ws.recv()
                        result = parse_response(res)
                        if 'payload_msg' in result and result['payload_msg']['code'] != self.success_code:
                            return result
        return result

    async def execute(self):
        with open(self.audio_path, mode="rb") as _f:
            data = _f.read()
        audio_data = bytes(data)
        if self.format == "mp3":
            segment_size = self.mp3_seg_size
            return await self.segment_data_processor(audio_data, segment_size)
        if self.format != "wav":
            raise Exception("format should in wav or mp3")
        nchannels, sampwidth, framerate, nframes, wav_len = read_wav_info(
            audio_data)
        size_per_sec = nchannels * sampwidth * framerate
        segment_size = int(size_per_sec * self.seg_duration / 1000)
        return await self.segment_data_processor(audio_data, segment_size)


def execute_one(audio_item, cluster, **kwargs):
    """

    :param audio_item: {"id": xxx, "path": "xxx"}
    :param cluster:集群名称
    :return:
    """
    assert 'id' in audio_item
    assert 'path' in audio_item
    audio_id = audio_item['id']
    audio_path = audio_item['path']
    audio_type = AudioType.LOCAL
    
    start_time = time.time()
    asr_http_client = AsrWsClient(
        audio_path=audio_path,
        cluster=cluster,
        audio_type=audio_type,
        **kwargs
    )
    result = asyncio.run(asr_http_client.execute())
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    return {
        "id": audio_id, 
        "path": audio_path, 
        "result": result,
        "processing_time": processing_time
    }

def batch_process_emotion_audio():
    """
    批量处理所有情感文件夹下的音频文件
    """
    ground_truth = load_ground_truth_texts()
    all_results = []
    
    for emotion in emotion_folders:
        emotion_path = os.path.join(base_audio_path, emotion)
        if not os.path.exists(emotion_path):
            continue
            
        # 获取该情感文件夹下的所有音频文件
        audio_files = glob.glob(os.path.join(emotion_path, "*.wav")) + glob.glob(os.path.join(emotion_path, "*.mp3"))
        
        print(f"处理 {emotion} 文件夹，共 {len(audio_files)} 个音频文件...")
        
        for audio_file in audio_files:
            audio_filename = os.path.basename(audio_file)
            audio_id = os.path.splitext(audio_filename)[0]
            
            print(f"正在处理: {audio_filename}")
            
            try:
                result = execute_one(
                    {
                        'id': audio_id,
                        'path': audio_file
                    },
                    cluster=cluster,
                    appid=appid,
                    token=token,
                    format=audio_format,
                )
                
                # 提取识别结果
                recognized_text = ""
                
                # 调试：打印完整的识别结果结构
                print(f"  调试 - 完整结果结构:")
                print(f"    result keys: {list(result['result'].keys()) if isinstance(result['result'], dict) else 'Not a dict'}")
                
                if isinstance(result['result'], dict):
                    if 'payload_msg' in result['result']:
                        payload_msg = result['result']['payload_msg']
                        print(f"    payload_msg keys: {list(payload_msg.keys()) if isinstance(payload_msg, dict) else 'Not a dict'}")
                        
                        if isinstance(payload_msg, dict):
                            if 'data' in payload_msg:
                                data = payload_msg['data']
                                print(f"    data type: {type(data)}, content: {data}")
                                
                                if isinstance(data, list) and len(data) > 0:
                                    first_item = data[0]
                                    print(f"    first item: {first_item}")
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
                
                print(f"   最终提取的文本: '{recognized_text}'")
                
                # 获取真实文本
                actual_text = ground_truth.get(audio_id, "")
                
                # 计算准确率
                accuracy = calculate_accuracy(recognized_text, actual_text)
                
                result_item = {
                    'filename': audio_filename,
                    'recognized_text': recognized_text,
                    'actual_text': actual_text,
                    'processing_time': result['processing_time'],
                    'accuracy': accuracy,
                    'emotion': emotion
                }
                
                all_results.append(result_item)
                
                # 打印单个结果
                print(f"  识别结果: {recognized_text}")
                print(f"  真实文本: {actual_text}")
                print(f"  处理时间: {result['processing_time']:.3f}秒")
                print(f"  准确率: {accuracy:.3f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"处理 {audio_filename} 时出错: {str(e)}")
                continue
    
    return all_results

def calculate_statistics(results):
    """
    计算统计指标
    """
    if not results:
        return {}
    
    processing_times = [r['processing_time'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # 排序用于计算百分位数
    processing_times.sort()
    accuracies.sort()
    
    stats = {
        'total_files': len(results),
        'avg_processing_time': sum(processing_times) / len(processing_times),
        'p50_processing_time': processing_times[len(processing_times) // 2],
        'p95_processing_time': processing_times[int(len(processing_times) * 0.95)],
        'avg_accuracy': sum(accuracies) / len(accuracies),
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies)
    }
    
    return stats

def save_results_to_file(results, stats):
    """
    保存结果到文件
    """
    output_file = "emotion_asr_results.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("情感语音识别结果报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入统计信息
        f.write("统计摘要:\n")
        f.write(f"总文件数: {stats['total_files']}\n")
        f.write(f"平均处理时间: {stats['avg_processing_time']:.3f}秒\n")
        f.write(f"50%分位处理时间: {stats['p50_processing_time']:.3f}秒\n")
        f.write(f"95%分位处理时间: {stats['p95_processing_time']:.3f}秒\n")
        f.write(f"平均准确率: {stats['avg_accuracy']:.3f}\n")
        f.write(f"最低准确率: {stats['min_accuracy']:.3f}\n")
        f.write(f"最高准确率: {stats['max_accuracy']:.3f}\n\n")
        
        # 写入详细结果
        f.write("详细结果:\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"文件名: {result['filename']}\n")
            f.write(f"情感类别: {result['emotion']}\n")
            f.write(f"识别文本: {result['recognized_text']}\n")
            f.write(f"真实文本: {result['actual_text']}\n")
            f.write(f"处理时间: {result['processing_time']:.3f}秒\n")
            f.write(f"准确率: {result['accuracy']:.3f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"结果已保存到: {output_file}")

def test_batch():
    """
    批量测试所有情感音频文件
    """
    print("开始批量处理情感音频文件...")
    print("=" * 60)
    
    # 批量处理
    results = batch_process_emotion_audio()
    
    if results:
        # 计算统计指标
        stats = calculate_statistics(results)
        
        # 打印统计结果
        print("\n" + "=" * 60)
        print("统计结果:")
        print(f"总文件数: {stats['total_files']}")
        print(f"平均处理时间: {stats['avg_processing_time']:.3f}秒")
        print(f"50%分位处理时间: {stats['p50_processing_time']:.3f}秒")
        print(f"95%分位处理时间: {stats['p95_processing_time']:.3f}秒")
        print(f"平均准确率: {stats['avg_accuracy']:.3f}")
        print(f"最低准确率: {stats['min_accuracy']:.3f}")
        print(f"最高准确率: {stats['max_accuracy']:.3f}")
        
        # 保存结果到文件
        save_results_to_file(results, stats)
    else:
        print("没有成功处理任何音频文件")

if __name__ == '__main__':
    test_batch()
