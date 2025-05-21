"""
实时语音识别、翻译与语音合成系统 - 集成版
------------------------------------------
这个版本集成了语音识别、翻译和自动语音合成功能，实现类似实时口译的体验。

功能：
- 实时中文语音识别
- 实时文本翻译
- 自动语音合成（TTS）
- 支持多语言翻译和语音合成

使用方法:
- 从下拉菜单选择目标翻译语言
- 配置自动TTS选项（启用/禁用、语音选择、速度等）
- 点击"开始录音"按钮开始录音
- 对着麦克风说话，系统会自动识别、翻译并播放译文
- 点击"停止录音"按钮停止录音

依赖库:
- funasr (语音识别)
- sounddevice (音频录制)
- numpy (数据处理)
- requests (API通信)
- edge_tts (文本到语音)
- pygame (音频播放)
- tkinter (GUI界面)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import asyncio
import time
import queue
import os
import sys
import io
from datetime import datetime
import numpy as np
import json
import requests
import base64
import hmac
import hashlib
from time import mktime
from wsgiref.handlers import format_date_time
from urllib.parse import urlencode
import re

# 尝试导入语音识别相关库
try:
    from funasr import AutoModel
    import sounddevice as sd
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("警告: funasr或sounddevice未安装，语音识别功能将无法使用")

# 尝试导入语音合成相关库
try:
    import edge_tts
    import pygame
    from pygame import mixer
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("警告: edge_tts或pygame未安装，语音合成功能将无法使用")

# 全局日志队列
log_queue = queue.Queue()

def log(message, show_time=True):
    """添加日志消息到队列"""
    if show_time:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        full_message = f"[{timestamp}] {message}"
    else:
        full_message = message
    log_queue.put(full_message)
    print(full_message)


#----------------------------------------
# 语音识别模块 (从translator.py导入)
#----------------------------------------
class FastLoadASR:
    def __init__(self, use_vad=True, use_punc=True, disable_update=True):
        """初始化快速加载版语音识别系统"""
        log("初始化语音识别模块...")

        if not FUNASR_AVAILABLE:
            log("错误: FunASR库未安装")
            return

        # 功能开关
        self.use_vad = use_vad
        self.use_punc = use_punc
        self.disable_update = disable_update

        # 语音识别参数设置
        self.sample_rate = 16000  # 采样率(Hz)

        # ASR参数
        self.asr_chunk_size = [0, 10, 5]  # 流式设置：[0, 10, 5] = 600ms
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        # VAD参数
        self.vad_chunk_duration_ms = 200  # VAD每个音频块的持续时间(毫秒)
        self.vad_chunk_samples = int(self.sample_rate * self.vad_chunk_duration_ms / 1000)

        # ASR参数
        self.asr_chunk_duration_ms = 600  # 每个ASR音频块的持续时间(毫秒)
        self.asr_chunk_samples = int(self.sample_rate * self.asr_chunk_duration_ms / 1000)

        # 运行时变量
        self.running = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.complete_transcript = ""
        self.raw_transcript = ""
        self.is_speaking = False
        self.speech_buffer = np.array([], dtype=np.float32)

        # 模型变量
        self.asr_model = None
        self.vad_model = None
        self.punc_model = None
        self.vad_cache = {}
        self.asr_cache = {}

        # 设置环境变量以加快加载
        if self.disable_update:
            os.environ["FUNASR_DISABLE_UPDATE"] = "True"

        # 异步预加载ASR模型
        log("开始异步加载ASR模型...")
        self.asr_load_thread = threading.Thread(target=self.load_asr_model)
        self.asr_load_thread.daemon = True
        self.asr_load_thread.start()

    def load_asr_model(self):
        """加载ASR模型的线程函数"""
        try:
            log("正在加载ASR模型...")
            self.asr_model = AutoModel(model="paraformer-zh-streaming")
            log("ASR模型加载完成!")
        except Exception as e:
            log(f"ASR模型加载失败: {e}")

    def ensure_asr_model_loaded(self):
        """确保ASR模型已加载"""
        if self.asr_model is None:
            log("等待ASR模型加载完成...")
            if hasattr(self, 'asr_load_thread'):
                self.asr_load_thread.join()

            # 如果线程结束后模型仍未加载，再次尝试加载
            if self.asr_model is None:
                log("重新尝试加载ASR模型...")
                try:
                    self.asr_model = AutoModel(model="paraformer-zh-streaming")
                    log("ASR模型加载完成!")
                except Exception as e:
                    log(f"ASR模型加载失败: {e}")
                    return False
        return True

    def load_vad_model_if_needed(self):
        """仅在需要时加载VAD模型"""
        if self.use_vad and self.vad_model is None:
            log("加载VAD模型...")
            try:
                self.vad_model = AutoModel(model="fsmn-vad")
                log("VAD模型加载完成!")
                return True
            except Exception as e:
                log(f"VAD模型加载失败: {e}")
                return False
        return True

    def load_punc_model_if_needed(self):
        """仅在需要时加载标点恢复模型"""
        if self.use_punc and self.punc_model is None:
            log("加载标点恢复模型...")
            try:
                self.punc_model = AutoModel(model="ct-punc")
                log("标点恢复模型加载完成!")
                return True
            except Exception as e:
                log(f"标点恢复模型加载失败: {e}")
                return False
        return True

    def audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            log(f"音频状态: {status}")
        # 将音频数据放入队列
        self.audio_queue.put(indata.copy())

    def process_audio_thread(self):
        """音频处理线程"""
        vad_buffer = np.array([], dtype=np.float32)

        while self.running:
            try:
                # 从队列获取音频数据
                while not self.audio_queue.empty() and self.running:
                    chunk = self.audio_queue.get_nowait()
                    if self.use_vad:
                        vad_buffer = np.append(vad_buffer, chunk.flatten())
                    else:
                        # 不使用VAD时，直接将音频块添加到语音缓冲区
                        self.speech_buffer = np.append(self.speech_buffer, chunk.flatten())

                # 使用VAD处理
                if self.use_vad and self.vad_model is not None:
                    while len(vad_buffer) >= self.vad_chunk_samples and self.running:
                        # 提取一个VAD音频块
                        vad_chunk = vad_buffer[:self.vad_chunk_samples]
                        vad_buffer = vad_buffer[self.vad_chunk_samples:]

                        # 使用VAD模型处理
                        vad_res = self.vad_model.generate(
                            input=vad_chunk,
                            cache=self.vad_cache,
                            is_final=False,
                            chunk_size=self.vad_chunk_duration_ms
                        )

                        # 处理VAD结果
                        if len(vad_res[0]["value"]):
                            # 有语音活动检测结果
                            for segment in vad_res[0]["value"]:
                                if segment[0] != -1 and segment[1] == -1:
                                    # 检测到语音开始
                                    self.is_speaking = True
                                    log("检测到语音开始...")
                                elif segment[0] == -1 and segment[1] != -1:
                                    # 检测到语音结束
                                    self.is_speaking = False
                                    log("检测到语音结束...")
                                    # 处理积累的语音缓冲区
                                    if len(self.speech_buffer) > 0:
                                        self.process_asr_buffer(is_final=True)

                        # 如果正在说话，将当前块添加到语音缓冲区
                        if self.is_speaking:
                            self.speech_buffer = np.append(self.speech_buffer, vad_chunk)
                else:
                    # 不使用VAD时，总是处于"说话"状态
                    self.is_speaking = True

                # 如果语音缓冲区足够大，进行ASR处理
                if len(self.speech_buffer) >= self.asr_chunk_samples:
                    self.process_asr_buffer()

                # 短暂休眠以减少CPU使用
                time.sleep(0.01)
            except Exception as e:
                log(f"音频处理错误: {e}")

    def process_asr_buffer(self, is_final=False):
        """处理语音缓冲区进行ASR识别"""
        if self.asr_model is None:
            return

        try:
            # 如果没有足够的样本而且不是最终处理，则返回
            if len(self.speech_buffer) < self.asr_chunk_samples and not is_final:
                return

            # 如果不是最终处理，提取一个ASR块
            if not is_final:
                asr_chunk = self.speech_buffer[:self.asr_chunk_samples]
                self.speech_buffer = self.speech_buffer[self.asr_chunk_samples:]
            else:
                # 如果是最终处理，使用整个缓冲区
                asr_chunk = self.speech_buffer
                self.speech_buffer = np.array([], dtype=np.float32)

            # 使用ASR模型处理
            if len(asr_chunk) > 0:
                asr_res = self.asr_model.generate(
                    input=asr_chunk,
                    cache=self.asr_cache,
                    is_final=is_final,
                    chunk_size=self.asr_chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back
                )

                # 如果有识别结果，处理并应用标点
                if asr_res[0]["text"]:
                    text = asr_res[0]["text"]
                    self.raw_transcript += text

                    # 应用标点恢复 (如果启用)
                    if self.use_punc and self.punc_model is not None:
                        punc_res = self.punc_model.generate(input=self.raw_transcript)
                        if punc_res:
                            punctuated_text = punc_res[0]["text"]
                            # 更新完整转写并添加到结果队列
                            self.complete_transcript = punctuated_text
                            self.result_queue.put((text, punctuated_text))
                    else:
                        # 不使用标点恢复时，直接使用原始文本
                        self.complete_transcript = self.raw_transcript
                        self.result_queue.put((text, self.raw_transcript))
        except Exception as e:
            log(f"ASR处理错误: {e}")

    def start(self):
        """开始语音识别"""
        if self.running:
            return False

        # 确保ASR模型已加载
        if not self.ensure_asr_model_loaded():
            log("无法启动语音识别：ASR模型加载失败")
            return False

        # 根据需要加载其他模型
        if self.use_vad:
            self.load_vad_model_if_needed()

        if self.use_punc:
            self.load_punc_model_if_needed()

        # 重置状态变量
        self.running = True
        self.vad_cache = {}
        self.asr_cache = {}
        self.complete_transcript = ""
        self.raw_transcript = ""
        self.is_speaking = not self.use_vad  # 不使用VAD时默认为说话状态
        self.speech_buffer = np.array([], dtype=np.float32)

        # 启动处理线程
        self.process_thread = threading.Thread(target=self.process_audio_thread)
        self.process_thread.daemon = True
        self.process_thread.start()

        # 启动音频流
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms音频块
            )
            self.stream.start()
        except Exception as e:
            log(f"启动音频流失败: {e}")
            self.running = False
            return False

        # 显示启动状态
        features = []
        if self.use_vad and self.vad_model is not None:
            features.append("语音端点检测")
        features.append("语音识别")
        if self.use_punc and self.punc_model is not None:
            features.append("标点恢复")

        log(f"语音识别已启动，包含" + "、".join(features) + "功能")
        log("请对着麦克风说话...")
        return True

    # 更新 stop 方法
    def stop(self):
        """停止实时口译 - 更新句子级管理"""
        if not self.is_running:
            return False

        print("停止实时口译...")

        # 设置停止标志
        self.is_running = False

        # 停止ASR
        final_transcript = self.asr.stop()

        # 处理最终文本
        if final_transcript:
            # 发送最终原文更新事件
            self._send_event("source_text_update", {
                "text": final_transcript
            })

            # 如果源语言和目标语言相同，则不需要翻译
            if self.source_lang == self.target_lang:
                final_translation = final_transcript
            else:
                # 翻译最终文本
                final_translation = self.translator.translate(
                    final_transcript,
                    self.source_lang,
                    self.target_lang
                )

            if final_translation:
                # 发送最终译文更新事件
                self._send_event("translated_text_update", {
                    "text": final_translation
                })

                # 找出未播放的句子
                final_sentences = self._extract_new_sentences(final_translation)

                # 如果有未播放的句子，播放它们
                if final_sentences and not self.tts.is_playing():
                    # 将未播放句子组合成文本
                    sentences_to_play = " ".join(final_sentences)

                    # 播放未播放句子
                    self.tts.speak(sentences_to_play)

                    # 发送TTS播放事件
                    self._send_event("tts_play", {
                        "text": sentences_to_play
                    })

        # 等待处理线程结束
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)

        # 发送停止事件
        self._send_event("system_stop", {})

        return True


#----------------------------------------
# 翻译模块 (从translator.py导入)
#----------------------------------------
class SimpleTranslator:
    """简化版翻译模块"""

    def __init__(self, app_id, api_secret, api_key):
        """初始化翻译模块"""
        log("初始化翻译模块...")
        self.app_id = app_id
        self.api_secret = api_secret
        self.api_key = api_key
        self.url = 'https://itrans.xf-yun.com/v1/its'

    def translate(self, text, from_lang="cn", to_lang="en"):
        """翻译文本"""
        if not text or from_lang == to_lang:
            return text

        try:
            log(f"翻译文本: '{text[:20]}...' 从 {from_lang} 到 {to_lang}")

            # 生成URL和请求头
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            # 解析URL
            url_parts = self.parse_url(self.url)
            host = url_parts["host"]
            path = url_parts["path"]

            # 生成签名
            signature_origin = f"host: {host}\ndate: {date}\nPOST {path} HTTP/modules.modules"
            signature_sha = hmac.new(
                self.api_secret.encode('utf-8'),
                signature_origin.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
            signature_sha = base64.b64encode(signature_sha).decode('utf-8')

            # 构建authorization
            authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')

            # 构建URL
            request_url = self.url + "?" + urlencode({
                "host": host,
                "date": date,
                "authorization": authorization
            })

            # 构建请求体
            body = {
                "header": {
                    "app_id": self.app_id,
                    "status": 3
                },
                "parameter": {
                    "its": {
                        "from": from_lang,
                        "to": to_lang,
                        "result": {}
                    }
                },
                "payload": {
                    "input_data": {
                        "encoding": "utf8",
                        "status": 3,
                        "text": base64.b64encode(text.encode("utf-8")).decode('utf-8')
                    }
                }
            }

            # 准备请求头
            headers = {
                'content-type': "application/json",
                'host': host,
                'app_id': self.app_id
            }

            # 发送请求
            response = requests.post(
                request_url,
                data=json.dumps(body),
                headers=headers,
                timeout=5.0
            )

            # 解析响应
            if response.status_code == 200:
                result = json.loads(response.content.decode())

                if 'payload' in result and 'result' in result['payload'] and 'text' in result['payload']['result']:
                    translated_text_base64 = result['payload']['result']['text']
                    translated_text = base64.b64decode(translated_text_base64).decode()

                    try:
                        # 尝试解析JSON响应
                        json_result = json.loads(translated_text)
                        if 'trans_result' in json_result and 'dst' in json_result['trans_result']:
                            return json_result['trans_result']['dst']
                        elif 'dst' in json_result:
                            return json_result['dst']
                        else:
                            return translated_text
                    except:
                        # 不是JSON格式，返回原始文本
                        return translated_text
                else:
                    log(f"翻译API错误: {result}")
            else:
                log(f"翻译请求失败，状态码: {response.status_code}")

            return None

        except Exception as e:
            log(f"翻译过程出错: {str(e)}")
            return None

    def parse_url(self, url):
        """解析URL"""
        stidx = url.index("://")
        host = url[stidx + 3:]
        schema = url[:stidx + 3]
        edidx = host.index("/")
        if edidx <= 0:
            raise Exception("invalid request url:" + url)
        path = host[edidx:]
        host = host[:edidx]
        return {"host": host, "path": path, "schema": schema}


#----------------------------------------
# 语音合成模块 (基于edge_TTS.py)
#----------------------------------------
class TTSManager:
    """语音合成管理器"""

    def __init__(self):
        """初始化语音合成管理器"""
        if not EDGE_TTS_AVAILABLE:
            log("警告: Edge TTS库未安装，语音合成功能将无法使用")
            self.available = False
            return

        log("初始化TTS模块...")
        self.available = True
        self.voices_manager = None
        self.voices_by_language = {}  # 按语言存储可用声音
        self.voice_name_map = {}  # 声音ID到友好名称的映射
        self.languages = []  # 支持的语言列表
        self.current_voice = None  # 当前使用的声音
        self.tts_speed = "0%"  # 语速设置
        self.tts_volume = "+0%"  # 音量设置
        self.tts_pitch = "+0Hz"  # 音调设置

        # TTS任务队列和处理线程
        self.tts_queue = asyncio.Queue()
        self.last_tts_text = ""  # 上次合成的文本
        self.is_playing = False  # 是否正在播放
        self.stop_event = threading.Event()

        # 映射翻译API语言代码到TTS语言代码
        self.lang_code_map = {
            "en": "en-US",
            "cn": "zh-CN",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "fr": "fr-FR",
            "de": "de-DE",
            "es": "es-ES",
            "ru": "ru-RU",
            "it": "it-IT"
        }

        # 初始化pygame混音器
        if self.available:
            try:
                mixer.init()
                log("音频播放器初始化成功")
            except Exception as e:
                log(f"音频播放器初始化失败: {e}")

        # 启动异步加载声音列表的线程
        self.load_thread = threading.Thread(target=self._load_voices)
        self.load_thread.daemon = True
        self.load_thread.start()

    def _load_voices(self):
        """加载可用声音列表的线程函数"""
        try:
            # 创建一个新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 执行异步加载
            loop.run_until_complete(self._async_load_voices())

            # 关闭事件循环
            loop.close()
        except Exception as e:
            log(f"加载TTS声音列表时出错: {e}")

    async def _async_load_voices(self):
        """异步加载可用声音列表"""
        try:
            log("正在加载TTS声音列表...")
            self.voices_manager = await edge_tts.VoicesManager.create()

            # 按语言组织声音
            for voice in self.voices_manager.voices:
                lang_code = voice["Locale"]

                # 添加到按语言组织的字典
                if lang_code not in self.voices_by_language:
                    self.voices_by_language[lang_code] = []

                # 添加声音到相应语言
                self.voices_by_language[lang_code].append(voice)

                # 添加到声音名称映射
                self.voice_name_map[voice["ShortName"]] = voice.get("FriendlyName", voice["ShortName"])

            # 创建支持的语言列表
            self.languages = sorted(self.voices_by_language.keys())

            # 默认选择英语声音
            if "en-US" in self.voices_by_language and self.voices_by_language["en-US"]:
                self.current_voice = self.voices_by_language["en-US"][0]["ShortName"]

            log(f"TTS声音加载完成，支持{len(self.languages)}种语言")

            # 启动TTS处理线程
            self.tts_thread = threading.Thread(target=self._run_tts_loop)
            self.tts_thread.daemon = True
            self.tts_thread.start()

            return True
        except Exception as e:
            log(f"加载TTS声音列表出错: {e}")
            return False

    def _run_tts_loop(self):
        """运行TTS处理循环的线程函数"""
        try:
            # 创建一个新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 运行TTS任务处理循环
            loop.run_until_complete(self._process_tts_queue())

            # 关闭事件循环
            loop.close()
        except Exception as e:
            log(f"TTS处理线程出错: {e}")

    async def _process_tts_queue(self):
        """处理TTS队列的异步函数"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取TTS任务
                text, voice = await self.tts_queue.get()

                # 检查是否是重复文本
                if text == self.last_tts_text:
                    self.tts_queue.task_done()
                    await asyncio.sleep(0.1)
                    continue

                # 更新上次合成的文本
                self.last_tts_text = text

                # 合成并播放
                await self._synthesize_and_play(text, voice)

                # 标记任务完成
                self.tts_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"处理TTS任务时出错: {e}")

            # 短暂休眠以减少CPU使用
            await asyncio.sleep(0.1)

    # modules/tts_module.py 中 _synthesize_and_play 方法的修正版本

    async def _synthesize_and_play(self, text):
        """合成并播放文本的异步函数"""
        if not text or not self.current_voice:
            return

        print(f"使用音色 {self.current_voice} 合成文本: '{text[:30]}...'")
        self.is_playing_flag = True

        try:
            # 修改这里，移除 custom_headers 参数
            if "<speak" in text:
                # 如果已经是 SSML 格式，直接使用
                communicate = edge_tts.Communicate(text, self.current_voice)
            else:
                # 构建简单的文本请求，不使用完整的 SSML
                communicate = edge_tts.Communicate(
                    text,
                    self.current_voice,
                    rate=self.tts_speed,
                    volume=self.tts_volume,
                    pitch=self.tts_pitch
                )

            # 收集音频数据
            audio_data = bytes()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

                # 检查是否需要停止
                if self.stop_event.is_set():
                    break

            if not audio_data or self.stop_event.is_set():
                self.is_playing_flag = False
                return

            # 播放音频
            await self._play_audio_from_memory(audio_data)

        except Exception as e:
            print(f"TTS合成或播放出错: {e}")
        finally:
            self.is_playing_flag = False

    async def _play_audio_from_memory(self, audio_data):
        """从内存播放音频数据的异步函数"""
        try:
            # 创建内存缓冲区并加载音频数据
            audio_io = io.BytesIO(audio_data)

            # 加载和播放
            mixer.music.load(audio_io)
            mixer.music.play()

            # 等待播放完成
            while mixer.music.get_busy():
                await asyncio.sleep(0.1)
                if self.stop_event.is_set():
                    mixer.music.stop()
                    break

            # 显式关闭以确保资源被释放
            mixer.music.unload()

        except Exception as e:
            log(f"播放音频出错: {e}")

    def get_voices_for_language(self, language_code):
        """获取指定语言的可用声音列表"""
        if not self.available or not self.voices_manager:
            return []

        # 映射翻译API语言代码到TTS语言代码
        tts_lang_code = self.lang_code_map.get(language_code)
        if not tts_lang_code or tts_lang_code not in self.voices_by_language:
            return []

        # 返回格式化的声音列表 [("voice_id", "友好名称"), ...]
        return [(voice["ShortName"], voice.get("FriendlyName", voice["ShortName"]))
                for voice in self.voices_by_language[tts_lang_code]]

    def set_voice_for_language(self, language_code):
        """根据语言代码设置合适的声音"""
        if not self.available or not self.voices_manager:
            return False

        # 映射翻译API语言代码到TTS语言代码
        tts_lang_code = self.lang_code_map.get(language_code)
        if not tts_lang_code or tts_lang_code not in self.voices_by_language:
            log(f"没有找到语言{language_code}的TTS声音")
            return False

        # 选择该语言的第一个声音
        if self.voices_by_language[tts_lang_code]:
            self.current_voice = self.voices_by_language[tts_lang_code][0]["ShortName"]
            log(f"已为语言{language_code}设置声音: {self.current_voice}")
            return True

        return False

    def set_voice(self, voice_id):
        """设置当前使用的声音"""
        if not self.available:
            return False

        self.current_voice = voice_id
        log(f"已设置TTS声音: {voice_id}")
        return True

    def set_speech_parameters(self, speed=None, volume=None, pitch=None):
        """设置语音参数"""
        if speed is not None:
            self.tts_speed = f"{speed}%"
        if volume is not None:
            self.tts_volume = f"{'+' if volume >= 0 else ''}{volume}%"
        if pitch is not None:
            self.tts_pitch = f"{'+' if pitch >= 0 else ''}{pitch}Hz"

        log(f"设置TTS参数: 语速={self.tts_speed}, 音量={self.tts_volume}, 音调={self.tts_pitch}")

    def speak(self, text, voice=None):
        """将文本加入TTS队列"""
        if not self.available or not self.voices_manager:
            return False

        # 如果未指定声音，使用当前设置的声音
        if voice is None:
            voice = self.current_voice

        if not voice:
            log("错误: 未设置TTS声音")
            return False

        # 处理文本
        # 除去多余空格，分割过长文本
        text = re.sub(r'\s+', ' ', text).strip()

        # 限制文本长度，防止过长导致超时
        if len(text) > 1000:
            text = text[:1000] + "..."

        # 加入队列
        future = asyncio.run_coroutine_threadsafe(
            self.tts_queue.put((text, voice)),
            asyncio.get_event_loop()
        )
        future.result()  # 等待添加到队列完成

        return True

    def stop(self):
        """停止所有TTS活动"""
        if not self.available:
            return

        self.stop_event.set()

        # 停止正在播放的音频
        if mixer.get_init() and mixer.music.get_busy():
            mixer.music.stop()

        log("TTS系统已停止")


#----------------------------------------
# 文本分析模块 (用于智能分段)
#----------------------------------------
class TextSegmenter:
    """文本分析和分段工具"""

    @staticmethod
    def has_sentence_end(text):
        """检查文本是否以句号、问号、感叹号等结束"""
        if not text:
            return False

        # 包含中英文标点
        sentence_end_chars = ['.', '?', '!', '。', '？', '！', '…']
        return text[-1] in sentence_end_chars

    @staticmethod
    def count_sentences(text):
        """计算文本中句子的数量（简单实现）"""
        if not text:
            return 0

        # 根据常见句子结束符号分割
        pattern = r'[.。!！?？…]+'
        sentences = re.split(pattern, text)

        # 过滤空句子
        return sum(1 for s in sentences if s.strip())

    @staticmethod
    def is_good_tts_trigger_point(current_text, new_text):
        """判断是否是合适的TTS触发点"""
        # 如果当前没有文本，直接返回False
        if not current_text:
            return False

        # modules. 句子结束标志
        if TextSegmenter.has_sentence_end(current_text):
            return True

        # 2. 句子数量达到阈值
        if TextSegmenter.count_sentences(current_text) >= 2:
            return True

        # 3. 文本长度达到阈值（约50个字符）
        if len(current_text) >= 50:
            return True

        # 4. 新文本与当前文本差异巨大（可能是新的翻译）
        if new_text and len(new_text) > 2 * len(current_text):
            return True

        # 默认不触发
        return False


#----------------------------------------
# 集成GUI应用
#----------------------------------------
class ASRTranslatorTTSApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # 翻译API配置
        self.APP_ID = "86c79fb7"
        self.API_SECRET = "MDY3ZGFkYWEyZDBiOTJkOGIyOTllOWMz"
        self.API_KEY = "f4369644e37eddd43adfe436e7904cf1"

        # 配置主窗口
        self.title("语音识别翻译与自动语音合成系统")
        self.geometry("800x700")
        self.minsize(700, 500)

        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("TCheckbutton", font=("Arial", 10))
        self.style.configure("Bold.TLabel", font=("Arial", 10, "bold"))

        # 状态变量
        self.is_recording = False
        self.source_lang = "cn"  # 默认源语言：中文
        self.target_lang = "en"  # 默认目标语言：英语
        self.auto_tts = tk.BooleanVar(value=True)  # 自动TTS开关
        self.last_translation = ""  # 最后一次翻译的文本
        self.last_tts_trigger_time = 0  # 上次TTS触发时间

        # 语言映射
        self.languages = {
            "cn": "中文",
            "en": "英文",
            "ja": "日语",
            "ko": "韩语",
            "fr": "法语",
            "es": "西班牙语",
            "ru": "俄语",
            "de": "德语",
            "it": "意大利语"
        }

        # 初始化组件
        self.create_widgets()

        # 初始化翻译模块
        self.translator = SimpleTranslator(
            app_id=self.APP_ID,
            api_secret=self.API_SECRET,
            api_key=self.API_KEY
        )

        # 初始化TTS模块
        self.tts_manager = TTSManager()

        # 初始化语音识别模块（如果库可用）
        if FUNASR_AVAILABLE:
            self.asr = FastLoadASR(use_vad=True, use_punc=True)
            self.update_status("模型加载中，请稍候...")
        else:
            self.asr = None
            self.update_status("语音识别库未安装，仅翻译功能可用")
            self.record_button.config(state="disabled")

        # 启动日志更新线程
        self.update_log_thread = threading.Thread(target=self.update_log, daemon=True)
        self.update_log_thread.start()

        # 启动结果更新线程
        if FUNASR_AVAILABLE:
            self.update_result_thread = threading.Thread(target=self.update_result, daemon=True)
            self.update_result_thread.start()

        # 在关闭窗口时停止所有进程
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 定时检查TTS是否可用并更新UI
        self.after(1000, self.check_tts_availability)

    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 顶部控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 语言选择区域
        lang_frame = ttk.Frame(control_frame)
        lang_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 源语言标签（仅显示）
        ttk.Label(lang_frame, text="源语言:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.source_lang_var = tk.StringVar(value=self.languages[self.source_lang])
        ttk.Label(lang_frame, textvariable=self.source_lang_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # 目标语言下拉菜单
        ttk.Label(lang_frame, text="目标语言:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.target_lang_var = tk.StringVar(value=self.target_lang)

        # 将语言代码映射为显示名称
        lang_options = [f"{code} ({name})" for code, name in self.languages.items() if code != self.source_lang]

        self.target_lang_combo = ttk.Combobox(lang_frame, values=lang_options, state="readonly", width=15)
        self.target_lang_combo.set(f"{self.target_lang} ({self.languages[self.target_lang]})")
        self.target_lang_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.target_lang_combo.bind("<<ComboboxSelected>>", self.on_language_change)

        # 录音按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)

        self.record_button = ttk.Button(button_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.pack(side=tk.RIGHT, padx=5)

        # TTS控制区域
        tts_control_frame = ttk.LabelFrame(main_frame, text="语音合成设置", padding=5)
        tts_control_frame.pack(fill=tk.X, pady=(0, 10))

        # TTS启用开关
        self.tts_enable_check = ttk.Checkbutton(
            tts_control_frame,
            text="自动语音合成",
            variable=self.auto_tts,
            command=self.on_auto_tts_toggle
        )
        self.tts_enable_check.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # TTS声音选择
        ttk.Label(tts_control_frame, text="声音:").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(tts_control_frame, textvariable=self.voice_var, state="readonly", width=25)
        self.voice_combo.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_change)

        # TTS语速设置
        ttk.Label(tts_control_frame, text="语速:").grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.speed_var = tk.IntVar(value=0)  # 默认速度 0%
        self.speed_scale = ttk.Scale(
            tts_control_frame,
            from_=-50,
            to=50,
            variable=self.speed_var,
            command=self.on_speed_change
        )
        self.speed_scale.grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.speed_label = ttk.Label(tts_control_frame, text="0%", width=5)
        self.speed_label.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)

        # TTS音量设置
        ttk.Label(tts_control_frame, text="音量:").grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        self.volume_var = tk.IntVar(value=0)  # 默认音量 0%
        self.volume_scale = ttk.Scale(
            tts_control_frame,
            from_=-50,
            to=50,
            variable=self.volume_var,
            command=self.on_volume_change
        )
        self.volume_scale.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.volume_label = ttk.Label(tts_control_frame, text="0%", width=5)
        self.volume_label.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        # TTS立即播放按钮
        self.tts_play_button = ttk.Button(
            tts_control_frame,
            text="立即朗读",
            command=self.play_current_translation
        )
        self.tts_play_button.grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky=tk.E)

        # 状态条
        self.status_var = tk.StringVar(value="系统就绪")
        status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, padx=5, pady=2)

        # 创建中间部分的Notebook（选项卡）
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 创建"结果"选项卡
        results_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(results_frame, text="语音识别结果")

        # 原文区域
        original_frame = ttk.LabelFrame(results_frame, text=f"原文 ({self.languages[self.source_lang]})")
        original_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.original_text = scrolledtext.ScrolledText(original_frame, wrap=tk.WORD, height=5)
        self.original_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 翻译区域
        translation_frame = ttk.LabelFrame(results_frame, text=f"译文 ({self.languages[self.target_lang]})")
        translation_frame.pack(fill=tk.BOTH, expand=True)

        self.translation_text = scrolledtext.ScrolledText(translation_frame, wrap=tk.WORD, height=5)
        self.translation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建"日志"选项卡
        log_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(log_frame, text="系统日志")

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 初始状态下禁用TTS控件
        self.disable_tts_controls()

    def check_tts_availability(self):
        """检查TTS是否可用并更新UI"""
        if self.tts_manager.available and self.tts_manager.voices_manager:
            # 更新TTS声音下拉菜单
            self.update_voice_combobox()
            # 启用TTS控件
            self.enable_tts_controls()
        else:
            # 1秒后再次检查
            self.after(1000, self.check_tts_availability)

    def update_voice_combobox(self):
        """更新TTS声音下拉菜单"""
        # 获取当前目标语言的可用声音
        voices = self.tts_manager.get_voices_for_language(self.target_lang)

        if voices:
            # 构建下拉菜单选项
            voice_options = [f"{name}" for voice_id, name in voices]
            self.voice_combo['values'] = voice_options

            # 设置默认选择
            if self.tts_manager.current_voice:
                # 查找当前声音对应的友好名称
                current_name = self.tts_manager.voice_name_map.get(
                    self.tts_manager.current_voice,
                    self.tts_manager.current_voice
                )
                if current_name in voice_options:
                    self.voice_combo.set(current_name)
                else:
                    # 如果当前声音不在列表中，选择第一个
                    self.voice_combo.set(voice_options[0])
                    self.on_voice_change()
        else:
            self.voice_combo['values'] = ["无可用声音"]
            self.voice_combo.set("无可用声音")
            self.disable_tts_controls()

    def enable_tts_controls(self):
        """启用TTS控件"""
        self.tts_enable_check.config(state="normal")
        if self.auto_tts.get():
            self.voice_combo.config(state="readonly")
            self.speed_scale.config(state="normal")
            self.volume_scale.config(state="normal")
            self.tts_play_button.config(state="normal")
        else:
            self.voice_combo.config(state="readonly")
            self.speed_scale.config(state="normal")
            self.volume_scale.config(state="normal")
            self.tts_play_button.config(state="normal")

    def disable_tts_controls(self):
        """禁用TTS控件"""
        if not EDGE_TTS_AVAILABLE:
            self.tts_enable_check.config(state="disabled")
        self.voice_combo.config(state="disabled")
        self.speed_scale.config(state="disabled")
        self.volume_scale.config(state="disabled")
        self.tts_play_button.config(state="disabled")

    def on_language_change(self, event):
        """处理语言选择变化"""
        selection = self.target_lang_combo.get()
        self.target_lang = selection.split()[0]  # 从显示格式中提取语言代码

        # 更新翻译区域标题
        ttk.Label(self.notebook.nametowidget(self.notebook.select()).winfo_children()[1],
                 text=f"译文 ({self.languages[self.target_lang]})").config(
                     text=f"译文 ({self.languages[self.target_lang]})"
                 )

        # 更新TTS声音选择
        if self.tts_manager.available and self.tts_manager.voices_manager:
            self.tts_manager.set_voice_for_language(self.target_lang)
            self.update_voice_combobox()

        self.update_status(f"目标语言已更改为: {self.languages[self.target_lang]}")

    def on_auto_tts_toggle(self):
        """处理自动TTS开关状态变化"""
        if self.auto_tts.get():
            self.update_status("自动语音合成已启用")
        else:
            self.update_status("自动语音合成已禁用")

    def on_voice_change(self, event=None):
        """处理TTS声音选择变化"""
        if not self.tts_manager.available or not self.tts_manager.voices_manager:
            return

        # 获取选择的声音友好名称
        selected_name = self.voice_var.get()

        # 查找对应的声音ID
        voices = self.tts_manager.get_voices_for_language(self.target_lang)
        for voice_id, name in voices:
            if name == selected_name:
                self.tts_manager.set_voice(voice_id)
                self.update_status(f"已设置语音: {name}")
                break

    def on_speed_change(self, value):
        """处理TTS语速变化"""
        speed = int(float(value))
        self.speed_label.config(text=f"{speed}%")
        if self.tts_manager.available:
            self.tts_manager.set_speech_parameters(speed=speed)

    def on_volume_change(self, value):
        """处理TTS音量变化"""
        volume = int(float(value))
        self.volume_label.config(text=f"{volume}%")
        if self.tts_manager.available:
            self.tts_manager.set_speech_parameters(volume=volume)

    def play_current_translation(self):
        """播放当前翻译文本"""
        if not self.tts_manager.available or not self.tts_manager.voices_manager:
            self.update_status("TTS系统不可用")
            return

        # 获取当前翻译文本
        text = self.translation_text.get(1.0, tk.END).strip()
        if not text:
            self.update_status("没有可朗读的文本")
            return

        # 播放文本
        self.tts_manager.speak(text)
        self.update_status("正在朗读译文...")

    def toggle_recording(self):
        """切换录音状态"""
        if not FUNASR_AVAILABLE:
            self.update_status("错误: 语音识别库未安装")
            return

        if not self.is_recording:
            # 开始录音
            self.start_recording()
        else:
            # 停止录音
            self.stop_recording()

    def start_recording(self):
        """开始录音"""
        # 检查ASR模型是否已加载
        if not self.asr.ensure_asr_model_loaded():
            self.update_status("错误: ASR模型未加载")
            return

        # 清空结果文本
        self.original_text.delete(1.0, tk.END)
        self.translation_text.delete(1.0, tk.END)
        self.last_translation = ""

        # 更新UI
        self.is_recording = True
        self.record_button.config(text="停止录音")
        self.update_status("正在录音...")

        # 启动ASR
        success = self.asr.start()
        if not success:
            self.is_recording = False
            self.record_button.config(text="开始录音")
            self.update_status("启动语音识别失败")
            return

        # 切换到结果选项卡
        self.notebook.select(0)

    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return

        # 更新UI
        self.is_recording = False
        self.record_button.config(text="开始录音")
        self.update_status("正在处理最终结果...")

        # 停止ASR
        self.asr.stop()

        # 处理最终结果
        final_text = self.asr.complete_transcript
        if final_text:
            # 更新原文
            self.original_text.delete(1.0, tk.END)
            self.original_text.insert(tk.END, final_text)

            # 翻译最终文本
            final_translation = self.translator.translate(
                text=final_text,
                from_lang=self.source_lang,
                to_lang=self.target_lang
            )

            if final_translation:
                # 更新译文
                self.translation_text.delete(1.0, tk.END)
                self.translation_text.insert(tk.END, final_translation)

                # 如果启用了自动TTS，则朗读最终翻译
                if self.auto_tts.get() and self.tts_manager.available:
                    self.tts_manager.speak(final_translation)

        self.update_status("录音已停止，处理完成")

    def update_log(self):
        """更新日志显示的线程函数"""
        while True:
            try:
                # 从队列获取日志消息
                while not log_queue.empty():
                    message = log_queue.get_nowait()

                    # 更新日志文本
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)  # 滚动到最新内容

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"更新日志错误: {e}")

    def update_result(self):
        """更新识别和翻译结果的线程函数"""
        while True:
            try:
                if FUNASR_AVAILABLE and hasattr(self, 'asr') and self.asr and self.is_recording:
                    # 检查ASR结果队列
                    while not self.asr.result_queue.empty():
                        _, text = self.asr.result_queue.get_nowait()

                        # 更新原文
                        self.original_text.delete(1.0, tk.END)
                        self.original_text.insert(tk.END, text)

                        # 翻译文本
                        translated = self.translator.translate(
                            text=text,
                            from_lang=self.source_lang,
                            to_lang=self.target_lang
                        )

                        # 更新译文
                        if translated:
                            self.translation_text.delete(1.0, tk.END)
                            self.translation_text.insert(tk.END, translated)

                            # 检查是否需要触发TTS
                            current_time = time.time()
                            if (self.auto_tts.get() and self.tts_manager.available and
                                translated != self.last_translation and
                                not self.tts_manager.is_playing):

                                # 检查是否是合适的TTS触发点
                                is_trigger_point = TextSegmenter.is_good_tts_trigger_point(
                                    self.last_translation, translated
                                )

                                # 检查距离上次TTS的时间是否足够长（至少1秒）
                                time_since_last_tts = current_time - self.last_tts_trigger_time

                                if (is_trigger_point and time_since_last_tts >= 1.0) or time_since_last_tts >= 3.0:
                                    # 朗读翻译
                                    self.tts_manager.speak(translated)
                                    self.last_tts_trigger_time = current_time

                            # 更新最后的翻译文本
                            self.last_translation = translated

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"更新结果错误: {e}")

    def update_status(self, message):
        """更新状态栏消息"""
        self.status_var.set(message)
        log(message, show_time=False)

    def on_closing(self):
        """窗口关闭处理"""
        if hasattr(self, 'asr') and self.asr and self.asr.running:
            self.asr.stop()

        # 停止TTS系统
        if hasattr(self, 'tts_manager') and self.tts_manager.available:
            self.tts_manager.stop()

        self.destroy()


# 主函数
def main():
    """主函数"""
    # 打印系统信息
    print("=" * 80)
    print("🎤 语音识别翻译与自动语音合成系统")
    print("=" * 80)
    print(f"Python 版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")

    # 检查是否安装了必要的库
    if not FUNASR_AVAILABLE:
        print("警告: funasr或sounddevice未安装，语音识别功能将无法使用")
        print("请安装必要的库: pip install funasr sounddevice numpy requests")

    if not EDGE_TTS_AVAILABLE:
        print("警告: edge_tts或pygame未安装，语音合成功能将无法使用")
        print("请安装必要的库: pip install edge-tts pygame")

    # 创建并运行应用
    app = ASRTranslatorTTSApp()
    app.mainloop()


if __name__ == "__main__":
    main()