# modules/tts_module.py - 语音合成模块 (更新)

import asyncio
import threading
import time
import io
import re
from queue import Queue

# 尝试导入TTS相关库
try:
    import edge_tts
    from pygame import mixer

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("警告: edge_tts或pygame未安装，语音合成功能将无法使用")


class TTSModule:
    """语音合成模块"""

    def __init__(self):
        """初始化语音合成模块"""
        if not TTS_AVAILABLE:
            print("警告: TTS库未安装")
            self.available = False
            return

        print("初始化TTS模块...")
        self.available = True
        self.voices_manager = None
        self.voices_by_language = {}  # 按语言存储可用声音
        self.current_voice = None  # 当前使用的声音

        # TTS参数设置
        self.tts_speed = "0%"  # 语速设置
        self.tts_volume = "+0%"  # 音量设置
        self.tts_pitch = "+0Hz"  # 音调设置

        # TTS任务队列和处理线程
        self.tts_queue = Queue()
        self.is_playing_flag = False
        self.stop_event = threading.Event()

        # 播放历史记录
        self.play_history = []
        self.max_history = 10

        # 映射语言代码到TTS语言代码
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
                print("音频播放器初始化成功")
            except Exception as e:
                print(f"音频播放器初始化失败: {e}")
                self.available = False
                return

        # 启动异步加载声音列表的线程
        self.load_thread = threading.Thread(target=self._load_voices)
        self.load_thread.daemon = True
        self.load_thread.start()

        # 启动TTS处理线程
        self.tts_thread = threading.Thread(target=self._tts_worker)
        self.tts_thread.daemon = True
        self.tts_thread.start()

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
            print(f"加载TTS声音列表时出错: {e}")

    async def _async_load_voices(self):
        """异步加载可用声音列表"""
        try:
            print("正在加载TTS声音列表...")
            self.voices_manager = await edge_tts.VoicesManager.create()

            # 按语言组织声音
            for voice in self.voices_manager.voices:
                lang_code = voice["Locale"]

                # 添加到按语言组织的字典
                if lang_code not in self.voices_by_language:
                    self.voices_by_language[lang_code] = []

                # 添加声音到相应语言
                self.voices_by_language[lang_code].append(voice)

            # 默认选择英语声音
            if "en-US" in self.voices_by_language and self.voices_by_language["en-US"]:
                self.current_voice = self.voices_by_language["en-US"][0]["ShortName"]

            print(f"TTS声音加载完成，支持{len(self.voices_by_language)}种语言")
            return True
        except Exception as e:
            print(f"加载TTS声音列表出错: {e}")
            return False

    def _tts_worker(self):
        """TTS处理线程函数"""
        try:
            # 创建一个新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 运行TTS任务处理循环
            loop.run_until_complete(self._process_tts_queue())

            # 关闭事件循环
            loop.close()
        except Exception as e:
            print(f"TTS处理线程出错: {e}")

    async def _process_tts_queue(self):
        """处理TTS队列的异步函数"""
        while not self.stop_event.is_set():
            try:
                # 检查队列是否为空
                if not self.tts_queue.empty():
                    # 获取TTS任务
                    text_info = self.tts_queue.get()

                    # 解析文本信息
                    if isinstance(text_info, dict):
                        text = text_info.get('text', '')
                        priority = text_info.get('priority', False)
                    else:
                        text = text_info
                        priority = False

                    # 检查是否要跳过这个任务
                    should_skip = False

                    # 检查是否最近播放过类似内容
                    for history_item in self.play_history:
                        if self._is_similar_text(text, history_item):
                            print(f"跳过类似内容: {text[:30]}...")
                            should_skip = True
                            break

                    if not should_skip:
                        # 合成并播放
                        await self._synthesize_and_play(text)

                        # 添加到播放历史
                        self._add_to_play_history(text)

                    # 标记任务完成
                    self.tts_queue.task_done()

                # 短暂休眠以减少CPU使用
                await asyncio.sleep(0.1)

            except Exception as e:
                print("处理TTS任务时出错: {0}".format(e))  # 避免使用f-string，兼容性更高

    def _is_similar_text(self, text1, text2):
        """检查两段文本是否非常相似

        参数:
            text1: 第一段文本
            text2: 第二段文本

        返回:
            bool: 是否相似
        """
        # 简单比较，如果一个文本包含另一个，并且长度相差不大
        if text1 in text2 or text2 in text1:
            len_diff = abs(len(text1) - len(text2))
            shorter_len = min(len(text1), len(text2))
            if shorter_len > 0 and len_diff / shorter_len < 0.2:  # 长度差异小于20%
                return True
        return False

    def _add_to_play_history(self, text):
        """添加文本到播放历史

        参数:
            text: 播放的文本
        """
        # 保持历史记录在最大限制内
        if len(self.play_history) >= self.max_history:
            self.play_history.pop(0)  # 移除最旧的记录

        # 添加新记录
        self.play_history.append(text)

    async def _synthesize_and_play(self, text):
        """合成并播放文本的异步函数 - 简化版本"""
        if not text or not self.current_voice:
            return

        print(f"使用音色 {self.current_voice} 合成文本: '{text[:30]}...'")
        self.is_playing_flag = True

        try:
            # 创建通信对象 - 简化版，不使用 SSML 和自定义标头
            communicate = edge_tts.Communicate(text, self.current_voice)

            # 设置语音参数
            if self.tts_speed != "0%":
                communicate.rate = self.tts_speed
            if self.tts_volume != "+0%":
                communicate.volume = self.tts_volume
            if self.tts_pitch != "+0Hz":
                communicate.pitch = self.tts_pitch

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
            import traceback
            traceback.print_exc()  # 打印完整堆栈跟踪
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
            print(f"播放音频出错: {e}")

    def get_voices_for_language(self, language_code):
        """获取指定语言的可用声音列表"""
        if not self.available or not self.voices_manager:
            return []

        # 映射翻译API语言代码到TTS语言代码
        tts_lang_code = self.lang_code_map.get(language_code)
        if not tts_lang_code or tts_lang_code not in self.voices_by_language:
            return []

        # 返回该语言的所有声音信息 [(voice_id, name, gender), ...]
        voices = []
        for voice in self.voices_by_language[tts_lang_code]:
            voice_id = voice["ShortName"]
            name = voice.get("FriendlyName", voice_id)
            gender = voice.get("Gender", "Unknown")
            voices.append((voice_id, name, gender))

        return voices

    def set_voice_for_language(self, language_code):
        """根据语言代码设置合适的声音"""
        if not self.available or not self.voices_manager:
            return False

        # 映射翻译API语言代码到TTS语言代码
        tts_lang_code = self.lang_code_map.get(language_code)
        if not tts_lang_code or tts_lang_code not in self.voices_by_language:
            print(f"没有找到语言{language_code}的TTS声音")
            return False

        # 选择该语言的第一个声音
        if self.voices_by_language[tts_lang_code]:
            self.current_voice = self.voices_by_language[tts_lang_code][0]["ShortName"]
            print(f"已为语言{language_code}设置声音: {self.current_voice}")
            return True

        return False

    def set_voice(self, voice_id):
        """设置当前使用的声音"""
        if not self.available:
            return False

        self.current_voice = voice_id
        print(f"已设置TTS声音: {voice_id}")
        return True

    def set_speech_parameters(self, speed=None, volume=None, pitch=None):
        """设置语音参数"""
        if speed is not None:
            self.tts_speed = f"{speed}%"
        if volume is not None:
            self.tts_volume = f"{'+' if volume >= 0 else ''}{volume}%"
        if pitch is not None:
            self.tts_pitch = f"{'+' if pitch >= 0 else ''}{pitch}Hz"

        print(f"设置TTS参数: 语速={self.tts_speed}, 音量={self.tts_volume}, 音调={self.tts_pitch}")

    def speak(self, text, clear_queue=False, priority=False):
        """将文本加入TTS队列 - 增加优先级选项"""
        if not self.available or not self.current_voice:
            return False

        # 如果需要清空队列
        if clear_queue:
            self.stop_current_playback()

        # 处理文本
        # 除去多余空格，分割过长文本
        text = re.sub(r'\s+', ' ', text).strip()

        # 跳过空文本
        if not text:
            return False

        # 限制文本长度，防止过长导致超时
        if len(text) > 1000:
            text = text[:1000] + "..."

        # 如果是优先级播放且当前正在播放，则先停止当前播放
        if priority and self.is_playing_flag:
            self.stop_current_playback()

        # 封装文本信息
        text_info = {
            'text': text,
            'priority': priority,
            'timestamp': time.time()
        }

        # 加入队列
        self.tts_queue.put(text_info)
        return True

    def is_playing(self):
        """检查是否正在播放"""
        return self.is_playing_flag

    def stop_current_playback(self):
        """停止当前播放并清空队列"""
        if not self.available:
            return False

        # 清空队列
        try:
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except:
                    pass
        except:
            pass

        # 如果正在播放，停止播放
        if self.is_playing_flag and mixer.get_init() and mixer.music.get_busy():
            mixer.music.stop()
            self.is_playing_flag = False
            print("已停止当前语音播放")
            return True

        return False

    def stop(self):
        """停止所有TTS活动"""
        if not self.available:
            return

        self.stop_event.set()

        # 停止正在播放的音频
        if mixer.get_init() and mixer.music.get_busy():
            mixer.music.stop()

        print("TTS系统已停止")