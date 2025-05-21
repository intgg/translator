# controller.py - 优化版控制器模块

import threading
import time
import re
from queue import Queue

# 导入配置和模块
import config
from modules.asr_module import EnhancedASR
from modules.translation_module import TranslationModule
from modules.tts_module import TTSModule
from modules.sentence_manager import SentenceManager


class InterpreterController:
    """实时口译系统主控制器"""

    def __init__(self):
        """初始化控制器"""
        print("初始化实时口译系统控制器...")

        # 状态变量
        self.is_running = False
        self.last_tts_time = 0

        # 翻译缓存 - 减少重复翻译请求
        self.translation_cache = {}
        self.translation_history = {}  # 句子ID到翻译历史的映射

        # 事件队列 - 用于向UI线程传递事件
        self.event_queue = Queue()

        # 语言设置
        self.source_lang = config.DEFAULT_SOURCE_LANG
        self.target_lang = config.DEFAULT_TARGET_LANG

        # 初始化模块
        self.asr = None
        self.translator = None
        self.tts = None

        # 初始化句子管理器
        self.sentence_manager = SentenceManager()

        # TTS触发配置参数（从config.py加载或使用默认值）
        self.TTS_STRONG_TRIGGER_TIME = getattr(config, 'TTS_STRONG_TRIGGER_TIME', 3.0)  # 强触发时间阈值（秒）
        self.TTS_MEDIUM_TRIGGER_TIME = getattr(config, 'TTS_MEDIUM_TRIGGER_TIME', 2.0)  # 中等触发时间阈值（秒）
        self.TTS_TEXT_LENGTH_THRESHOLD = getattr(config, 'TTS_TEXT_LENGTH_THRESHOLD', 50)  # 文本长度触发阈值（字符数）
        self.TTS_MEDIUM_TEXT_LENGTH = getattr(config, 'TTS_MEDIUM_TEXT_LENGTH', 30)  # 中等文本长度阈值（字符数）
        self.TTS_MIN_TEXT_LENGTH = getattr(config, 'TTS_MIN_TEXT_LENGTH', 15)  # 最小文本长度（字符数）
        self.TEXT_STABILITY_TIME = getattr(config, 'TEXT_STABILITY_TIME', 0.8)  # 文本稳定时间（秒）

        # 翻译稳定性跟踪
        self.last_translation = ""
        self.last_translation_time = 0
        self.translation_consecutive_matches = 0

    def initialize(self):
        """初始化各个模块"""
        print("正在初始化各模块...")

        # 初始化ASR模块
        self.asr = EnhancedASR(
            use_vad=config.ASR_USE_VAD,
            use_punc=config.ASR_USE_PUNC,
            disable_update=config.ASR_DISABLE_UPDATE
        )

        # 初始化翻译模块
        self.translator = TranslationModule(
            app_id=config.TRANSLATION_APP_ID,
            api_secret=config.TRANSLATION_API_SECRET,
            api_key=config.TRANSLATION_API_KEY
        )

        # 初始化TTS模块
        self.tts = TTSModule()

        # 等待TTS模块加载声音列表
        print("等待TTS模块加载声音列表...")
        time.sleep(2)  # 给TTS加载声音的时间

        # 设置默认语音
        self.tts.set_voice_for_language(self.target_lang)

        # 设置TTS参数
        self.tts.set_speech_parameters(
            speed=config.DEFAULT_TTS_SPEED,
            volume=config.DEFAULT_TTS_VOLUME,
            pitch=config.DEFAULT_TTS_PITCH
        )

        print("各模块初始化完成")
        return True

    def start(self, target_lang=None):
        """开始实时口译"""
        if self.is_running:
            print("系统已经在运行中")
            return False

        # 设置目标语言
        if target_lang:
            self.target_lang = target_lang
            # 为目标语言选择合适的TTS声音
            self.tts.set_voice_for_language(target_lang)

        print(
            f"启动实时口译: {config.LANGUAGE_DISPLAY_NAMES[self.source_lang]} -> {config.LANGUAGE_DISPLAY_NAMES[self.target_lang]}")

        # 重置状态
        self.last_tts_time = 0
        self.last_translation = ""
        self.last_translation_time = 0
        self.translation_consecutive_matches = 0

        # 清空事件队列
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except:
                pass

        # 重置句子管理器
        self.sentence_manager.clear()

        # 启动ASR (已包含清空历史数据)
        if not self.asr.start():
            print("启动语音识别失败")
            return False

        # 设置运行标志
        self.is_running = True

        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

        # 发送启动事件
        self._send_event("system_start", {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang
        })

        return True

    def stop(self):
        """停止实时口译"""
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

            # 处理最终文本，提取句子
            result = self.sentence_manager.process_text(final_transcript, True)  # 强制标记为有停顿
            self._log_decision(
                f"最终处理结果: {len(result['sentences_to_translate'])}个待翻译句子, {len(result['sentences_to_play'])}个待播放句子")

            # 翻译未翻译的句子
            for sentence in result['sentences_to_translate']:
                if self.source_lang == self.target_lang:
                    translation = sentence
                else:
                    translation = self._translate_with_cache(sentence)

                if translation:
                    # 更新翻译结果
                    self.sentence_manager.update_translation(sentence, translation)

            # 获取最终的译文
            if self.source_lang == self.target_lang:
                final_translation = final_transcript
            else:
                final_translation = self._translate_with_cache(final_transcript)

            if final_translation:
                # 发送最终译文更新事件
                self._send_event("translated_text_update", {
                    "text": final_translation
                })

                # 获取可播放的句子
                playable_sentences = self.sentence_manager._get_sentences_to_play()

                if playable_sentences:
                    # 停止当前可能在播放的TTS
                    self.tts.stop_current_playback()

                    # 播放最高优先级的句子
                    sentence_info = playable_sentences[0]
                    self._log_decision(
                        f"最终播放决策: 句子ID={sentence_info['id']}, 优先级={sentence_info['priority']:.1f}, 长度={sentence_info['length']}")
                    self.tts.speak(sentence_info['translation'])

                    # 标记为已播放
                    self.sentence_manager.mark_as_played(sentence_info['text'])

                    # 发送TTS播放事件
                    self._send_event("tts_play", {
                        "text": sentence_info['translation']
                    })

        # 等待处理线程结束
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)

        # 发送停止事件
        self._send_event("system_stop", {})

        return True

    def _process_loop(self):
        """处理循环 - 智能句子管理版"""
        while self.is_running:
            try:
                # 检查ASR结果队列
                if not self.asr.result_queue.empty():
                    # 获取ASR结果 (增量文本, 当前文本, 带标点文本)
                    new_text, current_text, punctuated_text = self.asr.result_queue.get()

                    # 发送原文更新事件 - 使用带标点的文本
                    self._send_event("source_text_update", {
                        "text": punctuated_text
                    })

                    # 处理文本，更新句子状态
                    result = self.sentence_manager.process_text(
                        punctuated_text,
                        self.asr.pause_detected
                    )

                    # 记录状态变化
                    for change in result['status_changes']:
                        self._log_decision(f"句子管理: {change}")

                    # 翻译需要翻译的句子
                    for sentence in result['sentences_to_translate']:
                        if self.source_lang == self.target_lang:
                            translation = sentence
                        else:
                            # 使用带缓存的翻译
                            translation = self._translate_with_cache(sentence)

                        if translation:
                            # 记录句子翻译决策
                            self._log_decision(f"翻译句段: '{sentence[:30]}...' -> '{translation[:30]}...'")

                            # 更新翻译结果
                            updated = self.sentence_manager.update_translation(sentence, translation)
                            if updated:
                                self._log_decision(f"更新了句段翻译")

                    # 如果源语言和目标语言相同，则不需要翻译
                    if self.source_lang == self.target_lang:
                        translated_text = punctuated_text
                    else:
                        # 翻译完整文本 (用于UI显示)
                        translated_text = self._translate_with_cache(punctuated_text)

                    if translated_text:
                        # 检查翻译稳定性
                        translation_stable = self._check_translation_stability(translated_text)

                        # 发送译文更新事件
                        self._send_event("translated_text_update", {
                            "text": translated_text
                        })

                    # 检查是否有句子可以播放
                    playable_sentences = result['sentences_to_play']

                    # 当前时间
                    current_time = time.time()

                    # 距离上次TTS的时间
                    time_since_last_tts = current_time - self.last_tts_time

                    if playable_sentences and not self.tts.is_playing():
                        # 判断是否应该触发TTS
                        should_play, reason = self._should_trigger_tts(playable_sentences[0], time_since_last_tts,
                                                                       translation_stable)

                        # 记录TTS触发判断
                        self._log_decision(f"TTS触发判断: {reason}")

                        # 如果决定播放
                        if should_play:
                            # 获取要播放的句子
                            sentence_info = playable_sentences[0]
                            translation_text = sentence_info['translation']

                            # 播放前记录详细信息
                            self._log_decision(
                                f"TTS触发: 句子ID={sentence_info['id']}, 优先级={sentence_info['priority']:.1f}, 长度={sentence_info['length']}")

                            # 如果正在播放，先停止
                            if self.tts.is_playing():
                                self.tts.stop_current_playback()

                            # 播放新文本
                            self.tts.speak(translation_text)
                            self.last_tts_time = current_time

                            # 标记为已播放
                            self.sentence_manager.mark_as_played(sentence_info['text'])

                            # 发送TTS播放事件
                            self._send_event("tts_play", {
                                "text": translation_text
                            })

                # 短暂休眠以减少CPU使用
                time.sleep(0.05)

            except Exception as e:
                print(f"处理循环错误: {e}")
                import traceback
                traceback.print_exc()

    def _translate_with_cache(self, text):
        """带缓存的翻译函数

        参数:
            text: 要翻译的文本

        返回:
            str: 翻译结果
        """
        # 检查缓存
        if text in self.translation_cache:
            # 获取缓存的翻译
            cached_translation = self.translation_cache[text]
            return cached_translation

        # 如果没有缓存，执行翻译
        translation = self.translator.translate(text, self.source_lang, self.target_lang)

        # 如果翻译成功，更新缓存
        if translation:
            self.translation_cache[text] = translation

            # 限制缓存大小，防止内存泄漏
            if len(self.translation_cache) > 100:  # 限制缓存最多100项
                # 移除最早的缓存项
                keys = list(self.translation_cache.keys())
                self.translation_cache.pop(keys[0])

        return translation

    def _check_translation_stability(self, new_translation):
        """检查翻译结果的稳定性

        参数:
            new_translation: 新的翻译结果

        返回:
            bool: 翻译是否稳定
        """
        current_time = time.time()

        # 检查翻译内容是否变化
        if new_translation == self.last_translation:
            # 连续匹配计数增加
            self.translation_consecutive_matches += 1

            # 如果连续匹配超过3次且间隔超过稳定阈值，判定为稳定
            time_since_last_update = current_time - self.last_translation_time
            if self.translation_consecutive_matches >= 3 and time_since_last_update >= self.TEXT_STABILITY_TIME:
                return True
        else:
            # 翻译变化，重置计数
            self.translation_consecutive_matches = 0
            self.last_translation = new_translation
            self.last_translation_time = current_time

        return False

    def _should_trigger_tts(self, sentence_info, time_since_last_tts, translation_stable):
        """智能判断是否应该触发TTS

        参数:
            sentence_info: 句子信息字典
            time_since_last_tts: 距离上次TTS的时间
            translation_stable: 翻译是否稳定

        返回:
            tuple(bool, str): (是否应该触发, 原因)
        """
        # 获取句子信息
        translation_text = sentence_info['translation']
        text_length = sentence_info['length']
        is_complete = sentence_info['is_complete']
        wait_time = sentence_info['time_since_creation']

        # 检查各条件

        # 1. 检查最小长度
        if text_length < self.TTS_MIN_TEXT_LENGTH:
            return False, f"文本过短 ({text_length} < {self.TTS_MIN_TEXT_LENGTH})"

        # 2. 检查间隔时间
        if time_since_last_tts < 1.0:  # 至少等待1秒
            return False, f"距上次播放时间过短 ({time_since_last_tts:.2f}s < 1.0s)"

        # 3. 检查翻译稳定性（优先考虑）
        if not translation_stable and text_length < self.TTS_TEXT_LENGTH_THRESHOLD and wait_time < self.TTS_MEDIUM_TRIGGER_TIME:
            return False, f"翻译尚未稳定且文本较短 ({text_length}字符)"

        # 强触发条件（满足任一条件立即触发）

        # 1. 完整句子（以句号等结尾）且长度达到最小要求
        if is_complete and text_length >= self.TTS_MIN_TEXT_LENGTH:
            return True, f"文本以句号结尾，且长度足够 ({text_length} >= {self.TTS_MIN_TEXT_LENGTH})"

        # 2. 文本超长
        if text_length > self.TTS_TEXT_LENGTH_THRESHOLD:
            return True, f"文本长度超过高阈值 ({text_length} > {self.TTS_TEXT_LENGTH_THRESHOLD})"

        # 3. 等待时间过长
        if (time_since_last_tts > self.TTS_STRONG_TRIGGER_TIME and
                text_length > self.TTS_MEDIUM_TEXT_LENGTH):
            return True, f"距上次播放时间较长 ({time_since_last_tts:.2f}s > {self.TTS_STRONG_TRIGGER_TIME}s)"

        # 4. 句子创建时间过长
        if wait_time > self.TTS_STRONG_TRIGGER_TIME and text_length > self.TTS_MEDIUM_TEXT_LENGTH:
            return True, f"句子等待时间过长 ({wait_time:.2f}s > {self.TTS_STRONG_TRIGGER_TIME}s)"

        # 5. 稳定的翻译和中等长度
        if translation_stable and text_length >= self.TTS_MEDIUM_TEXT_LENGTH:
            return True, f"翻译稳定且长度适中 ({text_length} >= {self.TTS_MEDIUM_TEXT_LENGTH})"

        # 默认不触发
        return False, "不满足任何触发条件"

    def _send_event(self, event_type, data):
        """发送事件到事件队列"""
        self.event_queue.put({
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        })

    def _log_decision(self, message):
        """记录决策日志"""
        self._send_event("log", {
            "type": "SENTENCE",
            "message": message
        })

    def set_target_language(self, language_code):
        """设置目标语言"""
        if language_code not in config.LANGUAGE_DISPLAY_NAMES:
            print(f"不支持的语言代码: {language_code}")
            return False

        self.target_lang = language_code

        # 设置对应的TTS声音
        self.tts.set_voice_for_language(language_code)

        # 清空翻译缓存
        self.translation_cache.clear()
        self.last_translation = ""
        self.translation_consecutive_matches = 0

        print(f"目标语言已设置为: {config.LANGUAGE_DISPLAY_NAMES[language_code]}")
        return True

    def set_tts_parameters(self, speed=None, volume=None, pitch=None):
        """设置TTS参数"""
        self.tts.set_speech_parameters(speed, volume, pitch)
        return True

    def stop_current_tts(self):
        """停止当前正在播放的TTS内容"""
        if self.tts and self.tts.available:
            result = self.tts.stop_current_playback()
            if result:
                # 发送TTS停止事件
                self._send_event("tts_stop", {})
            return result
        return False