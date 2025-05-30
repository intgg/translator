import os
import sys

# 设置项目路径
project_root = os.getcwd()
os.environ["FUNASR_CACHE"] = os.path.join(project_root, "models", "cached_models")
os.environ["HF_HOME"] = os.path.join(project_root, "models", "hf_cache")
os.environ["MODELSCOPE_CACHE"] = os.path.join(project_root, "models", "modelscope_cache")

# 定义资源路径
resources_dir = os.path.join("resources")

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import asyncio
import threading
import queue
import time
import sounddevice as sd
import pygame
import traceback
import numpy as np

# 导入项目模块
try:
    from FunASR import FastLoadASR
except ImportError:
    print("警告: FunASR.py 未找到或无法导入。语音识别功能将不可用。")
    FastLoadASR = None

try:
    from translation_module import TranslationModule, LANGUAGE_CODES, LANGUAGE_NAMES

    # 替换为您的实际API密钥
    TRANSLATION_APP_ID = "86c79fb7"
    TRANSLATION_API_SECRET = "MDY3ZGFkYWEyZDBiOTJkOGIyOTllOWMz"
    TRANSLATION_API_KEY = "f4369644e37eddd43adfe436e7904cf1"
except ImportError:
    print("警告: translation_module.py 未找到或无法导入。翻译功能将不可用。")
    TranslationModule = None
    LANGUAGE_CODES = {"中文": "cn", "英语": "en"}
    LANGUAGE_NAMES = {"cn": "中文", "en": "英语"}

try:
    import edge_TTS
except ImportError:
    print("警告: edge_TTS.py 未找到或无法导入。语音合成功能将不可用。")
    edge_TTS = None


class LoadingAnimationTimer(QObject):
    """用于创建加载动画效果的计时器类"""
    
    def __init__(self, parent, base_text="⏳ 模型加载中", color="#F59E0B", interval=500):
        """
        初始化加载动画计时器
        
        参数:
            parent: 父窗口，必须有update_status方法
            base_text: 基础文本，点号将附加在此文本之后
            color: 文本颜色
            interval: 动画更新间隔（毫秒）
        """
        super().__init__()
        self.parent = parent
        self.base_text = base_text
        self.color = color
        self.interval = interval
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.dots_count = 0
        
    def start(self):
        """开始动画"""
        self.dots_count = 0
        self.update_animation()  # 立即显示初始状态
        self.timer.start(self.interval)
        
    def stop(self):
        """停止动画"""
        self.timer.stop()
        
    def update_animation(self):
        """更新动画状态"""
        self.dots_count = (self.dots_count % 3) + 1  # 循环 1-3 个点
        dots = "." * self.dots_count
        self.parent.update_status(f"{self.base_text}{dots}", self.color)


class TranslationCard(QFrame):
    """翻译结果卡片"""

    def __init__(self, time_str, original, translation):
        super().__init__()
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px;
                margin: 4px;
            }
        """)

        layout = QVBoxLayout()

        # 时间戳
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #64748B; font-size: 12px;")
        layout.addWidget(time_label)

        # 原文
        original_label = QLabel(f"原文：{original}")
        original_label.setWordWrap(True)
        original_label.setStyleSheet("color: #1E293B; font-size: 14px; margin-top: 8px;")
        layout.addWidget(original_label)

        # 译文
        translation_label = QLabel(f"译文：{translation}")
        translation_label.setWordWrap(True)
        translation_label.setStyleSheet("color: #1E3A8A; font-size: 14px; font-weight: 500; margin-top: 4px;")
        layout.addWidget(translation_label)

        self.setLayout(layout)


class WaveformWidget(QWidget):
    """音频波形显示控件"""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)
        self.setStyleSheet("background-color: #F1F5F9; border-radius: 4px;")
        self.volume_data = []
        self.max_samples = 50

    def update_volume(self, volume):
        """更新音量数据"""
        self.volume_data.append(volume)
        if len(self.volume_data) > self.max_samples:
            self.volume_data.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self.volume_data:
            return

        # 绘制波形
        pen = QPen(QColor("#3B82F6"), 2)
        painter.setPen(pen)

        width = self.width()
        height = self.height()
        center_y = height // 2

        if len(self.volume_data) > 1:
            step = width / (len(self.volume_data) - 1)
            for i in range(1, len(self.volume_data)):
                x1 = (i - 1) * step
                x2 = i * step
                y1 = center_y - self.volume_data[i - 1] * center_y * 0.8
                y2 = center_y - self.volume_data[i] * center_y * 0.8
                painter.drawLine(x1, y1, x2, y2)


class VolumeBarWidget(QWidget):
    """音量条显示控件 - 绿色分段式块状递增显示"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(150, 20)
        self.setStyleSheet("background-color: #F1F5F9; border-radius: 4px;")
        self.volume = 0.0
        self.segments = 20  # 分段数量
        self.segment_colors = [
            "#10B981",  # 浅绿色
            "#059669",  # 中绿色
            "#047857",  # 深绿色
            "#065F46",  # 更深绿色
            "#064E3B"   # 最深绿色
        ]

    def update_volume(self, volume):
        """更新音量数据"""
        self.volume = volume
        self.update()

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
    
            width = self.width()
            height = self.height()
            
            # 确保音量值在有效范围内
            safe_volume = min(1.0, max(0.0, self.volume))
            
            # 计算活跃的段数
            active_segments = min(self.segments, max(0, int(safe_volume * self.segments)))
            
            # 段宽度和间距
            segment_width = max(1, width / self.segments * 0.8)  # 确保至少为1像素
            segment_spacing = max(0, width / self.segments * 0.2)
            
            # 绘制段
            for i in range(self.segments):
                try:
                    # 确定位置
                    x = i * (segment_width + segment_spacing)
                    
                    # 确定颜色 - 根据位置选择不同深度的绿色
                    if i < active_segments:
                        # 安全地选择颜色索引
                        color_index = min(len(self.segment_colors) - 1, 
                                        max(0, int(i / max(1, self.segments) * len(self.segment_colors))))
                        color = QColor(self.segment_colors[color_index])
                    else:
                        # 非活跃段使用灰色
                        color = QColor("#E2E8F0")
                    
                    # 绘制矩形，确保坐标有效
                    rect_x = max(0, int(x))
                    rect_width = max(1, int(segment_width))
                    painter.fillRect(rect_x, 0, rect_width, height, color)
                    
                    # 对活跃段添加边框
                    if i < active_segments:
                        painter.setPen(QPen(QColor("#047857"), 1))
                        painter.drawRect(rect_x, 0, rect_width, height - 1)
                except Exception:
                    # 忽略单个段的绘制错误
                    continue
        except Exception:
            # 忽略整个绘制过程的错误
            pass


class CircularButton(QPushButton):
    """圆形按钮"""

    def __init__(self, text):
        super().__init__(text)
        self.setFixedSize(120, 40)  # 修改尺寸为统一的长方形
        self.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border-radius: 6px;  /* 从原来的40px圆形改为圆角矩形 */
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #94A3B8;
            }
        """)


class WorkerSignals(QObject):
    """工作线程信号"""
    log_message = pyqtSignal(str)
    update_recognized_text = pyqtSignal(str, str)  # text, mode
    update_translated_text = pyqtSignal(str, str, str)  # time, original, translation
    update_volume = pyqtSignal(float)
    update_status = pyqtSignal(str, str)  # status, color


class ASRWorker(QObject):
    """ASR工作线程"""
    finished = pyqtSignal()

    def __init__(self, asr_instance, signals):
        super().__init__()
        self.asr_instance = asr_instance
        self.signals = signals
        self.is_running = False
        self.volume_timer = None  # 添加音量检测定时器

    def start_asr(self):
        """启动ASR"""
        try:
            self.is_running = True
            self.asr_instance.start()
            self.signals.log_message.emit("ASR已启动")
            
            # 启动音量检测定时器
            try:
                self.volume_timer = QTimer()
                self.volume_timer.timeout.connect(self.check_volume)
                self.volume_timer.start(100)  # 每100毫秒检测一次音量
                self.signals.log_message.emit("音量监测已启动")
            except Exception as e:
                self.signals.log_message.emit(f"音量监测启动失败: {e}")
                # 即使音量监测失败，也不影响主要功能
            
        except Exception as e:
            self.signals.log_message.emit(f"ASR启动失败: {e}")
            self.is_running = False
        finally:
            self.finished.emit()
    
    def check_volume(self):
        """检测当前音频输入音量并发送信号"""
        if not self.is_running or not self.asr_instance:
            return
            
        try:
            # 从ASR实例中获取当前音量
            if hasattr(self.asr_instance, 'last_audio_volume') and self.asr_instance.last_audio_volume is not None:
                volume = float(self.asr_instance.last_audio_volume)  # 确保是浮点数
                
                # 检查音量是否是有效数值
                if not np.isnan(volume) and not np.isinf(volume):
                    # 对音量进行归一化处理，确保在0-1之间
                    normalized_volume = min(1.0, max(0.0, volume * 20))  # 放大音量并确保在0-1范围内
                    # 发送音量更新信号
                    self.signals.update_volume.emit(normalized_volume)
        except (AttributeError, TypeError, ValueError, Exception) as e:
            # 静默处理所有可能的异常，确保不会导致程序崩溃
            pass

    def stop_asr(self):
        """停止ASR"""
        self.is_running = False
        
        # 停止音量检测定时器
        if self.volume_timer and self.volume_timer.isActive():
            self.volume_timer.stop()
            
        if self.asr_instance:
            try:
                self.asr_instance.stop()
                self.signals.log_message.emit("ASR已停止")
            except Exception as e:
                self.signals.log_message.emit(f"ASR停止失败: {e}")


class TranslationWorker(QObject):
    """翻译工作线程"""
    finished = pyqtSignal()

    def __init__(self, translation_instance, asr_queue, tts_queue, signals, target_lang_func):
        super().__init__()
        self.translation_instance = translation_instance
        self.asr_queue = asr_queue
        self.tts_queue = tts_queue
        self.signals = signals
        self.get_target_lang = target_lang_func
        self.is_running = False

    def run(self):
        """运行翻译线程"""
        self.is_running = True
        while self.is_running:
            try:
                text = self.asr_queue.get(timeout=0.5)
                if not self.is_running:
                    break

                if text:
                    # 获取目标语言
                    target_lang_name = self.get_target_lang()
                    to_lang_code = LANGUAGE_CODES.get(target_lang_name, "en")

                    # 执行翻译
                    self.signals.log_message.emit(f"正在翻译: {text[:30]}...")
                    translated = self.translation_instance.translate(
                        text=text,
                        from_lang="cn",
                        to_lang=to_lang_code
                    )

                    if translated:
                        # 发送翻译结果
                        current_time = time.strftime("%H:%M:%S")
                        self.signals.update_translated_text.emit(current_time, text, translated)
                        self.tts_queue.put(translated)
                        self.signals.log_message.emit(f"翻译完成: {translated[:30]}...")

            except queue.Empty:
                continue
            except Exception as e:
                self.signals.log_message.emit(f"翻译错误: {e}")

        self.finished.emit()

    def stop(self):
        """停止翻译线程"""
        self.is_running = False


class TTSWorker(QObject):
    """TTS工作线程"""
    finished = pyqtSignal()

    def __init__(self, tts_queue, signals, voice_func, rate_func, volume_func):
        super().__init__()
        self.tts_queue = tts_queue
        self.signals = signals
        self.get_voice = voice_func
        self.get_rate = rate_func
        self.get_volume = volume_func
        self.is_running = False
        self.async_loop = None

    def run(self):
        """运行TTS线程"""
        self.is_running = True

        # 创建新的事件循环
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)

        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=0.5)
                if not self.is_running:
                    break

                if text and edge_TTS:
                    voice = self.get_voice()
                    rate = self.get_rate()
                    volume = self.get_volume()

                    self.signals.log_message.emit(
                        f"正在合成语音: {text[:30]}... (音色: {voice}, 语速: {rate}, 音量: {volume})")

                    try:
                        # 运行异步TTS任务
                        success = self.async_loop.run_until_complete(
                            edge_TTS.text_to_speech(text, voice, rate=rate, volume=volume)
                        )

                        if success:
                            self.signals.log_message.emit("语音播放完成")
                        else:
                            self.signals.log_message.emit("语音合成失败")
                    except Exception as e:
                        self.signals.log_message.emit(f"TTS执行错误: {str(e)}")
                        import traceback
                        self.signals.log_message.emit(f"详细错误: {traceback.format_exc()}")

            except queue.Empty:
                continue
            except Exception as e:
                self.signals.log_message.emit(f"TTS线程错误: {e}")

        # 关闭事件循环
        self.async_loop.close()
        self.finished.emit()

    def stop(self):
        """停止TTS线程"""
        self.is_running = False


class SettingsDialog(QDialog):
    """设置对话框"""
    
    def __init__(self, parent=None, noise_reduction_checked=True, tts_enabled=True, cache_enabled=True, on_clear_cache=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # 设置对话框图标
        settings_icon_path = os.path.join(resources_dir, "setting.ico")
        if os.path.exists(settings_icon_path):
            settings_icon = QIcon(settings_icon_path)
            self.setWindowIcon(settings_icon)
        elif hasattr(parent, 'app_icon') and parent.app_icon:
            # 如果找不到设置图标，则使用父窗口的图标作为备选
            self.setWindowIcon(parent.app_icon)
        
        self.noise_reduction_checked = noise_reduction_checked
        self.tts_enabled = tts_enabled
        self.cache_enabled = cache_enabled
        self.on_clear_cache = on_clear_cache
        
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("系统设置")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1E3A8A;")
        layout.addWidget(title_label)
        
        # 创建选项卡控件
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E2E8F0;
                border-radius: 4px;
                padding: 10px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #F1F5F9;
                border: 1px solid #E2E8F0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
        """)
        
        # 音频设置选项卡
        audio_tab = QWidget()
        audio_layout = QVBoxLayout(audio_tab)
        audio_layout.setSpacing(15)
        
        # 降噪设置
        noise_group = QGroupBox("降噪设置")
        noise_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        noise_layout = QVBoxLayout(noise_group)
        
        self.noise_reduction = QCheckBox("启用降噪")
        self.noise_reduction.setChecked(self.noise_reduction_checked)
        self.noise_reduction.setStyleSheet("font-weight: normal;")
        noise_layout.addWidget(self.noise_reduction)
        
        noise_desc = QLabel("启用降噪可以减少背景噪音，提高语音识别准确率")
        noise_desc.setStyleSheet("color: #64748B; font-weight: normal;")
        noise_layout.addWidget(noise_desc)
        
        audio_layout.addWidget(noise_group)
        
        # 语音合成设置
        tts_group = QGroupBox("语音合成设置")
        tts_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        tts_layout = QVBoxLayout(tts_group)
        
        self.tts_checkbox = QCheckBox("启用语音合成")
        self.tts_checkbox.setChecked(self.tts_enabled)
        self.tts_checkbox.setStyleSheet("font-weight: normal;")
        tts_layout.addWidget(self.tts_checkbox)
        
        tts_desc = QLabel("启用语音合成可以将翻译结果转换为语音输出")
        tts_desc.setStyleSheet("color: #64748B; font-weight: normal;")
        tts_layout.addWidget(tts_desc)
        
        audio_layout.addWidget(tts_group)
        audio_layout.addStretch()
        
        # 高级设置选项卡
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setSpacing(15)
        
        # 缓存设置
        cache_group = QGroupBox("缓存设置")
        cache_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        cache_layout = QVBoxLayout(cache_group)
        
        self.cache_checkbox = QCheckBox("启用翻译缓存")
        self.cache_checkbox.setChecked(self.cache_enabled)
        self.cache_checkbox.setStyleSheet("font-weight: normal;")
        cache_layout.addWidget(self.cache_checkbox)
        
        cache_desc = QLabel("启用缓存可以加快相同内容的翻译速度")
        cache_desc.setStyleSheet("color: #64748B; font-weight: normal;")
        cache_layout.addWidget(cache_desc)
        
        clear_cache_btn = QPushButton("清除缓存")
        clear_cache_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
                max-width: 120px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
        """)
        
        # 为清除缓存按钮添加事件处理
        if self.on_clear_cache:
            clear_cache_btn.clicked.connect(self.on_clear_cache)
        
        cache_layout.addWidget(clear_cache_btn)
        
        advanced_layout.addWidget(cache_group)
        advanced_layout.addStretch()
        
        # 添加选项卡
        tab_widget.addTab(audio_tab, "音频设置")
        tab_widget.addTab(advanced_tab, "高级设置")
        
        layout.addWidget(tab_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #E2E8F0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        
        save_btn = QPushButton("保存")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
        """)
        save_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_settings(self):
        """获取设置值"""
        return {
            "noise_reduction": self.noise_reduction.isChecked(),
            "tts_enabled": self.tts_checkbox.isChecked(),
            "cache_enabled": self.cache_checkbox.isChecked()
        }


class VolumeMonitorWorker(QObject):
    """独立的音量监测线程，直接从麦克风读取音量数据"""
    finished = pyqtSignal()
    volume_updated = pyqtSignal(float)
    
    def __init__(self, input_device_index=None):
        super().__init__()
        self.input_device_index = input_device_index
        self.is_running = False
        self.stream = None
        self.sample_rate = 16000
        self.block_size = 1024
        self.volume_scale = 20.0  # 增加音量放大倍数，使显示更明显
        self.debug_counter = 0  # 用于控制调试信息输出频率
        self.last_volume = 0.0  # 上一次的音量值，用于平滑处理
        self.smooth_factor = 0.3  # 平滑因子，值越小越平滑
        self.noise_floor = 0.005  # 噪声阈值，低于此值视为静音
        
    def start_monitoring(self):
        """开始音量监测"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # 列出所有可用的音频设备
        try:
            devices = sd.query_devices()
            print("\n===== 可用音频设备列表 =====")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"输入设备 {i}: {device['name']} (通道数: {device['max_input_channels']})")
            print("===========================\n")
        except Exception as e:
            print(f"无法查询音频设备: {e}")
        
        try:
            # 打开音频流
            print(f"尝试打开音频设备 ID: {self.input_device_index}")
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype='float32',
                device=self.input_device_index
            )
            self.stream.start()
            print(f"音量监测已启动 (设备ID: {self.input_device_index}, 采样率: {self.sample_rate}, 块大小: {self.block_size})")
            
            # 获取实际使用的设备信息
            if hasattr(self.stream, 'device'):
                actual_device = self.stream.device
                print(f"实际使用的设备: {actual_device}")
                
        except Exception as e:
            print(f"音量监测启动失败: {e}")
            # 尝试使用默认设备
            try:
                print("尝试使用默认音频设备...")
                self.stream = sd.InputStream(
                    callback=self.audio_callback,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    dtype='float32',
                    device=None  # 使用默认设备
                )
                self.stream.start()
                print("音量监测已使用默认设备启动")
                
                # 获取默认设备信息
                if hasattr(self.stream, 'device'):
                    actual_device = self.stream.device
                    print(f"实际使用的默认设备: {actual_device}")
            except Exception as e2:
                print(f"使用默认设备启动音量监测也失败: {e2}")
                self.is_running = False
                self.finished.emit()
            
    def audio_callback(self, indata, frames, time_info, status):
        """音频回调函数，计算音量并发送信号"""
        if not self.is_running:
            return
            
        try:
            # 计算音频块的RMS（均方根）能量
            if indata is not None and len(indata) > 0:
                # 将音频数据转换为一维数组
                audio_data = indata.flatten()
                
                # 计算RMS音量
                volume = np.sqrt(np.mean(audio_data ** 2))
                
                # 应用噪声阈值过滤
                if volume < self.noise_floor:
                    volume = 0.0
                
                # 应用平滑处理
                smoothed_volume = self.last_volume * (1 - self.smooth_factor) + volume * self.smooth_factor
                self.last_volume = smoothed_volume
                
                # 应用音量放大倍数并限制在0-1范围内
                normalized_volume = min(1.0, max(0.0, smoothed_volume * self.volume_scale))
                
                # 发送音量更新信号
                self.volume_updated.emit(normalized_volume)
                
                # 每50帧输出一次调试信息，避免控制台刷新过快
                self.debug_counter += 1
                if self.debug_counter % 50 == 0:
                    print(f"当前音量: 原始={volume:.6f}, 平滑后={smoothed_volume:.6f}, 归一化后={normalized_volume:.2f}")
                    self.debug_counter = 0
        except Exception as e:
            # 静默处理异常
            pass
            
    def stop_monitoring(self):
        """停止音量监测"""
        self.is_running = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                print("音量监测已停止")
            except Exception as e:
                print(f"停止音量监测失败: {e}")
                
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多语言同声传译系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用图标
        icon_path = os.path.join(resources_dir, "translator_icon.ico")
        if os.path.exists(icon_path):
            self.app_icon = QIcon(icon_path)
            self.setWindowIcon(self.app_icon)
            # 设置任务栏图标（仅Windows）
            try:
                import ctypes
                myappid = 'mycompany.translationsystem.app.1.0'  # 任意字符串
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception as e:
                self.log_message(f"设置任务栏图标失败: {e}")
        else:
            self.log_message(f"图标文件不存在: {icon_path}")

        # 确保numpy已正确导入
        try:
            import numpy as np
            self.log_message("NumPy已成功导入")
        except ImportError:
            print("警告: NumPy库未找到或无法导入。音量显示功能可能不可用。")
            
        # 初始化变量
        self.is_running = False
        self.asr_instance = None
        self.translation_instance = None
        self.selected_input_device_idx = None
        self.selected_output_device_name = None
        self.voice_name_mapping = {}
        self.all_voices_for_language = []
        self.mixer_initialized = False
        
        # 设置选项
        self.noise_reduction_enabled = True
        self.tts_enabled = True
        self.cache_enabled = True

        # 队列
        self.asr_output_queue = queue.Queue()
        self.tts_output_queue = queue.Queue()

        # 工作线程
        self.asr_thread = None
        self.asr_worker = None
        self.translation_thread = None
        self.translation_worker = None
        self.tts_thread = None
        self.tts_worker = None
        
        # 音量监测线程
        self.volume_monitor_thread = None
        self.volume_monitor_worker = None

        # 信号
        self.signals = WorkerSignals()
        self.signals.log_message.connect(self.log_message)
        self.signals.update_recognized_text.connect(self.update_recognized_text)
        self.signals.update_translated_text.connect(self.add_translation_card)
        self.signals.update_volume.connect(self.update_volume_display)
        self.signals.update_status.connect(self.update_status)

        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8FAFC;
            }
            QLabel {
                color: #1E293B;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                background-color: white;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #3B82F6;
            }
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                background-color: white;
                color: #1E293B;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
                border-color: #3B82F6;
            }
            QTextEdit {
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px;
                background-color: white;
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #E2E8F0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3B82F6;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)

        self.init_ui()
        self.init_modules()

    def init_ui(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(main_layout)

        # 创建工具栏
        self.create_toolbar()

        # 创建内容区域
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)

        # 创建输入面板
        input_panel = self.create_input_panel()
        content_layout.addWidget(input_panel, 40)

        # 创建输出面板
        output_panel = self.create_output_panel()
        content_layout.addWidget(output_panel, 60)

        main_layout.addLayout(content_layout)

        # 创建状态栏
        self.create_statusbar()

    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: white;
                border-bottom: 1px solid #E2E8F0;
                padding: 10px;
                spacing: 10px;
            }
        """)
        self.addToolBar(toolbar)

        # Logo和标题
        logo_label = QLabel("🎙️ 同声传译系统")
        logo_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1E3A8A; margin-left: 10px;")
        toolbar.addWidget(logo_label)

        toolbar.addSeparator()

        # 初始化模型按钮
        self.init_model_button = QPushButton("初始化模型")
        self.init_model_button.clicked.connect(self.init_models_manually)
        self.init_model_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:disabled {
                background-color: #94A3B8;
            }
        """)
        toolbar.addWidget(self.init_model_button)

        # 开始/停止按钮
        self.start_button = CircularButton("开始")
        self.start_button.clicked.connect(self.toggle_translation)
        self.start_button.setEnabled(False)  # 初始禁用，等待模型加载
        toolbar.addWidget(self.start_button)

        # 录音源选择
        toolbar.addWidget(QLabel("录音源:"))
        self.audio_source = QComboBox()
        self.audio_source.addItems(["系统音频", "麦克风", "混合音频"])
        toolbar.addWidget(self.audio_source)

        # 音量标签和绿色分段式音量条
        toolbar.addWidget(QLabel("音量:"))
        self.volume_bar = VolumeBarWidget()
        toolbar.addWidget(self.volume_bar)
        
        # 音量数值显示标签
        self.volume_value_label = QLabel("0%")
        self.volume_value_label.setStyleSheet("""
            color: #10B981;
            font-weight: bold;
            min-width: 40px;
            text-align: center;
        """)
        self.volume_value_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.volume_value_label)

        # 添加弹性空间
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        # 设置按钮
        settings_btn = QPushButton("⚙️ 设置")
        settings_btn.clicked.connect(self.show_settings_dialog)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #64748B;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        toolbar.addWidget(settings_btn)

    def create_input_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 20px;
            }
        """)

        layout = QVBoxLayout()

        # 标题
        title = QLabel("输入设置")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1E3A8A; margin-bottom: 20px;")
        layout.addWidget(title)

        # 语言选择区
        lang_layout = QHBoxLayout()

        # 源语言
        source_group = QVBoxLayout()
        source_label = QLabel("源语言")
        source_label.setStyleSheet("font-size: 14px; color: #64748B; margin-bottom: 8px;")
        source_group.addWidget(source_label)

        self.source_lang = QComboBox()
        self.source_lang.addItems(["中文"])
        self.source_lang.setEnabled(False)
        source_group.addWidget(self.source_lang)
        lang_layout.addLayout(source_group)

        # 箭头
        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("font-size: 24px; color: #3B82F6; margin: 0 20px;")
        arrow_label.setAlignment(Qt.AlignCenter)
        lang_layout.addWidget(arrow_label)

        # 目标语言
        target_group = QVBoxLayout()
        target_label = QLabel("目标语言")
        target_label.setStyleSheet("font-size: 14px; color: #64748B; margin-bottom: 8px;")
        target_group.addWidget(target_label)

        self.target_lang = QComboBox()
        target_group.addWidget(self.target_lang)
        lang_layout.addLayout(target_group)

        layout.addLayout(lang_layout)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #E2E8F0; margin: 20px 0;")
        layout.addWidget(line)

        # 实时转录区
        transcribe_label = QLabel("实时转录")
        transcribe_label.setStyleSheet("font-size: 16px; font-weight: 500; margin-bottom: 12px;")
        layout.addWidget(transcribe_label)

        self.transcribe_area = QTextEdit()
        self.transcribe_area.setPlaceholderText("等待语音输入...")
        self.transcribe_area.setReadOnly(True)
        layout.addWidget(self.transcribe_area)

        # 音频控制区
        audio_control_label = QLabel("音频控制")
        audio_control_label.setStyleSheet("font-size: 16px; font-weight: 500; margin: 20px 0 12px 0;")
        layout.addWidget(audio_control_label)

        # 输入设备
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("🎤 输入设备 :"))
        self.input_device = QComboBox()
        self.input_device.currentIndexChanged.connect(self.on_input_device_changed)
        device_layout.addWidget(self.input_device)
        layout.addLayout(device_layout)

        # 音量控制 (添加可视化数值)
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("麦克风输入音量 :"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        volume_layout.addWidget(self.volume_slider)
        
        # 添加音量数值显示
        self.mic_volume_label = QLabel("70%")
        self.mic_volume_label.setStyleSheet("""
            color: #3B82F6;
            font-weight: bold;
            min-width: 40px;
            text-align: center;
        """)
        self.mic_volume_label.setAlignment(Qt.AlignCenter)
        self.volume_slider.valueChanged.connect(lambda v: self.mic_volume_label.setText(f"{v}%"))
        volume_layout.addWidget(self.mic_volume_label)
        
        layout.addLayout(volume_layout)

        panel.setLayout(layout)
        return panel

    def create_output_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 10px 20px 20px 20px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 标题栏和按钮布局
        title_layout = QHBoxLayout()
        
        title = QLabel("翻译结果（实时）")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #1E3A8A;")
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        # 导出按钮和清空按钮移到这里
        export_btn = QPushButton("📥 导出记录")
        export_btn.clicked.connect(self.export_translation_cards)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
        """)
        title_layout.addWidget(export_btn)

        clear_btn = QPushButton("🗑️ 清空记录")
        clear_btn.clicked.connect(self.clear_translation_cards)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
        """)
        title_layout.addWidget(clear_btn)
        
        layout.addLayout(title_layout)

        # 翻译结果滚动区域
        results_frame = QFrame()
        results_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #E2E8F0;
                background-color: #F8FAFC;
                border-radius: 8px;
                padding: 0;
            }
        """)
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background-color: transparent;")

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)

        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)

        results_layout.addWidget(self.scroll_area)
        layout.addWidget(results_frame)

        # 输出控制区
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: #F8FAFC; border-radius: 8px; padding: 16px; margin-top: 20px;")
        control_layout = QVBoxLayout()

        # 输出设备
        output_device_layout = QHBoxLayout()
        output_device_layout.addWidget(QLabel("🔊 输出设备 :"))
        self.output_device = QComboBox()
        self.output_device.currentIndexChanged.connect(self.on_output_device_changed)
        output_device_layout.addWidget(self.output_device)
        control_layout.addLayout(output_device_layout)

        # 性别和音色选择布局
        voice_layout = QHBoxLayout()
        
        # 性别选择
        voice_layout.addWidget(QLabel("性别 :"))
        self.gender_combo = QComboBox()
        self.gender_combo.currentIndexChanged.connect(self.on_gender_changed)
        self.gender_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        voice_layout.addWidget(self.gender_combo)

        # 音色选择
        voice_layout.addWidget(QLabel("音色 :"))
        self.voice_combo = QComboBox()
        # 设置音色下拉框的尺寸策略，使其能够水平扩展
        self.voice_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        voice_layout.addWidget(self.voice_combo)
        # 移除这里的stretch，让音色下拉框自然扩展到右边界
        
        control_layout.addLayout(voice_layout)
        
        # 语速和音量控制在同一行
        speed_volume_layout = QHBoxLayout()
        
        # 语速控制
        speed_group = QHBoxLayout()
        speed_group.addWidget(QLabel("语速 :"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(-50, 50)
        self.speed_slider.setValue(0)
        self.speed_slider.setFixedWidth(150)
        speed_group.addWidget(self.speed_slider)
        self.speed_label = QLabel("0%")
        self.speed_label.setFixedWidth(50)
        self.speed_label.setStyleSheet("""
            color: #3B82F6;
            font-weight: bold;
            min-width: 40px;
            text-align: center;
        """)
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v:+d}%"))
        speed_group.addWidget(self.speed_label)
        
        speed_volume_layout.addLayout(speed_group)
        speed_volume_layout.addSpacing(20)  # 添加间距
        
        # 音量控制
        volume_group = QHBoxLayout()
        volume_group.addWidget(QLabel("音量 :"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(-50, 50)
        self.volume_slider.setValue(0)
        self.volume_slider.setFixedWidth(150)
        volume_group.addWidget(self.volume_slider)
        self.volume_label = QLabel("0%")
        self.volume_label.setFixedWidth(50)
        self.volume_label.setStyleSheet("""
            color: #3B82F6;
            font-weight: bold;
            min-width: 40px;
            text-align: center;
        """)
        self.volume_label.setAlignment(Qt.AlignCenter)
        self.volume_slider.valueChanged.connect(lambda v: self.volume_label.setText(f"{v:+d}%"))
        volume_group.addWidget(self.volume_label)
        
        speed_volume_layout.addLayout(volume_group)
        speed_volume_layout.addStretch()
        
        control_layout.addLayout(speed_volume_layout)

        # 移除旧的按钮布局，因为按钮已经移到标题栏了
        control_frame.setLayout(control_layout)
        layout.addWidget(control_frame)

        panel.setLayout(layout)
        return panel

    def create_statusbar(self):
        statusbar = self.statusBar()
        statusbar.setStyleSheet("""
            QStatusBar {
                background-color: white;
                border-top: 1px solid #E2E8F0;
                padding: 8px;
            }
        """)

        # 连接状态
        self.status_label = QLabel("🔄 等待模型初始化...")
        self.status_label.setStyleSheet("color: #3B82F6; font-weight: bold;")
        statusbar.addWidget(self.status_label)

        # 统计信息
        self.stats_label = QLabel("已翻译: 0 句")
        statusbar.addPermanentWidget(self.stats_label)
        statusbar.addPermanentWidget(QLabel("|"))
        self.time_label = QLabel("运行时间: 00:00:00")
        statusbar.addPermanentWidget(self.time_label)

        # 添加日志显示按钮
        self.log_button = QPushButton("显示日志")
        self.log_button.clicked.connect(self.toggle_log_window)
        statusbar.addPermanentWidget(self.log_button)

    def toggle_log_window(self):
        """切换日志窗口显示"""
        if not hasattr(self, 'log_dialog'):
            self.log_dialog = QDialog(self)
            self.log_dialog.setWindowTitle("系统日志")
            self.log_dialog.setGeometry(200, 200, 800, 400)
            
            # 设置对话框图标
            if hasattr(self, 'app_icon') and self.app_icon:
                self.log_dialog.setWindowIcon(self.app_icon)

            layout = QVBoxLayout()
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
            layout.addWidget(self.log_text)

            # 清空日志按钮
            clear_button = QPushButton("清空日志")
            clear_button.clicked.connect(self.log_text.clear)
            layout.addWidget(clear_button)

            self.log_dialog.setLayout(layout)

        if self.log_dialog.isVisible():
            self.log_dialog.hide()
            self.log_button.setText("显示日志")
        else:
            self.log_dialog.show()
            self.log_button.setText("隐藏日志")

    def init_modules(self):
        """初始化各个模块"""
        # 初始化翻译模块
        if TranslationModule:
            self.translation_instance = TranslationModule(
                app_id=TRANSLATION_APP_ID,
                api_secret=TRANSLATION_API_SECRET,
                api_key=TRANSLATION_API_KEY
            )
            self.log_message("翻译模块已初始化")

        # 填充设备列表
        self.populate_audio_devices()

        # 填充目标语言
        self.populate_target_languages()

        # 初始化Pygame
        if edge_TTS:
            try:
                # 先尝试预初始化
                pygame.mixer.pre_init(
                    frequency=22050,
                    size=-16,
                    channels=2,
                    buffer=512
                )
                pygame.mixer.init()
                self.mixer_initialized = True
                self.log_message("Pygame Mixer已初始化")

                # 检查初始化状态
                if pygame.mixer.get_init():
                    mixer_info = pygame.mixer.get_init()
                    self.log_message(f"Mixer配置: 频率={mixer_info[0]}Hz, 格式={mixer_info[1]}, 通道={mixer_info[2]}")
                else:
                    self.log_message("警告: Mixer可能未正确初始化")

            except Exception as e:
                self.log_message(f"Pygame Mixer初始化失败: {e}")
                self.mixer_initialized = False

        # 不再自动初始化ASR，等待用户点击按钮
        self.log_message("请点击'初始化模型'按钮加载ASR模型")

    def init_models_manually(self):
        """手动初始化模型"""
        if not FastLoadASR:
            self.log_message("错误：FunASR模块不可用")
            return

        self.init_model_button.setEnabled(False)
        self.init_model_button.setText("正在加载...")
        self.log_message("开始初始化模型...")
        
        # 创建并启动加载动画计时器
        self.loading_animation = LoadingAnimationTimer(self, "⏳ 模型加载中", "#F59E0B", 500)
        self.loading_animation.start()

        # 在后台线程中初始化
        threading.Thread(target=self._init_models_thread, daemon=True).start()

    def _init_models_thread(self):
        """在后台线程中初始化模型"""
        try:
            # 初始化ASR实例
            self.log_message("正在初始化ASR实例...")
            self.asr_instance = FastLoadASR(
                use_vad=True,
                use_punc=True,
                text_output_callback=self.asr_text_callback,
                input_device_index=self.selected_input_device_idx,
                max_segment_duration_seconds=5.0
            )
            self.log_message("ASR实例初始化完成")

            # 加载ASR模型
            self.log_message("正在加载ASR模型...")
            if self.asr_instance.ensure_asr_model_loaded():
                self.log_message("ASR模型加载完成")

                if self.asr_instance.use_vad:
                    self.log_message("正在加载VAD模型...")
                    self.asr_instance.load_vad_model_if_needed()
                    self.log_message("VAD模型加载完成")

                if self.asr_instance.use_punc:
                    self.log_message("正在加载标点模型...")
                    self.asr_instance.load_punc_model_if_needed()
                    self.log_message("标点模型加载完成")

                # 所有模型加载完成
                QMetaObject.invokeMethod(self, "_on_models_loaded", Qt.QueuedConnection)
            else:
                self.log_message("ASR模型加载失败")
                QMetaObject.invokeMethod(self, "_on_models_failed", Qt.QueuedConnection)

        except Exception as e:
            self.log_message(f"模型初始化错误: {e}")
            QMetaObject.invokeMethod(self, "_on_models_failed", Qt.QueuedConnection)

    @pyqtSlot()
    def _on_models_loaded(self):
        """模型加载成功的回调"""
        # 停止加载动画
        if hasattr(self, 'loading_animation'):
            self.loading_animation.stop()
            
        self.init_model_button.setText("模型已加载")
        self.init_model_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 6px;
            }
        """)
        self.start_button.setEnabled(True)
        self.update_status("✅ 已就绪", "#10B981")
        self.log_message("所有模型加载完成，可以开始同传")

    @pyqtSlot()
    def _on_models_failed(self):
        """模型加载失败的回调"""
        # 停止加载动画
        if hasattr(self, 'loading_animation'):
            self.loading_animation.stop()
            
        self.init_model_button.setText("重新加载")
        self.init_model_button.setEnabled(True)
        self.update_status("❌ 模型加载失败", "#EF4444")
        self.log_message("模型加载失败，请重试")

    def populate_audio_devices(self):
        """填充音频设备列表"""
        try:
            devices = sd.query_devices()

            # 输入设备
            input_devices = [(i, device['name']) for i, device in enumerate(devices)
                             if device['max_input_channels'] > 0]

            for idx, name in input_devices:
                self.input_device.addItem(f"{name} (ID: {idx})")

            # 设置默认输入设备
            if input_devices:
                default_input = sd.default.device[0]
                for i, (idx, name) in enumerate(input_devices):
                    if idx == default_input:
                        self.input_device.setCurrentIndex(i)
                        self.selected_input_device_idx = idx
                        break

            # 输出设备
            output_devices = [device['name'] for device in devices
                              if device['max_output_channels'] > 0]

            self.output_device.addItems(output_devices)

            if output_devices:
                self.selected_output_device_name = output_devices[0]

        except Exception as e:
            self.log_message(f"加载音频设备失败: {e}")

    def populate_target_languages(self):
        """填充目标语言列表"""
        if LANGUAGE_CODES:
            languages = list(LANGUAGE_CODES.keys())
            # 移除中文（源语言）
            if "中文" in languages:
                languages.remove("中文")
            self.target_lang.addItems(languages)

            # 设置默认为英语
            if "英语" in languages:
                self.target_lang.setCurrentText("英语")

            # 连接信号
            self.target_lang.currentTextChanged.connect(self.on_target_language_changed)

            # 初始加载音色
            if languages:
                self.on_target_language_changed()

    def on_target_language_changed(self):
        """目标语言改变时更新音色列表"""
        target_lang_name = self.target_lang.currentText()
        if not target_lang_name or not edge_TTS:
            return

        lang_code = LANGUAGE_CODES.get(target_lang_name)
        if not lang_code:
            return

        # 在后台线程中获取音色
        threading.Thread(target=self.fetch_voices_for_language,
                         args=(lang_code,), daemon=True).start()

    def fetch_voices_for_language(self, lang_code):
        """获取指定语言的音色列表"""
        try:
            # 创建异步事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 语言代码映射
            tts_lang_map = {
                "cn": "zh-CN", "en": "en-US", "ja": "ja-JP", "es": "es-ES",
                "fr": "fr-FR", "de": "de-DE", "ko": "ko-KR", "ru": "ru-RU"
            }

            effective_lang_code = tts_lang_map.get(lang_code, lang_code)

            # 获取音色列表
            voices_list = loop.run_until_complete(
                edge_TTS.list_voices_by_language(effective_lang_code)
            )

            # 保存音色信息
            self.all_voices_for_language = []
            genders = set()

            for voice in voices_list:
                voice_name = voice['ShortName']
                # 查找对应的音色信息
                for voice_info in edge_TTS.SUPPORTED_VOICES:
                    if voice_info["short_name"] == voice_name:
                        gender = voice_info["gender_display"]
                        genders.add(gender)
                        self.all_voices_for_language.append({
                            "voice_name": voice_name,
                            "gender": gender,
                            "display_name": voice_info["voice_display"],
                            "locale_display": voice_info["locale_display"]
                        })
                        break

            # 更新UI
            genders_list = sorted(list(genders))
            QMetaObject.invokeMethod(self, "update_gender_combo",
                                     Qt.QueuedConnection,
                                     Q_ARG(list, genders_list))

        except Exception as e:
            self.log_message(f"获取音色列表失败: {e}")

    @pyqtSlot(list)
    def update_gender_combo(self, genders):
        """更新性别下拉框"""
        self.gender_combo.clear()
        self.gender_combo.addItems(genders)
        if genders:
            self.on_gender_changed()

    def on_gender_changed(self):
        """性别改变时更新音色列表"""
        selected_gender = self.gender_combo.currentText()
        if not selected_gender:
            return

        # 筛选音色
        filtered_voices = [v for v in self.all_voices_for_language
                           if v["gender"] == selected_gender]

        # 格式化显示
        formatted_voices = []
        self.voice_name_mapping = {}

        for voice_info in filtered_voices:
            display_name = f"{voice_info['display_name']} ({voice_info['locale_display']})"
            formatted_voices.append(display_name)
            self.voice_name_mapping[display_name] = voice_info['voice_name']

        # 更新音色下拉框
        self.voice_combo.clear()
        self.voice_combo.addItems(formatted_voices)

    def on_input_device_changed(self):
        """输入设备改变"""
        text = self.input_device.currentText()
        if " (ID: " in text:
            try:
                self.selected_input_device_idx = int(text.split(' (ID: ')[-1][:-1])
                self.log_message(f"选择输入设备 ID: {self.selected_input_device_idx}")
            except ValueError:
                pass

    def on_output_device_changed(self):
        """输出设备改变"""
        self.selected_output_device_name = self.output_device.currentText()
        self.log_message(f"选择输出设备: {self.selected_output_device_name}")

        # 重新初始化Pygame Mixer
        if self.mixer_initialized:
            try:
                pygame.mixer.quit()
                self.mixer_initialized = False
            except:
                pass

        try:
            # 尝试使用选定的设备初始化
            pygame.mixer.pre_init(
                frequency=22050,
                size=-16,
                channels=2,
                buffer=512,
                devicename=self.selected_output_device_name
            )
            pygame.mixer.init()
            self.mixer_initialized = True
            self.log_message(f"Pygame Mixer已使用设备 '{self.selected_output_device_name}' 初始化")

            # 测试mixer是否真的工作
            if pygame.mixer.get_init():
                self.log_message(f"Mixer状态: {pygame.mixer.get_init()}")
            else:
                self.log_message("警告: Mixer未正确初始化")

        except Exception as e:
            self.log_message(f"Pygame Mixer初始化失败: {e}")
            # 尝试默认设备
            try:
                pygame.mixer.init()
                self.mixer_initialized = True
                self.log_message("已回退到默认音频设备")
            except Exception as e2:
                self.log_message(f"默认设备也失败: {e2}")

    def asr_text_callback(self, segment, full_sentence, is_sentence_end):
        """ASR文本回调"""
        if not self.is_running:
            return

        if is_sentence_end:
            # 句子结束，添加到队列
            if full_sentence:
                self.asr_output_queue.put(full_sentence)
                self.signals.update_recognized_text.emit(full_sentence + "\n", "append")
        else:
            # 实时更新
            self.signals.update_recognized_text.emit(full_sentence, "update")

    def toggle_translation(self):
        """开始/停止翻译"""
        if self.is_running:
            self.stop_translation()
        else:
            self.start_translation()

    def start_translation(self):
        """开始翻译"""
        if not self.asr_instance:
            self.log_message("ASR实例未初始化")
            return

        if not self.translation_instance:
            self.log_message("翻译模块未初始化")
            return

        if not edge_TTS or not self.tts_enabled:
            self.log_message("警告：TTS未启用或不可用")

        self.is_running = True
        self.start_button.setText("停止")
        self.update_status("🔊 正在同传", "#3B82F6")

        # 清空文本区域
        self.transcribe_area.clear()

        # 启动计时器
        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        
        # 启动独立的音量监测线程
        self.volume_monitor_thread = QThread()
        self.volume_monitor_worker = VolumeMonitorWorker(self.selected_input_device_idx)
        self.volume_monitor_worker.moveToThread(self.volume_monitor_thread)
        
        # 连接信号
        self.volume_monitor_thread.started.connect(self.volume_monitor_worker.start_monitoring)
        self.volume_monitor_worker.finished.connect(self.volume_monitor_thread.quit)
        self.volume_monitor_worker.volume_updated.connect(self.update_volume_display)
        
        # 启动线程
        self.volume_monitor_thread.start()
        self.log_message("音量监测已启动")

        # 启动ASR
        self.asr_thread = QThread()
        self.asr_worker = ASRWorker(self.asr_instance, self.signals)
        self.asr_worker.moveToThread(self.asr_thread)

        self.asr_thread.started.connect(self.asr_worker.start_asr)
        self.asr_worker.finished.connect(self.asr_thread.quit)

        self.asr_thread.start()

        # 启动翻译线程
        self.translation_thread = QThread()
        self.translation_worker = TranslationWorker(
            self.translation_instance,
            self.asr_output_queue,
            self.tts_output_queue,
            self.signals,
            lambda: self.target_lang.currentText()
        )
        self.translation_worker.moveToThread(self.translation_thread)

        self.translation_thread.started.connect(self.translation_worker.run)
        self.translation_worker.finished.connect(self.translation_thread.quit)

        self.translation_thread.start()

        # 设置翻译缓存状态
        if self.translation_instance:
            try:
                # 查看是否有设置缓存使用的方法
                if hasattr(self.translation_instance, 'set_use_cache'):
                    self.translation_instance.set_use_cache(self.cache_enabled)
                    self.log_message(f"翻译缓存已{'启用' if self.cache_enabled else '禁用'}")
            except Exception as e:
                self.log_message(f"设置缓存状态失败: {str(e)}")

        # 启动TTS线程
        if self.tts_enabled and edge_TTS:
            self.tts_thread = QThread()
            self.tts_worker = TTSWorker(
                self.tts_output_queue,
                self.signals,
                self.get_selected_voice,
                lambda: f"{self.speed_slider.value():+d}%",
                lambda: f"{self.volume_slider.value():+d}%"  # 使用实际的音量滑块值
            )
            self.tts_worker.moveToThread(self.tts_thread)

            self.tts_thread.started.connect(self.tts_worker.run)
            self.tts_worker.finished.connect(self.tts_thread.quit)

            self.tts_thread.start()

        self.log_message("同声传译已启动")

    def stop_translation(self):
        """停止翻译"""
        self.is_running = False
        self.start_button.setText("开始")
        self.update_status("⏹️ 已停止", "#EF4444")

        # 停止计时器
        if hasattr(self, 'timer'):
            self.timer.stop()
            
        # 停止音量监测线程
        if self.volume_monitor_worker:
            self.volume_monitor_worker.stop_monitoring()
            
        if self.volume_monitor_thread and self.volume_monitor_thread.isRunning():
            self.volume_monitor_thread.quit()
            self.volume_monitor_thread.wait(1000)
            self.log_message("音量监测已停止")

        # 停止ASR
        if self.asr_worker:
            self.asr_worker.stop_asr()

        # 停止翻译线程
        if self.translation_worker:
            self.translation_worker.stop()

        # 停止TTS线程
        if self.tts_worker:
            self.tts_worker.stop()

        # 等待线程结束
        for thread in [self.asr_thread, self.translation_thread, self.tts_thread]:
            if thread and thread.isRunning():
                thread.quit()
                thread.wait(1000)

        # 清空队列
        while not self.asr_output_queue.empty():
            self.asr_output_queue.get_nowait()
        while not self.tts_output_queue.empty():
            self.tts_output_queue.get_nowait()

        self.log_message("同声传译已停止")

    def get_selected_voice(self):
        """获取选择的音色"""
        display_name = self.voice_combo.currentText()
        if display_name in self.voice_name_mapping:
            return self.voice_name_mapping[display_name]
        return display_name

    def update_recognized_text(self, text, mode):
        """更新识别文本"""
        if mode == "append":
            self.transcribe_area.append(text)
        elif mode == "update":
            # 更新最后一行
            cursor = self.transcribe_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text)

    def add_translation_card(self, time_str, original, translation):
        """添加翻译卡片"""
        card = TranslationCard(time_str, original, translation)
        self.scroll_layout.addWidget(card)

        # 滚动到底部
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

        # 更新统计
        count = self.scroll_layout.count()
        self.stats_label.setText(f"已翻译: {count} 句")

    def clear_translation_cards(self):
        """清空翻译卡片"""
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.stats_label.setText("已翻译: 0 句")

    def update_volume_display(self, volume):
        """更新音量显示"""
        try:
            # 确保音量是有效的浮点数
            if volume is None or np.isnan(volume) or np.isinf(volume):
                return
                
            # 确保音量在0-1范围内
            safe_volume = min(1.0, max(0.0, float(volume)))
            
            # 更新音量条
            self.volume_bar.update_volume(safe_volume)
            
            # 更新音量数值显示
            volume_percentage = int(safe_volume * 100)
            self.volume_value_label.setText(f"{volume_percentage}%")
        except Exception:
            # 忽略音量更新错误，确保不会导致程序崩溃
            pass

    def update_status(self, status, color):
        """更新状态显示"""
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def update_time(self):
        """更新运行时间"""
        if hasattr(self, 'start_time'):
            elapsed = int(time.time() - self.start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.time_label.setText(f"运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def log_message(self, message):
        """记录日志"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            print(log_entry)
    
            # 如果日志窗口存在，添加到日志窗口
            if hasattr(self, 'log_text') and self.log_text is not None:
                self.log_text.append(log_entry)
        except Exception as e:
            # 确保日志记录不会导致程序崩溃
            print(f"日志记录失败: {e}")

    def closeEvent(self, event):
        """关闭事件"""
        if self.is_running:
            self.stop_translation()

        if self.mixer_initialized:
            pygame.mixer.quit()

        event.accept()

    def export_translation_cards(self):
        """导出翻译记录到文件"""
        if self.scroll_layout.count() == 0:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("导出记录")
            msg_box.setText("当前没有翻译记录可导出")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.button(QMessageBox.Ok).setText("确定")

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #FFFFFF;
                    border-radius: 12px;
                    font-size: 15px;
                    padding: 12px;
                    min-width: 360px;
                    min-height: 180px;
                }
                QLabel {
                    font-size: 15px;
                    min-width: 220px;
                    margin-top: 20px;
                    qproperty-alignment: AlignCenter;
                }
                QMessageBox QLabel[objectName="qt_msgbox_label"] {
                    qproperty-alignment: AlignVCenter;
                    
                }
                QMessageBox QLabel[objectName="qt_msgboxex_icon_label"] {
                    qproperty-alignment: AlignVCenter;
                    margin-top: 20px;
                }
                QPushButton {
                    background-color: #3B82F6;
                    color: white;
                    padding: 8px 20px;
                    font-weight: bold;
                    border-radius: 6px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #2563EB;
                }
            """)

            msg_box.exec_()

            return

        # 创建格式选择对话框
        format_dialog = QDialog(self)
        format_dialog.setWindowTitle("选择导出格式")
        format_dialog.setMinimumWidth(400)
        format_dialog.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
            QLabel {
                font-size: 14px;
                margin-bottom: 10px;
            }
            QRadioButton {
                font-size: 14px;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
        """)
        
        layout = QVBoxLayout(format_dialog)
        
        # 标题
        title_label = QLabel("请选择导出格式：")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title_label)
        
        # 格式选项
        txt_radio = QRadioButton("文本文件 (TXT) - 简单文本格式")
        txt_radio.setChecked(True)
        layout.addWidget(txt_radio)
        
        csv_radio = QRadioButton("CSV文件 - 可在Excel中打开")
        layout.addWidget(csv_radio)
        
        html_radio = QRadioButton("HTML文件 - 美观的网页格式")
        layout.addWidget(html_radio)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            background-color: #EF4444;
            color: white;
        """)
        cancel_btn.clicked.connect(format_dialog.reject)
        
        confirm_btn = QPushButton("确定")
        confirm_btn.clicked.connect(format_dialog.accept)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(confirm_btn)
        
        layout.addSpacing(10)
        layout.addLayout(button_layout)
        
        # 显示对话框
        if format_dialog.exec_() != QDialog.Accepted:
            return
        
        # 确定导出格式
        export_format = "txt"
        if csv_radio.isChecked():
            export_format = "csv"
        elif html_radio.isChecked():
            export_format = "html"
        
        # 获取当前时间作为默认文件名
        current_time = time.strftime("%Y%m%d_%H%M%S")
        
        # 设置文件对话框选项
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("导出翻译记录")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        
        if export_format == "txt":
            default_filename = f"翻译记录_{current_time}.txt"
            file_dialog.setDefaultSuffix("txt")
            file_dialog.setNameFilter("文本文件 (*.txt);;所有文件 (*.*)")
        elif export_format == "csv":
            default_filename = f"翻译记录_{current_time}.csv"
            file_dialog.setDefaultSuffix("csv")
            file_dialog.setNameFilter("CSV文件 (*.csv);;所有文件 (*.*)")
        else:  # html
            default_filename = f"翻译记录_{current_time}.html"
            file_dialog.setDefaultSuffix("html")
            file_dialog.setNameFilter("HTML文件 (*.html);;所有文件 (*.*)")
            
        file_dialog.selectFile(default_filename)
        
        if file_dialog.exec_() != QFileDialog.Accepted:
            return
            
        filename = file_dialog.selectedFiles()[0]
        if not filename:
            return
        
        # 获取记录总数
        total_records = self.scroll_layout.count()
        
        # 创建进度对话框
        progress_dialog = QProgressDialog("正在导出翻译记录...", "取消", 0, total_records, self)
        progress_dialog.setWindowTitle("导出进度")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(500)  # 只有当操作超过500ms才显示
        progress_dialog.setAutoClose(True)
        progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: #FFFFFF;
                font-size: 14px;
            }
            QProgressBar {
                border: 1px solid #E2E8F0;
                border-radius: 4px;
                text-align: center;
                background-color: #F1F5F9;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #EF4444;
                color: white;
                padding: 6px 16px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
        """)
            
        try:
            # 收集翻译记录数据
            records = []
            for i in range(total_records):
                # 更新进度条
                progress_dialog.setValue(i)
                QApplication.processEvents()
                
                # 检查是否取消
                if progress_dialog.wasCanceled():
                    self.log_message("导出已取消")
                    return
                
                widget = self.scroll_layout.itemAt(i).widget()
                if isinstance(widget, TranslationCard):
                    # 从布局中获取卡片内容
                    layout = widget.layout()
                    time_label = layout.itemAt(0).widget()
                    original_label = layout.itemAt(1).widget()
                    translation_label = layout.itemAt(2).widget()
                    
                    # 提取文本内容
                    time_text = time_label.text()
                    original_text = original_label.text().replace("原文：", "")
                    translation_text = translation_label.text().replace("译文：", "")
                    
                    records.append({
                        "time": time_text,
                        "original": original_text,
                        "translation": translation_text
                    })
            
            # 根据不同格式进行导出
            if export_format == "txt":
                self._export_as_txt(filename, records)
            elif export_format == "csv":
                self._export_as_csv(filename, records)
            else:  # html
                self._export_as_html(filename, records)
                
            # 完成进度
            progress_dialog.setValue(total_records)
            
            self.log_message(f"翻译记录已导出到 {filename}")
            
            # 美化成功提示框
            success_box = QMessageBox(self)
            success_box.setWindowTitle("导出成功")
            success_box.setText(f"已成功导出 {total_records} 条翻译记录至:\n{filename}")
            success_box.setIcon(QMessageBox.Information)
            success_box.setStandardButtons(QMessageBox.Ok)
            success_box.button(QMessageBox.Ok).setText("确定")
            success_box.setStyleSheet("""
                QMessageBox {
                    background-color: #FFFFFF;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #10B981;
                    color: white;
                    padding: 6px 16px;
                    font-weight: bold;
                    border-radius: 4px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
            success_box.exec_()
            
        except Exception as e:
            self.log_message(f"导出失败: {str(e)}")
            
            # 美化错误提示框
            error_box = QMessageBox(self)
            error_box.setWindowTitle("导出失败")
            error_box.setText(f"导出记录时出错:\n{str(e)}")
            error_box.setIcon(QMessageBox.Critical)
            error_box.setStandardButtons(QMessageBox.Ok)
            error_box.button(QMessageBox.Ok).setText("确定")
            error_box.setStyleSheet("""
                QMessageBox {
                    background-color: #FFFFFF;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #EF4444;
                    color: white;
                    padding: 6px 16px;
                    font-weight: bold;
                    border-radius: 4px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #DC2626;
                }
            """)
            error_box.exec_()
    
    def _export_as_txt(self, filename, records):
        """将记录导出为TXT格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"翻译记录 - 导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for record in records:
                f.write(f"[{record['time']}]\n")
                f.write(f"原文：{record['original']}\n")
                f.write(f"译文：{record['translation']}\n")
                f.write("-" * 50 + "\n\n")
    
    def _export_as_csv(self, filename, records):
        """将记录导出为CSV格式"""
        import csv
        with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["时间", "原文", "译文"])
            
            for record in records:
                writer.writerow([record['time'], record['original'], record['translation']])
    
    def _export_as_html(self, filename, records):
        """将记录导出为HTML格式"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>翻译记录</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #F8FAFC;
        }}
        h1 {{
            color: #1E3A8A;
            text-align: center;
            padding-bottom: 10px;
            border-bottom: 2px solid #E2E8F0;
        }}
        .info {{
            text-align: center;
            color: #64748B;
            margin-bottom: 30px;
        }}
        .card {{
            background-color: white;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3B82F6;
        }}
        .time {{
            color: #64748B;
            font-size: 12px;
            margin-bottom: 10px;
        }}
        .original {{
            color: #1E293B;
            margin-bottom: 8px;
        }}
        .translation {{
            color: #1E3A8A;
            font-weight: 500;
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: #94A3B8;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>翻译记录</h1>
    <div class="info">导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 共 {len(records)} 条记录</div>
"""
        
        for record in records:
            html_content += f"""
    <div class="card">
        <div class="time">{record['time']}</div>
        <div class="original">原文：{record['original']}</div>
        <div class="translation">译文：{record['translation']}</div>
    </div>
"""
        
        html_content += """
    <footer>
        由同声传译系统导出
    </footer>
</body>
</html>
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def show_settings_dialog(self):
        """显示设置对话框"""
        # 获取当前设置
        noise_reduction_checked = True
        tts_enabled = True
        cache_enabled = True
        
        # 检查是否已有这些控件
        if hasattr(self, 'noise_reduction'):
            noise_reduction_checked = self.noise_reduction.isChecked()
        
        if hasattr(self, 'tts_checkbox'):
            tts_enabled = self.tts_checkbox.isChecked()
        
        if hasattr(self, 'cache_checkbox'):
            cache_enabled = self.cache_checkbox.isChecked()
        
        # 检查设置图标是否存在
        settings_icon_path = os.path.join(resources_dir, "setting.ico")
        if not os.path.exists(settings_icon_path):
            self.log_message(f"设置图标文件不存在: {settings_icon_path}")
        
        # 创建并显示设置对话框
        dialog = SettingsDialog(
            self,
            noise_reduction_checked=noise_reduction_checked,
            tts_enabled=tts_enabled,
            cache_enabled=cache_enabled,
            on_clear_cache=self.clear_cache
        )
        
        if dialog.exec_() == QDialog.Accepted:
            # 应用设置
            settings = dialog.get_settings()
            
            # 保存设置到实例变量
            self.noise_reduction_enabled = settings["noise_reduction"]
            self.tts_enabled = settings["tts_enabled"]
            self.cache_enabled = settings["cache_enabled"]
            
            self.log_message(f"已更新设置: 降噪={self.noise_reduction_enabled}, 语音合成={self.tts_enabled}, 缓存={self.cache_enabled}")
            
            # 更新UI状态
            if hasattr(self, 'noise_reduction'):
                self.noise_reduction.setChecked(self.noise_reduction_enabled)
                
            if hasattr(self, 'tts_checkbox'):
                self.tts_checkbox.setChecked(self.tts_enabled)
            
            if hasattr(self, 'cache_checkbox'):
                self.cache_checkbox.setChecked(self.cache_enabled)

    def clear_cache(self):
        """清除翻译缓存"""
        if self.translation_instance:
            try:
                # 调用翻译模块的清空缓存方法
                self.translation_instance.clear_cache()
                
                # 显示成功提示
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("清除缓存")
                msg_box.setText("翻译缓存已成功清除")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.button(QMessageBox.Ok).setText("确定")
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #FFFFFF;
                        border-radius: 12px;
                        font-size: 15px;
                        padding: 12px;
                        min-width: 360px;
                        min-height: 180px;
                    }
                    QLabel {
                        font-size: 15px;
                        min-width: 220px;
                        margin-top: 20px;
                        qproperty-alignment: AlignCenter;
                    }
                    QMessageBox QLabel[objectName="qt_msgbox_label"] {
                        qproperty-alignment: AlignVCenter;
                    }
                    QMessageBox QLabel[objectName="qt_msgboxex_icon_label"] {
                        qproperty-alignment: AlignVCenter;
                        margin-top: 20px;
                    }
                    QPushButton {
                        background-color: #10B981;
                        color: white;
                        padding: 8px 20px;
                        font-weight: bold;
                        border-radius: 6px;
                        min-width: 100px;
                    }
                    QPushButton:hover {
                        background-color: #059669;
                    }
                """)
                msg_box.exec_()
                
                # 记录日志
                self.log_message("翻译缓存已清除")
                
                # 获取缓存统计信息
                cache_stats = self.translation_instance.get_cache_stats()
                self.log_message(f"缓存状态: 当前大小={cache_stats['current_size']}, 容量={cache_stats['capacity']}")
            except Exception as e:
                self.log_message(f"清除缓存失败: {str(e)}")
        else:
            self.log_message("翻译模块未初始化，无法清除缓存")


def main():
    app = QApplication(sys.argv)

    # 设置应用程序字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    # 设置应用程序图标
    icon_path = os.path.join(resources_dir, "translator_icon.ico")
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()