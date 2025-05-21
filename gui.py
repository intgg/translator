# gui.py - 支持智能句子管理和增强日志的图形用户界面

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import queue
import sys
import os
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和控制器
import config
from controller import InterpreterController
from utils.debug_utils import log, LOG_CATEGORY_GENERAL


class InterpreterGUI(tk.Tk):
    """实时口译系统的图形用户界面"""

    def __init__(self):
        super().__init__()

        # 配置主窗口
        self.title("实时口译系统 - 智能句子管理版")
        self.geometry("900x650")
        self.minsize(700, 500)

        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("TCombobox", font=("Arial", 10))

        # 定义日志类型样式
        self.log_tags = {
            "ASR": "asr",  # 语音识别相关
            "TTS": "tts",  # 语音合成相关
            "TRANSLATOR": "trans",  # 翻译相关
            "SYSTEM": "system",  # 系统状态相关
            "ERROR": "error",  # 错误信息
            "SENTENCE": "sentence",  # 句子管理相关
            "INFO": "info"  # 一般信息
        }

        # 状态变量
        self.is_running = False
        self.voice_id_map = {}  # 音色显示名称到ID的映射

        # 日志队列和历史
        self.log_queue = queue.Queue()
        self.log_history = []  # 存储所有日志历史，用于过滤

        # 创建控制器
        self.controller = InterpreterController()

        # 添加自定义的事件拦截方法
        self.original_send_event = self.controller._send_event
        self.controller._send_event = self._intercepted_send_event

        # 创建用户界面
        self.create_widgets()

        # 启动事件处理线程
        self.event_thread = threading.Thread(target=self.process_events, daemon=True)
        self.event_thread.start()

        # 启动日志更新线程
        self.log_thread = threading.Thread(target=self.update_log, daemon=True)
        self.log_thread.start()

        # 在关闭窗口时停止所有进程
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 初始化控制器
        self.initialize_controller()

    def _intercepted_send_event(self, event_type, data):
        """拦截控制器事件以记录到日志"""
        # 首先调用原始方法
        self.original_send_event(event_type, data)

        # 然后根据事件类型记录日志
        if event_type == "source_text_update":
            self.add_log(f"语音识别: {data['text'][:50]}{'...' if len(data['text']) > 50 else ''}", "ASR")

        elif event_type == "translated_text_update":
            self.add_log(f"翻译结果: {data['text'][:50]}{'...' if len(data['text']) > 50 else ''}", "TRANSLATOR")

        elif event_type == "tts_play":
            self.add_log(f"开始语音合成: {data['text'][:50]}{'...' if len(data['text']) > 50 else ''}", "TTS")

        elif event_type == "tts_stop":
            self.add_log("语音播放停止", "TTS")

        elif event_type == "system_start":
            self.add_log(f"系统启动 - 源语言: {data['source_lang']}, 目标语言: {data['target_lang']}", "SYSTEM")

        elif event_type == "system_stop":
            self.add_log("系统停止", "SYSTEM")

        elif event_type == "log":
            # 处理直接发送的日志事件
            self.add_log(data['message'], data.get('type', 'INFO'))

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
        source_lang_name = config.LANGUAGE_DISPLAY_NAMES[config.DEFAULT_SOURCE_LANG]
        ttk.Label(lang_frame, text=source_lang_name).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # 目标语言下拉菜单
        ttk.Label(lang_frame, text="目标语言:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        # 准备目标语言选项（如果允许相同语言，则包括源语言）
        if hasattr(config, 'ALLOW_SAME_SOURCE_TARGET') and config.ALLOW_SAME_SOURCE_TARGET:
            self.target_lang_options = {code: name for code, name in config.LANGUAGE_DISPLAY_NAMES.items()}
        else:
            self.target_lang_options = {code: name for code, name in config.LANGUAGE_DISPLAY_NAMES.items()
                                        if code != config.DEFAULT_SOURCE_LANG}

        self.target_lang_var = tk.StringVar(value=config.DEFAULT_TARGET_LANG)
        self.target_lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.target_lang_var,
            values=[f"{code} ({name})" for code, name in self.target_lang_options.items()],
            state="readonly",
            width=15
        )
        self.target_lang_combo.set(
            f"{config.DEFAULT_TARGET_LANG} ({self.target_lang_options[config.DEFAULT_TARGET_LANG]})")
        self.target_lang_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.target_lang_combo.bind("<<ComboboxSelected>>", self.on_language_change)

        # 音色选择区域
        voice_frame = ttk.Frame(control_frame)
        voice_frame.pack(side=tk.LEFT, fill=tk.X, padx=10)

        ttk.Label(voice_frame, text="音色:").pack(side=tk.LEFT, padx=5, pady=5)

        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(
            voice_frame,
            textvariable=self.voice_var,
            state="readonly",
            width=25
        )
        self.voice_combo.pack(side=tk.LEFT, padx=5, pady=5)
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_change)

        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)

        # 添加停止播放按钮
        self.stop_playback_button = ttk.Button(
            button_frame,
            text="停止播放",
            command=self.stop_current_playback
        )
        self.stop_playback_button.pack(side=tk.RIGHT, padx=5)

        self.start_stop_button = ttk.Button(button_frame, text="开始录音", command=self.toggle_recording)
        self.start_stop_button.pack(side=tk.RIGHT, padx=5)

        # 创建主内容区域分隔器（可调整比例）
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # 创建文本区域框架
        text_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(text_frame, weight=60)

        # 创建日志区域框架
        log_frame = ttk.LabelFrame(self.paned_window, text="系统日志")
        self.paned_window.add(log_frame, weight=40)

        # 文本区域的内容
        # 原文文本框
        source_frame = ttk.LabelFrame(text_frame, text=f"原文 ({source_lang_name})")
        source_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.source_text = scrolledtext.ScrolledText(source_frame, wrap=tk.WORD, height=8)
        self.source_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 译文文本框
        target_lang_name = self.target_lang_options[config.DEFAULT_TARGET_LANG]
        self.translation_frame = ttk.LabelFrame(text_frame, text=f"译文 ({target_lang_name})")
        self.translation_frame.pack(fill=tk.BOTH, expand=True)

        self.translation_text = scrolledtext.ScrolledText(self.translation_frame, wrap=tk.WORD, height=8)
        self.translation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 日志区域内容
        # 日志工具栏
        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, side=tk.TOP, padx=5, pady=2)

        # 日志过滤器
        ttk.Label(log_toolbar, text="过滤:").pack(side=tk.LEFT, padx=(0, 5))

        self.filter_var = tk.StringVar(value="全部")
        self.filter_combo = ttk.Combobox(
            log_toolbar,
            textvariable=self.filter_var,
            values=["全部", "语音识别", "翻译", "语音合成", "句子管理", "系统状态", "错误"],
            state="readonly",
            width=10
        )
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.apply_log_filter)

        # 自动滚动
        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_check = ttk.Checkbutton(
            log_toolbar,
            text="自动滚动",
            variable=self.auto_scroll_var
        )
        auto_scroll_check.pack(side=tk.LEFT, padx=10)

        # 清空日志按钮
        clear_log_button = ttk.Button(
            log_toolbar,
            text="清空日志",
            command=self.clear_log
        )
        clear_log_button.pack(side=tk.RIGHT, padx=5)

        # 导出日志按钮
        export_log_button = ttk.Button(
            log_toolbar,
            text="导出日志",
            command=self.export_log
        )
        export_log_button.pack(side=tk.RIGHT, padx=5)

        # 日志文本区域
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

        # 配置日志文本标签样式
        self.log_text.tag_configure(self.log_tags["ASR"], foreground="#0066CC")
        self.log_text.tag_configure(self.log_tags["TTS"], foreground="#009933")
        self.log_text.tag_configure(self.log_tags["TRANSLATOR"], foreground="#9900CC")
        self.log_text.tag_configure(self.log_tags["SYSTEM"], foreground="#FF6600")
        self.log_text.tag_configure(self.log_tags["ERROR"], foreground="#FF0000")
        self.log_text.tag_configure(self.log_tags["SENTENCE"], foreground="#666699")
        self.log_text.tag_configure(self.log_tags["INFO"], foreground="#000000")

        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, padx=5, pady=2)

    def initialize_controller(self):
        """初始化控制器"""
        self.update_status("正在初始化系统...")
        self.add_log("开始初始化系统...", "SYSTEM")

        # 禁用控件
        self.start_stop_button.config(state="disabled")
        self.stop_playback_button.config(state="disabled")  # 禁用停止播放按钮
        self.target_lang_combo.config(state="disabled")
        self.voice_combo.config(state="disabled")

        # 初始化控制器
        if not self.controller.initialize():
            self.update_status("系统初始化失败")
            self.add_log("系统初始化失败", "ERROR")
            return

        # 启用控件
        self.start_stop_button.config(state="normal")
        self.stop_playback_button.config(state="normal")  # 启用停止播放按钮
        self.target_lang_combo.config(state="readonly")

        # 更新音色列表
        self.update_voice_list()
        self.voice_combo.config(state="readonly")

        self.update_status("系统初始化完成，可以开始录音")
        self.add_log("系统初始化完成", "SYSTEM")

    def update_voice_list(self):
        """根据选择的语言更新音色列表"""
        # 获取当前选择的目标语言
        selection = self.target_lang_combo.get()
        lang_code = selection.split()[0]

        # 获取该语言的可用音色
        voices = self.controller.tts.get_voices_for_language(lang_code)
        self.add_log(f"获取语言 {lang_code} 的可用音色", "TTS")

        if voices:
            # 构建音色选项 [f"{name} ({gender})"]
            voice_options = [f"{name} ({gender})" for voice_id, name, gender in voices]
            self.voice_combo['values'] = voice_options

            # 记录音色ID与显示名称的映射
            self.voice_id_map = {f"{name} ({gender})": voice_id for voice_id, name, gender in voices}

            # 选择默认音色
            if voice_options:
                self.voice_combo.set(voice_options[0])

                # 设置控制器使用的音色
                voice_id = self.voice_id_map[voice_options[0]]
                self.controller.tts.set_voice(voice_id)
                self.add_log(f"已设置默认音色: {voice_options[0]} (ID: {voice_id})", "TTS")
        else:
            self.voice_combo['values'] = ["无可用音色"]
            self.voice_combo.set("无可用音色")
            self.add_log(f"语言 {lang_code} 没有可用音色", "TTS")

    def toggle_recording(self):
        """切换录音状态"""
        if not self.is_running:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """开始录音 - 改进版"""
        self.update_status("正在启动...")
        self.add_log("准备开始录音...", "SYSTEM")

        # 获取当前选择的目标语言
        target_lang = self.target_lang_var.get().split()[0]
        self.add_log(f"目标语言: {target_lang}", "SYSTEM")

        # 清空文本框
        self.source_text.delete(1.0, tk.END)
        self.translation_text.delete(1.0, tk.END)

        # 确保停止任何正在播放的内容
        self.controller.tts.stop_current_playback()
        self.add_log("停止之前的TTS播放", "TTS")

        # 启动控制器
        if not self.controller.start(target_lang):
            self.update_status("启动失败")
            self.add_log("系统启动失败", "ERROR")
            return

        # 更新UI状态
        self.is_running = True
        self.start_stop_button.config(text="停止录音")
        self.target_lang_combo.config(state="disabled")
        self.voice_combo.config(state="disabled")  # 禁用音色选择

        self.update_status("录音中...")
        self.add_log("录音已开始，等待语音输入...", "SYSTEM")

    def stop_recording(self):
        """停止录音"""
        self.update_status("正在停止...")
        self.add_log("准备停止录音...", "SYSTEM")

        # 停止控制器
        self.controller.stop()

        # 更新UI状态
        self.is_running = False
        self.start_stop_button.config(text="开始录音")
        self.target_lang_combo.config(state="readonly")
        self.voice_combo.config(state="readonly")  # 启用音色选择

        self.update_status("已停止录音")
        self.add_log("录音已停止", "SYSTEM")

    def stop_current_playback(self):
        """停止当前播放并清空队列"""
        self.add_log("用户请求停止当前TTS播放", "TTS")
        if self.controller.stop_current_tts():
            self.update_status("已停止当前语音播放")
            self.add_log("成功停止当前TTS播放", "TTS")
        else:
            self.update_status("当前没有正在播放的内容")
            self.add_log("当前没有正在播放的内容", "TTS")

    def on_language_change(self, event):
        """处理语言选择变化"""
        # 获取选择的语言代码
        selection = self.target_lang_combo.get()
        lang_code = selection.split()[0]
        lang_name = self.target_lang_options[lang_code]

        # 更新目标语言框标题
        self.translation_frame.config(text=f"译文 ({lang_name})")

        # 如果系统已启动，则不更新控制器中的语言设置
        if not self.is_running:
            self.add_log(f"切换目标语言: {lang_code} ({lang_name})", "SYSTEM")
            self.controller.set_target_language(lang_code)

            # 更新音色列表
            self.update_voice_list()

        self.update_status(f"目标语言已设置为: {lang_name}")

    def on_voice_change(self, event):
        """处理音色选择变化"""
        selection = self.voice_combo.get()

        # 检查选择是否有效
        if selection in self.voice_id_map:
            voice_id = self.voice_id_map[selection]

            # 设置TTS使用的音色
            self.controller.tts.set_voice(voice_id)
            self.update_status(f"已设置音色: {selection}")
            self.add_log(f"切换语音音色: {selection} (ID: {voice_id})", "TTS")

    def process_events(self):
        """处理控制器事件的线程函数"""
        while True:
            try:
                # 检查事件队列
                if hasattr(self.controller, 'event_queue') and not self.controller.event_queue.empty():
                    event = self.controller.event_queue.get()

                    # 处理不同类型的事件
                    if event["type"] == "source_text_update":
                        self.update_source_text(event["data"]["text"])

                    elif event["type"] == "translated_text_update":
                        self.update_translation_text(event["data"]["text"])

                    elif event["type"] == "tts_play":
                        self.update_status(f"正在朗读译文...")

                    elif event["type"] == "tts_stop":
                        self.update_status(f"已停止语音播放")

                    elif event["type"] == "system_start":
                        self.update_status("系统已启动，请对着麦克风说话")

                    elif event["type"] == "system_stop":
                        self.update_status("系统已停止")

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"事件处理错误: {e}")
                self.add_log(f"事件处理错误: {str(e)}", "ERROR")

    def update_source_text(self, text):
        """更新原文文本框"""
        # 在主线程中更新UI
        self.after(0, lambda: self._update_text(self.source_text, text))

    def update_translation_text(self, text):
        """更新译文文本框"""
        # 在主线程中更新UI
        self.after(0, lambda: self._update_text(self.translation_text, text))

    def _update_text(self, text_widget, new_text):
        """更新文本控件的通用方法"""
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, new_text)
        text_widget.see(tk.END)  # 滚动到最新内容

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        log(message, LOG_CATEGORY_GENERAL)

    def add_log(self, message, log_type="INFO"):
        """添加日志消息到队列"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # 特殊处理决策日志
        if log_type == "SENTENCE" and message.startswith("TTS决策:"):
            # 突出显示决策日志
            log_message = f"[{timestamp}] [决策] {message}"
        else:
            log_message = f"[{timestamp}] [{log_type}] {message}"

        # 将日志消息添加到历史记录
        self.log_history.append((log_message, log_type))

        # 将日志消息放入队列
        self.log_queue.put((log_message, log_type))

    def update_log(self):
        """更新日志显示的线程函数"""
        while True:
            try:
                # 检查日志队列
                if not self.log_queue.empty():
                    log_message, log_type = self.log_queue.get()

                    # 在主线程中更新UI
                    self.after(0, lambda m=log_message, t=log_type: self._update_log_text(m, t))

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"更新日志错误: {e}")

    def _update_log_text(self, message, log_type="INFO"):
        """更新日志文本控件"""
        # 获取当前过滤器
        current_filter = self.filter_var.get()

        # 检查是否应该显示这个日志
        should_display = True

        if current_filter != "全部":
            filter_map = {
                "语音识别": "ASR",
                "翻译": "TRANSLATOR",
                "语音合成": "TTS",
                "句子管理": "SENTENCE",
                "系统状态": "SYSTEM",
                "错误": "ERROR"
            }

            if filter_map.get(current_filter) != log_type:
                should_display = False

        if should_display:
            # 获取对应的标签
            tag = self.log_tags.get(log_type, self.log_tags["INFO"])

            # 添加消息
            self.log_text.insert(tk.END, message + "\n", tag)

            # 如果启用了自动滚动，则滚动到最新内容
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)

    def apply_log_filter(self, event):
        """应用日志过滤器"""
        selected_filter = self.filter_var.get()
        self.add_log(f"应用日志过滤器: {selected_filter}", "SYSTEM")

        # 清空日志文本框
        self.log_text.delete(1.0, tk.END)

        # 获取过滤器对应的日志类型
        filter_map = {
            "全部": None,
            "语音识别": "ASR",
            "翻译": "TRANSLATOR",
            "语音合成": "TTS",
            "句子管理": "SENTENCE",
            "系统状态": "SYSTEM",
            "错误": "ERROR"
        }

        filter_type = filter_map.get(selected_filter)

        # 重新显示符合条件的日志
        for log_message, log_type in self.log_history:
            if filter_type is None or log_type == filter_type:
                tag = self.log_tags.get(log_type, self.log_tags["INFO"])
                self.log_text.insert(tk.END, log_message + "\n", tag)

        # 滚动到最新内容
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)

    def clear_log(self):
        """清空日志文本框"""
        self.log_text.delete(1.0, tk.END)
        self.log_history = []  # 清空历史记录
        self.add_log("日志已清空", "SYSTEM")

    def export_log(self):
        """导出日志到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interpreter_log_{timestamp}.txt"

            with open(filename, "w", encoding="utf-8") as f:
                for log_message, _ in self.log_history:
                    f.write(log_message + "\n")

            self.add_log(f"日志已导出到文件: {filename}", "SYSTEM")
            self.update_status(f"日志已导出到: {filename}")
        except Exception as e:
            self.add_log(f"导出日志失败: {str(e)}", "ERROR")
            self.update_status("导出日志失败")

    def on_closing(self):
        """窗口关闭处理"""
        if self.is_running:
            self.controller.stop()
            self.add_log("关闭程序，停止系统", "SYSTEM")

        self.destroy()


def main():
    """主函数"""
    app = InterpreterGUI()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback

        traceback.print_exc()