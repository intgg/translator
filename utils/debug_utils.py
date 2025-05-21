# utils/debug_utils.py - 调试工具函数

import time
from datetime import datetime

# 全局配置
ENABLE_DEBUG = True
LOG_TO_FILE = False
LOG_FILE = "debug.log"

# 日志级别
LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3
CURRENT_LOG_LEVEL = LOG_LEVEL_INFO  # 默认日志级别

# 日志类别
LOG_CATEGORY_ASR = "ASR"
LOG_CATEGORY_TTS = "TTS"
LOG_CATEGORY_TRANSLATOR = "TRANSLATOR"
LOG_CATEGORY_CONTROLLER = "CONTROLLER"
LOG_CATEGORY_SENTENCE = "SENTENCE"
LOG_CATEGORY_GENERAL = "GENERAL"


def log(message, category=LOG_CATEGORY_GENERAL, level=LOG_LEVEL_INFO, show_time=True):
    """记录日志消息

    参数:
        message: 日志消息
        category: 日志类别
        level: 日志级别
        show_time: 是否显示时间戳
    """
    if not ENABLE_DEBUG or level < CURRENT_LOG_LEVEL:
        return

    # 构建日志消息
    if show_time:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        formatted_message = f"[{timestamp}] [{category}] {message}"
    else:
        formatted_message = f"[{category}] {message}"

    # 打印到控制台
    print(formatted_message)

    # 写入日志文件
    if LOG_TO_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(formatted_message + "\n")
        except Exception as e:
            print(f"写入日志文件失败: {e}")


def get_timestamp():
    """获取当前时间戳"""
    return time.time()


def format_time_diff(start_time):
    """格式化时间差

    参数:
        start_time: 开始时间戳

    返回:
        str: 格式化的时间差
    """
    diff = time.time() - start_time
    if diff < 1:
        return f"{diff * 1000:.2f}ms"
    elif diff < 60:
        return f"{diff:.2f}s"
    else:
        minutes = int(diff // 60)
        seconds = diff % 60
        return f"{minutes}m {seconds:.2f}s"


class Timer:
    """计时器类，用于性能分析"""

    def __init__(self, name, category=LOG_CATEGORY_GENERAL):
        """初始化计时器

        参数:
            name: 计时器名称
            category: 日志类别
        """
        self.name = name
        self.category = category
        self.start_time = None

    def __enter__(self):
        """进入上下文"""
        self.start_time = get_timestamp()
        log(f"开始 {self.name}", self.category, LOG_LEVEL_DEBUG)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self.start_time:
            duration = format_time_diff(self.start_time)
            log(f"完成 {self.name}: {duration}", self.category, LOG_LEVEL_DEBUG)


def dump_object_state(obj, prefix="", exclude_attrs=None):
    """转储对象状态用于调试

    参数:
        obj: 要转储的对象
        prefix: 属性名前缀
        exclude_attrs: 要排除的属性列表

    返回:
        dict: 对象状态字典
    """
    if exclude_attrs is None:
        exclude_attrs = []

    # 排除常见的不需要转储的属性
    exclude_attrs.extend(['__dict__', '__class__', '__module__', '__weakref__'])

    state = {}

    for attr in dir(obj):
        if attr.startswith('_') or attr in exclude_attrs:
            continue

        try:
            value = getattr(obj, attr)

            # 排除可调用对象
            if callable(value):
                continue

            # 对于简单类型直接添加
            if isinstance(value, (str, int, float, bool, type(None))):
                state[f"{prefix}{attr}"] = value
            # 对于列表、元组、集合和字典等容器类型
            elif isinstance(value, (list, tuple, set)):
                state[f"{prefix}{attr}"] = f"{type(value).__name__}[{len(value)}]"
            elif isinstance(value, dict):
                state[f"{prefix}{attr}"] = f"dict[{len(value)}]"
            # 对于其他复杂对象
            else:
                state[f"{prefix}{attr}"] = f"{type(value).__name__}"
        except Exception as e:
            state[f"{prefix}{attr}"] = f"<访问错误: {e}>"

    return state


def print_object_state(obj, category=LOG_CATEGORY_GENERAL, exclude_attrs=None):
    """打印对象状态用于调试

    参数:
        obj: 要打印的对象
        category: 日志类别
        exclude_attrs: 要排除的属性列表
    """
    state = dump_object_state(obj, exclude_attrs=exclude_attrs)

    log(f"对象 {obj.__class__.__name__} 状态:", category, LOG_LEVEL_DEBUG)
    for attr, value in state.items():
        log(f"  {attr}: {value}", category, LOG_LEVEL_DEBUG)


def set_log_level(level):
    """设置日志级别

    参数:
        level: 日志级别
    """
    global CURRENT_LOG_LEVEL
    CURRENT_LOG_LEVEL = level
    log(f"日志级别设置为: {level}", LOG_CATEGORY_GENERAL, LOG_LEVEL_INFO)


def enable_file_logging(filename=None):
    """启用文件日志

    参数:
        filename: 日志文件名
    """
    global LOG_TO_FILE, LOG_FILE
    LOG_TO_FILE = True
    if filename:
        LOG_FILE = filename
    log(f"文件日志已启用，写入到: {LOG_FILE}", LOG_CATEGORY_GENERAL, LOG_LEVEL_INFO)


def disable_file_logging():
    """禁用文件日志"""
    global LOG_TO_FILE
    LOG_TO_FILE = False
    log("文件日志已禁用", LOG_CATEGORY_GENERAL, LOG_LEVEL_INFO)