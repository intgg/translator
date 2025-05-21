# config.py - 配置管理模块 - 优化参数版

# 翻译API配置
TRANSLATION_APP_ID = "86c79fb7"
TRANSLATION_API_SECRET = "MDY3ZGFkYWEyZDBiOTJkOGIyOTllOWMz"
TRANSLATION_API_KEY = "f4369644e37eddd43adfe436e7904cf1"

# 语言配置
DEFAULT_SOURCE_LANG = "cn"  # 默认源语言（中文）
DEFAULT_TARGET_LANG = "en"  # 默认目标语言（英语）

# 是否允许源语言和目标语言相同
ALLOW_SAME_SOURCE_TARGET = True  # 设置为True，允许中文→中文的翻译

# 语言映射
# 将语言代码映射为显示名称
LANGUAGE_DISPLAY_NAMES = {
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

# 将语言代码映射为TTS语言代码
TTS_LANGUAGE_MAPPING = {
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

# TTS配置
DEFAULT_TTS_SPEED = 0      # 默认语速（0%）
DEFAULT_TTS_VOLUME = 0     # 默认音量（0%）
DEFAULT_TTS_PITCH = 0      # 默认音调（0Hz）

# ASR配置
ASR_USE_VAD = True         # 是否使用VAD
ASR_USE_PUNC = True        # 是否使用标点恢复
ASR_DISABLE_UPDATE = True  # 是否禁用FunASR更新检查

# TTS触发配置 - 优化后的参数
TTS_STRONG_TRIGGER_TIME = 3.0    # 强触发时间阈值（秒）
TTS_MEDIUM_TRIGGER_TIME = 2.0    # 中等触发时间阈值（秒）
TTS_TEXT_LENGTH_THRESHOLD = 50   # 文本长度触发阈值（字符数）
TTS_MEDIUM_TEXT_LENGTH = 30      # 中等文本长度阈值（字符数）
TTS_MIN_TEXT_LENGTH = 15         # 最小文本长度（字符数），提高到15
TEXT_STABILITY_TIME = 0.8        # 文本稳定时间（秒）

# 句子管理配置
SENTENCE_MIN_LENGTH = 15          # 最小有效句子长度
SENTENCE_MERGE_THRESHOLD = 0.7    # 句子合并阈值
SENTENCE_SIMILARITY_THRESHOLD = 0.9  # 相似度阈值

# 日志配置
ENABLE_DEBUG_LOGS = True   # 是否启用调试日志