import os

project_root = os.getcwd()

os.environ["FUNASR_CACHE"] = os.path.join(project_root, "models", "cached_models")
os.environ["HF_HOME"] = os.path.join(project_root, "models", "hf_cache")
os.environ["MODELSCOPE_CACHE"] = os.path.join(project_root, "models", "modelscope_cache")
# 导入所需的库
import asyncio
import edge_tts
import pygame
import io
from pygame import mixer
import sys

# 定义支持的语音列表
# 结构: {"language_display": "用户界面语言名", "gender_display": "性别", "locale_display": "地区/方言", "voice_display": "语音名", "short_name": "edge-tts短名称"}
SUPPORTED_VOICES = [
    {"language_display": "南非荷兰语", "gender_display": "女性", "locale_display": "南非", "voice_display": "AdriNeural", "short_name": "af-ZA-AdriNeural"},
    {"language_display": "南非荷兰语", "gender_display": "男性", "locale_display": "南非", "voice_display": "WillemNeural", "short_name": "af-ZA-WillemNeural"},
    {"language_display": "阿姆哈拉语", "gender_display": "男性", "locale_display": "埃塞俄比亚", "voice_display": "AmehaNeural", "short_name": "am-ET-AmehaNeural"},
    {"language_display": "阿姆哈拉语", "gender_display": "女性", "locale_display": "埃塞俄比亚", "voice_display": "MekdesNeural", "short_name": "am-ET-MekdesNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "阿联酋", "voice_display": "FatimaNeural", "short_name": "ar-AE-FatimaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "阿联酋", "voice_display": "HamdanNeural", "short_name": "ar-AE-HamdanNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "巴林", "voice_display": "AliNeural", "short_name": "ar-BH-AliNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "巴林", "voice_display": "LailaNeural", "short_name": "ar-BH-LailaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "阿尔及利亚", "voice_display": "AminaNeural", "short_name": "ar-DZ-AminaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "阿尔及利亚", "voice_display": "IsmaelNeural", "short_name": "ar-DZ-IsmaelNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "埃及", "voice_display": "SalmaNeural", "short_name": "ar-EG-SalmaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "埃及", "voice_display": "ShakirNeural", "short_name": "ar-EG-ShakirNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "伊拉克", "voice_display": "BasselNeural", "short_name": "ar-IQ-BasselNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "伊拉克", "voice_display": "RanaNeural", "short_name": "ar-IQ-RanaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "约旦", "voice_display": "SanaNeural", "short_name": "ar-JO-SanaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "约旦", "voice_display": "TaimNeural", "short_name": "ar-JO-TaimNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "科威特", "voice_display": "FahedNeural", "short_name": "ar-KW-FahedNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "科威特", "voice_display": "NouraNeural", "short_name": "ar-KW-NouraNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "黎巴嫩", "voice_display": "LaylaNeural", "short_name": "ar-LB-LaylaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "黎巴嫩", "voice_display": "RamiNeural", "short_name": "ar-LB-RamiNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "利比亚", "voice_display": "ImanNeural", "short_name": "ar-LY-ImanNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "利比亚", "voice_display": "OmarNeural", "short_name": "ar-LY-OmarNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "摩洛哥", "voice_display": "JamalNeural", "short_name": "ar-MA-JamalNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "摩洛哥", "voice_display": "MounaNeural", "short_name": "ar-MA-MounaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "阿曼", "voice_display": "AbdullahNeural", "short_name": "ar-OM-AbdullahNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "阿曼", "voice_display": "AyshaNeural", "short_name": "ar-OM-AyshaNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "卡塔尔", "voice_display": "AmalNeural", "short_name": "ar-QA-AmalNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "卡塔尔", "voice_display": "MoazNeural", "short_name": "ar-QA-MoazNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "沙特阿拉伯", "voice_display": "HamedNeural", "short_name": "ar-SA-HamedNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "沙特阿拉伯", "voice_display": "ZariyahNeural", "short_name": "ar-SA-ZariyahNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "叙利亚", "voice_display": "AmanyNeural", "short_name": "ar-SY-AmanyNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "叙利亚", "voice_display": "LaithNeural", "short_name": "ar-SY-LaithNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "突尼斯", "voice_display": "HediNeural", "short_name": "ar-TN-HediNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "突尼斯", "voice_display": "ReemNeural", "short_name": "ar-TN-ReemNeural"},
    {"language_display": "阿拉伯语", "gender_display": "女性", "locale_display": "也门", "voice_display": "MaryamNeural", "short_name": "ar-YE-MaryamNeural"},
    {"language_display": "阿拉伯语", "gender_display": "男性", "locale_display": "也门", "voice_display": "SalehNeural", "short_name": "ar-YE-SalehNeural"},
    {"language_display": "阿塞拜疆语", "gender_display": "男性", "locale_display": "阿塞拜疆", "voice_display": "BabekNeural", "short_name": "az-AZ-BabekNeural"},
    {"language_display": "阿塞拜疆语", "gender_display": "女性", "locale_display": "阿塞拜疆", "voice_display": "BanuNeural", "short_name": "az-AZ-BanuNeural"},
    {"language_display": "保加利亚语", "gender_display": "男性", "locale_display": "保加利亚", "voice_display": "BorislavNeural", "short_name": "bg-BG-BorislavNeural"},
    {"language_display": "保加利亚语", "gender_display": "女性", "locale_display": "保加利亚", "voice_display": "KalinaNeural", "short_name": "bg-BG-KalinaNeural"},
    {"language_display": "孟加拉语", "gender_display": "女性", "locale_display": "孟加拉国", "voice_display": "NabanitaNeural", "short_name": "bn-BD-NabanitaNeural"},
    {"language_display": "孟加拉语", "gender_display": "男性", "locale_display": "孟加拉国", "voice_display": "PradeepNeural", "short_name": "bn-BD-PradeepNeural"},
    {"language_display": "孟加拉语", "gender_display": "男性", "locale_display": "印度", "voice_display": "BashkarNeural", "short_name": "bn-IN-BashkarNeural"},
    {"language_display": "孟加拉语", "gender_display": "女性", "locale_display": "印度", "voice_display": "TanishaaNeural", "short_name": "bn-IN-TanishaaNeural"},
    {"language_display": "波斯尼亚语", "gender_display": "男性", "locale_display": "波斯尼亚和黑塞哥维那", "voice_display": "GoranNeural", "short_name": "bs-BA-GoranNeural"},
    {"language_display": "波斯尼亚语", "gender_display": "女性", "locale_display": "波斯尼亚和黑塞哥维那", "voice_display": "VesnaNeural", "short_name": "bs-BA-VesnaNeural"},
    {"language_display": "加泰罗尼亚语", "gender_display": "男性", "locale_display": "西班牙", "voice_display": "EnricNeural", "short_name": "ca-ES-EnricNeural"},
    {"language_display": "加泰罗尼亚语", "gender_display": "女性", "locale_display": "西班牙", "voice_display": "JoanaNeural", "short_name": "ca-ES-JoanaNeural"},
    {"language_display": "捷克语", "gender_display": "男性", "locale_display": "捷克", "voice_display": "AntoninNeural", "short_name": "cs-CZ-AntoninNeural"},
    {"language_display": "捷克语", "gender_display": "女性", "locale_display": "捷克", "voice_display": "VlastaNeural", "short_name": "cs-CZ-VlastaNeural"},
    {"language_display": "威尔士语", "gender_display": "男性", "locale_display": "英国", "voice_display": "AledNeural", "short_name": "cy-GB-AledNeural"},
    {"language_display": "威尔士语", "gender_display": "女性", "locale_display": "英国", "voice_display": "NiaNeural", "short_name": "cy-GB-NiaNeural"},
    {"language_display": "丹麦语", "gender_display": "女性", "locale_display": "丹麦", "voice_display": "ChristelNeural", "short_name": "da-DK-ChristelNeural"},
    {"language_display": "丹麦语", "gender_display": "男性", "locale_display": "丹麦", "voice_display": "JeppeNeural", "short_name": "da-DK-JeppeNeural"},
    {"language_display": "德语", "gender_display": "女性", "locale_display": "奥地利", "voice_display": "IngridNeural", "short_name": "de-AT-IngridNeural"},
    {"language_display": "德语", "gender_display": "男性", "locale_display": "奥地利", "voice_display": "JonasNeural", "short_name": "de-AT-JonasNeural"},
    {"language_display": "德语", "gender_display": "男性", "locale_display": "瑞士", "voice_display": "JanNeural", "short_name": "de-CH-JanNeural"},
    {"language_display": "德语", "gender_display": "女性", "locale_display": "瑞士", "voice_display": "LeniNeural", "short_name": "de-CH-LeniNeural"},
    {"language_display": "德语", "gender_display": "女性", "locale_display": "德国", "voice_display": "AmalaNeural", "short_name": "de-DE-AmalaNeural"},
    {"language_display": "德语", "gender_display": "男性", "locale_display": "德国", "voice_display": "ConradNeural", "short_name": "de-DE-ConradNeural"},
    {"language_display": "德语", "gender_display": "男性", "locale_display": "德国", "voice_display": "FlorianMultilingualNeural", "short_name": "de-DE-FlorianMultilingualNeural"},
    {"language_display": "德语", "gender_display": "女性", "locale_display": "德国", "voice_display": "KatjaNeural", "short_name": "de-DE-KatjaNeural"},
    {"language_display": "德语", "gender_display": "男性", "locale_display": "德国", "voice_display": "KillianNeural", "short_name": "de-DE-KillianNeural"},
    {"language_display": "德语", "gender_display": "女性", "locale_display": "德国", "voice_display": "SeraphinaMultilingualNeural", "short_name": "de-DE-SeraphinaMultilingualNeural"},
    {"language_display": "希腊语", "gender_display": "女性", "locale_display": "希腊", "voice_display": "AthinaNeural", "short_name": "el-GR-AthinaNeural"},
    {"language_display": "希腊语", "gender_display": "男性", "locale_display": "希腊", "voice_display": "NestorasNeural", "short_name": "el-GR-NestorasNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "澳大利亚", "voice_display": "NatashaNeural", "short_name": "en-AU-NatashaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "澳大利亚", "voice_display": "WilliamNeural", "short_name": "en-AU-WilliamNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "加拿大", "voice_display": "ClaraNeural", "short_name": "en-CA-ClaraNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "加拿大", "voice_display": "LiamNeural", "short_name": "en-CA-LiamNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "英国", "voice_display": "LibbyNeural", "short_name": "en-GB-LibbyNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "英国", "voice_display": "MaisieNeural", "short_name": "en-GB-MaisieNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "英国", "voice_display": "RyanNeural", "short_name": "en-GB-RyanNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "英国", "voice_display": "SoniaNeural", "short_name": "en-GB-SoniaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "英国", "voice_display": "ThomasNeural", "short_name": "en-GB-ThomasNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "中国香港", "voice_display": "SamNeural", "short_name": "en-HK-SamNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "中国香港", "voice_display": "YanNeural", "short_name": "en-HK-YanNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "爱尔兰", "voice_display": "ConnorNeural", "short_name": "en-IE-ConnorNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "爱尔兰", "voice_display": "EmilyNeural", "short_name": "en-IE-EmilyNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "印度", "voice_display": "NeerjaExpressiveNeural", "short_name": "en-IN-NeerjaExpressiveNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "印度", "voice_display": "NeerjaNeural", "short_name": "en-IN-NeerjaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "印度", "voice_display": "PrabhatNeural", "short_name": "en-IN-PrabhatNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "肯尼亚", "voice_display": "AsiliaNeural", "short_name": "en-KE-AsiliaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "肯尼亚", "voice_display": "ChilembaNeural", "short_name": "en-KE-ChilembaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "尼日利亚", "voice_display": "AbeoNeural", "short_name": "en-NG-AbeoNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "尼日利亚", "voice_display": "EzinneNeural", "short_name": "en-NG-EzinneNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "新西兰", "voice_display": "MitchellNeural", "short_name": "en-NZ-MitchellNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "新西兰", "voice_display": "MollyNeural", "short_name": "en-NZ-MollyNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "菲律宾", "voice_display": "JamesNeural", "short_name": "en-PH-JamesNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "菲律宾", "voice_display": "RosaNeural", "short_name": "en-PH-RosaNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "新加坡", "voice_display": "LunaNeural", "short_name": "en-SG-LunaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "新加坡", "voice_display": "WayneNeural", "short_name": "en-SG-WayneNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "坦桑尼亚", "voice_display": "ElimuNeural", "short_name": "en-TZ-ElimuNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "坦桑尼亚", "voice_display": "ImaniNeural", "short_name": "en-TZ-ImaniNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "AnaNeural", "short_name": "en-US-AnaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "AndrewMultilingualNeural", "short_name": "en-US-AndrewMultilingualNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "AndrewNeural", "short_name": "en-US-AndrewNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "AriaNeural", "short_name": "en-US-AriaNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "AvaMultilingualNeural", "short_name": "en-US-AvaMultilingualNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "AvaNeural", "short_name": "en-US-AvaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "BrianMultilingualNeural", "short_name": "en-US-BrianMultilingualNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "BrianNeural", "short_name": "en-US-BrianNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "ChristopherNeural", "short_name": "en-US-ChristopherNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "EmmaMultilingualNeural", "short_name": "en-US-EmmaMultilingualNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "EmmaNeural", "short_name": "en-US-EmmaNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "EricNeural", "short_name": "en-US-EricNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "GuyNeural", "short_name": "en-US-GuyNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "JennyNeural", "short_name": "en-US-JennyNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "美国", "voice_display": "MichelleNeural", "short_name": "en-US-MichelleNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "RogerNeural", "short_name": "en-US-RogerNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "美国", "voice_display": "SteffanNeural", "short_name": "en-US-SteffanNeural"},
    {"language_display": "英语", "gender_display": "女性", "locale_display": "南非", "voice_display": "LeahNeural", "short_name": "en-ZA-LeahNeural"},
    {"language_display": "英语", "gender_display": "男性", "locale_display": "南非", "voice_display": "LukeNeural", "short_name": "en-ZA-LukeNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "阿根廷", "voice_display": "ElenaNeural", "short_name": "es-AR-ElenaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "阿根廷", "voice_display": "TomasNeural", "short_name": "es-AR-TomasNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "玻利维亚", "voice_display": "MarceloNeural", "short_name": "es-BO-MarceloNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "玻利维亚", "voice_display": "SofiaNeural", "short_name": "es-BO-SofiaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "智利", "voice_display": "CatalinaNeural", "short_name": "es-CL-CatalinaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "智利", "voice_display": "LorenzoNeural", "short_name": "es-CL-LorenzoNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "哥伦比亚", "voice_display": "GonzaloNeural", "short_name": "es-CO-GonzaloNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "哥伦比亚", "voice_display": "SalomeNeural", "short_name": "es-CO-SalomeNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "哥斯达黎加", "voice_display": "JuanNeural", "short_name": "es-CR-JuanNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "哥斯达黎加", "voice_display": "MariaNeural", "short_name": "es-CR-MariaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "古巴", "voice_display": "BelkysNeural", "short_name": "es-CU-BelkysNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "古巴", "voice_display": "ManuelNeural", "short_name": "es-CU-ManuelNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "多米尼加共和国", "voice_display": "EmilioNeural", "short_name": "es-DO-EmilioNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "多米尼加共和国", "voice_display": "RamonaNeural", "short_name": "es-DO-RamonaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "厄瓜多尔", "voice_display": "AndreaNeural", "short_name": "es-EC-AndreaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "厄瓜多尔", "voice_display": "LuisNeural", "short_name": "es-EC-LuisNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "西班牙", "voice_display": "AlvaroNeural", "short_name": "es-ES-AlvaroNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "西班牙", "voice_display": "ElviraNeural", "short_name": "es-ES-ElviraNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "西班牙", "voice_display": "XimenaNeural", "short_name": "es-ES-XimenaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "赤道几内亚", "voice_display": "JavierNeural", "short_name": "es-GQ-JavierNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "赤道几内亚", "voice_display": "TeresaNeural", "short_name": "es-GQ-TeresaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "危地马拉", "voice_display": "AndresNeural", "short_name": "es-GT-AndresNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "危地马拉", "voice_display": "MartaNeural", "short_name": "es-GT-MartaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "洪都拉斯", "voice_display": "CarlosNeural", "short_name": "es-HN-CarlosNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "洪都拉斯", "voice_display": "KarlaNeural", "short_name": "es-HN-KarlaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "墨西哥", "voice_display": "DaliaNeural", "short_name": "es-MX-DaliaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "墨西哥", "voice_display": "JorgeNeural", "short_name": "es-MX-JorgeNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "尼加拉瓜", "voice_display": "FedericoNeural", "short_name": "es-NI-FedericoNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "尼加拉瓜", "voice_display": "YolandaNeural", "short_name": "es-NI-YolandaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "巴拿马", "voice_display": "MargaritaNeural", "short_name": "es-PA-MargaritaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "巴拿马", "voice_display": "RobertoNeural", "short_name": "es-PA-RobertoNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "秘鲁", "voice_display": "AlexNeural", "short_name": "es-PE-AlexNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "秘鲁", "voice_display": "CamilaNeural", "short_name": "es-PE-CamilaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "波多黎各", "voice_display": "KarinaNeural", "short_name": "es-PR-KarinaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "波多黎各", "voice_display": "VictorNeural", "short_name": "es-PR-VictorNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "巴拉圭", "voice_display": "MarioNeural", "short_name": "es-PY-MarioNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "巴拉圭", "voice_display": "TaniaNeural", "short_name": "es-PY-TaniaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "萨尔瓦多", "voice_display": "LorenaNeural", "short_name": "es-SV-LorenaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "萨尔瓦多", "voice_display": "RodrigoNeural", "short_name": "es-SV-RodrigoNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "美国", "voice_display": "AlonsoNeural", "short_name": "es-US-AlonsoNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "美国", "voice_display": "PalomaNeural", "short_name": "es-US-PalomaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "乌拉圭", "voice_display": "MateoNeural", "short_name": "es-UY-MateoNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "乌拉圭", "voice_display": "ValentinaNeural", "short_name": "es-UY-ValentinaNeural"},
    {"language_display": "西班牙语", "gender_display": "女性", "locale_display": "委内瑞拉", "voice_display": "PaolaNeural", "short_name": "es-VE-PaolaNeural"},
    {"language_display": "西班牙语", "gender_display": "男性", "locale_display": "委内瑞拉", "voice_display": "SebastianNeural", "short_name": "es-VE-SebastianNeural"},
    {"language_display": "爱沙尼亚语", "gender_display": "女性", "locale_display": "爱沙尼亚", "voice_display": "AnuNeural", "short_name": "et-EE-AnuNeural"},
    {"language_display": "爱沙尼亚语", "gender_display": "男性", "locale_display": "爱沙尼亚", "voice_display": "KertNeural", "short_name": "et-EE-KertNeural"},
    {"language_display": "波斯语", "gender_display": "女性", "locale_display": "伊朗", "voice_display": "DilaraNeural", "short_name": "fa-IR-DilaraNeural"},
    {"language_display": "波斯语", "gender_display": "男性", "locale_display": "伊朗", "voice_display": "FaridNeural", "short_name": "fa-IR-FaridNeural"},
    {"language_display": "芬兰语", "gender_display": "男性", "locale_display": "芬兰", "voice_display": "HarriNeural", "short_name": "fi-FI-HarriNeural"},
    {"language_display": "芬兰语", "gender_display": "女性", "locale_display": "芬兰", "voice_display": "NooraNeural", "short_name": "fi-FI-NooraNeural"},
    {"language_display": "菲律宾语", "gender_display": "男性", "locale_display": "菲律宾", "voice_display": "AngeloNeural", "short_name": "fil-PH-AngeloNeural"},
    {"language_display": "菲律宾语", "gender_display": "女性", "locale_display": "菲律宾", "voice_display": "BlessicaNeural", "short_name": "fil-PH-BlessicaNeural"},
    {"language_display": "法语", "gender_display": "女性", "locale_display": "比利时", "voice_display": "CharlineNeural", "short_name": "fr-BE-CharlineNeural"},
    {"language_display": "法语", "gender_display": "男性", "locale_display": "比利时", "voice_display": "GerardNeural", "short_name": "fr-BE-GerardNeural"},
    {"language_display": "法语", "gender_display": "男性", "locale_display": "加拿大", "voice_display": "AntoineNeural", "short_name": "fr-CA-AntoineNeural"},
    {"language_display": "法语", "gender_display": "男性", "locale_display": "加拿大", "voice_display": "JeanNeural", "short_name": "fr-CA-JeanNeural"},
    {"language_display": "法语", "gender_display": "女性", "locale_display": "加拿大", "voice_display": "SylvieNeural", "short_name": "fr-CA-SylvieNeural"},
    {"language_display": "法语", "gender_display": "男性", "locale_display": "加拿大", "voice_display": "ThierryNeural", "short_name": "fr-CA-ThierryNeural"},
    {"language_display": "法语", "gender_display": "女性", "locale_display": "瑞士", "voice_display": "ArianeNeural", "short_name": "fr-CH-ArianeNeural"},
    {"language_display": "法语", "gender_display": "男性", "locale_display": "瑞士", "voice_display": "FabriceNeural", "short_name": "fr-CH-FabriceNeural"},
    {"language_display": "法语", "gender_display": "女性", "locale_display": "法国", "voice_display": "DeniseNeural", "short_name": "fr-FR-DeniseNeural"},
    {"language_display": "法语", "gender_display": "女性", "locale_display": "法国", "voice_display": "EloiseNeural", "short_name": "fr-FR-EloiseNeural"},
    {"language_display": "法语", "gender_display": "男性", "locale_display": "法国", "voice_display": "HenriNeural", "short_name": "fr-FR-HenriNeural"},
    {"language_display": "法语", "gender_display": "男性", "locale_display": "法国", "voice_display": "RemyMultilingualNeural", "short_name": "fr-FR-RemyMultilingualNeural"},
    {"language_display": "法语", "gender_display": "女性", "locale_display": "法国", "voice_display": "VivienneMultilingualNeural", "short_name": "fr-FR-VivienneMultilingualNeural"},
    {"language_display": "爱尔兰语", "gender_display": "男性", "locale_display": "爱尔兰", "voice_display": "ColmNeural", "short_name": "ga-IE-ColmNeural"},
    {"language_display": "爱尔兰语", "gender_display": "女性", "locale_display": "爱尔兰", "voice_display": "OrlaNeural", "short_name": "ga-IE-OrlaNeural"},
    {"language_display": "加利西亚语", "gender_display": "男性", "locale_display": "西班牙", "voice_display": "RoiNeural", "short_name": "gl-ES-RoiNeural"},
    {"language_display": "加利西亚语", "gender_display": "女性", "locale_display": "西班牙", "voice_display": "SabelaNeural", "short_name": "gl-ES-SabelaNeural"},
    {"language_display": "古吉拉特语", "gender_display": "女性", "locale_display": "印度", "voice_display": "DhwaniNeural", "short_name": "gu-IN-DhwaniNeural"},
    {"language_display": "古吉拉特语", "gender_display": "男性", "locale_display": "印度", "voice_display": "NiranjanNeural", "short_name": "gu-IN-NiranjanNeural"},
    {"language_display": "希伯来语", "gender_display": "男性", "locale_display": "以色列", "voice_display": "AvriNeural", "short_name": "he-IL-AvriNeural"},
    {"language_display": "希伯来语", "gender_display": "女性", "locale_display": "以色列", "voice_display": "HilaNeural", "short_name": "he-IL-HilaNeural"},
    {"language_display": "印地语", "gender_display": "男性", "locale_display": "印度", "voice_display": "MadhurNeural", "short_name": "hi-IN-MadhurNeural"},
    {"language_display": "印地语", "gender_display": "女性", "locale_display": "印度", "voice_display": "SwaraNeural", "short_name": "hi-IN-SwaraNeural"},
    {"language_display": "克罗地亚语", "gender_display": "女性", "locale_display": "克罗地亚", "voice_display": "GabrijelaNeural", "short_name": "hr-HR-GabrijelaNeural"},
    {"language_display": "克罗地亚语", "gender_display": "男性", "locale_display": "克罗地亚", "voice_display": "SreckoNeural", "short_name": "hr-HR-SreckoNeural"},
    {"language_display": "匈牙利语", "gender_display": "女性", "locale_display": "匈牙利", "voice_display": "NoemiNeural", "short_name": "hu-HU-NoemiNeural"},
    {"language_display": "匈牙利语", "gender_display": "男性", "locale_display": "匈牙利", "voice_display": "TamasNeural", "short_name": "hu-HU-TamasNeural"},
    {"language_display": "印尼语", "gender_display": "男性", "locale_display": "印度尼西亚", "voice_display": "ArdiNeural", "short_name": "id-ID-ArdiNeural"},
    {"language_display": "印尼语", "gender_display": "女性", "locale_display": "印度尼西亚", "voice_display": "GadisNeural", "short_name": "id-ID-GadisNeural"},
    {"language_display": "冰岛语", "gender_display": "女性", "locale_display": "冰岛", "voice_display": "GudrunNeural", "short_name": "is-IS-GudrunNeural"},
    {"language_display": "冰岛语", "gender_display": "男性", "locale_display": "冰岛", "voice_display": "GunnarNeural", "short_name": "is-IS-GunnarNeural"},
    {"language_display": "意大利语", "gender_display": "男性", "locale_display": "意大利", "voice_display": "DiegoNeural", "short_name": "it-IT-DiegoNeural"},
    {"language_display": "意大利语", "gender_display": "女性", "locale_display": "意大利", "voice_display": "ElsaNeural", "short_name": "it-IT-ElsaNeural"},
    {"language_display": "意大利语", "gender_display": "男性", "locale_display": "意大利", "voice_display": "GiuseppeMultilingualNeural", "short_name": "it-IT-GiuseppeMultilingualNeural"},
    {"language_display": "意大利语", "gender_display": "女性", "locale_display": "意大利", "voice_display": "IsabellaNeural", "short_name": "it-IT-IsabellaNeural"},
    {"language_display": "因纽特语", "gender_display": "女性", "locale_display": "加拿大 (加拿大统一音节文字)", "voice_display": "SiqiniqNeural", "short_name": "iu-Cans-CA-SiqiniqNeural"},
    {"language_display": "因纽特语", "gender_display": "男性", "locale_display": "加拿大 (加拿大统一音节文字)", "voice_display": "TaqqiqNeural", "short_name": "iu-Cans-CA-TaqqiqNeural"},
    {"language_display": "因纽特语", "gender_display": "女性", "locale_display": "加拿大 (拉丁字母)", "voice_display": "SiqiniqNeural", "short_name": "iu-Latn-CA-SiqiniqNeural"},
    {"language_display": "因纽特语", "gender_display": "男性", "locale_display": "加拿大 (拉丁字母)", "voice_display": "TaqqiqNeural", "short_name": "iu-Latn-CA-TaqqiqNeural"},
    {"language_display": "日语", "gender_display": "男性", "locale_display": "日本", "voice_display": "KeitaNeural", "short_name": "ja-JP-KeitaNeural"},
    {"language_display": "日语", "gender_display": "女性", "locale_display": "日本", "voice_display": "NanamiNeural", "short_name": "ja-JP-NanamiNeural"},
    {"language_display": "爪哇语", "gender_display": "男性", "locale_display": "印度尼西亚", "voice_display": "DimasNeural", "short_name": "jv-ID-DimasNeural"},
    {"language_display": "爪哇语", "gender_display": "女性", "locale_display": "印度尼西亚", "voice_display": "SitiNeural", "short_name": "jv-ID-SitiNeural"},
    {"language_display": "格鲁吉亚语", "gender_display": "女性", "locale_display": "格鲁吉亚", "voice_display": "EkaNeural", "short_name": "ka-GE-EkaNeural"},
    {"language_display": "格鲁吉亚语", "gender_display": "男性", "locale_display": "格鲁吉亚", "voice_display": "GiorgiNeural", "short_name": "ka-GE-GiorgiNeural"},
    {"language_display": "哈萨克语", "gender_display": "女性", "locale_display": "哈萨克斯坦", "voice_display": "AigulNeural", "short_name": "kk-KZ-AigulNeural"},
    {"language_display": "哈萨克语", "gender_display": "男性", "locale_display": "哈萨克斯坦", "voice_display": "DauletNeural", "short_name": "kk-KZ-DauletNeural"},
    {"language_display": "高棉语", "gender_display": "男性", "locale_display": "柬埔寨", "voice_display": "PisethNeural", "short_name": "km-KH-PisethNeural"},
    {"language_display": "高棉语", "gender_display": "女性", "locale_display": "柬埔寨", "voice_display": "SreymomNeural", "short_name": "km-KH-SreymomNeural"},
    {"language_display": "卡纳达语", "gender_display": "男性", "locale_display": "印度", "voice_display": "GaganNeural", "short_name": "kn-IN-GaganNeural"},
    {"language_display": "卡纳达语", "gender_display": "女性", "locale_display": "印度", "voice_display": "SapnaNeural", "short_name": "kn-IN-SapnaNeural"},
    {"language_display": "韩语", "gender_display": "男性", "locale_display": "韩国", "voice_display": "HyunsuMultilingualNeural", "short_name": "ko-KR-HyunsuMultilingualNeural"},
    {"language_display": "韩语", "gender_display": "男性", "locale_display": "韩国", "voice_display": "InJoonNeural", "short_name": "ko-KR-InJoonNeural"},
    {"language_display": "韩语", "gender_display": "女性", "locale_display": "韩国", "voice_display": "SunHiNeural", "short_name": "ko-KR-SunHiNeural"},
    {"language_display": "老挝语", "gender_display": "男性", "locale_display": "老挝", "voice_display": "ChanthavongNeural", "short_name": "lo-LA-ChanthavongNeural"},
    {"language_display": "老挝语", "gender_display": "女性", "locale_display": "老挝", "voice_display": "KeomanyNeural", "short_name": "lo-LA-KeomanyNeural"},
    {"language_display": "立陶宛语", "gender_display": "男性", "locale_display": "立陶宛", "voice_display": "LeonasNeural", "short_name": "lt-LT-LeonasNeural"},
    {"language_display": "立陶宛语", "gender_display": "女性", "locale_display": "立陶宛", "voice_display": "OnaNeural", "short_name": "lt-LT-OnaNeural"},
    {"language_display": "拉脱维亚语", "gender_display": "女性", "locale_display": "拉脱维亚", "voice_display": "EveritaNeural", "short_name": "lv-LV-EveritaNeural"},
    {"language_display": "拉脱维亚语", "gender_display": "男性", "locale_display": "拉脱维亚", "voice_display": "NilsNeural", "short_name": "lv-LV-NilsNeural"},
    {"language_display": "马其顿语", "gender_display": "男性", "locale_display": "北马其顿", "voice_display": "AleksandarNeural", "short_name": "mk-MK-AleksandarNeural"},
    {"language_display": "马其顿语", "gender_display": "女性", "locale_display": "北马其顿", "voice_display": "MarijaNeural", "short_name": "mk-MK-MarijaNeural"},
    {"language_display": "马拉雅拉姆语", "gender_display": "男性", "locale_display": "印度", "voice_display": "MidhunNeural", "short_name": "ml-IN-MidhunNeural"},
    {"language_display": "马拉雅拉姆语", "gender_display": "女性", "locale_display": "印度", "voice_display": "SobhanaNeural", "short_name": "ml-IN-SobhanaNeural"},
    {"language_display": "蒙古语", "gender_display": "男性", "locale_display": "蒙古", "voice_display": "BataaNeural", "short_name": "mn-MN-BataaNeural"},
    {"language_display": "蒙古语", "gender_display": "女性", "locale_display": "蒙古", "voice_display": "YesuiNeural", "short_name": "mn-MN-YesuiNeural"},
    {"language_display": "马拉地语", "gender_display": "女性", "locale_display": "印度", "voice_display": "AarohiNeural", "short_name": "mr-IN-AarohiNeural"},
    {"language_display": "马拉地语", "gender_display": "男性", "locale_display": "印度", "voice_display": "ManoharNeural", "short_name": "mr-IN-ManoharNeural"},
    {"language_display": "马来语", "gender_display": "男性", "locale_display": "马来西亚", "voice_display": "OsmanNeural", "short_name": "ms-MY-OsmanNeural"},
    {"language_display": "马来语", "gender_display": "女性", "locale_display": "马来西亚", "voice_display": "YasminNeural", "short_name": "ms-MY-YasminNeural"},
    {"language_display": "马耳他语", "gender_display": "女性", "locale_display": "马耳他", "voice_display": "GraceNeural", "short_name": "mt-MT-GraceNeural"},
    {"language_display": "马耳他语", "gender_display": "男性", "locale_display": "马耳他", "voice_display": "JosephNeural", "short_name": "mt-MT-JosephNeural"},
    {"language_display": "缅甸语", "gender_display": "女性", "locale_display": "缅甸", "voice_display": "NilarNeural", "short_name": "my-MM-NilarNeural"},
    {"language_display": "缅甸语", "gender_display": "男性", "locale_display": "缅甸", "voice_display": "ThihaNeural", "short_name": "my-MM-ThihaNeural"},
    {"language_display": "挪威博克马尔语", "gender_display": "男性", "locale_display": "挪威", "voice_display": "FinnNeural", "short_name": "nb-NO-FinnNeural"},
    {"language_display": "挪威博克马尔语", "gender_display": "女性", "locale_display": "挪威", "voice_display": "PernilleNeural", "short_name": "nb-NO-PernilleNeural"},
    {"language_display": "尼泊尔语", "gender_display": "女性", "locale_display": "尼泊尔", "voice_display": "HemkalaNeural", "short_name": "ne-NP-HemkalaNeural"},
    {"language_display": "尼泊尔语", "gender_display": "男性", "locale_display": "尼泊尔", "voice_display": "SagarNeural", "short_name": "ne-NP-SagarNeural"},
    {"language_display": "荷兰语", "gender_display": "男性", "locale_display": "比利时", "voice_display": "ArnaudNeural", "short_name": "nl-BE-ArnaudNeural"},
    {"language_display": "荷兰语", "gender_display": "女性", "locale_display": "比利时", "voice_display": "DenaNeural", "short_name": "nl-BE-DenaNeural"},
    {"language_display": "荷兰语", "gender_display": "女性", "locale_display": "荷兰", "voice_display": "ColetteNeural", "short_name": "nl-NL-ColetteNeural"},
    {"language_display": "荷兰语", "gender_display": "女性", "locale_display": "荷兰", "voice_display": "FennaNeural", "short_name": "nl-NL-FennaNeural"},
    {"language_display": "荷兰语", "gender_display": "男性", "locale_display": "荷兰", "voice_display": "MaartenNeural", "short_name": "nl-NL-MaartenNeural"},
    {"language_display": "波兰语", "gender_display": "男性", "locale_display": "波兰", "voice_display": "MarekNeural", "short_name": "pl-PL-MarekNeural"},
    {"language_display": "波兰语", "gender_display": "女性", "locale_display": "波兰", "voice_display": "ZofiaNeural", "short_name": "pl-PL-ZofiaNeural"},
    {"language_display": "普什图语", "gender_display": "男性", "locale_display": "阿富汗", "voice_display": "GulNawazNeural", "short_name": "ps-AF-GulNawazNeural"},
    {"language_display": "普什图语", "gender_display": "女性", "locale_display": "阿富汗", "voice_display": "LatifaNeural", "short_name": "ps-AF-LatifaNeural"},
    {"language_display": "葡萄牙语", "gender_display": "男性", "locale_display": "巴西", "voice_display": "AntonioNeural", "short_name": "pt-BR-AntonioNeural"},
    {"language_display": "葡萄牙语", "gender_display": "女性", "locale_display": "巴西", "voice_display": "FranciscaNeural", "short_name": "pt-BR-FranciscaNeural"},
    {"language_display": "葡萄牙语", "gender_display": "女性", "locale_display": "巴西", "voice_display": "ThalitaMultilingualNeural", "short_name": "pt-BR-ThalitaMultilingualNeural"},
    {"language_display": "葡萄牙语", "gender_display": "男性", "locale_display": "葡萄牙", "voice_display": "DuarteNeural", "short_name": "pt-PT-DuarteNeural"},
    {"language_display": "葡萄牙语", "gender_display": "女性", "locale_display": "葡萄牙", "voice_display": "RaquelNeural", "short_name": "pt-PT-RaquelNeural"},
    {"language_display": "罗马尼亚语", "gender_display": "女性", "locale_display": "罗马尼亚", "voice_display": "AlinaNeural", "short_name": "ro-RO-AlinaNeural"},
    {"language_display": "罗马尼亚语", "gender_display": "男性", "locale_display": "罗马尼亚", "voice_display": "EmilNeural", "short_name": "ro-RO-EmilNeural"},
    {"language_display": "俄语", "gender_display": "男性", "locale_display": "俄罗斯", "voice_display": "DmitryNeural", "short_name": "ru-RU-DmitryNeural"},
    {"language_display": "俄语", "gender_display": "女性", "locale_display": "俄罗斯", "voice_display": "SvetlanaNeural", "short_name": "ru-RU-SvetlanaNeural"},
    {"language_display": "僧伽罗语", "gender_display": "男性", "locale_display": "斯里兰卡", "voice_display": "SameeraNeural", "short_name": "si-LK-SameeraNeural"},
    {"language_display": "僧伽罗语", "gender_display": "女性", "locale_display": "斯里兰卡", "voice_display": "ThiliniNeural", "short_name": "si-LK-ThiliniNeural"},
    {"language_display": "斯洛伐克语", "gender_display": "男性", "locale_display": "斯洛伐克", "voice_display": "LukasNeural", "short_name": "sk-SK-LukasNeural"},
    {"language_display": "斯洛伐克语", "gender_display": "女性", "locale_display": "斯洛伐克", "voice_display": "ViktoriaNeural", "short_name": "sk-SK-ViktoriaNeural"},
    {"language_display": "斯洛文尼亚语", "gender_display": "女性", "locale_display": "斯洛文尼亚", "voice_display": "PetraNeural", "short_name": "sl-SI-PetraNeural"},
    {"language_display": "斯洛文尼亚语", "gender_display": "男性", "locale_display": "斯洛文尼亚", "voice_display": "RokNeural", "short_name": "sl-SI-RokNeural"},
    {"language_display": "索马里语", "gender_display": "男性", "locale_display": "索马里", "voice_display": "MuuseNeural", "short_name": "so-SO-MuuseNeural"},
    {"language_display": "索马里语", "gender_display": "女性", "locale_display": "索马里", "voice_display": "UbaxNeural", "short_name": "so-SO-UbaxNeural"},
    {"language_display": "阿尔巴尼亚语", "gender_display": "女性", "locale_display": "阿尔巴尼亚", "voice_display": "AnilaNeural", "short_name": "sq-AL-AnilaNeural"},
    {"language_display": "阿尔巴尼亚语", "gender_display": "男性", "locale_display": "阿尔巴尼亚", "voice_display": "IlirNeural", "short_name": "sq-AL-IlirNeural"},
    {"language_display": "塞尔维亚语", "gender_display": "男性", "locale_display": "塞尔维亚", "voice_display": "NicholasNeural", "short_name": "sr-RS-NicholasNeural"},
    {"language_display": "塞尔维亚语", "gender_display": "女性", "locale_display": "塞尔维亚", "voice_display": "SophieNeural", "short_name": "sr-RS-SophieNeural"},
    {"language_display": "巽他语", "gender_display": "男性", "locale_display": "印度尼西亚", "voice_display": "JajangNeural", "short_name": "su-ID-JajangNeural"},
    {"language_display": "巽他语", "gender_display": "女性", "locale_display": "印度尼西亚", "voice_display": "TutiNeural", "short_name": "su-ID-TutiNeural"},
    {"language_display": "瑞典语", "gender_display": "男性", "locale_display": "瑞典", "voice_display": "MattiasNeural", "short_name": "sv-SE-MattiasNeural"},
    {"language_display": "瑞典语", "gender_display": "女性", "locale_display": "瑞典", "voice_display": "SofieNeural", "short_name": "sv-SE-SofieNeural"},
    {"language_display": "斯瓦希里语", "gender_display": "男性", "locale_display": "肯尼亚", "voice_display": "RafikiNeural", "short_name": "sw-KE-RafikiNeural"},
    {"language_display": "斯瓦希里语", "gender_display": "女性", "locale_display": "肯尼亚", "voice_display": "ZuriNeural", "short_name": "sw-KE-ZuriNeural"},
    {"language_display": "斯瓦希里语", "gender_display": "男性", "locale_display": "坦桑尼亚", "voice_display": "DaudiNeural", "short_name": "sw-TZ-DaudiNeural"},
    {"language_display": "斯瓦希里语", "gender_display": "女性", "locale_display": "坦桑尼亚", "voice_display": "RehemaNeural", "short_name": "sw-TZ-RehemaNeural"},
    {"language_display": "泰米尔语", "gender_display": "女性", "locale_display": "印度", "voice_display": "PallaviNeural", "short_name": "ta-IN-PallaviNeural"},
    {"language_display": "泰米尔语", "gender_display": "男性", "locale_display": "印度", "voice_display": "ValluvarNeural", "short_name": "ta-IN-ValluvarNeural"},
    {"language_display": "泰米尔语", "gender_display": "男性", "locale_display": "斯里兰卡", "voice_display": "KumarNeural", "short_name": "ta-LK-KumarNeural"},
    {"language_display": "泰米尔语", "gender_display": "女性", "locale_display": "斯里兰卡", "voice_display": "SaranyaNeural", "short_name": "ta-LK-SaranyaNeural"},
    {"language_display": "泰米尔语", "gender_display": "女性", "locale_display": "马来西亚", "voice_display": "KaniNeural", "short_name": "ta-MY-KaniNeural"},
    {"language_display": "泰米尔语", "gender_display": "男性", "locale_display": "马来西亚", "voice_display": "SuryaNeural", "short_name": "ta-MY-SuryaNeural"},
    {"language_display": "泰米尔语", "gender_display": "男性", "locale_display": "新加坡", "voice_display": "AnbuNeural", "short_name": "ta-SG-AnbuNeural"},
    {"language_display": "泰米尔语", "gender_display": "女性", "locale_display": "新加坡", "voice_display": "VenbaNeural", "short_name": "ta-SG-VenbaNeural"},
    {"language_display": "泰卢固语", "gender_display": "男性", "locale_display": "印度", "voice_display": "MohanNeural", "short_name": "te-IN-MohanNeural"},
    {"language_display": "泰卢固语", "gender_display": "女性", "locale_display": "印度", "voice_display": "ShrutiNeural", "short_name": "te-IN-ShrutiNeural"},
    {"language_display": "泰语", "gender_display": "男性", "locale_display": "泰国", "voice_display": "NiwatNeural", "short_name": "th-TH-NiwatNeural"},
    {"language_display": "泰语", "gender_display": "女性", "locale_display": "泰国", "voice_display": "PremwadeeNeural", "short_name": "th-TH-PremwadeeNeural"},
    {"language_display": "土耳其语", "gender_display": "男性", "locale_display": "土耳其", "voice_display": "AhmetNeural", "short_name": "tr-TR-AhmetNeural"},
    {"language_display": "土耳其语", "gender_display": "女性", "locale_display": "土耳其", "voice_display": "EmelNeural", "short_name": "tr-TR-EmelNeural"},
    {"language_display": "乌克兰语", "gender_display": "男性", "locale_display": "乌克兰", "voice_display": "OstapNeural", "short_name": "uk-UA-OstapNeural"},
    {"language_display": "乌克兰语", "gender_display": "女性", "locale_display": "乌克兰", "voice_display": "PolinaNeural", "short_name": "uk-UA-PolinaNeural"},
    {"language_display": "乌尔都语", "gender_display": "女性", "locale_display": "印度", "voice_display": "GulNeural", "short_name": "ur-IN-GulNeural"},
    {"language_display": "乌尔都语", "gender_display": "男性", "locale_display": "印度", "voice_display": "SalmanNeural", "short_name": "ur-IN-SalmanNeural"},
    {"language_display": "乌尔都语", "gender_display": "男性", "locale_display": "巴基斯坦", "voice_display": "AsadNeural", "short_name": "ur-PK-AsadNeural"},
    {"language_display": "乌尔都语", "gender_display": "女性", "locale_display": "巴基斯坦", "voice_display": "UzmaNeural", "short_name": "ur-PK-UzmaNeural"},
    {"language_display": "乌兹别克语", "gender_display": "女性", "locale_display": "乌兹别克斯坦", "voice_display": "MadinaNeural", "short_name": "uz-UZ-MadinaNeural"},
    {"language_display": "乌兹别克语", "gender_display": "男性", "locale_display": "乌兹别克斯坦", "voice_display": "SardorNeural", "short_name": "uz-UZ-SardorNeural"},
    {"language_display": "越南语", "gender_display": "女性", "locale_display": "越南", "voice_display": "HoaiMyNeural", "short_name": "vi-VN-HoaiMyNeural"},
    {"language_display": "越南语", "gender_display": "男性", "locale_display": "越南", "voice_display": "NamMinhNeural", "short_name": "vi-VN-NamMinhNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "普通话 - 中国", "voice_display": "XiaoxiaoNeural", "short_name": "zh-CN-XiaoxiaoNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "普通话 - 中国", "voice_display": "XiaoyiNeural", "short_name": "zh-CN-XiaoyiNeural"},
    {"language_display": "中文", "gender_display": "男性", "locale_display": "普通话 - 中国", "voice_display": "YunjianNeural", "short_name": "zh-CN-YunjianNeural"},
    {"language_display": "中文", "gender_display": "男性", "locale_display": "普通话 - 中国", "voice_display": "YunxiNeural", "short_name": "zh-CN-YunxiNeural"},
    {"language_display": "中文", "gender_display": "男性", "locale_display": "普通话 - 中国", "voice_display": "YunxiaNeural", "short_name": "zh-CN-YunxiaNeural"},
    {"language_display": "中文", "gender_display": "男性", "locale_display": "普通话 - 中国", "voice_display": "YunyangNeural", "short_name": "zh-CN-YunyangNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "辽宁话 - 中国", "voice_display": "XiaobeiNeural", "short_name": "zh-CN-liaoning-XiaobeiNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "陕西话 - 中国", "voice_display": "XiaoniNeural", "short_name": "zh-CN-shaanxi-XiaoniNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "粤语 - 中国香港", "voice_display": "HiuGaaiNeural", "short_name": "zh-HK-HiuGaaiNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "粤语 - 中国香港", "voice_display": "HiuMaanNeural", "short_name": "zh-HK-HiuMaanNeural"},
    {"language_display": "中文", "gender_display": "男性", "locale_display": "粤语 - 中国香港", "voice_display": "WanLungNeural", "short_name": "zh-HK-WanLungNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "中文 - 中国台湾", "voice_display": "HsiaoChenNeural", "short_name": "zh-TW-HsiaoChenNeural"},
    {"language_display": "中文", "gender_display": "女性", "locale_display": "中文 - 中国台湾", "voice_display": "HsiaoYuNeural", "short_name": "zh-TW-HsiaoYuNeural"},
    {"language_display": "中文", "gender_display": "男性", "locale_display": "中文 - 中国台湾", "voice_display": "YunJheNeural", "short_name": "zh-TW-YunJheNeural"},
    {"language_display": "祖鲁语", "gender_display": "女性", "locale_display": "南非", "voice_display": "ThandoNeural", "short_name": "zu-ZA-ThandoNeural"},
    {"language_display": "祖鲁语", "gender_display": "男性", "locale_display": "南非", "voice_display": "ThembaNeural", "short_name": "zu-ZA-ThembaNeural"},
]

# 添加根据语言代码筛选语音的函数
async def list_voices_by_language(language_code):
    """
    根据语言代码返回对应的语音列表
    
    参数:
        language_code (str): 语言代码，如 'zh-CN', 'en-US' 等
    
    返回:
        list: 包含该语言可用语音的列表，每个语音是一个字典，包含 ShortName 等信息
    """
    voices = []
    language_code_lower = language_code.lower()
    
    # 处理特殊情况
    if language_code_lower == "cn":
        language_code_lower = "zh-cn"
    elif language_code_lower == "en":
        language_code_lower = "en-us"
    
    # 尝试不同的匹配方式
    for voice in SUPPORTED_VOICES:
        short_name = voice["short_name"].lower()
        
        # 1. 直接匹配开头
        if short_name.startswith(language_code_lower):
            voices.append({
                "ShortName": voice["short_name"],
                "Gender": "Female" if voice["gender_display"] == "女性" else "Male",
                "Locale": voice["short_name"].split("-")[0] + "-" + voice["short_name"].split("-")[1],
                "DisplayName": f"{voice['voice_display']} ({voice['locale_display']})",
                "LocalName": voice["voice_display"]
            })
            continue
            
        # 2. 匹配语言代码的第一部分（如 "en" 匹配 "en-US", "en-GB" 等）
        if "-" in language_code_lower:
            lang_prefix = language_code_lower.split("-")[0]
            if short_name.startswith(lang_prefix + "-"):
                voices.append({
                    "ShortName": voice["short_name"],
                    "Gender": "Female" if voice["gender_display"] == "女性" else "Male",
                    "Locale": voice["short_name"].split("-")[0] + "-" + voice["short_name"].split("-")[1],
                    "DisplayName": f"{voice['voice_display']} ({voice['locale_display']})",
                    "LocalName": voice["voice_display"]
                })
        
        # 3. 处理特殊映射（如 "zh" 匹配 "zh-CN", "zh-TW", "zh-HK" 等）
        if language_code_lower == "zh" and short_name.startswith("zh-"):
            voices.append({
                "ShortName": voice["short_name"],
                "Gender": "Female" if voice["gender_display"] == "女性" else "Male",
                "Locale": voice["short_name"].split("-")[0] + "-" + voice["short_name"].split("-")[1],
                "DisplayName": f"{voice['voice_display']} ({voice['locale_display']})",
                "LocalName": voice["voice_display"]
            })
    
    print(f"为语言 {language_code} 找到 {len(voices)} 个音色")
    return voices

async def play_audio_from_memory(audio_data):
    """直接从内存播放音频数据 (假定mixer已初始化)"""
    if not mixer.get_init():
        print("错误: Pygame Mixer 未初始化。")
        return

    audio_io = io.BytesIO(audio_data)
    print("正在播放语音...")
    mixer.music.load(audio_io)
    mixer.music.play()

    while mixer.music.get_busy():
        await asyncio.sleep(0.1)
    mixer.music.unload()
    print("播放完成！")

async def text_to_speech(text, voice, rate=None, volume=None):
    """将文本转换为语音并直接播放（不保存文件）

    参数:
        text (str): 要转换的文本.
        voice (str): 使用的语音名称 (e.g., \'en-US-AriaNeural\').
        rate (str, optional): 语速调整 (e.g., \'+20%\', \'-10%\'). 默认为 None.
        volume (str, optional): 音量调整 (e.g., \'+15%\', \'-5%\'). 默认为 None.
    """
    log_message = f"正在使用音色 {voice}"
    if rate is not None:
        log_message += f", 语速: {rate}"
    if volume is not None:
        log_message += f", 音量: {volume}"
    log_message += " 生成语音..."
    print(log_message)

    try:
        tts_options = {}
        if rate is not None:
            tts_options['rate'] = rate
        if volume is not None:
            tts_options['volume'] = volume
        communicate = edge_tts.Communicate(text, voice, **tts_options)
        audio_data = bytes()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        if not audio_data:
            print("警告：未生成音频数据，可能是文本或语音选择有问题")
            return False

        print("语音生成完成，准备播放...")
        await play_audio_from_memory(audio_data)
        return True

    except edge_tts.exceptions.NoAudioReceived:
        print("错误：未收到音频数据。可能的原因：")
        print("1. 所选语音不支持输入的文本")
        print("2. 网络连接问题")
        print("3. 请尝试不同的语音或更简短的文本")
        return False
    except Exception as e:
        print(f"错误：生成语音时发生异常: {str(e)}")
        return False

async def get_supported_languages():
    """
    获取所有支持的语言列表
    
    返回:
        list: 包含所有支持的语言代码的列表
    """
    languages = set()
    for voice in SUPPORTED_VOICES:
        short_name = voice["short_name"]
        if "-" in short_name:
            lang_code = short_name.split("-")[0] + "-" + short_name.split("-")[1]
            languages.add(lang_code)
    
    return sorted(list(languages))

async def main():
    """主函数，交互式运行TTS"""
    print("=== Edge TTS 交互式演示程序 (新流程) ===")

    try:
        mixer.init()  # 初始化 Pygame Mixer

        while True:
            # 1. 获取用户输入的文本
            text = input("请输入要转换为语音的文本 (输入\'退出\'结束程序): ")
            if text.lower() in ['退出', 'exit', 'quit']:
                print("程序已退出。")
                break

            # 2. 选择语种
            available_languages = sorted(list(set(v["language_display"] for v in SUPPORTED_VOICES)))
            print("\n可用的语种:")
            for i, lang_name in enumerate(available_languages, 1):
                print(f"{i}. {lang_name}")

            selected_language_display = None
            while True:
                try:
                    choice = int(input("\n请选择语种编号: "))
                    if 1 <= choice <= len(available_languages):
                        selected_language_display = available_languages[choice - 1]
                        break
                    else:
                        print("无效的选择，请重试。")
                except ValueError:
                    print("请输入数字。")
                except KeyboardInterrupt: print("\n程序已中断。"); sys.exit(0)


            # 3. 选择性别
            voices_in_lang = [v for v in SUPPORTED_VOICES if v["language_display"] == selected_language_display]
            available_genders = sorted(list(set(v["gender_display"] for v in voices_in_lang)))
            print(f"\n可用的性别 ({selected_language_display}):")
            for i, gender_name in enumerate(available_genders, 1):
                print(f"{i}. {gender_name}")
            
            selected_gender_display = None
            while True:
                try:
                    choice = int(input("\n请选择性别编号: "))
                    if 1 <= choice <= len(available_genders):
                        selected_gender_display = available_genders[choice - 1]
                        break
                    else:
                        print("无效的选择，请重试。")
                except ValueError:
                    print("请输入数字。")
                except KeyboardInterrupt: print("\n程序已中断。"); sys.exit(0)

            # 4. 选择音色
            voices_for_selection = [
                v for v in voices_in_lang if v["gender_display"] == selected_gender_display
            ]
            print(f"\n可用的音色 ({selected_language_display} - {selected_gender_display}):")
            for i, voice_info in enumerate(voices_for_selection, 1):
                # 组合显示名称：地区/方言 - 语音名
                display_name = f"{voice_info['locale_display']} - {voice_info['voice_display']}"
                print(f"{i}. {display_name}")

            selected_voice_short_name = None
            while True:
                try:
                    choice = int(input("\n请选择音色编号: "))
                    if 1 <= choice <= len(voices_for_selection):
                        selected_voice_short_name = voices_for_selection[choice - 1]["short_name"]
                        break
                    else:
                        print("无效的选择，请重试。")
                except ValueError:
                    print("请输入数字。")
                except KeyboardInterrupt: print("\n程序已中断。"); sys.exit(0)

            # 5. 执行文本到语音转换并播放
            if selected_voice_short_name:
                success = await text_to_speech(text, selected_voice_short_name)
                if success:
                    choice = input("\n是否继续? (y/n): ")
                    if choice.lower() not in ['y', 'yes', '是']:
                        print("程序已退出。")
                        break
                else:
                    print("TTS失败，请尝试其他选项。")
            else:
                print("未选择有效的音色。")

    except Exception as e:
        print(f"主程序发生错误: {e}")
    finally:
        if mixer.get_init():
            mixer.quit() # 关闭 Pygame Mixer

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已被用户中断。")
    except Exception as e:
        print(f"程序启动时发生错误: {str(e)}")