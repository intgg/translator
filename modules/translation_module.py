# modules/translation_module.py - 翻译模块

import time
import hashlib
import base64
import hmac
import json
import re
from datetime import datetime
from wsgiref.handlers import format_date_time
from time import mktime
from urllib.parse import urlencode

# 尝试导入更高效的HTTP库
try:
    import httpx

    use_httpx = True
    print("使用httpx加速HTTP请求")
except ImportError:
    import requests

    use_httpx = False
    print("使用标准requests库")


class TranslationModule:
    """翻译模块"""

    def __init__(self, app_id, api_secret, api_key):
        """初始化翻译模块"""
        print("初始化翻译模块...")
        self.app_id = app_id
        self.api_secret = api_secret
        self.api_key = api_key
        self.url = 'https://itrans.xf-yun.com/v1/its'

        # HTTP客户端设置
        self.timeout = 5.0  # 请求超时设置

        # 缓存最近的翻译以减少重复请求
        self.cache = {}
        self.cache_size = 50  # 最大缓存条目数

        # 如果使用httpx，创建一个客户端实例
        if use_httpx:
            self.client = httpx.Client(timeout=self.timeout)
        else:
            self.client = None

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

    def translate(self, text, from_lang="cn", to_lang="en", use_cache=True):
        """翻译文本"""
        if not text or not text.strip():
            return ""

        # 源语言与目标语言相同，直接返回原文
        if from_lang == to_lang:
            return text

        # 检查缓存
        if use_cache:
            cache_key = f"{text}_{from_lang}_{to_lang}"
            if cache_key in self.cache:
                return self.cache[cache_key]

        try:
            # 生成URL
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            # 解析URL
            url_parts = self.parse_url(self.url)
            host = url_parts["host"]
            path = url_parts["path"]

            # 生成签名
            signature_origin = f"host: {host}\ndate: {date}\nPOST {path} HTTP/1.1"
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
            if use_httpx and self.client:
                response = self.client.post(
                    request_url,
                    json=body,
                    headers=headers
                )
                result = response.json()
            else:
                json_data = json.dumps(body)
                response = requests.post(
                    request_url,
                    data=json_data,
                    headers=headers,
                    timeout=self.timeout
                )
                result = json.loads(response.content.decode())

            # 解析响应
            if 'payload' in result and 'result' in result['payload'] and 'text' in result['payload']['result']:
                translated_text_base64 = result['payload']['result']['text']
                translated_text = base64.b64decode(translated_text_base64).decode()

                try:
                    # 尝试解析JSON响应
                    json_result = json.loads(translated_text)
                    if 'trans_result' in json_result and 'dst' in json_result['trans_result']:
                        translated = json_result['trans_result']['dst']
                    elif 'dst' in json_result:
                        translated = json_result['dst']
                    else:
                        translated = translated_text
                except:
                    # 不是JSON格式，返回原始文本
                    translated = translated_text

                # 更新缓存
                if use_cache:
                    # 如果缓存已满，移除最早的条目
                    if len(self.cache) >= self.cache_size:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                    self.cache[cache_key] = translated

                return translated
            else:
                print(f"翻译API错误: {result}")
                return None
        except Exception as e:
            print(f"翻译过程出错: {e}")
            return None

    def __del__(self):
        """清理资源"""
        if use_httpx and self.client:
            try:
                self.client.close()
            except:
                pass