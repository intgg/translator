# modules/sentence_manager.py - 优化版智能句子管理模块

import re
import time
import difflib
import hashlib


class SentenceManager:
    """智能句子管理器 - 处理文本分割、状态跟踪和TTS触发决策"""

    def __init__(self):
        """初始化句子管理器"""
        # 句子状态字典 - 键为句子文本，值为状态信息
        self.sentences = {}
        # 句子ID映射 - 用于追踪句子及其变体
        self.sentence_ids = {}

        # 配置参数
        self.stability_threshold = 0.8  # 稳定性阈值(秒)
        self.max_wait_time = 3.0  # 最大等待时间(秒)
        self.min_diff_ratio = 0.2  # 最小差异比例(播放新句子的阈值)
        self.min_sentence_length = 15  # 最小句子长度阈值 - 增加到15字符
        self.merge_threshold = 0.7  # 句子合并阈值 - 两个句子相似度超过此值可以合并

        # 跟踪变量
        self.played_sentences = set()  # 已播放句子集合
        self.last_played_time = 0  # 上次播放时间
        self.last_update_time = 0  # 上次更新时间
        self.processed_text = ""  # 最近处理的完整文本

    def process_text(self, text, vad_pause_detected=False):
        """处理输入文本，更新句子状态

        参数:
            text: 完整文本 (ASR结果)
            vad_pause_detected: 是否检测到VAD停顿

        返回:
            dict: {
                'sentences_to_translate': 需要翻译的句子列表,
                'sentences_to_play': 可以播放的句子列表,
                'status_changes': 状态变化信息
            }
        """
        current_time = time.time()
        self.last_update_time = current_time

        # 记录状态变化
        status_changes = []

        # 如果文本与上次完全相同，直接返回当前状态
        if text == self.processed_text:
            sentences_to_translate = self._get_sentences_to_translate()
            sentences_to_play = self._get_sentences_to_play()
            return {
                'sentences_to_translate': sentences_to_translate,
                'sentences_to_play': sentences_to_play,
                'status_changes': ['文本未变化']
            }

        # 更新最近处理的文本
        self.processed_text = text

        # 分割文本获取句子
        sentences = self._split_into_sentences(text)

        # 合并短句
        merged_sentences = self._merge_short_sentences(sentences)
        if len(merged_sentences) != len(sentences):
            status_changes.append(f'合并了{len(sentences) - len(merged_sentences)}个短句')

        # 更新句子状态
        for sentence in merged_sentences:
            # 清理句子文本
            sentence = sentence.strip()
            if not sentence:
                continue

            # 生成或获取句子ID
            sentence_id = self._get_sentence_id(sentence)

            # 如果是新句子，添加到状态跟踪
            if sentence_id not in self.sentences:
                self.sentences[sentence_id] = {
                    'text': sentence,
                    'created_at': current_time,
                    'updated_at': current_time,
                    'completion_time': None,  # 完整性确认时间
                    'stability_time': None,  # 稳定性确认时间
                    'is_complete': self._is_complete_sentence(sentence),
                    'is_stable': False,
                    'translated': False,
                    'translation': None,
                    'played': False,
                    'length': len(sentence)  # 记录句子长度
                }

                status_changes.append(f'新句子: {sentence[:20]}{"..." if len(sentence) > 20 else ""}')

                # 检查新句子是否已经完整
                if self.sentences[sentence_id]['is_complete'] or vad_pause_detected:
                    self.sentences[sentence_id]['completion_time'] = current_time
                    status_changes.append(f'句子已标记为完整')
            else:
                # 更新已有句子状态
                state = self.sentences[sentence_id]
                state['updated_at'] = current_time

                # 检查完整性状态是否变化
                is_complete = self._is_complete_sentence(sentence)
                if is_complete and not state['is_complete']:
                    state['is_complete'] = True
                    state['completion_time'] = current_time
                    status_changes.append(f'句子变为完整: {sentence[:20]}{"..." if len(sentence) > 20 else ""}')
                elif vad_pause_detected and not state['is_complete']:
                    state['is_complete'] = True
                    state['completion_time'] = current_time
                    status_changes.append(f'句子因VAD停顿标记为完整')

                # 更新文本和长度
                if sentence != state['text']:
                    old_length = state['length']
                    state['text'] = sentence
                    state['length'] = len(sentence)
                    status_changes.append(f'句子更新: {old_length} -> {state["length"]}字符')

        # 更新所有句子的状态
        stability_updates = self._update_sentence_states(current_time, vad_pause_detected)
        status_changes.extend(stability_updates)

        # 获取需要翻译和播放的句子
        sentences_to_translate = self._get_sentences_to_translate()
        sentences_to_play = self._get_sentences_to_play()

        return {
            'sentences_to_translate': sentences_to_translate,
            'sentences_to_play': sentences_to_play,
            'status_changes': status_changes
        }

    def _update_sentence_states(self, current_time, vad_pause_detected):
        """更新所有句子的状态

        参数:
            current_time: 当前时间戳
            vad_pause_detected: 是否检测到VAD停顿

        返回:
            list: 状态变化信息列表
        """
        status_changes = []

        for sentence_id, state in list(self.sentences.items()):
            # 已播放的句子跳过处理
            if state['played']:
                continue

            # 检查句子是否稳定 (1. 已标记完整且经过稳定时间, 或 2. 达到最大等待时间)
            if not state['is_stable']:
                time_since_completion = (current_time - state['completion_time']) if state['completion_time'] else 0
                time_since_creation = current_time - state['created_at']

                if (state['is_complete'] and time_since_completion >= self.stability_threshold) or \
                        (time_since_creation >= self.max_wait_time):
                    state['is_stable'] = True
                    state['stability_time'] = current_time
                    status_changes.append(f'句子稳定: {state["text"][:20]}... (等待: {time_since_creation:.2f}秒)')

        return status_changes

    def _get_sentences_to_translate(self):
        """获取需要翻译的句子列表

        返回:
            list: 需要翻译的句子列表
        """
        sentences_to_translate = []

        for sentence_id, state in self.sentences.items():
            # 只翻译稳定且未翻译的句子，并且长度达到最小要求
            if state['is_stable'] and not state['translated'] and not state['played'] and state[
                'length'] >= self.min_sentence_length:
                sentences_to_translate.append(state['text'])

        return sentences_to_translate

    def _get_sentences_to_play(self):
        """获取可以播放的句子列表

        返回:
            list: 可以播放的句子列表
        """
        sentences_to_play = []
        current_time = time.time()

        for sentence_id, state in self.sentences.items():
            # 只播放已翻译但未播放的句子，并且长度达到最小要求
            if state['is_stable'] and state['translated'] and not state['played'] and state[
                'length'] >= self.min_sentence_length:
                # 检查是否是已播放句子的微小扩展
                if not self._is_minor_extension(state['text']):
                    sentences_to_play.append({
                        'text': state['text'],
                        'translation': state['translation'],
                        'priority': self._calculate_priority(state, current_time),
                        'length': state['length'],
                        'is_complete': state['is_complete'],
                        'id': sentence_id,
                        'time_since_creation': current_time - state['created_at']
                    })

        # 按优先级排序
        if sentences_to_play:
            sentences_to_play.sort(key=lambda x: x['priority'], reverse=True)

        return sentences_to_play

    def update_translation(self, sentence, translation):
        """更新句子的翻译结果

        参数:
            sentence: 句子文本
            translation: 翻译结果

        返回:
            bool: 是否成功更新
        """
        # 获取句子ID
        sentence_id = self._get_sentence_id(sentence)

        if sentence_id in self.sentences:
            # 检查翻译是否变化
            if self.sentences[sentence_id].get('translation') != translation:
                self.sentences[sentence_id]['translated'] = True
                self.sentences[sentence_id]['translation'] = translation
                return True
        else:
            # 尝试找到最相似的句子
            best_match = None
            best_score = 0

            for sid, state in self.sentences.items():
                score = difflib.SequenceMatcher(None, sentence, state['text']).ratio()
                if score > 0.8 and score > best_score:  # 80%以上相似度
                    best_match = sid
                    best_score = score

            if best_match:
                self.sentences[best_match]['translated'] = True
                self.sentences[best_match]['translation'] = translation
                # 添加新句子文本到相应ID的映射
                self.sentence_ids[sentence] = best_match
                return True

        return False

    def mark_as_played(self, sentence_text):
        """标记句子为已播放

        参数:
            sentence_text: 句子文本

        返回:
            bool: 是否成功标记
        """
        # 尝试直接查找
        sentence_id = self._get_sentence_id(sentence_text)

        if sentence_id in self.sentences:
            self.sentences[sentence_id]['played'] = True
            self.played_sentences.add(sentence_text)
            self.last_played_time = time.time()
            return True

        # 如果找不到，尝试找最相似的句子
        for sid, state in self.sentences.items():
            if difflib.SequenceMatcher(None, sentence_text, state['text']).ratio() > 0.8:
                self.sentences[sid]['played'] = True
                self.played_sentences.add(state['text'])
                self.last_played_time = time.time()
                return True

        return False

    def _split_into_sentences(self, text):
        """将文本分割为句子

        参数:
            text: 完整文本

        返回:
            list: 句子列表
        """
        if not text:
            return []

        # 句子分隔符（中英文标点符号）
        sentence_delimiters = r'(?<=[.。!！?？;；])\s*'

        # 分割文本获取句子列表
        parts = re.split(sentence_delimiters, text)
        sentences = []

        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

        return sentences

    def _merge_short_sentences(self, sentences):
        """合并短句为更有意义的单元

        参数:
            sentences: 原始句子列表

        返回:
            list: 合并后的句子列表
        """
        if not sentences:
            return []

        # 如果只有一个句子，直接返回
        if len(sentences) <= 1:
            return sentences

        result = []
        current = sentences[0]

        for i in range(1, len(sentences)):
            # 如果当前句子或下一个句子都很短，考虑合并
            if len(current) < self.min_sentence_length or len(sentences[i]) < self.min_sentence_length:
                # 合并短句
                combined = current + sentences[i]
                current = combined
            else:
                # 当前句子足够长，添加到结果并开始新句子
                result.append(current)
                current = sentences[i]

        # 添加最后一个句子
        if current:
            result.append(current)

        return result

    def _is_complete_sentence(self, sentence):
        """判断句子是否完整（以标点符号结尾）

        参数:
            sentence: 句子文本

        返回:
            bool: 是否完整
        """
        return bool(re.search(r'[.。!！?？;；]\s*$', sentence))

    def _is_minor_extension(self, new_sentence):
        """检查是否是已播放句子的微小扩展

        参数:
            new_sentence: 新句子文本

        返回:
            bool: 是否为微小扩展
        """
        # 检查句子是否与已播放句子有很高的相似度
        for played in self.played_sentences:
            # 计算相似度比例
            similarity = difflib.SequenceMatcher(None, played, new_sentence).ratio()

            # 如果新句子是已播放句子的扩展
            if new_sentence.startswith(played):
                # 计算新增部分的比例
                diff_ratio = (len(new_sentence) - len(played)) / max(len(played), 1)

                # 如果新增部分很小，认为是微小扩展
                if diff_ratio < self.min_diff_ratio:
                    return True

            # 如果两句话非常相似
            elif similarity > 0.9:
                return True

        return False

    def _calculate_priority(self, state, current_time):
        """计算句子的播放优先级

        参数:
            state: 句子状态
            current_time: 当前时间

        返回:
            float: 优先级分数
        """
        # 基础优先级分数
        priority = 0

        sentence = state['text']

        # 1. 完整句子优先
        if state['is_complete']:
            priority += 100

        # 2. 等待时间越长优先级越高
        wait_time = current_time - state['created_at']
        priority += min(wait_time * 10, 50)  # 最多加50分

        # 3. 句子长度适中的优先级高 - 修改优先级计算
        # 太短的句子优先级低，太长的句子也降低优先级
        length_factor = len(sentence) / 80.0  # 标准化长度 (理想长度约80字符)
        if length_factor < 0.5:  # 太短
            priority -= 20 * (0.5 - length_factor)  # 惩罚太短的句子
        elif length_factor > 1.5:  # 太长
            priority -= 10 * (length_factor - 1.5)  # 轻微惩罚太长的句子

        return priority

    def _get_sentence_id(self, sentence_text):
        """获取或生成句子ID

        参数:
            sentence_text: 句子文本

        返回:
            str: 句子ID
        """
        # 如果已有映射，直接返回
        if sentence_text in self.sentence_ids:
            return self.sentence_ids[sentence_text]

        # 为新句子生成唯一ID
        sentence_id = hashlib.md5(sentence_text.encode('utf-8')).hexdigest()[:8]
        self.sentence_ids[sentence_text] = sentence_id
        return sentence_id

    def clear(self):
        """清空所有状态"""
        self.sentences = {}
        self.sentence_ids = {}
        self.played_sentences = set()
        self.last_played_time = 0
        self.last_update_time = 0
        self.processed_text = ""