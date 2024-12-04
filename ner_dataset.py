import os
import random
import ahocorasick
import re
from tqdm import tqdm
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

class BuildNERData:
    """
    NER 数据生成类。主要作用是将 data 文件夹下的 Chinese_medical.json 文件中的文本打上标签。
    将各类实体导入到 Aho-Corasick 自动机中，对每个文本进行模式匹配。
    """
    def __init__(self):
        # 实体类型列表
        self.entity_types = ["disease", "symptom", "check", "department", "food", "producer", "cure", "drug"]
        self.max_len = 30
        self.punctuation = ['，', '。', '！', '；', '：', ',', '.', '?', '!', ';']
        self.automaton = ahocorasick.Automaton()
        # 加载所有实体到 Aho-Corasick 自动机
        for entity_type in self.entity_types:
            entity_file = os.path.join('data', 'ents', f'{entity_type}.txt')
            if not os.path.exists(entity_file):
                logging.warning(f"Entity file not found: {entity_file}")
                continue
            with open(entity_file, encoding='utf-8') as f:
                entities = f.read().split('\n')
            for entity in entities:
                if len(entity) >= 2:
                    self.automaton.add_word(entity, (entity, entity_type))
        self.automaton.make_automaton()
        logging.info("Aho-Corasick automaton has been built successfully.")

    def split_text(self, text):
        """
        将长文本随机分割为短文本

        Args:
            text (str): 长文本

        Returns:
            list: 分割后的短文本列表
        """
        text = text.replace('\n', ',')
        pattern = r'([，。！；：,.?!;])'
        sentences = re.split(pattern, text)

        split_sentences = []
        temp_sentence = ''

        for idx, segment in enumerate(sentences):
            if segment in self.punctuation:
                temp_sentence += segment
                if (len(temp_sentence) > self.max_len and random.random() < 0.9) or random.random() < 0.15:
                    split_sentences.append(temp_sentence)
                    temp_sentence = ''
            else:
                temp_sentence += segment

        if temp_sentence:
            split_sentences.append(temp_sentence)

        # 随机将部分句子的末尾标点改为 '。'
        for i in range(len(split_sentences)):
            if random.random() < 0.3:
                if split_sentences[i][-1] in self.punctuation:
                    split_sentences[i] = split_sentences[i][:-1] + '。'
                else:
                    split_sentences[i] += '。'

        return split_sentences

    def make_text_label(self, text):
        """
        对文本进行实体识别，生成 NER 标签

        Args:
            text (str): 文本

        Returns:
            list: 标签列表
            int: 匹配到的实体数量
        """
        label = ['O'] * len(text)
        matches = list(self.automaton.iter(text))
        flag = 0
        occupied = [False] * len(text)

        # 按匹配的实体长度从长到短排序，优先标注长的实体
        matches.sort(key=lambda x: len(x[1][0]), reverse=True)

        for end_idx, (entity, entity_type) in matches:
            start_idx = end_idx - len(entity) + 1
            if any(occupied[start_idx:end_idx + 1]):
                continue
            label[start_idx] = f'B-{entity_type}'
            for idx in range(start_idx + 1, end_idx + 1):
                label[idx] = f'I-{entity_type}'
            for idx in range(start_idx, end_idx + 1):
                occupied[idx] = True
            flag += 1

        return label, flag

def build_file(all_texts, all_labels, output_path):
    """
    将文本和对应的标签写入文件

    Args:
        all_texts (list): 文本列表
        all_labels (list): 标签列表
        output_path (str): 输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for text, label in zip(all_texts, all_labels):
            for char, tag in zip(text, label):
                f.write(f'{char} {tag}\n')
            f.write('\n')
    logging.info(f"NER data has been written to {output_path}")

def main():
    # 初始化 NER 数据生成器
    ner_data_builder = BuildNERData()

    all_texts = []
    all_labels = []

    # 读取数据
    data_file = os.path.join('data', 'Chinese_medical.json')
    if not os.path.exists(data_file):
        logging.error(f"Data file not found: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        all_data = f.read().splitlines()
    logging.info(f"Successfully read data from {data_file}")

    for data_line in tqdm(all_data, desc="Processing data"):
        if not data_line.strip():
            continue
        try:
            data = json.loads(data_line.strip())
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse data: {data_line}, error: {e}")
            continue

        data_texts = [data.get("description", ""), data.get("prevent", ""), data.get("cause", "")]

        for text in data_texts:
            if not text:
                continue
            split_texts = ner_data_builder.split_text(text)
            for segment in split_texts:
                if not segment:
                    continue
                label, flag = ner_data_builder.make_text_label(segment)
                if flag >= 1:
                    assert len(segment) == len(label)
                    all_texts.append(segment)
                    all_labels.append(label)

    # 写入 NER 数据文件
    output_file = os.path.join('data', 'ner_dataset.txt')
    build_file(all_texts, all_labels, output_file)

if __name__ == "__main__":
    main()