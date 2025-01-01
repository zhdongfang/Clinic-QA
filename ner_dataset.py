import os
import random
import ahocorasick
import re
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO)

class BuildNERData:
    """
    Label the text in the file. Import entities into the Aho Corasick automaton and perform pattern matching on each text.
    """
    def __init__(self):
        self.entity_types = ["disease", "symptom", "check", "department", "food", "producer", "cure", "drug"]
        self.max_len = 30
        self.punctuation = ['，', '。', '！', '；', '：', ',', '.', '?', '!', ';']
        self.automaton = ahocorasick.Automaton()
        # Load entities into the Aho Corasick automaton
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
        Randomly split long text into short text
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

        # Randomly change the punctuation at the end of some sentences to '.'
        for i in range(len(split_sentences)):
            if random.random() < 0.3:
                if split_sentences[i][-1] in self.punctuation:
                    split_sentences[i] = split_sentences[i][:-1] + '。'
                else:
                    split_sentences[i] += '。'

        return split_sentences

    def make_text_label(self, text):
        """
        Perform entity recognition on text and generate NER tags
        """
        label = ['O'] * len(text)
        matches = list(self.automaton.iter(text))
        flag = 0
        occupied = [False] * len(text)

        # Sort by matching entity length from long to short, prioritizing long entities for annotation
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
    with open(output_path, "w", encoding="utf-8") as f:
        for text, label in zip(all_texts, all_labels):
            for char, tag in zip(text, label):
                f.write(f'{char} {tag}\n')
            f.write('\n')
    logging.info(f"NER data has been written to {output_path}")

def load_medical_data(json_path):
    data = []
    decoder = json.JSONDecoder()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            idx = 0
            length = len(content)
            while idx < length:
                # Skip any leading commas and whitespace
                while idx < length and content[idx] in ', \n\r\t':
                    idx += 1
                if idx >= length:
                    break
                try:
                    obj, end = decoder.raw_decode(content, idx)
                    data.append(obj)
                    idx = end
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse data at index {idx}: {e}")
                    # Try to skip to the next character after the current index
                    idx += 1
    except Exception as e:
        logging.error(f"Failed to read file {json_path}: {e}")
    logging.info(f"Loaded {len(data)} records from {json_path}")
    return data

def main():
    ner_data_builder = BuildNERData()
    all_texts = []
    all_labels = []
    data_file = os.path.join('data', 'Chinese_medical.json')
    if not os.path.exists(data_file):
        logging.error(f"Data file not found: {data_file}")
        return

    all_data = load_medical_data(data_file)
    if not all_data:
        logging.error("No data loaded from the JSON file. Exiting.")
        return
    logging.info(f"Successfully loaded {len(all_data)} records from {data_file}")

    for data in tqdm(all_data, desc="Processing data"):
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

    output_file = os.path.join('data', 'ner_dataset.txt')
    build_file(all_texts, all_labels, output_file)

if __name__ == "__main__":
    main()