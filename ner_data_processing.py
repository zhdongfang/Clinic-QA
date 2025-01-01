# ner_data_processing.py

import os
import random
import torch
import ahocorasick
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_data(path, max_len=None):

    all_text, all_tag = [], []
    with open(path, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen, tag = [], []
    for data in all_data:
        data = data.split(' ')
        if len(data) != 2:
            if len(sen) > 2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te, ta = data
        sen.append(te)
        tag.append(ta)
    if max_len is not None:
        return all_text[:max_len], all_tag[:max_len]
    return all_text, all_tag


class rule_find:
    """
    Rule matching method based on Aho Corasick automaton
    """
    def __init__(self):
        self.idx2type = idx2type = ["food", "producer", "cure", "drug", "check", "disease", "symptom", "department"]
        self.type2idx = type2idx = {
            "food": 0, "producer": 1, "cure": 2, "drug": 3,
            "check": 4, "disease": 5, "symptom": 6, "department": 7
        }
        self.ahos = [ahocorasick.Automaton() for _ in range(len(self.type2idx))]

        # Read entity files and build automata
        for etype in idx2type:
            with open(os.path.join('data', 'ents', f'{etype}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en) >= 2:
                    self.ahos[type2idx[etype]].add_word(en, en)
        for aho in self.ahos:
            aho.make_automaton()

    def find(self, sen):
        """
        Use an AC automaton to match entities in a known lexicon.
        Return [(start, end, type, text),...]
        """
        rule_result = []
        mp = {}
        all_res = []
        all_ty = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for _ in range(len(now)):
                all_ty.append(self.idx2type[i])

        if len(all_res) != 0:
            all_res = sorted(all_res, key=lambda x: len(x[1]), reverse=True)
            for idx, res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[idx], res[1]))
                # Mark the location of the entity as occupied to prevent overlap
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


def find_entities(tag):
    result = []
    label_len = len(tag)
    i = 0
    while i < label_len:
        if tag[i].startswith('B'):
            etype = tag[i].lstrip('B-')
            j = i + 1
            while j < label_len and tag[j].startswith('I'):
                j += 1
            # Starting and ending indices of entities, as well as their types
            result.append((i, j - 1, etype))
            i = j
        else:
            i += 1
    return result


class tfidf_alignment:
    """
    Align and optimize the initially identified entities based on TF-IDF vectors and cosine similarity.
    Construct corresponding TF-IDF vector sets.
    """
    def __init__(self):
        entities_path = os.path.join('data', 'ents')
        files = os.listdir(entities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.tag_2_embs = {}
        self.tag_2_tfidf_model = {}
        self.tag_2_entity = {}
        for ty in files:
            with open(os.path.join(entities_path, ty), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                entities = [
                    ent for ent in entities
                    if len(ent.split(' ')[0]) <= 15 and len(ent.split(' ')[0]) >= 1
                ]
                en_name = [ent.split(' ')[0] for ent in entities]
                etype = ty.strip('.txt')
                self.tag_2_entity[etype] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[etype] = embs
                self.tag_2_tfidf_model[etype] = tfidf_model

    def align(self, ent_list):
        """
        Perform cosine similarity calculation on the vocabulary of each entity and corresponding type
        """
        new_result = {}
        for s, e, cls, ent in ent_list:
            if cls not in self.tag_2_tfidf_model:
                continue
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]
            if max_score >= 0.5:
                new_result[cls] = self.tag_2_entity[cls][max_idx]
        return new_result


class Entity_Extend:
    """
    Data augmentation: methods such as entity replacement, masking, and concatenation are used to expand training data through random operations.
    """
    def __init__(self):
        entities_path = os.path.join('data', 'ents')
        files = os.listdir(entities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.type2entity = {}
        self.type2weight = {}
        for f in files:
            with open(os.path.join(entities_path, f), 'r', encoding='utf-8') as ff:
                entities = ff.read().split('\n')
                en_name = [
                    ent for ent in entities
                    if len(ent.split(' ')[0]) <= 15 and len(ent.split(' ')[0]) >= 1
                ]
                en_weight = [1] * len(en_name)
                etype = f.strip('.txt')
                self.type2entity[etype] = en_name
                self.type2weight[etype] = en_weight

    def no_work(self, te, tag, etype):
        return te, tag

    def entity_replace(self, te, ta, etype):
        if etype not in self.type2entity:
            return te, ta
        choice_ent = random.choices(self.type2entity[etype],
                                    weights=self.type2weight[etype], k=1)[0]
        new_tag = ["B-" + etype] + ["I-" + etype] * (len(choice_ent) - 1)
        return list(choice_ent), new_tag

    def entity_mask(self, te, ta, etype):
        if len(te) <= 3:
            return te, ta
        elif len(te) <= 5:
            te.pop(random.randint(0, len(te) - 1))
        else:
            te.pop(random.randint(0, len(te) - 1))
            te.pop(random.randint(0, len(te) - 1))
        new_tag = ["B-" + etype] + ["I-" + etype] * (len(te) - 1)
        return te, new_tag

    def entity_union(self, te, ta, etype):
        if etype not in self.type2entity:
            return te, ta
        words = ['和', '与', '以及']
        wor = random.choice(words)
        choice_ent = random.choices(self.type2entity[etype],
                                    weights=self.type2weight[etype], k=1)[0]
        te = te + list(wor) + list(choice_ent)
        new_tag = ta + ['O'] * len(wor) + ["B-" + etype] + ["I-" + etype] * (len(choice_ent) - 1)
        return te, new_tag

    def entities_extend(self, text, tag, ents):
        """
        Randomly apply one of the five strategies mentioned above to each entity in the annotation sequence.
        This includes two rounds of now_work.
        """
        strategies = [self.no_work, self.entity_union, self.entity_mask,
                      self.entity_replace, self.no_work]

        new_text = text.copy()
        new_tag = tag.copy()
        offset = 0
        for ent in ents:
            p = random.choice(strategies)
            te, ta = p(new_text[ent[0] + offset: ent[1] + 1 + offset],
                       new_tag[ent[0] + offset: ent[1] + 1 + offset],
                       ent[2])
            # Replace text and labels
            new_text[ent[0] + offset: ent[1] + 1 + offset] = te
            new_tag[ent[0] + offset: ent[1] + 1 + offset] = ta

            offset += len(te) - (ent[1] - ent[0] + 1)
        return new_text, new_tag


class Nerdataset(Dataset):
    def __init__(self, all_text, all_label, tokenizer, max_len, tag2idx,
                 is_dev=False, enhance_data=False):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx
        self.is_dev = is_dev
        self.enhance_data = enhance_data
        self.entity_extend = Entity_Extend()
        # Perception of epochs during data augmentation control
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        text, label = self.all_text[idx], self.all_label[idx]

        if self.is_dev:
            local_max_len = min(len(text) + 2, 500)
        else:
            if self.enhance_data and self.current_epoch >= 7 and (self.current_epoch % 2 == 1):
                ents = find_entities(label)
                text, label = self.entity_extend.entities_extend(text, label, ents)
            local_max_len = self.max_len

        text, label = text[:local_max_len - 2], label[:local_max_len - 2]

        x_len = len(text)
        assert len(text) == len(label), "Inconsistent length between text and label"

        text_idx = self.tokenizer.encode(text, add_special_token=True)
        # Add a<PAD>(or [CLS]/[SEP] tag) before and after as a placeholder
        label_idx = [self.tag2idx['<PAD>']] + [self.tag2idx[i] for i in label] + [self.tag2idx['<PAD>']]

        # Fill to a fixed length
        pad_len = local_max_len - len(text_idx)
        if pad_len > 0:
            text_idx += [0] * pad_len
            label_idx += [self.tag2idx['<PAD>']] * pad_len

        return torch.tensor(text_idx),torch.tensor(label_idx),x_len

    def __len__(self):
        return len(self.all_text)


def build_tag2idx(all_tag):
    tag2idx = {'<PAD>': 0}
    for sen in all_tag:
        for tg in sen:
            if tg not in tag2idx:
                tag2idx[tg] = len(tag2idx)
    return tag2idx