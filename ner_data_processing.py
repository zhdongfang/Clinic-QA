import os
import pickle

from sklearn.model_selection import train_test_split
import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 获取数据并分割成句子和标签
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


# Rule-based entity extraction
class RuleFind:
    def __init__(self):
        self.idx2type = ["food", "producer", "cure", "drug", "check", "disease", "symptom", "department"]
        self.type2idx = {type: idx for idx, type in enumerate(self.idx2type)}
        self.ahos = [ahocorasick.Automaton() for _ in range(len(self.type2idx))]

        for type in self.idx2type:
            with open(os.path.join('data', 'ents', f'{type}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en) >= 2:
                    self.ahos[self.type2idx[type]].add_word(en, en)
        for aho in self.ahos:
            aho.make_automaton()

    def find(self, sen):
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
            for i, res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[i], res[1]))
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


# TF-IDF 对齐策略
class TfidfAlignment:
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
                entities = [ent for ent in entities if len(ent.split(' ')[0]) <= 15 and len(ent.split(' ')[0]) >= 1]
                en_name = [ent.split(' ')[0] for ent in entities]
                ty = ty.strip('.txt')
                self.tag_2_entity[ty] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[ty] = embs
                self.tag_2_tfidf_model[ty] = tfidf_model

    def align(self, ent_list):
        new_result = {}
        for s, e, cls, ent in ent_list:
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]

            if max_score >= 0.5:
                new_result[cls] = self.tag_2_entity[cls][max_idx]
        return new_result


# 生成 tag2idx 映射
def build_tag2idx(all_tag):
    tag2idx = {'<PAD>': 0}
    for sen in all_tag:
        for tag in sen:
            tag2idx[tag] = tag2idx.get(tag, len(tag2idx))
    return tag2idx


def main():
    # 读取数据
    all_text, all_label = get_data(os.path.join('data', 'ner_dataset.txt'))
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_label, test_size=0.02, random_state=42)

    # 构建tag2idx映射
    if os.path.exists('tmp_data/tag2idx.npy'):
        with open('tmp_data/tag2idx.npy', 'rb') as f:
            tag2idx = pickle.load(f)
    else:
        tag2idx = build_tag2idx(all_label)
        with open('tmp_data/tag2idx.npy', 'wb') as f:
            pickle.dump(tag2idx, f)

    idx2tag = list(tag2idx)
    print(f"数据准备完成: {len(train_text)} 条训练数据, {len(dev_text)} 条验证数据")


if __name__ == "__main__":
    main()