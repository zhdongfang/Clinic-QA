import os
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score

from ner_data_processing import (
    get_data, rule_find, find_entities, tfidf_alignment,
    Nerdataset, build_tag2idx
)

cache_model = 'best_roberta_rnn_model_ent_aug'


class Bert_Model(nn.Module):
    def __init__(self, model_name, hidden_size, tag_num, bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.gru = nn.RNN(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bi
        )
        out_dim = hidden_size * 2 if bi else hidden_size
        self.classifier = nn.Linear(out_dim, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        # x: [batch_size, seq_len]
        # label: [batch_size, seq_len]
        bert_0, _ = self.bert(x, attention_mask=(x > 0), return_dict=False)
        gru_0, _ = self.gru(bert_0)  # [batch_size, seq_len, hidden_dim * 2]
        pre = self.classifier(gru_0)  # [batch_size, seq_len, tag_num]

        if label is not None:
            loss = self.loss_fn(
                pre.reshape(-1, pre.shape[-1]),
                label.reshape(-1)
            )
            return loss
        else:
            # return predicted labels (argmax)
            return torch.argmax(pre, dim=-1).squeeze(0)


def merge(model_result_word, rule_result):
    """
    Merge model prediction results and rule-based results to avoid entity overlaps.
    Sort entities by length in descending order, skip if positions conflict.
    """
    result = model_result_word + rule_result
    result = sorted(result, key=lambda x: len(x[-1]), reverse=True)
    check_result = []
    used_map = {}

    for res in result:
        if res[0] in used_map or res[1] in used_map:
            continue
        check_result.append(res)
        for i in range(res[0], res[1] + 1):
            used_map[i] = 1
    return check_result


def get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag):
    """
     Perform model prediction + rule-based recognition + merging + TF-IDF alignment
    on the input sentence, and return the final recognition results.
    """
    sen_ids = tokenizer.encode(sen, add_special_tokens=True, return_tensors='pt').to(device)
    pre = model(sen_ids).tolist()
    # Remove [CLS], [SEP]
    pre_tag = [idx2tag[i] for i in pre[1:-1]]

    model_result = find_entities(pre_tag)
    model_result_word = []
    for (start, end, etype) in model_result:
        word = sen[start:end+1]
        model_result_word.append((start, end, etype, word))

    rule_result = rule.find(sen)

    merged_result = merge(model_result_word, rule_result)

    # TF-IDF alignment
    aligned_result = tfidf_r.align(merged_result)
    return aligned_result


if __name__ == "__main__":
    all_text, all_label = get_data(os.path.join('data', 'ner_dataset.txt'))
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_label, test_size=0.02, random_state=42)

    if os.path.exists('tmp_data/tag2idx.npy'):
        with open('tmp_data/tag2idx.npy', 'rb') as f:
            tag2idx = pickle.load(f)
    else:
        tag2idx = build_tag2idx(all_label)
        with open('tmp_data/tag2idx.npy', 'wb') as f:
            pickle.dump(tag2idx, f)

    idx2tag = list(tag2idx)

    max_len = 50
    epoch = 30
    batch_size = 60
    hidden_size = 128
    bi = True
    model_name = 'model/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    lr = 1e-5
    is_train = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = Nerdataset(train_text, train_label, tokenizer, max_len, tag2idx, enhance_data=True)
    dev_dataset = Nerdataset(dev_text, dev_label, tokenizer, max_len, tag2idx, is_dev=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = Bert_Model(model_name, hidden_size, len(tag2idx), bi).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1

    # 训练
    if is_train:
        for e in range(epoch):
            train_dataset.set_epoch(e)

            model.train()
            loss_sum = 0.0
            ba = 0

            for x, y, batch_len in tqdm(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                loss = model(x, y)
                loss.backward()
                opt.step()

                loss_sum += loss.item()
                ba += 1

            model.eval()
            all_pre = []
            all_label_ref = []
            with torch.no_grad():
                for x, y, batch_len in tqdm(dev_dataloader, desc="Validating"):
                    x = x.to(device)
                    pred = model(x)
                    pred = pred[1:batch_len[0] + 1].tolist()

                    # Convert to label strings
                    pred_tags = [idx2tag[i] for i in pred]
                    all_pre.append(pred_tags)

                    label_cpu = y[0][1:batch_len[0] + 1].tolist()
                    label_str = [idx2tag[i] for i in label_cpu]
                    all_label_ref.append(label_str)

            f1 = f1_score(all_pre, all_label_ref)
            if f1 > best_f1:
                best_f1 = f1
                print(f"Epoch={e}, loss={loss_sum/ba:.5f}, f1={f1:.5f}  ---> best")
                torch.save(model.state_dict(), f"model/{cache_model}.pt")
            else:
                print(f"Epoch={e}, loss={loss_sum/ba:.5f}, f1={f1:.5f}")

    rule = rule_find()
    tfidf_r = tfidf_alignment()

    while True:
        sen = input("Please input: ")
        if not sen.strip():
            break
        result = get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag)
        print("Entities:", result)