import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizer
from seqeval.metrics import f1_score
from ner_data_processing import get_data, build_tag2idx, RuleFind, TfidfAlignment


class BertModelNN(nn.Module):
    def __init__(self, model_name, hidden_size, tag_num, bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.gru = nn.RNN(input_size=768, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=bi)
        self.classifier = nn.Linear(hidden_size * 2 if bi else hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        bert_0, _ = self.bert(x, attention_mask=(x > 0), return_dict=False)
        gru_0, _ = self.gru(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1).squeeze(0)


class NerDataset(Dataset):
    def __init__(self, all_text, all_label, tokenizer, max_len, tag2idx):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx

    def __getitem__(self, idx):
        text, label = self.all_text[idx], self.all_label[idx]
        text_idx = self.tokenizer.encode(text, add_special_tokens=True)
        label_idx = [self.tag2idx['<PAD>']] + [self.tag2idx[i] for i in label] + [self.tag2idx['<PAD>']]
        text_idx += [0] * (self.max_len - len(text_idx))
        label_idx += [self.tag2idx['<PAD>']] * (self.max_len - len(label_idx))
        return torch.tensor(text_idx), torch.tensor(label_idx)

    def __len__(self):
        return len(self.all_text)


def main():
    # 加载数据
    all_text, all_label = get_data('data/ner_dataset.txt')
    tag2idx = build_tag2idx(all_label)
    idx2tag = list(tag2idx)

    # 初始化模型和训练参数
    model_name = 'model/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    max_len = 100
    epoch = 30
    batch_size = 60
    hidden_size = 128
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_label, test_size=0.02,
                                                                    random_state=42)

    # 构建数据集
    train_dataset = NerDataset(train_text, train_label, tokenizer, max_len, tag2idx)
    dev_dataset = NerDataset(dev_text, dev_label, tokenizer, max_len, tag2idx)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # train_dataset = Nerdataset(train_text, train_label, tokenizer, max_len, tag2idx, enhance_data=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = BertModelNN(model_name, hidden_size, len(tag2idx), bi=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 训练模型
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        for text, label in train_loader:
            optimizer.zero_grad()
            loss = model(text, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epoch}: Loss = {total_loss / len(train_loader)}")

    # 模型评估
    model.eval()
    pred_tags, true_tags = [], []
    with torch.no_grad():
        for text, label in dev_loader:
            pred = model(text)
            pred_tags.append(pred)
            true_tags.append(label)

    print(f"F1 Score: {f1_score(true_tags, pred_tags)}")


if __name__ == "__main__":
    main()