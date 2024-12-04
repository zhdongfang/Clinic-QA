from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import ollama
import pickle
import ner as nm
import streamlit as st

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    choice: str


# 模型加载
def load_model(cache_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open('tmp_data/tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    rule = nm.rule_find()
    tfidf_r = nm.tfidf_alignment()
    model_name = 'model/chinese-roberta-wwm-ext'
    bert_tokenizer = nm.BertTokenizer.from_pretrained(model_name)
    bert_model = nm.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
    bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt'))

    bert_model = bert_model.to(device)
    bert_model.eval()
    return bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device


# 处理用户查询的意图识别
@app.post("/intent_recognition/")
async def intent_recognition(request: QueryRequest):
    try:
        prompt = f"""<指令>你是一个医疗问答机器人，请根据以下提示和你掌握的医学知识，尽可能详细、准确地回答用户的问题。如果提示信息不足，你可以结合自身的知识为用户提供帮助。</指令>"""
        rec_result = ollama.generate(model=request.choice, prompt=prompt)['response']
        return {"intent_result": rec_result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


# 其它相关 API 逻辑


def main():
    st.title(f"医疗智能问答机器人")

    # 直接跳过登录界面，进入主页
    if 'chat_windows' not in st.session_state:
        st.session_state.chat_windows = [[]]
        st.session_state.messages = [[]]

    if st.button('新建对话窗口'):
        st.session_state.chat_windows.append([])
        st.session_state.messages.append([])

    window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
    selected_window = st.selectbox('请选择对话窗口:', window_options)
    active_window_index = int(selected_window.split()[1]) - 1

    selected_option = st.selectbox(
        label='请选择大语言模型:',
        options=['qwen2:0.5b', 'qwen2.5:7b-instruct', 'Llama3.1-8B-Chinese-Chat', 'llama2-chinese:13b-chat-q8_0']
    )
    choice = selected_option

    glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model(
        'best_roberta_rnn_model_ent_aug')

    # 此部分不再需要登录验证，直接跳过登录逻辑
    # ...
