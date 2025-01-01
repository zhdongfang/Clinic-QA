# chat_api.py

import json
import logging
import re
import ollama
import intent_recognition
import ner as zwk
from fastapi.middleware.cors import CORSMiddleware
import pickle
from transformers import BertTokenizer
import torch
import py2neo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    query: str
    selected_model: str

class ChatResponse(BaseModel):
    answer: str
    entities: str = None
    intent: str = None
    prompt: str = None

def load_model(cache_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    glm_model = None
    glm_tokenizer= None
    with open('tmp_data/tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()
    model_name = 'model/chinese-roberta-wwm-ext'
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
    bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt', weights_only=True))
    bert_model = bert_model.to(device)
    bert_model.eval()
    return glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device


cache_model = 'best_roberta_rnn_model_ent_aug'
glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model(cache_model)
client = py2neo.Graph('http://localhost:7474', user='neo4j', password='12345678', name='chinese_medical')

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.query
    choice = request.selected_model

    logger.info(f"Received query: {query}")
    logger.info(f"Selected model: {choice}")

    # Extract Entity
    try:
        entities = zwk.get_ner_result(bert_model, bert_tokenizer, query, rule, tfidf_r, device, idx2tag)
        print(f"Extracting Entity: {entities}")
    except Exception as e:
        logger.error(f"Failed to extract entity: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract entity.")

    # intention recognition
    try:
        intent_result = intent_recognition.intent_recognition(query, choice, entities)
        print(f"Intention recognition: {intent_result}")
    except Exception as e:
        logger.error(f"Intention recognition failed: {e}")
        intent_result = {"intent": None}

    intent_result = json.loads(intent_result)
    if not isinstance(intent_result, dict):
        logger.error(f"Intention recognition return type error: {type(intent_result)}")
        raise HTTPException(status_code=500, detail="Failed to extract entity.")

    english_intents = intent_result.get("intent")
    print(f"Intention recognition: {english_intents}")

    if not english_intents:
        final_answer = "抱歉，我无法识别您的意图。请咨询专业医疗人员。"
        purp = "意图识别失败"
        knowledge_content = ""
    else:
        # Search the knowledge base
        knowledge_base_info = query_knowledge_base(entities, english_intents)
        print(f"Information retrieved from the knowledge base: {knowledge_base_info}")

        # Prompt
        prompt, purp, intents_returned = generate_prompt(english_intents, knowledge_base_info, query)
        print("Prompt："+ prompt)

        # LLM
        try:
            ollama_response = ollama.generate(model=choice, prompt=prompt)
            final_answer = ollama_response['response']
            print("final_answer"+final_answer)
        except ollama.ResponseError as e:
            logger.error(f"RAG model response error: {e}")
            final_answer = "抱歉，我无法处理您的请求。"
        except Exception as e:
            logger.error(f"RAG model response error: {e}")
            final_answer = "抱歉，我无法处理您的请求。"

        # Extract knowledge base information
        knowledge = re.findall(r'<提示>(.*?)</提示>', prompt)
        knowledge_content = "\n".join([f"提示{idx + 1}: {kn}" for idx, kn in enumerate(knowledge) if len(kn) >= 3])

    return ChatResponse(
        answer=final_answer,
        entities=str(entities) if entities else "",
        intent="、".join(english_intents) if english_intents else None,
        prompt=knowledge_content
    )


def query_knowledge_base(entities, intents):
    knowledge_info = []
    disease_name = entities.get('disease')

    if not disease_name:
        logger.warning("Disease name not extracted")
        return knowledge_info

    for intent in intents:
        if intent == "QUERY_DISEASE_DESCRIPTION":
            query = "MATCH (d:disease {name: $disease}) RETURN d.description AS description"
            parameters = {"disease": disease_name}
            result = execute_neo4j_query(query, parameters)
            if result:
                knowledge_info.append(f"疾病详情: {result['description']}")

        elif intent == "QUERY_DISEASE_CAUSE":
            query = "MATCH (d:disease {name: $disease}) RETURN d.cause AS cause"
            parameters = {"disease": disease_name}
            result = execute_neo4j_query(query, parameters)
            if result:
                knowledge_info.append(f"病因: {result['cause']}")

        elif intent == "QUERY_PREVENTION_MEASURES":
            query = "MATCH (d:disease {name: $disease}) RETURN d.prevent AS prevention"
            parameters = {"disease": disease_name}
            result = execute_neo4j_query(query, parameters)
            if result:
                knowledge_info.append(f"预防措施: {result['prevention']}")

        elif intent == "QUERY_TREATMENT_DURATION":
            query = "MATCH (d:disease {name: $disease}) RETURN d.treatment_duration AS duration"
            parameters = {"disease": disease_name}
            result = execute_neo4j_query(query, parameters)
            if result:
                knowledge_info.append(f"治疗周期: {result['duration']}")

        elif intent == "QUERY_CURE_PROBABILITY":
            query = "MATCH (d:disease {name: $disease}) RETURN d.cure_rate AS probability"
            parameters = {"disease": disease_name}
            result = execute_neo4j_query(query, parameters)
            if result:
                knowledge_info.append(f"治愈率: {result['probability']}")

        elif intent == "QUERY_SUSCEPTIBLE_POPULATION":
            query = "MATCH (d:disease {name: $disease}) RETURN d.susceptible_populations AS population"
            parameters = {"disease": disease_name}
            result = execute_neo4j_query(query, parameters)
            if result:
                knowledge_info.append(f"易感人群: {result['population']}")

        elif intent == "QUERY_REQUIRED_DRUGS":
            query = "MATCH (d:disease {name: $disease})-[:use_drug]->(p:drug) RETURN p.name AS drug"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            drugs = [res['drug'] for res in results]
            if drugs:
                knowledge_info.append(f"所需药物: {', '.join(drugs)}")

        elif intent == "QUERY_RECOMMENDED_FOODS":
            query = "MATCH (d:disease {name: $disease})-[:recommended_food]->(f:food) RETURN f.name AS food"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            foods = [res['food'] for res in results]
            if foods:
                knowledge_info.append(f"推荐食物: {', '.join(foods)}")

        elif intent == "QUERY_AVOIDED_FOODS":
            query = "MATCH (d:disease {name: $disease})-[:prohibited_food]->(f:food) RETURN f.name AS food"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            foods = [res['food'] for res in results]
            if foods:
                knowledge_info.append(f"忌吃食物: {', '.join(foods)}")

        elif intent == "QUERY_REQUIRED_TESTS":
            query = "MATCH (d:disease {name: $disease})-[:need_check]->(t:check) RETURN t.name AS test"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            tests = [res['test'] for res in results]
            if tests:
                knowledge_info.append(f"所需检查: {', '.join(tests)}")

        elif intent == "QUERY_DISEASE_SYMPTOMS":
            query = "MATCH (d:disease {name: $disease})-[:has_symptom]->(s:symptom) RETURN s.name AS symptom"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            symptoms = [res['symptom'] for res in results]
            if symptoms:
                knowledge_info.append(f"疾病症状: {', '.join(symptoms)}")

        elif intent == "QUERY_TREATMENT_METHODS":
            query = "MATCH (d:disease {name: $disease})-[:treatment_method]->(t:cure) RETURN t.name AS treatment"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            treatments = [res['treatment'] for res in results]
            if treatments:
                knowledge_info.append(f"治疗方法: {', '.join(treatments)}")

        elif intent == "QUERY_COMORBIDITIES":
            query = "MATCH (d:disease {name: $disease})-[:comorbidity_with]->(c:disease) RETURN c.name AS comorbidity"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            comorbidities = [res['comorbidity'] for res in results]
            if comorbidities:
                knowledge_info.append(f"并发症: {', '.join(comorbidities)}")

        elif intent == "QUERY_DRUG_MANUFACTURER":
            query = "MATCH (d:disease {name: $disease})-[:use_drug]->(p:drug)-[:produces]->(m:producer) RETURN m.name AS manufacturer"
            parameters = {"disease": disease_name}
            results = execute_neo4j_query_multiple(query, parameters)
            manufacturers = [res['manufacturer'] for res in results]
            if manufacturers:
                knowledge_info.append(f"药物生产商: {', '.join(manufacturers)}")

        else:
            logger.warning(f"Unknown intent: {intent}")

    return knowledge_info

def execute_neo4j_query(query, parameters=None):
    try:
        logger.debug(f"Executing query: {query} with parameters: {parameters}")
        result = client.run(query, **(parameters or {})).data()
        logger.debug(f"Query result: {result}")
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return None

def execute_neo4j_query_multiple(query, parameters=None):
    try:
        logger.debug(f"Executing multiple queries: {query} with parameters: {parameters}")
        results = client.run(query, **(parameters or {})).data()
        logger.debug(f"Multiple query results: {results}")
        return results
    except Exception as e:
        logger.error(f"Error executing multiple queries: {e}")
        return []

def generate_prompt(intents, knowledge_info, query):
    prompt = "<指令>你是一个专业的医学专家，请根据以下提示信息和你掌握的医学知识，详细、准确地回答用户的问题。如果提示信息不足，你可以结合自身的知识为用户提供帮助。</指令>"
    prompt += "<指令>请你仅针对医疗类问题提供详细回答。如果问题不是医学或医疗类相关的，你要回答“我只能回答医疗相关的问题。”，以明确告知你的回答限制。</指令>"
    for info in knowledge_info:
        prompt += f"<提示>{info}</提示>"
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += "注意：请确保你的回答详细、准确，你回答的内容中不要带有<>和</>这种形式的标签！"
    return prompt, "、".join(intents), intents