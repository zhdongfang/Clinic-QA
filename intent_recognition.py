# intent_recognition.py

import ollama
import json
import logging
import re

logger = logging.getLogger(__name__)

def intent_recognition(query: str, choice: str, entities: dict) -> dict:
    prompt = f"""
    阅读下列提示，回答问题（问题在输入的最后）:
    当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

    **提取的实体**
    {", ".join([f"{k}: {v}" for k, v in entities.items()])}
    
    **查询类别**
    - "QUERY_DISEASE_DESCRIPTION"
    - "QUERY_DISEASE_CAUSE"
    - "QUERY_PREVENTION_MEASURES"
    - "QUERY_TREATMENT_DURATION"
    - "QUERY_CURE_PROBABILITY"
    - "QUERY_SUSCEPTIBLE_POPULATION"
    - "QUERY_REQUIRED_DRUGS"
    - "QUERY_RECOMMENDED_FOODS"
    - "QUERY_AVOIDED_FOODS"
    - "QUERY_REQUIRED_TESTS"
    - "QUERY_DISEASE_SYMPTOMS"
    - "QUERY_TREATMENT_METHODS"
    - "QUERY_COMORBIDITIES"
    - "QUERY_DRUG_MANUFACTURER"

    在处理用户的问题时，请按照以下步骤操作：
    - 仔细阅读用户的问题。
    - 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
    - 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
    - 确保最终的输出列表包含了所有与用户问题相关的类别描述。

    以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
    **示例1：**
    输入："睡眠不好，这是为什么？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_DISEASE_CAUSE"]
    **示例2：**
    输入："感冒了，怎么办才好？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_REQUIRED_DRUGS", "QUERY_TREATMENT_METHODS"]
    **示例3：**
    输入："跑步后膝盖痛，需要吃点什么？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_RECOMMENDED_FOODS", "QUERY_REQUIRED_DRUGS"]
    **示例4：**
    输入："我怎样才能避免冬天的流感和感冒？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_PREVENTION_MEASURES"]
    **示例5：**
    输入："头疼是什么原因，应该怎么办？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_DISEASE_CAUSE", "QUERY_TREATMENT_METHODS"]
    **示例6：**
    输入："如何知道自己是不是有艾滋病？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_REQUIRED_TESTS","QUERY_DISEASE_CAUSE"]
    **示例7：**
    输入："我该怎么知道我自己是否得了21三体综合症呢？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_REQUIRED_TESTS","QUERY_DISEASE_CAUSE"]
    **示例8：**
    输入："感冒了，怎么办？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_TREATMENT_METHODS","QUERY_REQUIRED_DRUGS","QUERY_REQUIRED_TESTS","QUERY_RECOMMENDED_FOODS"]
    **示例9：**
    输入："癌症会引发其他疾病吗？"
    输出：["QUERY_DISEASE_DESCRIPTION","QUERY_COMORBIDITIES","QUERY_DISEASE_DESCRIPTION"]
    **示例10：**
    输入："三九感冒灵的生产者是谁？三九感冒灵是谁生产的？"
    输出：["QUERY_DRUG_MANUFACTURER"]
    **示例11：**
    输入："肺栓塞，应该吃什么"
    输出：["QUERY_DISEASE_DESCRIPTION", "QUERY_RECOMMENDED_FOODS", "QUERY_REQUIRED_DRUGS"]
    #肺栓塞是一种严重的疾病，用户可能在询问相关的饮食建议和必要的药物治疗，因此包括了"QUERY_RECOMMENDED_FOODS"和"QUERY_REQUIRED_DRUGS"。
    通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。
    
    **注意：**
    - 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
    - 参考上述11个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
    - 你的输出的类别数量不应该超过5个，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。
    
    现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
    问题输入："{query}"
    输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"QUERY_DISEASE_DESCRIPTION"这个需求。
    再次检查你的输出都包含在**查询类别**:"QUERY_DISEASE_DESCRIPTION","QUERY_DISEASE_CAUSE","QUERY_PREVENTION_MEASURES","QUERY_TREATMENT_DURATION","QUERY_CURE_PROBABILITY","QUERY_SUSCEPTIBLE_POPULATION","QUERY_REQUIRED_DRUGS","QUERY_RECOMMENDED_FOODS","QUERY_AVOIDED_FOODS","QUERY_REQUIRED_TESTS","QUERY_DISEASE_SYMPTOMS","QUERY_TREATMENT_METHODS","QUERY_COMORBIDITIES","QUERY_DRUG_MANUFACTURER".
    """
    try:
        rec_result = ollama.generate(model=choice, prompt=prompt)['response']
        match = re.search(r'\[.*\]', rec_result)
        if match:
            intent_list = eval(match.group(0))
            rec_result = {"intent": intent_list}
            intent_dict = json.dumps(rec_result, ensure_ascii=False)
        return intent_dict
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing intent recognition result error: {e}")
        logger.error(f"Intention recognition of original results: {rec_result}")
        return {"intent": None}
    except Exception as e:
        logger.error(f"Intention recognition failed : {e}")
        return {"intent": None}