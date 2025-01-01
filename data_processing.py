import os
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def load_medical_data(json_path):
    data = []
    decoder = json.JSONDecoder()
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            idx = 0
            length = len(content)
            while idx < length:
                obj, idx_new = decoder.raw_decode(content, idx)
                data.append(obj)
                idx = idx_new
                # Skip commas and whitespace characters
                while idx < length and content[idx] in ', \n\r\t':
                    idx += 1
        logging.info(f"Loaded {len(data)} records from {json_path}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
    except Exception as e:
        logging.error(f"fail to read file: {e}")
    return data


def extract_entities_and_relationships(data):
    """
    Args:
        data (list):  Medical data list.
    Returns:
        dict:  Entity collection.
        list:  Relationship list.
    """
    all_entities = {
        "disease": [],
        "drug": [],
        "food": [],
        "check": [],
        "department": [],
        "symptom": [],
        "cure": [],
        "producer": [],
    }
    relationships = []

    for entry in tqdm(data, desc="Processing data"):
        disease_name = entry.get("name", "")
        if not disease_name:
            continue

        disease_entity = {
            "name": disease_name,
            "description": entry.get("description", "").strip(),
            "cause": entry.get("cause", "").strip(),
            "prevent": entry.get("prevent", "").strip(),
            "treatment_duration": entry.get("treatment_duration", "").strip(),
            "cure_rate": entry.get("cure_rate", "").strip(),
            "susceptible_populations": entry.get("susceptible_populations", "").strip(),
        }
        all_entities["disease"].append(disease_entity)

        # Drug entities and relationships
        drugs = entry.get("common_drug", []) + entry.get("recommand_drug", [])
        cleaned_drugs = [drug.strip() for drug in drugs if drug.strip()]
        all_entities["drug"].extend(cleaned_drugs)
        relationships.extend([("disease", disease_name, "use_drug", "drug", drug) for drug in cleaned_drugs])

        # Food entities and relationships
        recommended_eat = entry.get("recommended_food", []) + entry.get("recommended_recipe", [])
        prohibited_eat = entry.get("prohibited_food", [])
        all_entities["food"].extend(recommended_eat + prohibited_eat)
        relationships.extend([("disease", disease_name, "recommended_food", "food", food) for food in recommended_eat])
        relationships.extend([("disease", disease_name, "prohibited_food", "food", food) for food in prohibited_eat])

        # Check entities and relationships
        checks = entry.get("check", [])
        all_entities["check"].extend(checks)
        relationships.extend([("disease", disease_name, "need_check", "check", check) for check in checks])

        # department entities and relationships
        departments = entry.get("department", [])
        all_entities["department"].extend(departments)
        if departments:
            relationships.append(("disease", disease_name, "association", "department", departments[-1]))

        # symptom entities and relationships
        symptoms = [symptom.rstrip('...') for symptom in entry.get("symptom", [])]
        all_entities["symptom"].extend(symptoms)
        relationships.extend([("disease", disease_name, "has_symptom", "symptom", symptom) for symptom in symptoms])

        # cure entities and relationships
        cure_methods = entry.get("cure", [])
        if cure_methods:
            cure_methods = [method[0] if isinstance(method, list) else method for method in cure_methods]
            cure_methods = [method for method in cure_methods if len(method) >= 2]
            all_entities["cure"].extend(cure_methods)
            relationships.extend(
                [("disease", disease_name, "treatment_method", "cure", method) for method in cure_methods])

        # comorbidity_with relationships
        comorbidity_with = entry.get("comorbidity_with", [])
        relationships.extend([("disease", disease_name, "comorbidity_with", "disease", cw) for cw in comorbidity_with])

        # producer entities and relationships
        producers = entry.get("producer", [])
        for detail in producers:
            # Process two formats: Drug, producer, and 'producer (drug)'
            if '(' in detail and ')' in detail:
                parts = detail.split('(')
                if len(parts) != 2:
                    logging.warning(f"Invalid producer detail format (producer(drug)): {detail}")
                    continue
                producer = parts[0].strip()
                drug = parts[1].rstrip(')').strip()
            elif ',' in detail:
                parts = detail.split(',')
                if len(parts) != 2:
                    logging.warning(f"Invalid producer detail format (drug,producer): {detail}")
                    continue
                drug = parts[0].strip()
                producer = parts[1].strip()
            else:
                logging.warning(f"Unknown producer detail format: {detail}")
                continue

            if not drug or not producer:
                logging.warning(f"Incomplete producer detail: {detail}")
                continue

            all_entities["drug"].append(drug.strip())
            all_entities["producer"].append(producer.strip())
            relationships.append(("producer", producer.strip(), "produces", "drug", drug.strip()))

    # Remove duplicate entities
    for entity_type in all_entities:
        if entity_type != "disease":
            all_entities[entity_type] = list(set(all_entities[entity_type]))

    return all_entities, relationships


def save_entities(all_entities, output_dir='data/ents'):
    os.makedirs(output_dir, exist_ok=True)
    for entity_type, entities in all_entities.items():
        file_path = os.path.join(output_dir, f'{entity_type}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            if entity_type == "disease":
                for disease in entities:
                    f.write(disease['name'] + '\n')
            else:
                f.write('\n'.join(entities))
    logging.info(f"Entities saved to {output_dir}")


def save_relationships(relationships, file_path='data/ents_rel.txt'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for rel in relationships:
            f.write(' '.join(rel) + '\n')
    logging.info(f"Relationships saved to {file_path}")