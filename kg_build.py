import py2neo
import argparse
import logging
from collections import defaultdict

from data_processing import load_medical_data, extract_entities_and_relationships, save_entities, save_relationships

logging.basicConfig(level=logging.INFO)

def import_entities_batch(graph, label, entities):
    """
    Batch import entities into Neo4j
    """
    unique_entities = list(set(entities))
    query = f"""
    UNWIND $entities AS name
    MERGE (n:`{label}` {{name: name}})
    """
    graph.run(query, entities=unique_entities)
    logging.info(f'Imported {len(unique_entities)} {label} entities')

def import_disease_entities_batch(graph, diseases):
    """
    Batch import disease entities and their attributes into Neo4j.
    """
    query = """
    UNWIND $diseases AS disease
    MERGE (n:disease {name: disease.name})
    SET n += disease.properties
    """
    disease_data = [
        {
            'name': disease['name'],
            'properties': {k: v for k, v in disease.items() if k != 'name'}
        }
        for disease in diseases
    ]
    graph.run(query, diseases=disease_data)
    logging.info(f'Imported {len(disease_data)} disease entities.')

def create_relationships_batch(graph, relationships):
    """
    Batch create relationships between entities
    """
    rel_groups = defaultdict(list)
    for rel in relationships:
        start_label, start_name, rel_type, end_label, end_name = rel
        key = (start_label, rel_type, end_label)
        rel_groups[key].append({'start_name': start_name, 'end_name': end_name})

    total_rels = 0
    for (start_label, rel_type, end_label), rels in rel_groups.items():
        query = f"""
        UNWIND $relationships AS rel
        MATCH (a:`{start_label}` {{name: rel.start_name}}), (b:`{end_label}` {{name: rel.end_name}})
        MERGE (a)-[r:`{rel_type}`]->(b)
        """
        graph.run(query, relationships=rels)
        total_rels += len(rels)
    logging.info(f'Created {total_rels} relationships.')

def main():
    parser = argparse.ArgumentParser(description="Building Knowledge Graph.")
    parser.add_argument('--website', type=str, default='http://localhost:7474', help='neo4j connection address')
    parser.add_argument('--username', type=str, default='neo4j', help='username')
    parser.add_argument('--password', type=str, default='12345678', help='password')
    parser.add_argument('--database', type=str, default='chinese_medical', help='database')
    parser.add_argument('--json_path', type=str, default='data/Chinese_medical.json', help='file path')
    args = parser.parse_args()

    try:
        graph = py2neo.Graph(args.website, user=args.username, password=args.password, name=args.database)
        logging.info("Successfully connected to Neo4j database!")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        exit(1)

    # clear db or not ?
    is_delete = input('Do you want to delete all entities and relationships on Neo4j? (y/n): ')
    if is_delete.lower() == 'y':
        graph.run("MATCH (n) DETACH DELETE n")
        logging.info("All nodes and relationships in the database have been cleared！")

    # read data
    all_data = load_medical_data(args.json_path)
    if not all_data:
        logging.error("No data loaded. Exiting.")
        exit(1)
    logging.info("Successfully loaded the data file")

    all_entities, relationships = extract_entities_and_relationships(all_data)

    # Save entities and relationships to files
    save_entities(all_entities, output_dir='data/ents')
    save_relationships(relationships, file_path='data/ent_rel.txt')

    # Import entities into Neo4j
    for label, entities in all_entities.items():
        if label == "disease":
            import_disease_entities_batch(graph, entities)
        else:
            import_entities_batch(graph, label, entities)

    create_relationships_batch(graph, relationships)
    logging.info("Knowledge graph construction completed!")

if __name__ == "__main__":
    main()