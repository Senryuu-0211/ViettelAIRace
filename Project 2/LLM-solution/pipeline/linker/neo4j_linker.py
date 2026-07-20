import os
from neo4j import GraphDatabase
from pipeline.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ICD_MIN_SCORE, ICD_TOP_K
from pipeline.linker.embedder import embed_query

# Khởi tạo một driver kết nối tới Neo4j (dùng chung)
_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

def lookup_diagnosis(diagnosis_text: str) -> list[str]:
    """
    Nhúng text chẩn đoán và tìm mã bệnh trên Neo4j bằng Vector Search.
    Thay thế cho lookup_diagnosis cũ trong icd10.py
    """
    if not diagnosis_text or not diagnosis_text.strip():
        return []

    # 1. Embed query
    try:
        # embed_query trả về np.ndarray (vector 1 chiều)
        qv = embed_query(diagnosis_text)
        query_vector = qv.tolist()
    except Exception as e:
        print(f"Error embedding '{diagnosis_text}': {e}")
        return []

    # 2. Vector search trên Neo4j
    driver = get_driver()
    candidates = []
    
    query = """
    CALL db.index.vector.queryNodes('disease_name_idx', $top_k, $query_vector)
    YIELD node, score
    WHERE score >= $min_score
    RETURN node.code AS code, score
    ORDER BY score DESC
    """
    
    try:
        with driver.session() as session:
            result = session.run(query, top_k=ICD_TOP_K, query_vector=query_vector, min_score=ICD_MIN_SCORE)
            for record in result:
                code = record["code"]
                if code:
                    candidates.append(code)
    except Exception as e:
        print(f"Error querying Neo4j for '{diagnosis_text}': {e}")
        return []

    return candidates

def lookup_batch(diagnosis_texts: list[str]) -> dict[str, list[str]]:
    result = {}
    for text in diagnosis_texts:
        codes = lookup_diagnosis(text)
        if codes:
            result[text] = codes
    return result
