import os
import sys
import time
import numpy as np

# Thêm thư mục gốc vào sys.path để import pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from neo4j import GraphDatabase
from pipeline.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

def embed_batch(texts):
    resp = httpx.post(OLLAMA_EMBED_URL, json={"model": OLLAMA_EMBED_MODEL, "input": texts}, timeout=300)
    resp.raise_for_status()
    return resp.json()["embeddings"]

def embed_single(text):
    return embed_batch([text])[0]

def process_batch(tx, nodes_batch, embeddings):
    # Cập nhật embedding vào Neo4j
    query = """
    UNWIND $batch AS item
    MATCH (d:Disease {code: item.code})
    SET d.embedding = item.embedding
    """
    tx.run(query, batch=[{ "code": item["code"], "embedding": emb } for item, emb in zip(nodes_batch, embeddings)])

def init_neo4j_vectors():
    print(f"--- DEBUG ---")
    print(f"OLLAMA_BASE_URL  = {OLLAMA_BASE_URL}")
    print(f"OLLAMA_EMBED_MODEL = {OLLAMA_EMBED_MODEL}")
    print(f"NEO4J_URI        = {NEO4J_URI}")
    print(f"NEO4J_USER       = {NEO4J_USER}")
    print(f"NEO4J_PASSWORD   = {'***' if NEO4J_PASSWORD else 'EMPTY!'}")
    print(f"---")
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print(f"Verifying Ollama at {OLLAMA_BASE_URL}...")
    try:
        tags_resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        tags_resp.raise_for_status()
        models = [m["name"] for m in tags_resp.json().get("models", [])]
        print(f"Ollama OK, models: {models}")
    except Exception as e:
        print(f"WARNING: Ollama check failed: {e}")
    
    with driver.session() as session:
        # Lấy tổng số Disease nodes có name_vi
        result = session.run("MATCH (d:Disease) WHERE d.name_vi IS NOT NULL AND d.embedding IS NULL RETURN count(d) as total")
        total = result.single()["total"]
        print(f"Found {total} Disease nodes to embed.")
        
        batch_size = 500
        offset = 0
        
        while offset < total:
            print(f"Processing batch {offset} to {offset + batch_size}...")
            # Lấy batch
            nodes = session.run(
                "MATCH (d:Disease) WHERE d.name_vi IS NOT NULL AND d.embedding IS NULL "
                "RETURN d.code AS code, d.name_vi AS name SKIP $skip LIMIT $limit",
                skip=offset, limit=batch_size
            ).data()
            
            if not nodes:
                break
                
            texts = [node["name"] for node in nodes]
            print(f"  Embedding {len(texts)} texts...")
            
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    vectors = embed_batch(texts)
                    
                    vecs_np = np.array(vectors, dtype=np.float32)
                    norms = np.linalg.norm(vecs_np, axis=-1, keepdims=True)
                    norms[norms == 0] = 1.0
                    vecs_np = vecs_np / norms
                    vectors_normalized = vecs_np.tolist()
                    
                    print("  Updating Neo4j...")
                    session.execute_write(process_batch, nodes, vectors_normalized)
                    print(f"  Batch complete.")
                    break
                    
                except Exception as e:
                    print(f"Error embedding batch (attempt {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        wait = 5 * attempt
                        print(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print("  Max retries reached, falling back to single-node embedding...")
                        for i, node in enumerate(nodes):
                            try:
                                v = embed_single(node["name"])
                                v_np = np.array(v, dtype=np.float32)
                                n = np.linalg.norm(v_np)
                                v_np = v_np / n if n > 0 else v_np
                                session.execute_write(process_batch, [node], [v_np.tolist()])
                            except Exception as e2:
                                print(f"Error on code {node['code']}: {e2}")
            
            offset += batch_size

        print("Creating Vector Index 'disease_name_idx'...")
        try:
            # Tạo vector index (chiều mặc định của bge-m3 là 1024)
            session.run("CREATE VECTOR INDEX disease_name_idx IF NOT EXISTS FOR (d:Disease) ON (d.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}")
            print("Vector Index created successfully.")
        except Exception as e:
            print(f"Error creating index: {e}")

    driver.close()
    print("Done!")

if __name__ == "__main__":
    init_neo4j_vectors()
