#!/usr/bin/env python3
"""
Step 4: Import ICD-10 Knowledge Graph into Neo4j AuraDB.
Reads data/icd10_knowledge_graph.json and imports all nodes + edges.
"""

import json
import os
import sys
import time

from neo4j import GraphDatabase

HERE = os.path.dirname(os.path.abspath(__file__))
KG_PATH = os.path.join(HERE, "..", "data", "icd10_knowledge_graph.json")

BATCH_SIZE = 500
DELAY_BETWEEN_BATCHES = 0.5


def connect():
    URI = "neo4j+ssc://a9ab656d.databases.neo4j.io"
    user = "a9ab656d"
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        import getpass
        password = getpass.getpass(f"Password for {user}@{URI}: ")

    print(f"Connecting to {URI} ...")
    with GraphDatabase.driver(URI, auth=(user, password)) as driver:
        driver.verify_connectivity()
    print("  Connected OK")
    # Re-create: with block closes the driver
    driver = GraphDatabase.driver(URI, auth=(user, password),
                                  max_connection_lifetime=3600)
    return driver


def clean(driver):
    """Delete all existing nodes and edges."""
    print("  Cleaning existing data ...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  Done")


def create_constraints(driver):
    statements = [
        "CREATE CONSTRAINT chapter_code IF NOT EXISTS FOR (n:Chapter) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT section_code IF NOT EXISTS FOR (n:Section) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT category_code IF NOT EXISTS FOR (n:Category) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT disease_code  IF NOT EXISTS FOR (n:Disease)  REQUIRE n.code IS UNIQUE",
    ]

    with driver.session() as session:
        for stmt in statements:
            try:
                session.run(stmt)
                print(f"  OK: {stmt[:80]}...")
            except Exception as e:
                print(f"  SKIP constraint: {e}")

        try:
            session.run(
                "CREATE FULLTEXT INDEX DiseaseName IF NOT EXISTS "
                "FOR (d:Disease) ON EACH [d.name_vi]"
            )
            print("  OK: FULLTEXT INDEX DiseaseName")
        except Exception as e:
            print(f"  SKIP fulltext index: {e}")

    print("  Constraints done")


def import_nodes(driver, nodes):
    labels = ["Chapter", "Section", "Category", "Disease"]
    for label in labels:
        label_nodes = [n for n in nodes if n["label"] == label]
        if not label_nodes:
            continue

        print(f"\n  Importing {len(label_nodes)} {label} nodes ...")
        for i in range(0, len(label_nodes), BATCH_SIZE):
            batch = label_nodes[i : i + BATCH_SIZE]
            with driver.session() as session:
                result = session.execute_write(
                    _batch_merge_nodes,
                    [{"id": n["id"], "label": n["label"], "props": n["properties"]}
                     for n in batch],
                )
            if (i + 1) % 2000 == 0:
                print(f"    {min(i + BATCH_SIZE, len(label_nodes))}/{len(label_nodes)}")


def _batch_merge_nodes(tx, batch):
    label_map = {
        "Chapter":  "UNWIND $batch AS row MERGE (n:Chapter  {code: row.props.code}) SET n += row.props",
        "Section":  "UNWIND $batch AS row MERGE (n:Section  {code: row.props.code}) SET n += row.props",
        "Category": "UNWIND $batch AS row MERGE (n:Category {code: row.props.code}) SET n += row.props",
        "Disease":  "UNWIND $batch AS row MERGE (n:Disease  {code: row.props.code}) SET n += row.props",
    }

    label = batch[0]["label"]
    query = label_map.get(label)
    if not query:
        return

    tx.run(query, batch=batch)


def import_edges(driver, edges):
    print(f"\n  Importing {len(edges)} HAS_CHILD edges ...")

    for i in range(0, len(edges), BATCH_SIZE):
        batch = edges[i : i + BATCH_SIZE]
        with driver.session() as session:
            session.execute_write(_batch_create_edges, batch)
        if (i + 1) % 5000 == 0:
            print(f"    {min(i + BATCH_SIZE, len(edges))}/{len(edges)}")
        time.sleep(DELAY_BETWEEN_BATCHES)


def _batch_create_edges(tx, batch):
    query = """
    UNWIND $batch AS edge
    MATCH (a)
    WHERE (edge.from_label = 'Chapter'  AND a:Chapter  AND a.code = edge.from_code)
       OR (edge.from_label = 'Section'  AND a:Section  AND a.code = edge.from_code)
       OR (edge.from_label = 'Category' AND a:Category AND a.code = edge.from_code)
       OR (edge.from_label = 'Disease'  AND a:Disease  AND a.code = edge.from_code)
    MATCH (b)
    WHERE (edge.to_label = 'Chapter'  AND b:Chapter  AND b.code = edge.to_code)
       OR (edge.to_label = 'Section'  AND b:Section  AND b.code = edge.to_code)
       OR (edge.to_label = 'Category' AND b:Category AND b.code = edge.to_code)
       OR (edge.to_label = 'Disease'  AND b:Disease  AND b.code = edge.to_code)
    MERGE (a)-[r:HAS_CHILD]->(b)
    """
    mapped = [{"from_code": e["from_code"], "from_label": e["from_label"],
               "to_code": e["to_code"], "to_label": e["to_label"]} for e in batch]
    tx.run(query, batch=mapped)


def verify(driver):
    print("\n  Verifying ...")
    queries = {
        "Chapter":  "MATCH (n:Chapter)  RETURN count(n) AS cnt",
        "Section":  "MATCH (n:Section)  RETURN count(n) AS cnt",
        "Category": "MATCH (n:Category) RETURN count(n) AS cnt",
        "Disease":  "MATCH (n:Disease)  RETURN count(n) AS cnt",
        "HAS_CHILD": "MATCH ()-[r:HAS_CHILD]->() RETURN count(r) AS cnt",
    }

    with driver.session() as session:
        for name, query in queries.items():
            result = session.run(query)
            count = result.single()["cnt"]
            print(f"    {name}: {count}")

        # Sample fulltext search
        try:
            result = session.run(
                "CALL db.index.fulltext.queryNodes('DiseaseName', 'viêm phổi') "
                "YIELD node, score RETURN node.code, node.name_vi, score LIMIT 5"
            )
            rows = list(result)
            if rows:
                print("\n  Sample search 'viem phoi' (fulltext index):")
                for r in rows:
                    name_clean = r['node.name_vi'].encode('ascii', 'replace').decode('ascii')
                    print(f"    {r['node.code']}: {name_clean[:60]} (score={r['score']:.3f})")
        except Exception as e:
            print(f"    Fulltext search test skipped: {e}")


def transform_kg(kg):
    """Deduplicate nodes by (label, code) and add label info to edges."""
    seen = set()
    deduped = []
    for n in kg["nodes"]:
        key = (n["label"], n["properties"]["code"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(n)

    # Build lookup: node_id -> (label, code)
    id_info = {}
    for n in deduped:
        id_info[n["id"]] = (n["label"], n["properties"]["code"])

    new_edges = []
    for e in kg["edges"]:
        from_label, from_code = id_info[e["from"]]
        to_label, to_code = id_info[e["to"]]
        new_edges.append({
            "from_code": from_code,
            "from_label": from_label,
            "to_code": to_code,
            "to_label": to_label,
        })

    print(f"  Deduped: {len(kg['nodes'])} -> {len(deduped)} nodes, "
          f"{len(kg['edges'])} -> {len(new_edges)} edges")
    return deduped, new_edges


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()

    if not os.path.exists(KG_PATH):
        print(f"ERROR: KG not found at {KG_PATH}")
        print("Run step3_build_kg.py first.")
        sys.exit(1)

    print("Loading knowledge graph ...")
    with open(KG_PATH, "r", encoding="utf-8") as f:
        kg = json.load(f)

    stats = kg["stats"]
    print(f"  {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    for label, count in stats["by_label"].items():
        print(f"    {label}: {count}")

    nodes, edges = transform_kg(kg)

    driver = connect()
    try:
        clean(driver)
        create_constraints(driver)
        import_nodes(driver, nodes)
        import_edges(driver, edges)
        verify(driver)
    finally:
        driver.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
