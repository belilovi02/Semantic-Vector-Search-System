"""Weaviate helper utilities: start, schema, ingest, query wrappers.

Schema notes:
- Class `Document` with text, category, timestamp, source.
- vectorIndexConfig tuned (HNSW): efConstruction and maxConnections adjust recall/ingest tradeoffs.
"""
import time
import subprocess
from typing import List, Dict, Optional

import weaviate


def start_weaviate_via_docker(compose_path: str = "./weaviate/docker-compose.yml"):
    """Start local Weaviate using docker-compose. Requires Docker to be installed.
    This is optional — you can use a cloud Weaviate by setting WEAVIATE_URL in env.
    """
    cmd = ["docker", "compose", "-f", compose_path, "up", "-d"]
    subprocess.check_call(cmd)
    print("Weaviate started (give it a few seconds to be ready)...")


def get_client(url: str = "http://localhost:8080", api_key: Optional[str] = None):
    auth = None
    if api_key:
        auth = weaviate.AuthApiKey(api_key)
    client = weaviate.Client(url, auth_client=auth)
    return client


def create_schema(client: weaviate.Client, class_name: str = "Document"):
    # Delete if already exists
    try:
        client.schema.delete_class(class_name)
    except Exception:
        pass

    class_obj = {
        "class": class_name,
        "vectorizer": "none",  # we'll provide precomputed vectors
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "category", "dataType": ["text"]},
            {"name": "timestamp", "dataType": ["date"]},
            {"name": "source", "dataType": ["text"]},
        ],
        "vectorIndexConfig": {
            "hnsw": {
                # M (max connections) controls graph connectivity; higher M => higher recall, slower ingestion and more RAM
                "m": 32,
                # efConstruction controls quality of graph during build
                "efConstruction": 128
            }
        },
    }
    client.schema.create_class(class_obj)
    print(f"Created class {class_name} in Weaviate with HNSW parameters m=32 efConstruction=128")


def batch_insert_documents(client: weaviate.Client, docs: List[Dict], vectors: List[List[float]], class_name: str = "Document", batch_size: int = 128):
    """Insert docs with vectors in batches and measure ingestion latency/throughput.
    Expects docs list of dicts with keys id, text, category, timestamp, source.
    Returns list of (start_time, end_time, inserted)
    """
    importer = client.batch
    start = time.time()
    timings = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        t0 = time.time()
        with importer as b:
            for d, vec in zip(batch_docs, batch_vecs):
                properties = {"text": d["text"], "category": d["category"], "timestamp": d["timestamp"], "source": d["source"]}
                b.add_data_object(properties, class_name=class_name, uuid=d["id"], vector=vec)
        t1 = time.time()
        timings.append((t0, t1, len(batch_docs)))
    total_time = time.time() - start
    return timings, total_time


def query_vector_search(client: weaviate.Client, query_vec, top_k: int = 10, class_name: str = "Document", filters: Dict = None):
    # Build where filter if provided
    where = None
    if filters:
        # Example: {"path": ["category"], "operator": "Equal", "valueText": "sports"}
        where = {"operator": "And", "operands": filters}
    res = client.query.get(class_name, ["id", "text", "category"]).with_near_vector({"vector": query_vec, "certainty": None}).with_limit(top_k)
    if where:
        res = res.with_where(where)
    out = res.do()
    return out
