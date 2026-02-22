"""Search execution and latency measurement.

Supports both Weaviate and Pinecone wrappers.

Functions:
- run_search_experiment(db_type, client, queries, model_encoder, top_k, filter_fn)
  -> returns retrievals dict, latency statistics
"""
import time
import numpy as np
from typing import List, Dict, Callable
from tqdm import tqdm


def latency_stats(times):
    arr = np.array(times)
    return {"mean_ms": float(np.mean(arr) * 1000), "p95_ms": float(np.percentile(arr, 95) * 1000), "p99_ms": float(np.percentile(arr, 99) * 1000)}


def run_search_weaviate(client, weaviate_wrapper, queries: List[Dict], encoder, top_k: int = 10, metadata_filter_fn: Callable = None):
    retrievals = {}
    latencies = []
    for q in tqdm(queries):
        qid = q["id"]
        qtext = q["query"]
        vec = encoder.encode([qtext], batch_size=1, show_progress=False)[0]
        t0 = time.time()
        where = None
        if metadata_filter_fn:
            where = metadata_filter_fn(q)
        res = weaviate_wrapper.query_vector_search(client, vec, top_k=top_k, filters=where)
        t1 = time.time()
        latencies.append(t1 - t0)
        hits = []
        try:
            for h in res.get("data", {}).get("Get", {}).get("Document", []) or []:
                hits.append(h.get("id"))
        except Exception:
            # fallback parse
            pass
        retrievals[qid] = hits
    stats = latency_stats(latencies)
    stats["qps"] = len(queries) / sum(latencies) if sum(latencies) > 0 else None
    return retrievals, stats


def run_search_pinecone(index, queries: List[Dict], encoder, top_k: int = 10, metadata_filter: Dict = None):
    retrievals = {}
    latencies = []
    for q in tqdm(queries):
        qid = q["id"]
        qtext = q["query"]
        vec = encoder.encode([qtext], batch_size=1, show_progress=False)[0]
        t0 = time.time()
        res = index.query(query_vector=vec, top_k=top_k, filter=metadata_filter) if hasattr(index, 'query') else index.query(vec, top_k=top_k)
        t1 = time.time()
        latencies.append(t1 - t0)
        hits = []
        try:
            # pinecone returns results in a nested structure
            matches = res['results'][0]['matches'] if 'results' in res else res['matches']
            for m in matches:
                hits.append(m['id'])
        except Exception:
            pass
        retrievals[qid] = hits
    stats = latency_stats(latencies)
    stats["qps"] = len(queries) / sum(latencies) if sum(latencies) > 0 else None
    return retrievals, stats
