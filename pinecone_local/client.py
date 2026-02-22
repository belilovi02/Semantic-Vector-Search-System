"""Wrapper around the installed `pinecone` package for project ingestion experiments.

This module imports the real `pinecone` package (installed via pip) and exposes
helper functions used by the experiments code.
"""
import time
from typing import List, Dict, Optional
import os

import pinecone

from config import PINECONE_API_KEY, PINECONE_ENV


def init_pinecone(api_key: Optional[str] = None, env: Optional[str] = None):
    api_key = api_key or PINECONE_API_KEY
    env = env or PINECONE_ENV
    if not api_key or not env:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV in environment or config.")
    pinecone.init(api_key=api_key, environment=env)


def create_index(name: str, dimension: int, metric: str = "cosine", shards: int = 1):
    if name in pinecone.list_indexes():
        print(f"Index {name} already exists")
    else:
        pinecone.create_index(name, dimension=dimension, metric=metric)
        print(f"Created Pinecone index {name} with dim={dimension}, metric={metric}")
    return pinecone.Index(name)


def delete_index(name: str):
    if name in pinecone.list_indexes():
        pinecone.delete_index(name)


def batch_upsert(index, items: List[Dict], batch_size: int = 128, max_workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _upsert_batch(batch):
        t0 = time.time()
        index.upsert(vectors=[(it["id"], it["vector"], it.get("metadata")) for it in batch])
        t1 = time.time()
        return (t0, t1, len(batch))

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    timings = []
    t_start = time.time()
    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_upsert_batch, b): idx for idx, b in enumerate(batches)}
            for fut in as_completed(futures):
                try:
                    timings.append(fut.result())
                except Exception:
                    timings.append((0, 0, 0))
    else:
        for b in batches:
            timings.append(_upsert_batch(b))

    total = time.time() - t_start
    return timings, total


def query_index(index, query_vector, top_k: int = 10, filter: Optional[Dict] = None):
    res = index.query(queries=[query_vector], top_k=top_k, include_metadata=True, filter=filter)
    return res
