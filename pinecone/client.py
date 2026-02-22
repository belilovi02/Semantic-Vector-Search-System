"""Pinecone helper utilities: init, create index, upsert, query.

Pinecone usage notes:
- Requires PINECONE_API_KEY and PINECONE_ENV to be set (see .env.example)
- Index dimension must match embedding dim. Metric recommended: 'cosine' or 'dotproduct'
"""
import time
from typing import List, Dict, Optional
import os

import importlib
import importlib.util
import sys
import os

from config import PINECONE_API_KEY, PINECONE_ENV


def _load_external_pinecone():
    # Attempt to locate the installed `pinecone` package in site-packages and load it
    for p in sys.path:
        if not p:
            continue
        # look for typical site-packages locations
        if ("site-packages" in p) or ("dist-packages" in p):
            candidate = os.path.join(p, "pinecone", "__init__.py")
            if os.path.isfile(candidate):
                spec = importlib.util.spec_from_file_location("pinecone_ext", candidate)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    # fallback: try a normal import from sys.path (may raise)
    return importlib.import_module("pinecone")


pinecone_ext = _load_external_pinecone()


def init_pinecone(api_key: Optional[str] = None, env: Optional[str] = None):
    api_key = api_key or PINECONE_API_KEY
    env = env or PINECONE_ENV
    if not api_key or not env:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV in environment or config.")
    pinecone_ext.init(api_key=api_key, environment=env)


def create_index(name: str, dimension: int, metric: str = "cosine", shards: int = 1):
    if name in pinecone_ext.list_indexes():
        print(f"Index {name} already exists")
    else:
        pinecone_ext.create_index(name, dimension=dimension, metric=metric)
        print(f"Created Pinecone index {name} with dim={dimension}, metric={metric}")
    return pinecone_ext.Index(name)


def delete_index(name: str):
    if name in pinecone_ext.list_indexes():
        pinecone_ext.delete_index(name)


def batch_upsert(index, items: List[Dict], batch_size: int = 128, max_workers: int = 4):
    """Upsert items in batches. Use a ThreadPoolExecutor to parallelize batch upserts for throughput tests.
    Returns timings per batch and total time.
    """
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
                except Exception as e:
                    # record failed batch as zero time
                    timings.append((0, 0, 0))
    else:
        for b in batches:
            timings.append(_upsert_batch(b))

    total = time.time() - t_start
    return timings, total


def query_index(index, query_vector, top_k: int = 10, filter: Optional[Dict] = None):
    # filter expects Pinecone metadata filter dict
    res = index.query(queries=[query_vector], top_k=top_k, include_metadata=True, filter=filter)
    return res
