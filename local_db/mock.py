"""Simple in-memory mock implementations for Weaviate and Pinecone-like APIs.
Used when external services aren't available to allow full H1 measurements locally.
"""
import time
from typing import List, Dict, Tuple
import numpy as np

class SimpleWeaviateMock:
    def __init__(self):
        # store docs as list of dicts and vectors as numpy array
        self.docs = []  # list of dicts with id
        self.vectors = []  # list of numpy arrays

    def batch_insert_documents(self, docs: List[Dict], vectors: List[List[float]], batch_size: int = 128) -> Tuple[List[Tuple[float,float,int]], float]:
        timings = []
        t_start = time.time()
        for i in range(0, len(docs), batch_size):
            bdocs = docs[i:i+batch_size]
            bvecs = vectors[i:i+batch_size]
            t0 = time.time()
            for d, v in zip(bdocs, bvecs):
                self.docs.append(d)
                self.vectors.append(np.array(v, dtype=np.float32))
            t1 = time.time()
            timings.append((t0, t1, len(bdocs)))
        total = time.time() - t_start
        return timings, total

    def query_vector_search(self, query_vec: List[float], top_k: int = 10):
        if len(self.vectors) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        mat = np.vstack(self.vectors)
        # compute dot product
        scores = mat.dot(q)
        idx = np.argsort(-scores)[:top_k]
        return [self.docs[i]["id"] for i in idx]


class SimplePineconeMock:
    def __init__(self):
        self.ids = []
        self.vectors = []

    def create_index(self, name: str, dimension: int = 512, metric: str = "cosine"):
        # return self as index
        return self

    def batch_upsert(self, index, items: List[Dict], batch_size: int = 128) -> Tuple[List[Tuple[float,float,int]], float]:
        timings = []
        t_start = time.time()
        for i in range(0, len(items), batch_size):
            b = items[i:i+batch_size]
            t0 = time.time()
            for it in b:
                self.ids.append(it["id"])
                self.vectors.append(np.array(it["vector"], dtype=np.float32))
            t1 = time.time()
            timings.append((t0, t1, len(b)))
        total = time.time() - t_start
        return timings, total

    def query(self, queries: List[List[float]], top_k: int = 10, include_metadata: bool = False):
        if len(self.vectors) == 0:
            return {"results": []}
        q = np.array(queries[0], dtype=np.float32)
        mat = np.vstack(self.vectors)
        scores = mat.dot(q)
        idx = np.argsort(-scores)[:top_k]
        res = [{"id": self.ids[i], "score": float(scores[i])} for i in idx]
        return {"matches": res}
