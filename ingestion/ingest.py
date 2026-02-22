"""Ingestion experiments: measure ingestion latency and throughput across dataset sizes.

Functions:
- run_ingestion_experiment(db_client, docs, vectors, batch_size, db_type) -> DataFrame with timings and summary

This module is database-agnostic: relies on weaviate.client and pinecone.client wrappers.
"""
import time
import pandas as pd
from typing import List, Dict


def summarize_timings(timings):
    # timings: list of (t0, t1, count)
    rows = []
    total_items = 0
    total_time = 0.0
    for t0, t1, count in timings:
        delta = t1 - t0
        rows.append({"batch_time": delta, "batch_size": count, "throughput_vps": count / delta if delta > 0 else None})
        total_time += delta
        total_items += count
    summary = {
        "total_items": total_items,
        "total_time_s": total_time,
        "overall_throughput_vps": total_items / total_time if total_time > 0 else None,
    }
    df = pd.DataFrame(rows)
    return df, summary


def run_weaviate_ingest(weaviate_client, wrapper_module, docs: List[Dict], vectors: List[List[float]], batch_size: int = 128):
    timings, total_time = wrapper_module.batch_insert_documents(weaviate_client, docs, vectors, batch_size=batch_size)
    df, summary = summarize_timings(timings)
    summary.update({"db": "weaviate", "batch_size": batch_size})
    return df, summary


def run_pinecone_ingest(index, pinecone_wrapper, docs: List[Dict], vectors: List[List[float]], batch_size: int = 128):
    # Build items
    items = []
    for d, v in zip(docs, vectors):
        items.append({"id": d["id"], "vector": v, "metadata": {"category": d["category"], "timestamp": d["timestamp"], "source": d["source"]}})
    timings, total_time = pinecone_wrapper.batch_upsert(index, items, batch_size=batch_size)
    df, summary = summarize_timings(timings)
    summary.update({"db": "pinecone", "batch_size": batch_size})
    return df, summary


def run_weaviate_ingest_stream(weaviate_client, wrapper_module, docs: List[Dict], encoder, batch_size: int = 128):
    """Stream-encode docs and insert in batches to Weaviate; useful to avoid large memmap files on Windows.
    Returns (df, summary) similar to run_weaviate_ingest.
    """
    timings = []
    n = len(docs)
    for i in range(0, n, batch_size):
        batch_docs = docs[i : i + batch_size]
        texts = [d["text"] for d in batch_docs]
        embs = encoder.encode(texts, batch_size=len(texts), show_progress=False)
        # wrapper_module.batch_insert_documents expects lists of vectors
        batch_timings, _ = wrapper_module.batch_insert_documents(weaviate_client, batch_docs, embs.tolist(), batch_size=len(batch_docs))
        timings.extend(batch_timings)
    df, summary = summarize_timings(timings)
    summary.update({"db": "weaviate", "batch_size": batch_size})
    return df, summary


def run_pinecone_ingest_stream(index, pinecone_wrapper, docs: List[Dict], encoder, batch_size: int = 128):
    """Stream-encode docs and upsert in batches to Pinecone; avoids creating a full vectors buffer in memory or on disk.
    """
    timings = []
    n = len(docs)
    for i in range(0, n, batch_size):
        batch_docs = docs[i : i + batch_size]
        texts = [d["text"] for d in batch_docs]
        embs = encoder.encode(texts, batch_size=len(texts), show_progress=False)
        items = []
        for d, v in zip(batch_docs, embs):
            items.append({"id": d["id"], "vector": v.tolist(), "metadata": {"category": d["category"], "timestamp": d["timestamp"], "source": d["source"]}})
        batch_timings, _ = pinecone_wrapper.batch_upsert(index, items, batch_size=len(items))
        timings.extend(batch_timings)
    df, summary = summarize_timings(timings)
    summary.update({"db": "pinecone", "batch_size": batch_size})
    return df, summary


# Note: In experiments we will orchestrate calling the above wrappers and storing results to disk.
