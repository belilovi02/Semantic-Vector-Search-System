"""Orchestrator for running ingestion, query and evaluation experiments.

This script ties together data, embeddings, db clients, ingestion, querying and evaluation.

It produces CSV/JSON summaries under project/experiments/ for later analysis and plotting.
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from data.dataset import DATA_DIR
from embeddings.encoder import SentenceTransformerEncoder, BertEncoder, DummyEncoder
from weaviate import client as weav_client_module
from pinecone_local import client as pinecone_client_module
from ingestion.ingest import run_weaviate_ingest, run_pinecone_ingest
from evaluation.search_eval import run_search_weaviate, run_search_pinecone
from evaluation.metrics import evaluate_all
from config import DEFAULT_ST_MODEL, DEFAULT_BERT_MODEL, DEFAULT_BATCH_SIZE


EXPERIMENTS_DIR = Path(__file__).resolve().parents[0]
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_documents(n_docs: int):
    docs_file = DATA_DIR / f"documents_{n_docs}.jsonl"
    if not docs_file.exists():
        raise FileNotFoundError(f"Documents file {docs_file} not found. Run data.prepare_dataset first.")
    docs = [json.loads(line) for line in docs_file.open("r", encoding="utf-8")]
    return docs


def _load_queries_and_qrels():
    qfile = DATA_DIR / "queries.jsonl"
    qrels_file = DATA_DIR / "qrels.json"
    queries = [json.loads(line) for line in qfile.open("r", encoding="utf-8")] if qfile.exists() else []
    qrels = json.load(qrels_file.open("r", encoding="utf-8")) if qrels_file.exists() else {}
    return queries, qrels


def encode_documents(encoder, docs: List[Dict], model_name: str, batch_size: int = 256, chunk_size: int = 10000):
    """Encode documents in streaming/chunked fashion and persist to disk using numpy memmap.

    This avoids keeping all vectors in memory for very large corpora (e.g., 1M documents).
    - `chunk_size` controls how many documents are encoded and written per iteration.
    """
    texts_iter = (d["text"] for d in docs)
    n = len(docs)
    # temporary encoder dimension determination (encode a small batch)
    sample_texts = [d["text"] for d in docs[: min(8, n)]]
    sample_emb = encoder.encode(sample_texts, batch_size=8, show_progress=False)
    dim = sample_emb.shape[1]

    out_file = DATA_DIR / f"vectors_{model_name}_{n}.dat"
    if out_file.exists():
        print(f"Loading existing memmap vectors from {out_file}")
        vectors = np.memmap(out_file, dtype="float32", mode="r", shape=(n, dim))
        return vectors

    print(f"Encoding {n} documents using {model_name} into memmap {out_file} (dim={dim})...")
    # create memmap file
    mmap = np.memmap(out_file, dtype="float32", mode="w+", shape=(n, dim))

    idx = 0
    texts = []
    for d in docs:
        texts.append(d["text"])
        if len(texts) >= chunk_size:
            embs = encoder.encode(texts, batch_size=batch_size, show_progress=True)
            mmap[idx : idx + len(embs), :] = embs.astype("float32")
            idx += len(embs)
            texts = []
    # final batch
    if texts:
        embs = encoder.encode(texts, batch_size=batch_size, show_progress=True)
        mmap[idx : idx + len(embs), :] = embs.astype("float32")
        idx += len(embs)

    # flush to disk and reopen read-only
    mmap.flush()
    vectors = np.memmap(out_file, dtype="float32", mode="r", shape=(n, dim))
    return vectors


def run_all_experiments(models: List[str] = ["sentence_transformer", "bert"], sizes: List[int] = [10000, 50000, 100000], sample_queries: int = 200):
    # Load a maximal set of documents (largest size)
    max_size = max(sizes)
    docs = _load_documents(max_size)
    queries, qrels = _load_queries_and_qrels()
    queries_sample = queries[:sample_queries]

    results = []

    # Fallback: if LOCAL_ONLY and FORCE_HASHING_ENCODER are set, only use DummyEncoder and skip all others
    use_fallback = (
        os.environ.get("LOCAL_ONLY", "0") == "1" and os.environ.get("FORCE_HASHING_ENCODER", "0") == "1"
    )
    if use_fallback:
        print("[INFO] Fallback mode: using DummyEncoder only (no torch/transformers will be imported).")
        models = ["dummy"]
    for model in models:
        if use_fallback or model == "dummy":
            encoder = DummyEncoder(max_dim=512)
            model_name = "dummy_tfidf"
        elif model == "sentence_transformer":
            encoder = SentenceTransformerEncoder(model_name=DEFAULT_ST_MODEL)
            model_name = DEFAULT_ST_MODEL.replace('/', '_')
        elif model == "bert":
            encoder = BertEncoder(model_name=DEFAULT_BERT_MODEL)
            model_name = DEFAULT_BERT_MODEL.replace('/', '_')
        else:
            raise ValueError(f"Unsupported model {model}")

        # Encode all docs once (up to max_size)
        vectors = encode_documents(encoder, docs, model_name, batch_size=DEFAULT_BATCH_SIZE)

        # Setup Weaviate (local or cloud)
        try:
            w_client = weav_client_module.get_client()
            weav_client_module.create_schema(w_client)
            print("Weaviate client ready")
        except Exception as e:
            print("Weaviate not available or schema creation failed:", e)
            w_client = None

        # Setup Pinecone
        pine_index = None
        try:
            pinecone_client_module.init_pinecone()
            idx_name = f"exp-{model_name}-{int(time.time())}"
            pine_index = pinecone_client_module.create_index(idx_name, dimension=encoder.dim, metric="cosine")
        except Exception as e:
            print("Pinecone not available or failed to create index:", e)
            pine_index = None

        for size in sizes:
            docs_subset = docs[:size]
            vecs_subset = vectors[:size]

            # Ingest to Weaviate
            if w_client:
                df_w, summary_w = run_weaviate_ingest(w_client, weav_client_module, docs_subset, vecs_subset, batch_size=256)
                summary_w.update({"model": model, "model_name": model_name, "size": size})
                results.append({"type": "ingest", **summary_w})
            # Ingest to Pinecone
            if pine_index:
                df_p, summary_p = run_pinecone_ingest(pine_index, pinecone_client_module, docs_subset, vecs_subset, batch_size=256)
                summary_p.update({"model": model, "model_name": model_name, "size": size})
                results.append({"type": "ingest", **summary_p})

        # Query / relevance evaluation
        # Pure vector search
        if w_client:
            retrievals_w, stats_w = run_search_weaviate(w_client, weav_client_module, queries_sample, encoder, top_k=10)
            metrics_w = evaluate_all(queries_sample, retrievals_w, qrels, k_values=[1,5,10])
            out = {"db": "weaviate", "model": model, "model_name": model_name, "type": "search_vector", **stats_w, **metrics_w}
            results.append(out)

        if pine_index:
            retrievals_p, stats_p = run_search_pinecone(pine_index, queries_sample, encoder, top_k=10)
            metrics_p = evaluate_all(queries_sample, retrievals_p, qrels, k_values=[1,5,10])
            out = {"db": "pinecone", "model": model, "model_name": model_name, "type": "search_vector", **stats_p, **metrics_p}
            results.append(out)

        # Hybrid search example (filter by category="sports")
        # For demonstration, we apply the same filter for all queries (real experiments should vary filters)
        def weav_filter_fn(q):
            return [{"path": ["category"], "operator": "Equal", "valueText": "sports"}]

        pinecone_filter = {"category": {"$eq": "sports"}}

        if w_client:
            retrievals_w_h, stats_w_h = run_search_weaviate(w_client, weav_client_module, queries_sample, encoder, top_k=10, metadata_filter_fn=weav_filter_fn)
            metrics_w_h = evaluate_all(queries_sample, retrievals_w_h, qrels, k_values=[1,5,10])
            out = {"db": "weaviate", "model": model, "model_name": model_name, "type": "search_hybrid", **stats_w_h, **metrics_w_h}
            results.append(out)

        if pine_index:
            retrievals_p_h, stats_p_h = run_search_pinecone(pine_index, queries_sample, encoder, top_k=10, metadata_filter=pinecone_filter)
            metrics_p_h = evaluate_all(queries_sample, retrievals_p_h, qrels, k_values=[1,5,10])
            out = {"db": "pinecone", "model": model, "model_name": model_name, "type": "search_hybrid", **stats_p_h, **metrics_p_h}
            results.append(out)

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = RESULTS_DIR / f"results_{int(time.time())}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved experiment summary to {results_csv}")
    return results_df


if __name__ == "__main__":
    run_all_experiments()
