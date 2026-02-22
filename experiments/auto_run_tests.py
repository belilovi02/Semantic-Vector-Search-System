"""Automated test runner for large-scale experiments.

Creates parameter grids per hypothesis, runs encoding and offline retrieval evaluations,
collects timings and system metrics, and writes per-run JSON and a summary CSV.

Usage: run as module from repository root under the project's venv.

NOTE: This runner uses the `DummyEncoder` by default for speed and reproducibility.
"""
import os
import time
import json
import heapq
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import psutil
import sys
from pathlib import Path as _Path

# ensure repository root is on sys.path so local packages can be imported when running as script
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate_synthetic import generate_documents
from data.dataset import DATA_DIR
from embeddings.encoder import DummyEncoder
from evaluation.metrics import evaluate_all

RESULTS_DIR = Path(__file__).resolve().parents[0] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sample_system_stats(interval: float, duration: float) -> List[Dict[str, Any]]:
    samples = []
    p = psutil.Process()
    t0 = time.time()
    while time.time() - t0 < duration:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()._asdict()
        try:
            gpu = None
        except Exception:
            gpu = None
        samples.append({"ts": time.time(), "cpu_percent": cpu, "mem": mem})
        time.sleep(interval)
    return samples


def encode_to_memmap(encoder, docs: List[Dict], model_name: str, batch_size: int = 256, chunk_size: int = 10000):
    # Simple wrapper similar to experiments.run_experiments.encode_documents but returns times
    n = len(docs)
    sample_texts = [d["text"] for d in docs[: min(8, n)]]
    sample_emb = encoder.encode(sample_texts, batch_size=8, show_progress=False)
    dim = sample_emb.shape[1]
    out_file = DATA_DIR / f"vectors_{model_name}_{n}.dat"
    out_path = str(out_file)
    # create memmap in a Windows-friendly way: pre-allocate file then open as memmap
    out_file.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = n * dim * np.dtype('float32').itemsize
    try:
        # pre-allocate by setting file size
        with open(out_path, 'wb') as fh:
            fh.seek(total_bytes - 1)
            fh.write(b'\0')
        mmap = np.memmap(out_path, dtype='float32', mode='r+', shape=(n, dim))
    except Exception as e:
        # log details for debugging before fallback
        print(f"DEBUG: memmap create failed for {out_path} n={n} dim={dim} total_bytes={total_bytes}: {e}")
        # fallback to open_memmap (may still fail on some platforms)
        try:
            mmap = np.lib.format.open_memmap(out_path, mode='w+', dtype='float32', shape=(n, dim))
        except Exception as e2:
            print(f"DEBUG: open_memmap also failed for {out_path}: {e2}")
            raise


    idx = 0
    timings = []
    texts = []
    docs_ids = []
    t_start = time.time()
    for d in docs:
        texts.append(d["text"])
        docs_ids.append(d["id"])
        if len(texts) >= chunk_size:
            t0 = time.time()
            embs = encoder.encode(texts, batch_size=batch_size, show_progress=False)
            t1 = time.time()
            mmap[idx : idx + len(embs), :] = embs.astype("float32")
            timings.append({"start": t0, "end": t1, "count": len(embs)})
            idx += len(embs)
            texts = []
    if texts:
        t0 = time.time()
        embs = encoder.encode(texts, batch_size=batch_size, show_progress=False)
        t1 = time.time()
        mmap[idx : idx + len(embs), :] = embs.astype("float32")
        timings.append({"start": t0, "end": t1, "count": len(embs)})
        idx += len(embs)

    mmap.flush()
    total = time.time() - t_start
    return str(out_file), dim, docs_ids, timings, total


def offline_search(memmap_path: str, docs_ids: List[str], query_embs: np.ndarray, qids: List[str] = None, top_k: int = 10, chunk_size: int = 10000):
    # memmap approach: iterate in chunks and compute dot-product scores
    mmap = np.memmap(memmap_path, dtype="float32", mode="r")
    # infer dim and n
    total_elems = mmap.size
    # try to infer shape using docs_ids length
    n = len(docs_ids)
    dim = int(total_elems // n)
    mmap = mmap.reshape((n, dim))

    retrievals = {}
    for qi, q_emb in enumerate(query_embs):
        # accumulate topk as (score, docid)
        heap = []
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            chunk = mmap[start:end]
            scores = chunk.dot(q_emb)
            # get local topk
            if scores.size == 0:
                continue
            idx = np.argpartition(-scores, min(top_k, scores.size)-1)[: min(top_k, scores.size)]
            for i in idx:
                s = float(scores[i])
                doc_index = start + int(i)
                if len(heap) < top_k:
                    heapq.heappush(heap, (s, docs_ids[doc_index]))
                else:
                    if s > heap[0][0]:
                        heapq.heapreplace(heap, (s, docs_ids[doc_index]))
        topk = [doc for (_s, doc) in sorted(heap, key=lambda x: -x[0])]
        # map retrieval to provided qid if present, else fallback to numeric q1.. style
        if qids and qi < len(qids):
            qid = qids[qi]
        else:
            qid = f"q{qi+1}"
        retrievals[qid] = topk
    return retrievals


def measure_offline_query_latency(memmap_path: str, docs_ids: List[str], query_embs: np.ndarray, sample_size: int = 100, top_k: int = 1, chunk_size: int = 10000) -> List[float]:
    """Measure per-query latency for offline memmap search.

    Loads the memmap once and times the nearest-neighbor loop for a sample of queries. Returns list of per-query latencies in seconds.
    """
    import random
    mmap = np.memmap(memmap_path, dtype="float32", mode="r")
    total_elems = mmap.size
    n = len(docs_ids)
    if n == 0:
        return []
    dim = int(total_elems // n)
    mmap = mmap.reshape((n, dim))

    latencies = []
    q_count = len(query_embs)
    if q_count == 0:
        return latencies
    # choose a sample of queries
    indices = list(range(q_count))
    if q_count > sample_size:
        indices = random.sample(indices, sample_size)

    for qi in indices:
        q_emb = query_embs[qi]
        t0 = time.time()
        # do same inner loop as offline_search but only for this query
        heap = []
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            chunk = mmap[start:end]
            scores = chunk.dot(q_emb)
            if scores.size == 0:
                continue
            idx_local = np.argpartition(-scores, min(top_k, scores.size)-1)[: min(top_k, scores.size)]
            for i in idx_local:
                s = float(scores[i])
                doc_index = start + int(i)
                if len(heap) < top_k:
                    heapq.heappush(heap, (s, doc_index))
                else:
                    if s > heap[0][0]:
                        heapq.heapreplace(heap, (s, doc_index))
        _ = [docs_ids[d] for (_s, d) in sorted(heap, key=lambda x: -x[0])]
        t1 = time.time()
        latencies.append(t1 - t0)
    return latencies


def run_configs_and_collect(configs: List[Dict], out_prefix: str = "auto_test"):
    # minimal run invocation log so we can debug runs outside stdout
    import datetime
    logp = Path(__file__).resolve().parents[0] / 'run_invocation.log'
    try:
        with open(logp, 'a', encoding='utf-8') as lf:
            lf.write(f"{datetime.datetime.now().isoformat()} START run_configs_and_collect out_prefix={out_prefix} n_cfgs={len(configs)}\n")
    except Exception:
        pass

    results = []
    for cfg in configs:
        print("Running config:", cfg)
        try:
            with open(logp, 'a', encoding='utf-8') as lf:
                lf.write(f"{datetime.datetime.now().isoformat()} START config {cfg}\n")
        except Exception:
            pass
        # skip only if an existing result for the same hypothesis, n_docs *and* target_db exists
        existing = list(RESULTS_DIR.glob(f"{out_prefix}_{cfg['hypothesis']}_{cfg['n_docs']}_*.json"))
        found_existing = None
        # Only skip if a prior result matches the key configuration fields exactly
        def same_config(rec_cfg, cfg):
            keys = ['hypothesis', 'n_docs', 'target_db', 'model_name', 'search_mode', 'batch_size']
            for k in keys:
                if rec_cfg.get(k) != cfg.get(k):
                    return False
            return True

        for fpath in existing:
            try:
                rec = json.load(open(fpath, 'r', encoding='utf-8'))
                rec_cfg = rec.get('config', {})
                if same_config(rec_cfg, cfg):
                    found_existing = (fpath, rec)
                    break
            except Exception:
                continue
        if found_existing:
            fpath, rec = found_existing
            print(f"Skipping config, exact results exist: {fpath}")
            try:
                metrics = rec.get('metrics', {})
                results.append({
                    "hypothesis": cfg["hypothesis"],
                    "n_docs": cfg["n_docs"],
                    "encode_total_s": rec.get('encode_total_s'),
                    **metrics,
                    "out_file": str(fpath),
                })
            except Exception:
                pass
            continue
        # else: no existing result for this exact DB+size; run the job
        try:
            n = cfg["n_docs"]
            docs_path = DATA_DIR / f"documents_{n}.jsonl"
            if not docs_path.exists():
                print(f"Docs not found for n={n}, generating...")
                generate_documents(str(docs_path), n)

            docs = [json.loads(line) for line in docs_path.open("r", encoding="utf-8")]
            # For reproducibility and correct qid->doc mapping, always regenerate queries/qrels
            # from the exact documents file for this config (so qrels reference docs present in the index).
            queries_path = DATA_DIR / "queries.jsonl"
            qrels_path = DATA_DIR / "qrels.json"
            from data.generate_synthetic import generate_queries_and_qrels_json
            # create q_count matching sample_queries or default 200
            q_count = max(cfg.get("sample_queries", 100), 100)
            print(f'Generating queries/qrels from {docs_path} (q_count={q_count})')
            generate_queries_and_qrels_json(str(docs_path), str(queries_path), str(qrels_path), q_count=q_count)
            queries = [json.loads(line) for line in queries_path.open("r", encoding="utf-8")] if queries_path.exists() else []
            queries_sample = queries[: cfg.get("sample_queries", 100)]

            # Allow using a real encoder (sentence-transformers) when model_name is set to a non-dummy value
            model_name_cfg = cfg.get("model_name", "dummy")
            if model_name_cfg == 'dummy':
                encoder = DummyEncoder(max_dim=cfg.get("dim", 512))
            else:
                try:
                    from embeddings.real_encoder import RealEncoder
                    encoder = RealEncoder(model_name=model_name_cfg)
                except Exception as e:
                    print(f"Real encoder unavailable ({e}), falling back to DummyEncoder")
                    encoder = DummyEncoder(max_dim=cfg.get("dim", 512))

            # Force hashing encoder at runtime when requested via env var (overrides sklearn if present)
            import os
            if os.environ.get('FORCE_HASHING_ENCODER') == '1' and hasattr(encoder, '_use_sklearn'):
                encoder._use_sklearn = False
                encoder.vectorizer = None

            # Prepare query encodings (small) for later query-latency probes
            q_texts = [q["query"] for q in queries_sample]
            q_embs = encoder.encode(q_texts, batch_size=cfg.get("batch_size", 256), show_progress=False)

            # Default placeholders
            encode_total = None
            timings = []
            sys_stats = []
            memmap_path = None
            dim = cfg.get("dim", 512)
            # ensure metrics exists for H1 branch (avoid UnboundLocalError)
            metrics = {}

            # If H1, perform streaming encode + ingest (avoid memmap file creation on Windows)
            ingest_summary = None
            query_latency = None
            if cfg.get('hypothesis') == 'H1_ingest' and cfg.get('target_db'):
                print(f"H1 streaming: encoding and ingesting in batches to {cfg['target_db']}")
                try:
                    import time as _time
                    from ingestion.ingest import summarize_timings
                    n = len(docs)
                    batch_size = cfg.get('batch_size', 128)

                    # system sampling during a short window of encode (lightweight)
                    # we'll sample for up to 5s or the time it takes to do the first few encodes
                    t_encode_start = _time.time()
                    ingest_timings = []

                    # Setup DB clients/index where needed
                    db_idx = None
                    if cfg.get('target_db') == 'local':
                        # Local-only mode: use simple in-memory mock (SimpleWeaviateMock)
                        from local_db.mock import SimpleWeaviateMock
                        vclient = SimpleWeaviateMock()
                        use_mock_db = True
                    elif cfg.get('target_db') == 'weaviate':
                        # Prefer local helper module at weaviate/client.py to avoid conflicts with external package
                        use_mock_db = False
                        try:
                            import importlib.util as _il
                            local_path = ROOT / 'weaviate' / 'client.py'
                            if local_path.exists():
                                spec = _il.spec_from_file_location('weaviate_local', str(local_path))
                                weav_local = _il.module_from_spec(spec)
                                spec.loader.exec_module(weav_local)
                                vclient = weav_local.get_client()
                                try:
                                    weav_local.create_schema(vclient)
                                except Exception:
                                    pass
                            else:
                                # fallback to installed package
                                from weaviate.client import get_client, create_schema
                                vclient = get_client()
                                try:
                                    create_schema(vclient)
                                except Exception:
                                    pass
                        except Exception as e:
                            print('Weaviate client unavailable, using local mock:', e)
                            from local_db.mock import SimpleWeaviateMock
                            vclient = SimpleWeaviateMock()
                            use_mock_db = True
                    elif cfg.get('target_db') == 'pinecone':
                        import pinecone.client as pine_mod
                        idx_name = f"exp_{out_prefix}_{cfg['n_docs']}_{int(time.time())}"
                        try:
                            pine_mod.init_pinecone()
                        except Exception:
                            pass
                        try:
                            db_idx = pine_mod.create_index(idx_name, dimension=dim)
                        except Exception:
                            try:
                                idx_list = pine_mod.pinecone_ext.list_indexes()
                                db_idx = pine_mod.pinecone_ext.Index(idx_list[0]) if idx_list else None
                            except Exception:
                                db_idx = None

                    # Stream encode + ingest per batch
                    for i in range(0, len(docs), batch_size):
                        batch_docs = docs[i : i + batch_size]
                        texts = [d['text'] for d in batch_docs]

                        t0 = _time.time()
                        embs = encoder.encode(texts, batch_size=len(texts), show_progress=False)
                        t1 = _time.time()
                        timings.append({"start": t0, "end": t1, "count": len(texts)})

                        # ingest this batch and measure ingestion latency
                        try:
                            if cfg.get('target_db') == 'weaviate' or cfg.get('target_db') == 'local':
                                # use mock or real client appropriately; local uses SimpleWeaviateMock
                                if 'use_mock_db' in locals() and use_mock_db:
                                    batch_timings, _ = vclient.batch_insert_documents(batch_docs, embs.tolist(), batch_size=len(batch_docs))
                                    ingest_timings.extend(batch_timings)
                                else:
                                    # call weaviate batch insert directly for this batch
                                    batch_timings, _ = __import__('weaviate.client').batch_insert_documents(vclient, batch_docs, embs.tolist(), batch_size=len(batch_docs))
                                    ingest_timings.extend(batch_timings)
                            elif cfg.get('target_db') == 'pinecone':
                                # attempt to use real pinecone if available, else use mock
                                try:
                                    import pinecone.client as pine_mod
                                    items = []
                                    for d, v in zip(batch_docs, embs):
                                        items.append({"id": d["id"], "vector": v.tolist(), "metadata": {"category": d["category"], "timestamp": d["timestamp"], "source": d["source"]}})
                                    batch_timings, _ = pine_mod.batch_upsert(db_idx, items, batch_size=len(items))
                                    ingest_timings.extend(batch_timings)
                                except Exception:
                                    # fallback to SimplePineconeMock
                                    from local_db.mock import SimplePineconeMock
                                    if db_idx is None or not hasattr(db_idx, 'batch_upsert'):
                                        db_idx = SimplePineconeMock()
                                    items = []
                                    for d, v in zip(batch_docs, embs):
                                        items.append({"id": d["id"], "vector": v.tolist(), "metadata": {"category": d["category"], "timestamp": d["timestamp"], "source": d["source"]}})
                                    batch_timings, _ = db_idx.batch_upsert(db_idx, items, batch_size=len(items))
                                    ingest_timings.extend(batch_timings)
                        except Exception as e:
                            print('Batch ingest failed for batch starting at', i, ':', e)
                            ingest_timings.append((0, 0, 0))

                        # light system sampling for first few seconds
                        if _time.time() - t_encode_start < 5.0:
                            try:
                                sys_stats = sample_system_stats(interval=1.0, duration=min(5.0, max(1.0, _time.time() - t_encode_start)))
                            except Exception:
                                sys_stats = []

                    encode_total = sum([b['end'] - b['start'] for b in timings])

                            # summarize ingest timings
                    df_ing, summary_ing = summarize_timings(ingest_timings)
                    summary_ing.update({"db": cfg.get('target_db'), "batch_size": batch_size})
                    ingest_summary = summary_ing

                    # measure query latency if ingestion succeeded
                    if ingest_summary and 'error' not in ingest_summary:
                        q_latencies = []
                        if cfg.get('target_db') == 'weaviate':
                            # if using mock weaviate, fall back to the mock API
                            if 'use_mock_db' in locals() and use_mock_db:
                                for q_emb in q_embs:
                                    t0 = _time.time()
                                    _ = vclient.query_vector_search(q_emb.tolist(), top_k=1)
                                    t1 = _time.time()
                                    q_latencies.append(t1 - t0)
                            else:
                                for q_emb in q_embs:
                                    t0 = _time.time()
                                    _ = vclient.query.get('Document', ['id']).with_near_vector({'vector': q_emb.tolist()}).with_limit(1).do()
                                    t1 = _time.time()
                                    q_latencies.append(t1 - t0)
                        elif cfg.get('target_db') == 'pinecone' and db_idx is not None:
                            for q_emb in q_embs:
                                t0 = _time.time()
                                _ = db_idx.query(queries=[q_emb.tolist()], top_k=1, include_metadata=False)
                                t1 = _time.time()
                                q_latencies.append(t1 - t0)
                        # Local-only mock DBs expose a simple query_vector_search API
                        elif cfg.get('target_db') == 'local':
                            for q_emb in q_embs:
                                t0 = _time.time()
                                _ = vclient.query_vector_search(q_emb.tolist(), top_k=1)
                                t1 = _time.time()
                                q_latencies.append(t1 - t0)
                        if q_latencies:
                            import numpy as _np
                            q_arr = _np.array(q_latencies)
                            query_latency = {
                                'mean_s': float(q_arr.mean()),
                                'p50_s': float(_np.percentile(q_arr, 50)),
                                'p90_s': float(_np.percentile(q_arr, 90)),
                                'p99_s': float(_np.percentile(q_arr, 99)),
                                'qps': float(len(q_arr) / float(q_arr.sum())) if q_arr.sum() > 0 else None,
                            }

                except Exception as e:
                    print('H1 streaming ingest unexpected error:', e)
                    ingest_summary = {'error': str(e)}

            # If this was an H1 ingest run, attach ingest & query latency to metrics
            if cfg.get('hypothesis') == 'H1_ingest':
                metrics = {'ingest': ingest_summary, 'query_latency': query_latency}

            else:
                # Non-H1: fallback to previous memmap + offline eval workflow
                memmap_path, dim, docs_ids, timings, encode_total = encode_to_memmap(encoder, docs, cfg.get("model_name", "dummy"), batch_size=cfg.get("batch_size", 256), chunk_size=cfg.get("chunk_size", 10000))

                # system sampling during encode (lightweight): sample a few times
                sys_stats = sample_system_stats(interval=1.0, duration=min(5.0, encode_total))

                # offline retrieval (request top-20 so we can evaluate up to k=20)
                qids = [q.get("id", f"q{i+1}") for i, q in enumerate(queries_sample)]
                retrievals = offline_search(memmap_path, docs_ids, q_embs, qids=qids, top_k=20, chunk_size=cfg.get("search_chunk", 5000))

                # measure offline query latency (time spent doing nearest-neighbor over memmap)
                try:
                    sample_q = min(len(q_embs), int(cfg.get('sample_queries', 100)))
                    q_lat_list = measure_offline_query_latency(memmap_path, docs_ids, q_embs, sample_size=sample_q, top_k=1, chunk_size=cfg.get("search_chunk", 5000))
                    if q_lat_list:
                        import numpy as _np
                        q_arr = _np.array(q_lat_list)
                        query_latency = {
                            'mean_s': float(q_arr.mean()),
                            'p50_s': float(_np.percentile(q_arr, 50)),
                            'p90_s': float(_np.percentile(q_arr, 90)),
                            'p99_s': float(_np.percentile(q_arr, 99)),
                            'qps': float(len(q_arr) / float(q_arr.sum())) if q_arr.sum() > 0 else None,
                        }
                    else:
                        query_latency = None
                except Exception as e:
                    print('H3 offline query latency measurement failed:', e)
                    query_latency = None

                # load qrels if available
                qrels_path = DATA_DIR / "qrels.json"
                qrels = json.load(open(qrels_path, "r", encoding="utf-8")) if qrels_path.exists() else {}

                metrics = evaluate_all(queries_sample, retrievals, qrels, k_values=[5,10,20])
                # attach query_latency into H3 metrics for downstream aggregation/plots
                metrics['query_latency'] = query_latency if 'query_latency' in locals() else None

            # Ensure H1 runs always write ingest + query_latency into metrics (even if earlier code path had errors)
            if cfg.get('hypothesis') == 'H1_ingest':
                metrics = {
                    'ingest': ingest_summary if 'ingest_summary' in locals() else None,
                    'query_latency': query_latency if 'query_latency' in locals() else None,
                }

            record = {
                "config": cfg,
                "encode_total_s": encode_total,
                "encode_batches": timings,
                "system_samples": sys_stats,
                "metrics": metrics,
                "memmap_path": memmap_path,
                "dim": dim,
                "n_docs": n,
            }
            ts = int(time.time())
            out_file = RESULTS_DIR / f"{out_prefix}_{cfg['hypothesis']}_{n}_{ts}.json"
            print(f"DEBUG: Attempting to write result JSON to {out_file}")
            try:
                with open(out_file, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2)
                print(f"DEBUG: JSON written to {out_file}")
            except Exception as e:
                print(f"ERROR: Failed to write JSON to {out_file}: {e}")
                # write error sentinel
                import traceback
                err = {"config": cfg, "error": str(e), "traceback": traceback.format_exc()}
                err_file = RESULTS_DIR / f"{out_prefix}_error_{cfg['hypothesis']}_{n}_{int(time.time())}.json"
                try:
                    with open(err_file, 'w', encoding='utf-8') as ef:
                        json.dump(err, ef, indent=2)
                    print(f"DEBUG: Wrote error JSON to {err_file}")
                except Exception as ee:
                    print(f"ERROR: Also failed to write error JSON to {err_file}: {ee}")
                raise
            results.append({
                "hypothesis": cfg["hypothesis"],
                "n_docs": n,
                "encode_total_s": encode_total,
                **metrics,
                "out_file": str(out_file),
            })
        except Exception as e:
            print(f"ERROR: Exception while running config {cfg}: {e}")
            import traceback
            traceback.print_exc()
            # write a config-level error file so we can inspect
            err = {"config": cfg, "error": str(e), "traceback": traceback.format_exc()}
            err_file = RESULTS_DIR / f"{out_prefix}_config_error_{cfg['hypothesis']}_{cfg.get('n_docs','unknown')}_{int(time.time())}.json"
            try:
                with open(err_file, 'w', encoding='utf-8') as ef:
                    json.dump(err, ef, indent=2)
                print(f"DEBUG: Wrote config error JSON to {err_file}")
            except Exception as ee:
                print(f"ERROR: Failed to write config error JSON to {err_file}: {ee}")
            # continue to next config
            continue
    # write CSV summary
    import pandas as pd

    df = pd.DataFrame(results)
    csv_out = RESULTS_DIR / f"{out_prefix}_summary_{int(time.time())}.csv"
    print(f"DEBUG: Attempting to write CSV summary to {csv_out}")
    try:
        df.to_csv(csv_out, index=False)
        print(f"DEBUG: CSV written to {csv_out}")
    except Exception as e:
        print(f"ERROR: Failed to write CSV to {csv_out}: {e}")
        raise
    print("Wrote summary:", csv_out)
    try:
        with open(logp, 'a', encoding='utf-8') as lf:
            lf.write(f"{datetime.datetime.now().isoformat()} FINISH run_configs_and_collect out_prefix={out_prefix} written_csv={csv_out}\n")
    except Exception:
        pass
    return df


def build_configs():
    # Build configs per user's suggested test plan
    configs = []
    # H1: Ingestion throughput — sizes x repeats x DBs
    h = "H1_ingest"
    sizes_h1 = [10_000, 50_000, 100_000, 500_000]
    # Use local-only DBs when LOCAL_ONLY is set in env (use mocks). Default is cloud DBs.
    if os.getenv("LOCAL_ONLY", "0").lower() in ("1", "true", "yes"):
        dbs = ["local"]
    else:
        dbs = ["pinecone", "weaviate"]
    batch_sizes = [100, 500, 1000]
    for n in sizes_h1:
        for repeat in range(3):
            for db in dbs:
                configs.append({
                    "hypothesis": h,
                    "n_docs": n,
                    "batch_size": batch_sizes[repeat % len(batch_sizes)],
                    "model_name": "dummy",
                    "dim": 512,
                    "target_db": db,
                    "sample_queries": 30,
                })

    # H2: Retrieval relevance — sizes x DBs x search modes
    h = "H2_relevance"
    sizes_h2 = [10_000, 100_000, 1_000_000]
    search_modes = ["vector", "hybrid"]
    for n in sizes_h2:
        for db in dbs:
            for mode in search_modes:
                configs.append({
                    "hypothesis": h,
                    "n_docs": n,
                    "batch_size": 256,
                    "model_name": "dummy",
                    "dim": 512,
                    "target_db": db,
                    "search_mode": mode,
                    "sample_queries": 30,
                })

    # H3: Embedding model effect — sizes x DBs x models
    h = "H3_model_effect"
    sizes_h3 = [10_000, 100_000, 1_000_000]
    models = ["bert", "sentence_transformer"]
    for n in sizes_h3:
        for db in dbs:
            for m in models:
                configs.append({
                    "hypothesis": h,
                    "n_docs": n,
                    "batch_size": 256,
                    "model_name": m,
                    "dim": 512,
                    "target_db": db,
                    "sample_queries": 30,
                })

    return configs


if __name__ == "__main__":
    cfgs = build_configs()
    df = run_configs_and_collect(cfgs, out_prefix="auto_test")
    print(df.head())
