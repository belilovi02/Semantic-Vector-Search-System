"""Add query_latency measurements to existing H3 run JSONs that have memmap files but lack query_latency.

Usage:
  py -3 -u -m experiments.add_missing_query_latency
"""
import json, os
from pathlib import Path
from experiments.auto_run_tests import measure_offline_query_latency
from data.generate_synthetic import generate_queries_and_qrels_json
from embeddings.encoder import DummyEncoder

RES = Path(__file__).resolve().parents[0] / 'results'
files = sorted(RES.glob('*H3_model_effect*.json'))
updated = []
for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        cfg = r.get('config', {})
        metrics = r.get('metrics', {}) if isinstance(r.get('metrics'), dict) else {}
        if 'query_latency' in metrics:
            continue
        memmap_path = r.get('memmap_path')
        if not memmap_path or not Path(memmap_path).exists():
            print(f"Skipping {f.name}: no memmap found")
            continue
        n = cfg.get('n_docs')
        # regenerate queries (use sample_queries or default small number)
        q_count = max(cfg.get('sample_queries', 100), 100)
        docs_path = Path(__file__).resolve().parents[0].parent / 'data' / f"documents_{n}.jsonl"
        queries_path = Path(__file__).resolve().parents[0].parent / 'data' / 'queries.jsonl'
        qrels_path = Path(__file__).resolve().parents[0].parent / 'data' / 'qrels.json'
        print(f"Generating queries for n={n} (q_count={q_count})")
        generate_queries_and_qrels_json(str(docs_path), str(queries_path), str(qrels_path), q_count=q_count)
        queries = [json.loads(line) for line in queries_path.open('r', encoding='utf-8')]
        queries_sample = queries[: cfg.get('sample_queries', 100)]
        q_texts = [q['query'] for q in queries_sample]
        # encode queries using DummyEncoder or RealEncoder if available
        model_name = cfg.get('model_name', 'dummy')
        if model_name == 'dummy':
            encoder = DummyEncoder(max_dim=cfg.get('dim', 512))
        else:
            try:
                from embeddings.real_encoder import RealEncoder
                encoder = RealEncoder(model_name=cfg.get('model_name'))
            except Exception:
                encoder = DummyEncoder(max_dim=cfg.get('dim', 512))
        # respect FORCE_HASHING_ENCODER env var
        if os.environ.get('FORCE_HASHING_ENCODER') == '1' and hasattr(encoder, '_use_sklearn'):
            encoder._use_sklearn = False
            encoder.vectorizer = None
        print(f"Encoding {len(q_texts)} queries for {f.name}")
        q_embs = encoder.encode(q_texts, batch_size=cfg.get('batch_size', 256), show_progress=False)
        # load docs ids
        docs = [json.loads(line) for line in docs_path.open('r', encoding='utf-8')]
        docs_ids = [d.get('id') for d in docs]
        # measure latencies
        print(f"Measuring offline query latency using memmap {memmap_path}")
        latencies = measure_offline_query_latency(memmap_path, docs_ids, q_embs, sample_size=min(len(q_embs), cfg.get('sample_queries', 100)), top_k=1)
        if not latencies:
            print(f"No latencies measured for {f.name}")
            continue
        import statistics
        mean_s = statistics.mean(latencies)
        p50 = sorted(latencies)[int(0.50 * len(latencies))]
        p90 = sorted(latencies)[int(0.90 * len(latencies))]
        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        qps = 1.0 / mean_s if mean_s > 0 else None
        qlat = {'mean_s': mean_s, 'p50_s': p50, 'p90_s': p90, 'p99_s': p99, 'qps': qps}
        # write back into file (update metrics)
        if not isinstance(r.get('metrics'), dict):
            r['metrics'] = {}
        r['metrics']['query_latency'] = qlat
        json.dump(r, open(f, 'w', encoding='utf-8'), indent=2)
        print(f"Updated {f.name} with query_latency: mean_s={mean_s:.4f}")
        updated.append(f.name)
    except Exception as e:
        print(f"Failed to update {f.name}: {e}")

print('Updated files:', updated)
