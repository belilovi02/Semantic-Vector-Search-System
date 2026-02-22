"""Scan H1 local results and write a concise summary of runs that have query latency recorded.

Usage: run via module after H1 local run finishes (or as part of a wrapper script).
Writes: experiments/results/auto_test_h1_local_3h_summary_<ts>.json
"""
import json
from pathlib import Path
import time

RESULTS_DIR = Path(__file__).resolve().parents[0] / 'results'
out = []
for p in sorted(RESULTS_DIR.glob('auto_test_h1_local_H1_ingest_*.json')):
    try:
        rec = json.load(open(p, 'r', encoding='utf-8'))
    except Exception:
        continue
    cfg = rec.get('config', {})
    m = rec.get('metrics', {}) or {}
    ingest = m.get('ingest') or {}
    qlat = m.get('query_latency')
    relevant = False
    if qlat and isinstance(qlat, dict):
        # treat zero or null mean as not relevant
        mean_s = qlat.get('mean_s')
        if mean_s is not None and float(mean_s) > 0:
            relevant = True
    if relevant:
        out.append({
            'file': str(p),
            'n_docs': cfg.get('n_docs'),
            'batch_size': cfg.get('batch_size'),
            'ingest_total_items': ingest.get('total_items'),
            'ingest_total_time_s': ingest.get('total_time_s'),
            'ingest_overall_throughput_vps': ingest.get('overall_throughput_vps'),
            'query_mean_s': qlat.get('mean_s'),
            'query_p50_s': qlat.get('p50_s'),
            'query_p90_s': qlat.get('p90_s'),
            'query_p99_s': qlat.get('p99_s'),
            'query_qps': qlat.get('qps'),
        })

ts = int(time.time())
out_file = RESULTS_DIR / f"auto_test_h1_local_3h_summary_{ts}.json"
with open(out_file, 'w', encoding='utf-8') as fh:
    json.dump({'generated_at': ts, 'relevant_runs': out}, fh, indent=2)
print(f"Wrote summary: {out_file} (relevant_runs={len(out)})")

# Also print a short table to stdout
for r in out:
    print(f"n={r['n_docs']:7} batch={r['batch_size']:4} mean_s={r['query_mean_s']:.6f} qps={r['query_qps']}")
