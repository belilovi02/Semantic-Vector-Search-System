"""Produce H1 summary CSV/JSON and plots (ingest throughput vs n_docs, query latency vs n_docs).

Usage: .\.venv\Scripts\python.exe -m experiments.produce_h1_summary_and_plots
Outputs:
 - experiments/results/h1_runs_detailed.csv (per-run rows)
 - experiments/results/h1_summary_by_n_docs.csv
 - experiments/results/h1_summary_by_n_docs.json
 - experiments/results/plot_h1_ingest_vs_n_docs.png
 - experiments/results/plot_h1_query_latency_vs_n_docs.png
"""
import json
from pathlib import Path
import statistics

RES = Path(__file__).resolve().parents[0] / 'results'
files = sorted(RES.glob('*H1_ingest*.json'))
rows = []
for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        cfg = r.get('config', {})
        n = cfg.get('n_docs')
        bs = cfg.get('batch_size')
        q = r.get('metrics', {}).get('query_latency')
        ingest = r.get('metrics', {}).get('ingest') if isinstance(r.get('metrics'), dict) else r.get('ingest')
        ingest_vps = None
        if ingest and isinstance(ingest, dict):
            ingest_vps = ingest.get('overall_throughput_vps')
        row = {
            'file': str(f),
            'n_docs': n,
            'batch_size': bs,
            'mean_s': q.get('mean_s') if q else None,
            'p50_s': q.get('p50_s') if q else None,
            'p90_s': q.get('p90_s') if q else None,
            'p99_s': q.get('p99_s') if q else None,
            'qps': q.get('qps') if q else None,
            'ingest_vps': ingest_vps
        }
        rows.append(row)
    except Exception as e:
        print('Skipping', f, 'error', e)

# write per-run detailed CSV
import csv
out_detailed = RES / 'h1_runs_detailed.csv'
with open(out_detailed, 'w', newline='', encoding='utf-8') as fh:
    cols = ['file','n_docs','batch_size','mean_s','p50_s','p90_s','p99_s','qps','ingest_vps']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# aggregate by n_docs
summary = {}
for r in rows:
    n = r['n_docs']
    if n is None:
        continue
    summary.setdefault(n, {'mean_s': [], 'p50_s': [], 'p90_s': [], 'p99_s': [], 'qps': [], 'ingest_vps': []})
    for k in ['mean_s','p50_s','p90_s','p99_s','qps','ingest_vps']:
        v = r.get(k)
        if v is not None:
            summary[n][k].append(v)

summ_rows = []
for n, vals in sorted(summary.items()):
    def safe_mean(lst):
        return statistics.mean(lst) if lst else None
    def safe_median(lst):
        return statistics.median(lst) if lst else None
    sr = {
        'n_docs': n,
        'mean_mean_s': safe_mean(vals['mean_s']),
        'median_p50_s': safe_median(vals['p50_s']),
        'median_p90_s': safe_median(vals['p90_s']),
        'median_p99_s': safe_median(vals['p99_s']),
        'mean_qps': safe_mean(vals['qps']),
        'mean_ingest_vps': safe_mean(vals['ingest_vps'])
    }
    summ_rows.append(sr)

# write summary CSV and JSON
out_summary_csv = RES / 'h1_summary_by_n_docs.csv'
out_summary_json = RES / 'h1_summary_by_n_docs.json'
with open(out_summary_csv, 'w', newline='', encoding='utf-8') as fh:
    cols = ['n_docs','mean_mean_s','median_p50_s','median_p90_s','median_p99_s','mean_qps','mean_ingest_vps']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in summ_rows:
        w.writerow(r)

with open(out_summary_json, 'w', encoding='utf-8') as fh:
    json.dump(summ_rows, fh, indent=2)

# produce plots
try:
    import matplotlib
    import matplotlib.pyplot as plt
    xs = [r['n_docs'] for r in summ_rows]
    ingest_ys = [r['mean_ingest_vps'] for r in summ_rows]
    mean_latency = [r['mean_mean_s'] for r in summ_rows]
    p50 = [r['median_p50_s'] for r in summ_rows]
    p90 = [r['median_p90_s'] for r in summ_rows]
    p99 = [r['median_p99_s'] for r in summ_rows]

    # ingest throughput vs n_docs
    plt.figure(figsize=(6,4))
    plt.plot(xs, ingest_ys, marker='o')
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('ingest throughput (vectors/sec)')
    plt.title('H1: ingest throughput vs n_docs')
    plt.grid(True, which='both', ls='--', lw=0.5)
    p1 = RES / 'plot_h1_ingest_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # query latency vs n_docs
    plt.figure(figsize=(6,4))
    plt.plot(xs, mean_latency, marker='o', label='mean')
    plt.plot(xs, p50, marker='x', linestyle='--', label='p50')
    plt.plot(xs, p90, marker='^', linestyle='--', label='p90')
    plt.plot(xs, p99, marker='s', linestyle='--', label='p99')
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('query latency (s)')
    plt.title('H1: query latency vs n_docs')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    p2 = RES / 'plot_h1_query_latency_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()
    print('Wrote plots:', p1, p2)
except Exception as e:
    print('Plotting failed:', e)

print('Wrote detailed CSV:', out_detailed)
print('Wrote summary CSV:', out_summary_csv)
print('Wrote summary JSON:', out_summary_json)
print('Done.')
