"""Produce H3 comparison CSV/JSON and plots grouped by model and n_docs.

Usage: py -3 -u -m experiments.produce_h3_model_comparison
Outputs:
 - experiments/results/h3_model_runs_detailed.csv (per-run rows incl. model_name)
 - experiments/results/h3_model_summary_by_n_docs_and_model.csv
 - experiments/results/h3_model_summary_by_n_docs_and_model.json
 - experiments/results/plot_h3_precision_p5_by_model_vs_n_docs.png
 - experiments/results/plot_h3_recall_r5_by_model_vs_n_docs.png
 - experiments/results/plot_h3_map_by_model_vs_n_docs.png
 - experiments/results/plot_h3_query_latency_by_model_vs_n_docs.png
"""
import json
from pathlib import Path
import statistics
import csv

RES = Path(__file__).resolve().parents[0] / 'results'
files = sorted(RES.glob('*H3_model_effect*.json')) + sorted(RES.glob('*H3_model_effect_*.json'))
rows = []
failed = []
for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        cfg = r.get('config', {})
        n = cfg.get('n_docs')
        bs = cfg.get('batch_size')
        model = cfg.get('model_name')
        metrics = r.get('metrics', {}) if isinstance(r.get('metrics'), dict) else {}
        p5 = metrics.get('p@5')
        r5 = metrics.get('r@5')
        mapv = metrics.get('map') or metrics.get('mrr')
        qlat = metrics.get('query_latency') if isinstance(metrics, dict) else None
        encode_total = r.get('encode_total_s')
        if any(v is not None for v in [p5, r5, mapv, qlat]):
            rows.append({'file': str(f), 'n_docs': n, 'batch_size': bs, 'model': model, 'p@5': p5, 'r@5': r5, 'map': mapv, 'query_latency_mean_s': qlat.get('mean_s') if qlat else None, 'encode_total_s': encode_total})
        else:
            failed.append({'file': str(f), 'n_docs': n, 'batch_size': bs, 'model': model})
    except Exception as e:
        failed.append({'file': str(f), 'error': str(e)})

# per-run detailed CSV
out_detailed = RES / 'h3_model_runs_detailed.csv'
with open(out_detailed, 'w', newline='', encoding='utf-8') as fh:
    cols = ['file', 'n_docs', 'batch_size', 'model', 'p@5', 'r@5', 'map', 'query_latency_mean_s', 'encode_total_s']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# failed
out_failed = RES / 'h3_model_runs_failed.csv'
with open(out_failed, 'w', newline='', encoding='utf-8') as fh:
    cols = ['file', 'n_docs', 'batch_size', 'model', 'error']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in failed:
        w.writerow({'file': r.get('file'), 'n_docs': r.get('n_docs'), 'batch_size': r.get('batch_size'), 'model': r.get('model'), 'error': r.get('error')})

# aggregate by (n_docs, model)
summary = {}
for r in rows:
    key = (r['n_docs'], r['model'])
    summary.setdefault(key, {'p@5': [], 'r@5': [], 'map': [], 'query_latency_mean_s': []})
    for k in ['p@5', 'r@5', 'map', 'query_latency_mean_s']:
        v = r.get(k)
        if v is not None:
            summary[key][k].append(v)

summ_rows = []
for (n, model), vals in sorted(summary.items()):
    def safe_mean(lst):
        return statistics.mean(lst) if lst else None
    sr = {
        'n_docs': n,
        'model': model,
        'mean_p@5': safe_mean(vals['p@5']),
        'mean_r@5': safe_mean(vals['r@5']),
        'mean_map': safe_mean(vals['map']),
        'mean_query_latency_s': safe_mean(vals['query_latency_mean_s'])
    }
    summ_rows.append(sr)

out_summary_csv = RES / 'h3_model_summary_by_n_docs_and_model.csv'
out_summary_json = RES / 'h3_model_summary_by_n_docs_and_model.json'
with open(out_summary_csv, 'w', newline='', encoding='utf-8') as fh:
    cols = ['n_docs', 'model', 'mean_p@5', 'mean_r@5', 'mean_map', 'mean_query_latency_s']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in summ_rows:
        w.writerow(r)

with open(out_summary_json, 'w', encoding='utf-8') as fh:
    json.dump(summ_rows, fh, indent=2)

# produce comparison plots
try:
    import matplotlib.pyplot as plt
    import numpy as np
    models = sorted({r['model'] for r in summ_rows})
    ns = sorted({r['n_docs'] for r in summ_rows})

    def series_for(metric):
        out = {}
        for m in models:
            vals = []
            for n in ns:
                row = next((x for x in summ_rows if x['n_docs'] == n and x['model'] == m), None)
                vals.append(row.get(metric) if row else None)
            out[m] = vals
        return out

    # precision p@5
    data = series_for('mean_p@5')
    plt.figure(figsize=(6,4))
    for m, vals in data.items():
        plt.plot(ns, [v if v is not None else np.nan for v in vals], marker='o', label=m)
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('mean p@5')
    plt.title('H3: mean p@5 by model vs n_docs')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    p1 = RES / 'plot_h3_precision_p5_by_model_vs_n_docs.png'
    plt.tight_layout(); plt.savefig(p1, dpi=150); plt.close()

    # recall r@5
    data = series_for('mean_r@5')
    plt.figure(figsize=(6,4))
    for m, vals in data.items():
        plt.plot(ns, [v if v is not None else np.nan for v in vals], marker='o', label=m)
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('mean r@5')
    plt.title('H3: mean r@5 by model vs n_docs')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    p2 = RES / 'plot_h3_recall_r5_by_model_vs_n_docs.png'
    plt.tight_layout(); plt.savefig(p2, dpi=150); plt.close()

    # MAP
    data = series_for('mean_map')
    plt.figure(figsize=(6,4))
    for m, vals in data.items():
        plt.plot(ns, [v if v is not None else np.nan for v in vals], marker='o', label=m)
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('mean MAP')
    plt.title('H3: mean MAP by model vs n_docs')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    p3 = RES / 'plot_h3_map_by_model_vs_n_docs.png'
    plt.tight_layout(); plt.savefig(p3, dpi=150); plt.close()

    # latency
    data = series_for('mean_query_latency_s')
    plt.figure(figsize=(6,4))
    for m, vals in data.items():
        plt.plot(ns, [v if v is not None else np.nan for v in vals], marker='o', label=m)
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('query latency (s)')
    plt.title('H3: mean query latency by model vs n_docs')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    p4 = RES / 'plot_h3_query_latency_by_model_vs_n_docs.png'
    plt.tight_layout(); plt.savefig(p4, dpi=150); plt.close()

    print('Wrote plots:', p1, p2, p3, p4)
except Exception as e:
    print('Plotting failed:', e)

print('Wrote detailed CSV:', out_detailed)
print('Wrote failed CSV:', out_failed)
print('Wrote summary CSV:', out_summary_csv)
print('Wrote summary JSON:', out_summary_json)
print('Done.')
