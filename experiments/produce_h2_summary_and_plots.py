"""Produce H2 summary CSV/JSON and plots (precision/recall/MRR vs n_docs).

Usage: .\.venv\Scripts\python.exe -m experiments.produce_h2_summary_and_plots
Outputs:
 - experiments/results/h2_runs_detailed.csv (per-run rows)
 - experiments/results/h2_summary_by_n_docs.csv
 - experiments/results/h2_summary_by_n_docs.json
 - experiments/results/plot_h2_precision_vs_n_docs.png
 - experiments/results/plot_h2_recall_vs_n_docs.png
 - experiments/results/plot_h2_mrr_vs_n_docs.png
 - experiments/results/h2_runs_failed.csv (runs missing metrics)
"""
import json
from pathlib import Path
import statistics
import csv

RES = Path(__file__).resolve().parents[0] / 'results'
files = sorted(RES.glob('*H2_relevance*.json')) + sorted(RES.glob('*H2_relevance_*.json'))
rows = []
failed = []
for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        cfg = r.get('config', {})
        n = cfg.get('n_docs')
        bs = cfg.get('batch_size')
        metrics = r.get('metrics', {}) if isinstance(r.get('metrics'), dict) else {}
        p1 = metrics.get('p@1')
        p5 = metrics.get('p@5')
        p10 = metrics.get('p@10')
        r1 = metrics.get('r@1')
        r5 = metrics.get('r@5')
        r10 = metrics.get('r@10')
        mrr = metrics.get('mrr') or metrics.get('map') or metrics.get('mrr@10')
        qlat = r.get('metrics', {}).get('query_latency') if isinstance(r.get('metrics'), dict) else None
        # consider run successful if any of precision/recall/mrr is not None
        if any(v is not None for v in [p1,p5,p10,r1,r5,r10,mrr]):
            rows.append({'file': str(f),'n_docs': n,'batch_size': bs,'p@1': p1,'p@5': p5,'p@10': p10,'r@1': r1,'r@5': r5,'r@10': r10,'mrr': mrr,'query_latency_mean_s': qlat.get('mean_s') if qlat else None,'encode_total_s': r.get('encode_total_s')})
        else:
            failed.append({'file': str(f),'n_docs': n,'batch_size': bs})
    except Exception as e:
        failed.append({'file': str(f),'error': str(e)})

# write per-run detailed CSV
out_detailed = RES / 'h2_runs_detailed.csv'
with open(out_detailed, 'w', newline='', encoding='utf-8') as fh:
    cols = ['file','n_docs','batch_size','p@1','p@5','p@10','r@1','r@5','r@10','mrr','query_latency_mean_s','encode_total_s']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# write failed runs CSV
out_failed = RES / 'h2_runs_failed.csv'
with open(out_failed, 'w', newline='', encoding='utf-8') as fh:
    cols = ['file','n_docs','batch_size','error']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in failed:
        w.writerow({'file': r.get('file'), 'n_docs': r.get('n_docs'), 'batch_size': r.get('batch_size'), 'error': r.get('error')})

# aggregate by n_docs
summary = {}
for r in rows:
    n = r['n_docs']
    summary.setdefault(n, {'p@1': [], 'p@5': [], 'p@10': [], 'r@1': [], 'r@5': [], 'r@10': [], 'mrr': [], 'query_latency_mean_s': []})
    for k in ['p@1','p@5','p@10','r@1','r@5','r@10','mrr','query_latency_mean_s']:
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
        'mean_p@1': safe_mean(vals['p@1']),
        'mean_p@5': safe_mean(vals['p@5']),
        'mean_p@10': safe_mean(vals['p@10']),
        'mean_r@1': safe_mean(vals['r@1']),
        'mean_r@5': safe_mean(vals['r@5']),
        'mean_r@10': safe_mean(vals['r@10']),
        'mean_mrr': safe_mean(vals['mrr']),
        'mean_query_latency_s': safe_mean(vals['query_latency_mean_s'])
    }
    summ_rows.append(sr)

# write summary CSV and JSON
out_summary_csv = RES / 'h2_summary_by_n_docs.csv'
out_summary_json = RES / 'h2_summary_by_n_docs.json'
with open(out_summary_csv, 'w', newline='', encoding='utf-8') as fh:
    cols = ['n_docs','mean_p@1','mean_p@5','mean_p@10','mean_r@1','mean_r@5','mean_r@10','mean_mrr','mean_query_latency_s']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in summ_rows:
        w.writerow(r)

with open(out_summary_json, 'w', encoding='utf-8') as fh:
    json.dump(summ_rows, fh, indent=2)

# produce plots
try:
    import matplotlib.pyplot as plt
    xs = [r['n_docs'] for r in summ_rows]
    # precision plot
    plt.figure(figsize=(6,4))
    plt.plot(xs, [r['mean_p@1'] for r in summ_rows], marker='o', label='p@1')
    plt.plot(xs, [r['mean_p@5'] for r in summ_rows], marker='o', label='p@5')
    plt.plot(xs, [r['mean_p@10'] for r in summ_rows], marker='o', label='p@10')
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('precision')
    plt.title('H2: precision@k vs n_docs')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    p1 = RES / 'plot_h2_precision_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # recall plot
    plt.figure(figsize=(6,4))
    plt.plot(xs, [r['mean_r@1'] for r in summ_rows], marker='o', label='r@1')
    plt.plot(xs, [r['mean_r@5'] for r in summ_rows], marker='o', label='r@5')
    plt.plot(xs, [r['mean_r@10'] for r in summ_rows], marker='o', label='r@10')
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('recall')
    plt.title('H2: recall@k vs n_docs')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    p2 = RES / 'plot_h2_recall_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    # mrr plot
    plt.figure(figsize=(6,4))
    plt.plot(xs, [r['mean_mrr'] for r in summ_rows], marker='o', label='mean MRR')
    plt.xscale('log')
    plt.xlabel('n_docs (log)')
    plt.ylabel('MRR')
    plt.title('H2: MRR vs n_docs')
    plt.grid(True, which='both', ls='--', lw=0.5)
    p3 = RES / 'plot_h2_mrr_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()
    print('Wrote plots:', p1, p2, p3)
except Exception as e:
    print('Plotting failed:', e)

print('Wrote detailed CSV:', out_detailed)
print('Wrote failed CSV:', out_failed)
print('Wrote summary CSV:', out_summary_csv)
print('Wrote summary JSON:', out_summary_json)
print('Done.')
