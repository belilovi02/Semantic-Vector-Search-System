"""Inspect which of the 24 H1 runs have completed and summary metrics.
Writes: experiments/results/h1_24_status.csv and prints a summary table.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.rerun_h1_pine_weaviate import build_h1_pine_weaviate
RES = ROOT / 'experiments' / 'results'

cfgs = build_h1_pine_weaviate()
# Map of (n_docs, target_db) -> list of result records
records = {}
for f in RES.glob('*.json'):
    try:
        obj = json.load(open(f, 'r', encoding='utf-8'))
    except Exception:
        continue
    # skip non-dict top-level JSONs
    if not isinstance(obj, dict):
        continue
    cfg = obj.get('config', {})
    if cfg.get('hypothesis') != 'H1_ingest':
        continue
    key = (cfg.get('n_docs'), cfg.get('target_db'))
    rec = {
        'file': str(f),
        'p@5': obj.get('metrics', {}).get('p@5'),
        'p@10': obj.get('metrics', {}).get('p@10'),
        'p@20': obj.get('metrics', {}).get('p@20'),
        'map': obj.get('metrics', {}).get('map'),
        'encode_total_s': obj.get('encode_total_s'),
    }
    records.setdefault(key, []).append(rec)

# build status list for expected 24
out = []
for cfg in cfgs:
    k = (cfg['n_docs'], cfg['target_db'])
    found = records.get(k, [])
    status = 'done' if found else 'missing'
    # if multiple runs per (n,db), pick latest by file mtime
    latest = None
    if found:
        latest = max(found, key=lambda r: Path(r['file']).stat().st_mtime)
    out.append({
        'n_docs': cfg['n_docs'],
        'target_db': cfg['target_db'],
        'batch_size': cfg.get('batch_size'),
        'status': status,
        'runs_found': len(found),
        'p@5': latest['p@5'] if latest else None,
        'p@10': latest['p@10'] if latest else None,
        'p@20': latest['p@20'] if latest else None,
        'map': latest['map'] if latest else None,
        'out_file': latest['file'] if latest else None,
    })

# write CSV
import csv
out_csv = RES / 'h1_24_status.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=['n_docs','target_db','batch_size','status','runs_found','p@5','p@10','p@20','map','out_file'])
    writer.writeheader()
    for r in out:
        writer.writerow(r)

# print summary
print('Wrote status CSV to', out_csv)
print()
print('Summary (grouped):')
from collections import Counter
cnt = Counter([r['status'] for r in out])
for k,v in cnt.items():
    print(f'  {k}: {v}')

print() 
print('Detailed list:')
for r in out:
    print(f"n={r['n_docs']:7} db={r['target_db']:8} status={r['status']:7} runs={r['runs_found']:2} p@5={r['p@5']} p@10={r['p@10']} p@20={r['p@20']}")

print()
print('If any are "missing", you can force them by running experiments/rerun_h1_pine_weaviate.py or experiments/do_regen_and_rerun.py')
