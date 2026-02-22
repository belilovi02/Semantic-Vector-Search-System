"""Rerun missing local H1 configs with adaptive sample_queries for large datasets.

Usage: .\.venv\Scripts\python.exe -m experiments.rerun_missing_local_h1_large
"""
import json
import time
from pathlib import Path

RES = Path(__file__).resolve().parents[0] / 'results'
BACKUP = RES / 'backup_before_rerun'

if not BACKUP.exists():
    print('No backup folder found:', BACKUP)
    raise SystemExit(1)

files = sorted(BACKUP.glob('auto_test_h1_local_H1_ingest_*.json'))
configs = []
seen = set()
for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        cfg = r.get('config', {})
        # normalize target and model
        cfg['target_db'] = 'local'
        cfg['model_name'] = 'dummy'
        n = cfg.get('n_docs', 0)
        # adaptive sample_queries
        if n >= 500_000:
            cfg['sample_queries'] = 5
        elif n >= 100_000:
            cfg['sample_queries'] = 10
        elif n >= 50_000:
            cfg['sample_queries'] = 15
        else:
            cfg['sample_queries'] = 30

        key = (cfg.get('hypothesis'), cfg.get('n_docs'), cfg.get('batch_size'), cfg.get('target_db'), cfg.get('model_name'))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    except Exception as e:
        print('Skipping file', f, 'parse error', e)

print('Will re-run', len(configs), 'configs with adaptive sample_queries sequentially.')
from experiments.auto_run_tests import run_configs_and_collect

results = []
for i, cfg in enumerate(configs, start=1):
    print(f"\n== Running {i}/{len(configs)}: n={cfg['n_docs']} batch={cfg.get('batch_size')} sample_queries={cfg.get('sample_queries')}")
    try:
        df = run_configs_and_collect([cfg], out_prefix='auto_test_h1_local')
        n = cfg.get('n_docs')
        cand = sorted(RES.glob(f"auto_test_h1_local_{cfg['hypothesis']}_{n}_*.json"))
        if cand:
            fpath = cand[-1]
            print('Wrote:', fpath)
            r = json.load(open(fpath, 'r', encoding='utf-8'))
            q = r.get('metrics', {}).get('query_latency')
            print('Query latency:', q)
            results.append({'file': str(fpath), 'n_docs': n, 'sample_queries': cfg.get('sample_queries'), 'query_latency': q})
        else:
            print('No output file found for config:', cfg)
    except Exception as e:
        print('Config failed:', cfg, 'error:', e)
        continue

# write a small summary JSON
summary = RES / f"auto_test_h1_local_query_latency_summary_large_{int(time.time())}.json"
with open(summary, 'w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=2)
print('\nWrote summary to', summary)
print('\nDone.')
