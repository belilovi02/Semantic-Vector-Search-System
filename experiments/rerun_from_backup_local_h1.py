"""Re-run H1 configs found in the backup folder to regenerate results with query_latency.

Usage:
    .\.venv\Scripts\python.exe -m experiments.rerun_from_backup_local_h1
"""
import json
from pathlib import Path
import time

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
        key = (cfg.get('hypothesis'), cfg.get('n_docs'), cfg.get('batch_size'), cfg.get('target_db'), cfg.get('model_name'))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    except Exception as e:
        print('Skipping file', f, 'parse error', e)

print('Found', len(configs), 'unique configs to re-run from backup.')
from experiments.auto_run_tests import run_configs_and_collect
new_files = []
for i, cfg in enumerate(configs, start=1):
    print(f"\n== Re-running {i}/{len(configs)}: n_docs={cfg.get('n_docs')} batch={cfg.get('batch_size')}")
    try:
        df = run_configs_and_collect([cfg], out_prefix='auto_test_h1_local')
        n = cfg.get('n_docs')
        cand = sorted(RES.glob(f"auto_test_h1_local_{cfg['hypothesis']}_{n}_*.json"))
        if cand:
            print('Wrote:', cand[-1])
            new_files.append(cand[-1])
        else:
            print('No output file found after run for config:', cfg)
    except Exception as e:
        print('Error running config', cfg, '->', e)
        continue

print('\nAll done. New files written:')
for f in new_files:
    print('-', f)
