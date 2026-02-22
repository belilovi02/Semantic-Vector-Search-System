"""Backup existing local H1 results missing query_latency and re-run them to capture latencies.

Usage: run as module from repo root under project's venv:
    .\.venv\Scripts\python.exe -m experiments.rerun_missing_local_h1
"""
import json
from pathlib import Path
import shutil
import time
from collections import OrderedDict

RES = Path(__file__).resolve().parents[0] / 'results'
BACKUP = RES / 'backup_before_rerun'
BACKUP.mkdir(parents=True, exist_ok=True)

files = sorted(RES.glob('auto_test_h1_local_H1_ingest_*.json'))
missing = []
configs = []
seen_cfgs = set()

for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        m = r.get('metrics', {})
        q = None
        if isinstance(m, dict):
            q = m.get('query_latency')
        if not q:
            missing.append(f)
            cfg = r.get('config', {})
            # use tuple key of key config fields to dedupe
            key = (cfg.get('hypothesis'), cfg.get('n_docs'), cfg.get('target_db'), cfg.get('model_name'), cfg.get('batch_size'))
            if key not in seen_cfgs:
                seen_cfgs.add(key)
                configs.append(cfg)
    except Exception:
        missing.append(f)

print('Found', len(missing), 'existing local H1 files missing query_latency.')
if len(missing) == 0:
    print('Nothing to do.')
    raise SystemExit(0)

# Backup files
ts = int(time.time())
for f in missing:
    dest = BACKUP / f.name
    print('Backing up', f, '->', dest)
    shutil.move(str(f), str(dest))

# Now re-run the configs sequentially (continue on failure)
print('\nRe-running', len(configs), 'unique configs to capture query_latency (sequential)...')
from experiments.auto_run_tests import run_configs_and_collect

new_files = []
for idx, cfg in enumerate(configs, start=1):
    try:
        print(f"\n== Running config {idx}/{len(configs)}: n_docs={cfg.get('n_docs')} batch={cfg.get('batch_size')}")
        df = run_configs_and_collect([cfg], out_prefix='auto_test_h1_local')
        # locate the most recent output file for this config
        n = cfg.get('n_docs')
        cand = sorted(RES.glob(f"auto_test_h1_local_{cfg['hypothesis']}_{n}_*.json"))
        if cand:
            new_files.append(cand[-1])
            print('Wrote:', cand[-1])
        else:
            print('No output file found for config:', cfg)
    except KeyboardInterrupt:
        print('Interrupted by user; stopping further runs.')
        break
    except Exception as e:
        print('Config failed:', cfg, 'error:', e)
        import traceback, json
        errf = RES / f"auto_test_h1_local_config_error_{cfg.get('n_docs','unknown')}_{int(time.time())}.json"
        with open(errf, 'w', encoding='utf-8') as ef:
            json.dump({"config": cfg, "error": str(e), "traceback": traceback.format_exc()}, ef, indent=2)
        print('Wrote config error to', errf)
        continue

print('\nRe-run finished; new/updated files:')
for f in new_files:
    print('-', f)

print('\nDone.')
