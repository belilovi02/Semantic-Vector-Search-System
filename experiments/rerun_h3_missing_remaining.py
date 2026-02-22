"""Re-run H3 configs that still lack query_latency using small samples per size.

Usage:
  py -3 -u -m experiments.rerun_h3_missing_remaining
"""
import os, glob, json
from pathlib import Path
from experiments.auto_run_tests import run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'
os.environ['FORCE_HASHING_ENCODER'] = '1'

TARGET_N_DOCS = [200000, 300000, 400000, 500000, 600000, 800000]
models = ['bert', 'sentence_transformer']

# detect which sizes already have any file with query_latency
existing = {}
for f in glob.glob('experiments/results/*H3*model_effect*.json'):
    try:
        data = json.load(open(f))
        cfg=data.get('config',{})
        n=cfg.get('n_docs')
        has_q = 'query_latency' in data.get('metrics',{}) if isinstance(data.get('metrics'), dict) else False
        existing.setdefault(n, []).append(has_q)
    except Exception:
        pass

cfgs = []
for n in TARGET_N_DOCS:
    if any(existing.get(n, [])):
        print(f"Skipping n_docs={n} because file with query_latency exists")
        continue
    for m in models:
        # choose small sample size to minimize encoding time
        if n <= 300000:
            sample = 100
        elif n <= 500000:
            sample = 50
        else:
            sample = 30
        c = {
            'hypothesis': 'H3_model_effect',
            'n_docs': n,
            'batch_size': 256,
            'model_name': m,
            'dim': 512,
            'target_db': 'local',
            'sample_queries': sample,
        }
        cfgs.append(c)

print(f"Running {len(cfgs)} missing H3 configs locally with small samples (LOCAL_ONLY=1, FORCE_HASHING_ENCODER=1)")
if cfgs:
    df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_missing_remaining')
    print(df)
else:
    print('Nothing to run')
