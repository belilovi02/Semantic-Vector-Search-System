"""Complete missing targeted H3 reruns: only run missing (size, model) combos.

Usage:
  $env:FORCE_HASHING_ENCODER='1'; py -3 -u -m experiments.complete_h3_targeted_reruns

This checks results dir for files prefixed with 'auto_test_H3_local_rerun_specific_H3_model_effect_{n}_' and
runs only the missing combinations. Uses sample_queries=500 for n < 300k, else 300 for speed.
"""
import os
from pathlib import Path
from experiments.auto_run_tests import run_configs_and_collect

RES = Path(__file__).resolve().parents[0] / 'results'
TARGET_N_DOCS = [50000, 200000, 300000, 400000, 500000, 600000, 800000]
models = ['bert', 'sentence_transformer']

existing = set()
for p in RES.glob('auto_test_H3_local_rerun_specific_H3_model_effect_*_*.json'):
    parts = p.name.split('_')
    # pattern: auto_test_H3_local_rerun_specific_H3_model_effect_{n}_{ts}.json
    try:
        n = int(parts[-2])
        existing.add((n,))
    except Exception:
        continue

# construct configs for missing (n, model) combos
cfgs = []
for n in TARGET_N_DOCS:
    for m in models:
        # detect if any existing for n; if yes, skip this model-n combo only if there are 2 existing runs for same n (both models)
        matches = list(RES.glob(f'auto_test_H3_local_rerun_specific_H3_model_effect_{n}_*.json'))
        # We will allow up to 2 existing runs per n; if <2, add missing models
        existing_models = set()
        for mm in matches:
            # attempt to read model name from the JSON 'config' inside
            try:
                import json
                j = json.load(open(mm, 'r', encoding='utf-8'))
                cfg = j.get('config', {})
                existing_models.add(cfg.get('model_name'))
            except Exception:
                pass
        if m not in existing_models:
            cfg = {
                'hypothesis': 'H3_model_effect',
                'n_docs': n,
                'batch_size': 256,
                'model_name': m,
                'dim': 512,
                'target_db': 'local',
                'sample_queries': 500 if n < 300_000 else (200 if n >= 500_000 else 300),
            }
            cfgs.append(cfg)

if not cfgs:
    print('No missing combos found; nothing to run.')
else:
    print(f'Running {len(cfgs)} missing H3 configs locally (FORCE_HASHING_ENCODER should be set).')
    df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_specific')
    print(df)
