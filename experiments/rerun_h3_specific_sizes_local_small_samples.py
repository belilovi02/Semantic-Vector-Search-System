"""Run H3 configs locally for specific sizes with smaller sample_queries to capture latency.

Usage:
  py -3 -u -m experiments.rerun_h3_specific_sizes_local_small_samples
"""
import os
from experiments.auto_run_tests import run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'
os.environ['FORCE_HASHING_ENCODER'] = '1'

TARGET_N_DOCS = [50000, 200000, 300000, 400000, 500000, 600000, 800000]
models = ['bert', 'sentence_transformer']

cfgs = []
for n in TARGET_N_DOCS:
    for m in models:
        sample = 200 if n < 100000 else (100 if n < 500000 else 50)
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

print(f"Running {len(cfgs)} targeted H3 configs locally with smaller samples: {TARGET_N_DOCS} (LOCAL_ONLY=1, FORCE_HASHING_ENCODER=1)")

df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_specific_small_samples')
print(df)
