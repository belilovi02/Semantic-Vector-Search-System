"""Run large H3 configs that still lack query_latency with tiny sample sizes.

Usage:
  py -3 -u -m experiments.rerun_h3_large_remaining
"""
import os
from experiments.auto_run_tests import run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'
os.environ['FORCE_HASHING_ENCODER'] = '1'

TARGET_N_DOCS = [400000, 500000, 600000, 800000]
models = ['bert', 'sentence_transformer']

cfgs = []
for n in TARGET_N_DOCS:
    for m in models:
        sample = 50 if n <= 500000 else 30
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

print(f"Running {len(cfgs)} large H3 configs locally with tiny samples (LOCAL_ONLY=1, FORCE_HASHING_ENCODER=1)")

df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_large_remaining')
print(df)
