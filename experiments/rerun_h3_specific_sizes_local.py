"""Run H3 configs locally for a specific list of n_docs (to replace zero-metric runs).

Usage:
  py -3 -u -m experiments.rerun_h3_specific_sizes_local

For each size this creates configs for models ['bert','sentence_transformer'] and runs them locally
with sample_queries=500. Results saved with prefix 'auto_test_H3_local_rerun_specific'.
"""
import os
from experiments.auto_run_tests import run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'

TARGET_N_DOCS = [50000, 200000, 300000, 400000, 500000, 600000, 800000]
models = ['bert', 'sentence_transformer']

cfgs = []
for n in TARGET_N_DOCS:
    for m in models:
        c = {
            'hypothesis': 'H3_model_effect',
            'n_docs': n,
            'batch_size': 256,
            'model_name': m,
            'dim': 512,
            'target_db': 'local',
            'sample_queries': 500,
        }
        cfgs.append(c)

print(f"Running {len(cfgs)} targeted H3 configs locally for sizes: {TARGET_N_DOCS} (sample_queries=500, LOCAL_ONLY=1)")

df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_specific')
print(df)
