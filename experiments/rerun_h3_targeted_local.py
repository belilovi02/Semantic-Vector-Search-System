"""Targeted H3 reruns (local) for sizes that had zero metrics.

Usage:
  py -3 -u -m experiments.rerun_h3_targeted_local

This forces LOCAL_ONLY=1, sets target_db='local', sample_queries=500 and runs only selected n_docs.
Writes results with prefix: auto_test_H3_local_rerun_targeted
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'

TARGET_N_DOCS = {50000, 200000, 300000, 400000, 500000, 600000, 800000}

cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H3_model_effect' and c.get('n_docs') in TARGET_N_DOCS]
# modify configs for local rerun: force local mock DB, increase sample queries
for c in cfgs:
    c['target_db'] = 'local'
    c['sample_queries'] = 500

print(f"Running {len(cfgs)} targeted H3 configs locally with sample_queries=500 (LOCAL_ONLY=1)")
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_targeted')
print(df)
