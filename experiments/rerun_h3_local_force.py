"""Force re-run of all H3 model effect experiments locally and overwrite results.

Usage:
  py -3 -u -m experiments.rerun_h3_local_force

This forces LOCAL_ONLY=1, sets target_db='local' and sample_queries=200 for H3 configs
and writes results with prefix: auto_test_H3_local_rerun_force
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'

cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H3_model_effect']
# modify configs for local rerun: force local mock DB, increase sample queries
for c in cfgs:
    c['target_db'] = 'local'
    # cap sample_queries for very large runs if present
    n = c.get('n_docs', 0)
    if n >= 500000:
        c['sample_queries'] = 100
    else:
        c['sample_queries'] = 200

print(f"Force-running {len(cfgs)} H3 configs locally with sample_queries as configured (LOCAL_ONLY=1)")
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_force')
print(df)
