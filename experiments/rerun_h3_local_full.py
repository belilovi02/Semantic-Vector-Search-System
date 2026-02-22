"""Re-run H3 model effect experiments locally with more queries to obtain stable metrics.

Usage:
  py -3 -u -m experiments.rerun_h3_local_full

This forces LOCAL_ONLY=1 and sets target_db='local' and sample_queries=200 for H3 configs.
Writes results with prefix: auto_test_H3_local_rerun
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'

cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H3_model_effect']
# modify configs for local rerun: force local mock DB, increase sample queries
for c in cfgs:
    c['target_db'] = 'local'
    c['sample_queries'] = 200

print(f"Running {len(cfgs)} H3 configs locally with sample_queries=200 (LOCAL_ONLY=1)")
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun')
print(df)
