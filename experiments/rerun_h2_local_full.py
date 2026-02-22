"""Re-run H2 relevance experiments locally with more queries to obtain stable metrics.

Usage:
  py -3 -u -m experiments.rerun_h2_local_full

This forces LOCAL_ONLY=1 and sets target_db='local' and sample_queries=200 for H2 configs.
Writes results with prefix: auto_test_H2_local_rerun
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'

cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H2_relevance']
# modify configs for local rerun: force local mock DB, increase sample queries
for c in cfgs:
    c['target_db'] = 'local'
    c['sample_queries'] = 200

print(f"Running {len(cfgs)} H2 configs locally with sample_queries=200 (LOCAL_ONLY=1)")
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H2_local_rerun')
print(df)
