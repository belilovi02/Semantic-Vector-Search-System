"""Re-run only H3 configs missing query_latency and overwrite results.

Usage:
  py -3 -u -m experiments.rerun_h3_local_missing_latency
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'
os.environ['FORCE_HASHING_ENCODER'] = '1'

# sizes that were found missing query latency
missing = {50000,200000,300000,400000,500000,600000,800000}

cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H3_model_effect' and c.get('n_docs') in missing]
for c in cfgs:
    c['target_db'] = 'local'
    n = c.get('n_docs',0)
    if n >= 500000:
        c['sample_queries'] = 50
    elif n >= 200000:
        c['sample_queries'] = 100
    else:
        c['sample_queries'] = 200

print(f"Re-running {len(cfgs)} missing H3 configs locally with sample_queries configured (LOCAL_ONLY=1, FORCE_HASHING_ENCODER=1)")
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_missing_latency')
print(df)
