"""Run only the H2 configs for n_docs=1_000_000 locally with lower sample_queries to finish reruns.

Usage:
  py -3 -u -m experiments.run_single_h2_1m_local
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'
cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H2_relevance' and c.get('n_docs') == 1000000]
for c in cfgs:
    c['target_db'] = 'local'
    c['sample_queries'] = 30

print('Running', len(cfgs), 'configs for n_docs=1000000 with sample_queries=30')
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H2_local_rerun')
print(df)
