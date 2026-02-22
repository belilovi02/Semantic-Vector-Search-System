"""Run H3 model-effect experiments locally using mock DBs only.

Usage: python -m experiments.run_h3_local
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect

# enforce local-only mode
os.environ['LOCAL_ONLY'] = '1'

cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H3_model_effect']
print(f"Running {len(cfgs)} H3 configs locally (LOCAL_ONLY=1)")
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local')
print(df.head())
