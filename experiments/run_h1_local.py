"""Run H1 suite forcing local mock DBs (so query latency is recorded reliably).

Usage: set LOCAL_ONLY=1 or the script will set it, then run as module.
"""
import os
from experiments.rerun_h1_pine_weaviate import build_h1_pine_weaviate
from experiments.auto_run_tests import run_configs_and_collect
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'experiments' / 'results'

os.environ['LOCAL_ONLY'] = os.environ.get('LOCAL_ONLY', '1')
cfgs = build_h1_pine_weaviate()
# force local target_db so SimpleWeaviateMock is used and ingestion+query latency succeed
for c in cfgs:
    c['target_db'] = 'local'

print('Running H1 locally with mock DBs (configs:', len(cfgs), ')')
df = run_configs_and_collect(cfgs, out_prefix='auto_test_h1_local')
print('Finished local H1; wrote summary CSV and JSONs under experiments/results')

out_json = RES / 'auto_test_h1_local_summary.json'
with open(out_json, 'w', encoding='utf-8') as fh:
    fh.write(df.to_json(orient='records'))
print('Wrote', out_json)
