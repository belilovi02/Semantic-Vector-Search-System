"""Run final H1 suite for Pinecone+Weaviate (24 configs) and produce final status CSV and a summary JSON."""
from pathlib import Path
from experiments.rerun_h1_pine_weaviate import build_h1_pine_weaviate
from experiments.auto_run_tests import run_configs_and_collect
import json

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'experiments' / 'results'

cfgs = build_h1_pine_weaviate()
print('Running final H1 suite (24 configs)')
df = run_configs_and_collect(cfgs, out_prefix='auto_test_h1_final')
print('Final H1 run finished; writing summary JSON')
out_json = RES / 'auto_test_h1_final_summary.json'
with open(out_json, 'w', encoding='utf-8') as fh:
    fh.write(df.to_json(orient='records'))
print('Wrote', out_json)
# refresh status CSV
from experiments.inspect_h1_results import __name__ as _; import experiments.inspect_h1_results as insp
print('Updated status CSV')
