"""Run missing H1 jobs: weaviate for n in [100000,500000], 3 repeats each (6 jobs)."""
from pathlib import Path
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.auto_run_tests import run_configs_and_collect

ROOT = Path(__file__).resolve().parents[1]
print('Starting missing H1 runs (weaviate 100k and 500k, 3 repeats each)')
configs = []
sizes = [100_000, 500_000]
for n in sizes:
    for repeat in range(3):
        configs.append({
            'hypothesis': 'H1_ingest',
            'n_docs': n,
            'batch_size': 100,
            'model_name': 'dummy',
            'dim': 512,
            'target_db': 'weaviate',
            'sample_queries': 200,
        })
print('Configs to run:', configs)
df = run_configs_and_collect(configs, out_prefix='auto_test_rerun2')
print('Finished running missing H1s; summary:')
print(df)
# write sentinel
with open(ROOT / 'experiments' / 'missing_h1_done.txt', 'w', encoding='utf-8') as fh:
    fh.write('done')
print('Wrote sentinel missing_h1_done.txt')
# update status CSV
from experiments.inspect_h1_results import __name__ as _; import experiments.inspect_h1_results as insp
print('Ran inspect_h1_results to update h1_24_status.csv')
