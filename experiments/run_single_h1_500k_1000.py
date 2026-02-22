"""Run single local H1 config: n_docs=500000, batch_size=1000, sample_queries=5"""
from experiments.auto_run_tests import run_configs_and_collect
cfg = {'hypothesis': 'H1_ingest', 'n_docs': 500000, 'batch_size': 1000, 'model_name': 'dummy', 'dim': 512, 'target_db': 'local', 'sample_queries': 5}
print('Running config:', cfg)
run_configs_and_collect([cfg], out_prefix='auto_test_h1_local')
print('Done.')
