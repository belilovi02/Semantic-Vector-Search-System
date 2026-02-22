from experiments.auto_run_tests import run_configs_and_collect

cfg = {
    "hypothesis": "H1_ingest",
    "n_docs": 10000,
    "batch_size": 1000,
    "model_name": "dummy",
    "dim": 512,
    "target_db": "local",
    "sample_queries": 30,
}

print('Starting focused local H1 run (10k, sample_queries=30)...')
df = run_configs_and_collect([cfg], out_prefix='auto_test_debug_10k')
print('Done. Summary DataFrame:')
print(df)
