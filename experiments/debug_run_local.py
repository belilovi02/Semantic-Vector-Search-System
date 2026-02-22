from experiments.auto_run_tests import run_configs_and_collect

cfg = {
    "hypothesis": "H1_ingest",
    "n_docs": 1000,
    "batch_size": 100,
    "model_name": "dummy",
    "dim": 512,
    "target_db": "local",
    "sample_queries": 5,
}

print('Starting debug local H1 run...')
df = run_configs_and_collect([cfg], out_prefix='auto_test_debug')
print('Done. Summary DataFrame:')
print(df)
