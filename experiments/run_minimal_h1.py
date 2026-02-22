from experiments.auto_run_tests import run_configs_and_collect
cfg = {
    "hypothesis": "H1_ingest",
    "n_docs": 5,
    "batch_size": 2,
    "model_name": "dummy",
    "dim": 128,
    "target_db": "pinecone",
    "sample_queries": 5,
}
print('Starting minimal run...')
df = run_configs_and_collect([cfg], out_prefix='auto_test_minimal')
print('Returned df:')
print(df)
