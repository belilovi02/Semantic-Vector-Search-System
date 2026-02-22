from experiments.auto_run_tests import run_configs_and_collect
cfg = {
    "hypothesis": "H1_ingest",
    "n_docs": 10000,
    "batch_size": 100,
    "model_name": "dummy",
    "dim": 512,
    "target_db": "pinecone",
    "sample_queries": 200,
}
df = run_configs_and_collect([cfg], out_prefix='auto_test_debug')
print(df.to_string())
print('Done')