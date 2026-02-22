from experiments.auto_run_tests import run_configs_and_collect

cfg = {
    "hypothesis": "H1_ingest",
    "n_docs": 1000,
    "batch_size": 50,
    "model_name": "bert-base-uncased",
    "dim": 768,
    "target_db": "local",
    "sample_queries": 10,
}

print('Starting debug local H1 run with HF BERT encoder...')
df = run_configs_and_collect([cfg], out_prefix='auto_test_debug_hfbert')
print('Done. Summary DataFrame:')
print(df)
