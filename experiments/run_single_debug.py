from pathlib import Path
import os, sys, traceback
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
from data.generate_synthetic import generate_queries_and_qrels_json
from experiments.auto_run_tests import run_configs_and_collect

# regen queries/qrels using documents_10000.jsonl
docs = DATA_DIR / 'documents_10000.jsonl'
if not docs.exists():
    raise SystemExit('documents_10000.jsonl not found')
print('Regenerating queries/qrels from', docs)
generate_queries_and_qrels_json(str(docs), str(DATA_DIR / 'queries.jsonl'), str(DATA_DIR / 'qrels.json'), q_count=50)
print('Regenerated queries/qrels')

cfg = {
    "hypothesis": "H1_ingest",
    "n_docs": 10000,
    "batch_size": 100,
    "model_name": "dummy",
    "dim": 512,
    "target_db": "pinecone",
    "sample_queries": 50,
}
print('Running single cfg:', cfg)
try:
    df = run_configs_and_collect([cfg], out_prefix='auto_test_debug')
    print('Done run; df head:')
    try:
        print(df.head())
    except Exception:
        print('Could not print df')
    with open('experiments/run_single_debug_ok.txt','w',encoding='utf-8') as fh:
        fh.write('ok')
except Exception as e:
    print('Run failed:')
    traceback.print_exc()
    with open('experiments/run_single_debug_error.txt','w',encoding='utf-8') as fh:
        fh.write(str(e) + '\n')
    raise
print('script done')
