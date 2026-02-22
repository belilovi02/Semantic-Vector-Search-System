"""Quick H1 verification: regenerate queries/qrels from small docs and run 4 quick H1 configs (n=5)"""
from pathlib import Path
from data.generate_synthetic import generate_queries_and_qrels_json
from experiments.auto_run_tests import run_configs_and_collect

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
qjson = DATA_DIR / 'queries.jsonl'
qrels = DATA_DIR / 'qrels.json'
# regenerate from the small doc set
docs = DATA_DIR / 'documents_5.jsonl'
if not docs.exists():
    raise SystemExit('Missing documents_5.jsonl; generate small docs first')
if qjson.exists():
    qjson.unlink()
if qrels.exists():
    qrels.unlink()
print('Regenerating queries/qrels from', docs)
generate_queries_and_qrels_json(str(docs), str(qjson), str(qrels), q_count=5)
print('Regenerated queries/qrels')

# build 4 quick configs
configs = []
sizes = [5]
dbs = ['pinecone','weaviate']
for n in sizes:
    for repeat in range(2):
        for db in dbs:
            configs.append({
                'hypothesis':'H1_ingest',
                'n_docs': n,
                'batch_size': 2,
                'model_name': 'dummy',
                'dim': 128,
                'target_db': db,
                'sample_queries': 5,
            })

print('Running quick configs:', configs)
df = run_configs_and_collect(configs, out_prefix='auto_test_quick_h1')
print('Run complete; summary:')
print(df)
