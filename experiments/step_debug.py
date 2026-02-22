import json
from pathlib import Path
from data.generate_synthetic import generate_queries_and_qrels_json
from experiments.auto_run_tests import encode_to_memmap, offline_search
from embeddings.encoder import DummyEncoder
from evaluation.metrics import evaluate_all

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = Path(__file__).resolve().parents[0] / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use small dataset documents_5.jsonl
docs_path = DATA_DIR / 'documents_5.jsonl'
if not docs_path.exists():
    raise SystemExit('documents_5.jsonl not found')
# regenerate small queries/qrels
generate_queries_and_qrels_json(str(docs_path), str(DATA_DIR / 'queries.jsonl'), str(DATA_DIR / 'qrels.json'), q_count=5)

# load docs and queries
docs = [json.loads(line) for line in docs_path.open('r', encoding='utf-8')]
queries = [json.loads(line) for line in (DATA_DIR / 'queries.jsonl').open('r', encoding='utf-8')]
qrels = json.load(open(DATA_DIR / 'qrels.json','r',encoding='utf-8'))

# encoder
enc = DummyEncoder(max_dim=128)
memmap_path, dim, docs_ids, timings, encode_total = encode_to_memmap(enc, docs, 'dummy', batch_size=2, chunk_size=10)
print('ENCODE OK', memmap_path, dim, len(docs_ids), 'encode_s', encode_total)

# encode queries
q_texts = [q['query'] for q in queries]
qids = [q['id'] for q in queries]
q_embs = enc.encode(q_texts, batch_size=2)

# offline search
retrievals = offline_search(memmap_path, docs_ids, q_embs, qids=qids, top_k=5, chunk_size=10)
print('RETRIEVALS:', retrievals)

metrics = evaluate_all(queries, retrievals, qrels, k_values=[1,3,5])
print('METRICS:', metrics)

record = {
    'config': {'hypothesis':'H1_ingest','n_docs':5,'batch_size':2,'model_name':'dummy','dim':128,'target_db':'pinecone','sample_queries':5},
    'encode_total_s': encode_total,
    'metrics': metrics,
}
out_file = RESULTS_DIR / f"debug_manual_{int(__import__('time').time())}.json"
with open(out_file,'w',encoding='utf-8') as fh:
    json.dump(record, fh, indent=2)
print('Wrote manual result to', out_file)
