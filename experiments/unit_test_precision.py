from embeddings.encoder import DummyEncoder
from experiments.auto_run_tests import encode_to_memmap, offline_search
from evaluation.metrics import evaluate_all
import json

# small synthetic docs
docs = [{'id': str(i), 'text': f'some unique content about topic {i} and keyword_{i}'} for i in range(1, 6)]
# create queries that are snippets of doc 3 and doc 5
queries = [{'id':'q1', 'query':'keyword_3'}, {'id':'q2', 'query':'keyword_5'}]
qrels = {'q1':['3'], 'q2':['5']}
encoder = DummyEncoder(max_dim=128)
memmap_path, dim, docs_ids, timings, encode_total = encode_to_memmap(encoder, docs, model_name='dummy_ut', batch_size=2, chunk_size=10)
q_texts = [q['query'] for q in queries]
q_embs = encoder.encode(q_texts, batch_size=2)
qids = [q['id'] for q in queries]
retrievals = offline_search(memmap_path, docs_ids, q_embs, qids=qids, top_k=5)
metrics = evaluate_all(queries, retrievals, qrels, k_values=[1,3,5])
import json
with open('experiments/unit_test_precision_result.json','w',encoding='utf-8') as fh:
    json.dump({'docs_ids':docs_ids,'retrievals':retrievals,'metrics':metrics}, fh, indent=2)
print('Wrote results to experiments/unit_test_precision_result.json')
