from pathlib import Path
import json
from random import shuffle

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
docs_path = DATA_DIR / 'documents_10000.jsonl'
queries_path = DATA_DIR / 'queries.jsonl'
qrels_path = DATA_DIR / 'qrels.json'

pairs = []
with docs_path.open('r', encoding='utf-8') as fh:
    for line in fh:
        try:
            obj = json.loads(line)
            txt = obj.get('text') or obj.get('title') or ''
            snippet = txt[:200]
            pairs.append((str(obj['id']), snippet))
        except Exception:
            continue
if not pairs:
    raise SystemExit('No docs')
shuffle(pairs)
q_count = min(200, len(pairs))
queries = []
qrels = {}
for i in range(q_count):
    docid, snippet = pairs[i]
    qid = f'q{i+1}'
    queries.append({'id': qid, 'query': snippet})
    qrels[qid] = [docid]
# write
with queries_path.open('w', encoding='utf-8') as fh:
    for q in queries:
        fh.write(json.dumps(q, ensure_ascii=False) + '\n')
with qrels_path.open('w', encoding='utf-8') as fh:
    json.dump(qrels, fh, ensure_ascii=False, indent=2)
print('Wrote queries and qrels to', queries_path, qrels_path)
