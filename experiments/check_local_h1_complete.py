"""Check which unique local H1 configs have query_latency and list missing ones."""
import json
from pathlib import Path
RES = Path(__file__).resolve().parents[0] / 'results'
files = sorted(RES.glob('auto_test_h1_local_H1_ingest_*.json'))
seen = {}
for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        cfg = r.get('config', {})
        key = (cfg.get('n_docs'), cfg.get('batch_size'))
        q = r.get('metrics', {}).get('query_latency')
        seen.setdefault(key, []).append((str(f), bool(q)))
    except Exception as e:
        seen.setdefault(('error', str(f)), []).append((str(f), False))

# expected H1 matrix: n_docs in [10000,50000,100000,500000] x batch_size in [100,500,1000] x 1 repeat
expected_docs = [10000,50000,100000,500000]
batches = [100,500,1000]
expected = set((d,b) for d in expected_docs for b in batches)

have = set(k for k,v in seen.items() if k in expected and any(q for (_,q) in v))
missing = sorted(list(expected - have))
print('Found unique local H1 configs scanned:', len(seen))
print('Unique expected configs with query_latency:', len(have), 'of', len(expected))
if missing:
    print('\nMissing configs (n_docs, batch_size):')
    for m in missing:
        print('-', m)
else:
    print('\nAll expected H1 local configs have query_latency.')

print('\nDetailed files with query_latency (examples):')
count=0
for k,v in seen.items():
    if k in expected:
        for p,has_q in v:
            if has_q:
                print('-', k, p)
                count+=1
                break
print('\nDone.')
