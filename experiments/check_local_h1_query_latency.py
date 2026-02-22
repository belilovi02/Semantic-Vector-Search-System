import json
from pathlib import Path

RES = Path(__file__).resolve().parents[0] / 'results'
files = sorted(RES.glob('auto_test_h1_local_H1_ingest_*.json'))

total = len(files)
with_latency = []
without_latency = []

for f in files:
    try:
        r = json.load(open(f, 'r', encoding='utf-8'))
        m = r.get('metrics', {})
        q = m.get('query_latency') if isinstance(m, dict) else None
        if q:
            with_latency.append((str(f), q))
        else:
            without_latency.append(str(f))
    except Exception as e:
        without_latency.append(str(f) + ' (parse error: ' + str(e) + ')')

print('Total local H1 files:', total)
print('With query_latency:', len(with_latency))
print('Without query_latency:', len(without_latency))
print('\nExamples with query_latency (up to 5):')
for p, q in with_latency[:5]:
    print('-', p)
    print('  ', q)

print('\nExamples missing query_latency (up to 5):')
for p in without_latency[:5]:
    print('-', p)

# exit code reflects success if at least one file has query_latency
import sys
sys.exit(0 if len(with_latency) > 0 else 2)
