"""Utility to delete queries/qrels, regenerate and run H1 pine/weaviate with verbose prints."""
from pathlib import Path
from data.generate_synthetic import generate_queries_and_qrels_json
from experiments.rerun_h1_pine_weaviate import build_h1_pine_weaviate
from experiments.auto_run_tests import run_configs_and_collect

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
qjson = DATA_DIR / 'queries.jsonl'
qrels = DATA_DIR / 'qrels.json'
for p in (qjson, qrels):
    if p.exists():
        p.unlink()
        print('Removed', p)
    else:
        print('Not present', p)

# regenerate using first doc size found
# pick a documents_N.jsonl file to use for generation
docs = list(DATA_DIR.glob('documents_*.jsonl'))
if not docs:
    raise SystemExit('No documents found')
# use smallest docs for speed
docs = sorted(docs, key=lambda x: int(x.stem.split('_')[1]))
docs_path = docs[0]
print('Using', docs_path, 'to regenerate queries and qrels')
# regenerate 200 queries
generate_queries_and_qrels_json(str(docs_path), str(qjson), str(qrels), q_count=200)
print('Regenerated queries and qrels')
# run configs
import sys, os, traceback
print('Python executable:', sys.executable)
print('CWD:', os.getcwd())
cfgs = build_h1_pine_weaviate()
# remove any pre-existing auto_test_rerun2 results to force full rerun
RES = Path(__file__).resolve().parents[0] / 'results'
for f in RES.glob('auto_test_rerun2_*'):
    try:
        f.unlink()
        print('Removed existing result', f)
    except Exception as e:
        print('Could not remove', f, e)

try:
    df = run_configs_and_collect(cfgs, out_prefix='auto_test_rerun2')
    print('Run complete, wrote summary to results folder')
    try:
        print(df.head())
    except Exception:
        print('Could not print df head')
    # also write a sentinel file to indicate success
    with open('experiments/do_regen_success.txt', 'w', encoding='utf-8') as fh:
        fh.write('success')
except Exception as e:
    print('Run failed with exception:')
    traceback.print_exc()
    with open('experiments/do_regen_error.txt', 'w', encoding='utf-8') as fh:
        fh.write(str(e) + '\n')
        traceback.print_exc(file=fh)
    raise

