"""Re-run H2 relevance experiments against real backends (Pinecone, Weaviate).

Usage:
  py -3 -u -m experiments.rerun_h2_remote_full

Requirements:
 - For Pinecone: set PINECONE_API_KEY and PINECONE_ENV in env
 - For Weaviate: set WEAVIATE_URL (and WEAVIATE_API_KEY if needed) in env

This script verifies credentials, adaptively sets sample_queries (higher for larger corpora), and writes results with prefixes:
 - auto_test_H2_remote_pinecone_rerun
 - auto_test_H2_remote_weaviate_rerun

Notes:
 - Running at large scale may be slow and may incur costs on Pinecone.
 - You must confirm you want both DBs run; set DRY_RUN=1 to only check configs and envs.
"""
import os
from experiments.auto_run_tests import build_configs, run_configs_and_collect
from pathlib import Path

DRY = os.environ.get('DRY_RUN') == '1'
RES = Path(__file__).resolve().parents[0] / 'results'

cfgs = [c for c in build_configs() if c.get('hypothesis') == 'H2_relevance']
print(f"Found {len(cfgs)} H2 configs to run.")

# credential checks
missing = []
can_run = {'pinecone': False, 'weaviate': False}
if os.environ.get('PINECONE_API_KEY') and os.environ.get('PINECONE_ENV'):
    can_run['pinecone'] = True
else:
    missing.append('PINECONE_API_KEY/PINECONE_ENV')

if os.environ.get('WEAVIATE_URL'):
    can_run['weaviate'] = True
else:
    missing.append('WEAVIATE_URL')

print('Credential check:')
for k,v in can_run.items():
    print(f" - {k}: {'OK' if v else 'MISSING'}")
if missing:
    print('Missing required envs for some DBs:', ', '.join(set(missing)))

if DRY:
    print('DRY_RUN=1 set; exiting after checks.')
    raise SystemExit(0)

# adaptive sample size: larger n_docs -> more sample queries for stability
def adaptive_sample_queries(n_docs):
    if not n_docs:
        return 200
    if n_docs <= 10000:
        return 200
    if n_docs <= 100000:
        return 400
    if n_docs <= 300000:
        return 600
    return 800

for db in ['pinecone', 'weaviate']:
    if not can_run[db]:
        print(f"Skipping {db}: missing credentials")
        continue
    cfgs_db = []
    for c in cfgs:
        nc = c.copy()
        nc['target_db'] = db
        nc['sample_queries'] = adaptive_sample_queries(nc.get('n_docs'))
        cfgs_db.append(nc)
    out_prefix = f'auto_test_H2_remote_{db}_rerun'
    print(f"Running {len(cfgs_db)} H2 configs against {db} with out_prefix={out_prefix} (this may take time)")
    try:
        df = run_configs_and_collect(cfgs_db, out_prefix=out_prefix)
        print('Done for', db)
    except Exception as e:
        print('Error running configs for', db, e)

print('All done. Check', RES, 'for new JSON/CSV outputs.')
print('If any runs failed, inspect the *_rerun JSON files and logs for error details.')
