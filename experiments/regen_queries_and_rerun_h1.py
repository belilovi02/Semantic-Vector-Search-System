"""Delete existing queries/qrels, regenerate during rerun, and run H1 pine/weaviate re-run."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generate_synthetic import generate_queries_and_qrels_json
from experiments.rerun_h1_pine_weaviate import build_h1_pine_weaviate
from experiments.auto_run_tests import run_configs_and_collect

DATA_DIR = ROOT / 'data'
qjson = DATA_DIR / 'queries.jsonl'
qrels = DATA_DIR / 'qrels.json'
if qjson.exists():
    qjson.unlink(); print('Removed', qjson)
if qrels.exists():
    qrels.unlink(); print('Removed', qrels)

# run all configs; auto_run_tests will detect missing queries and generate from docs per config
cfgs = build_h1_pine_weaviate()
run_configs_and_collect(cfgs, out_prefix='auto_test_rerun2')
print('Done rerun')
