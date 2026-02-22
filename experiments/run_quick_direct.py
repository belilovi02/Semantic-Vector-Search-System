"""Direct invocation of run_configs_and_collect with explicit writeouts for robust debugging."""
import json
from pathlib import Path
from experiments.auto_run_tests import run_configs_and_collect, RESULTS_DIR

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = RESULTS_DIR / 'local_quick_summary.csv'
OUT_JSON = RESULTS_DIR / 'local_quick_summary.json'

configs = [
    {"hypothesis":"H1_ingest","n_docs":5,"batch_size":2,"model_name":"dummy","dim":128,"target_db":"pinecone","sample_queries":5},
    {"hypothesis":"H1_ingest","n_docs":5,"batch_size":2,"model_name":"dummy","dim":128,"target_db":"weaviate","sample_queries":5},
]

print('Starting direct run, configs:', configs)
try:
    df = run_configs_and_collect(configs, out_prefix='auto_test_quick_h1_direct')
    print('Run returned dataframe with shape:', getattr(df, 'shape', 'no df'))
    try:
        df.to_csv(OUT_CSV, index=False)
        print('Wrote CSV to', OUT_CSV)
    except Exception as e:
        print('Failed to write CSV:', e)
    # also write JSON for inspection
    recs = json.loads(df.to_json(orient='records')) if hasattr(df, 'to_json') else []
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(recs, fh, indent=2)
    print('Wrote JSON to', OUT_JSON)
    with open(RESULTS_DIR / 'local_quick_ok.txt', 'w', encoding='utf-8') as fh:
        fh.write('ok')
except Exception as e:
    print('Direct run failed with exception:', e)
    import traceback
    traceback.print_exc()
    # write exception file
    with open(RESULTS_DIR / 'local_quick_error.txt', 'w', encoding='utf-8') as fh:
        fh.write(str(e) + '\n')
        traceback.print_exc(file=fh)
    raise
