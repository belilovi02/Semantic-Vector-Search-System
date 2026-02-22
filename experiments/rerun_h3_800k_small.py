"""Run H3 configs for n_docs=800000 with tiny sample_queries to capture query latency.
"""
import os
from experiments.auto_run_tests import run_configs_and_collect

os.environ['LOCAL_ONLY'] = '1'
os.environ['FORCE_HASHING_ENCODER'] = '1'

cfgs = []
for m in ['bert','sentence_transformer']:
    cfgs.append({'hypothesis':'H3_model_effect','n_docs':800000,'batch_size':256,'model_name':m,'dim':512,'target_db':'local','sample_queries':30})

print('Running 800k runs (2 models) with sample_queries=30')
df = run_configs_and_collect(cfgs, out_prefix='auto_test_H3_local_rerun_800k')
print(df)
