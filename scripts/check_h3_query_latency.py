import glob, json
from pathlib import Path
files=glob.glob('experiments/results/*H3*model_effect*.json')
summary={}
for f in files:
    try:
        data=json.load(open(f))
        cfg=data.get('config',{})
        n=cfg.get('n_docs')
        has_q='query_latency' in data.get('metrics',{}) if isinstance(data.get('metrics'), dict) else False
        summary.setdefault(n,[]).append((Path(f).name, has_q))
    except Exception as e:
        summary.setdefault('error',[]).append((f,str(e)))

for n in sorted([k for k in summary.keys() if isinstance(k,int)]):
    any_q = any(has for _,has in summary[n])
    print(f"n_docs={n:7} any_query_latency={any_q}")
    for fn,has in summary[n]:
        print('   ',fn,'query_latency=',has)
