from pathlib import Path
from experiments.auto_run_tests import RESULTS_DIR
p = RESULTS_DIR / 'write_test_sentinel.txt'
with open(p, 'w', encoding='utf-8') as fh:
    fh.write('ok')
print('Wrote', p)
print('Exists now:', p.exists())
