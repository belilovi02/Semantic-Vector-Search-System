"""Run the full set of 24 H1 ingest experiments, forcing one result file per config.

This script will remove placeholder `auto_test_fill_*` files, then run each H1 config
individually so every config produces its own JSON result file.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.rerun_h1_more_queries import build_h1_moreq
from experiments.auto_run_tests import run_configs_and_collect
from pathlib import Path as _P

RESULTS = _P(__file__).resolve().parents[0] / "results"

def remove_placeholders():
    removed = 0
    for p in RESULTS.glob('auto_test_fill_H1_ingest_*.json'):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    if removed:
        print('Removed', removed, 'placeholder files')

def main():
    remove_placeholders()
    cfgs = build_h1_moreq()
    for i, cfg in enumerate(cfgs):
        prefix = f"auto_test_h1real_{i}"
        print('Running config', i+1, 'of', len(cfgs), cfg)
        # run single-config to guarantee separate output
        run_configs_and_collect([cfg], out_prefix=prefix)

if __name__ == '__main__':
    main()
