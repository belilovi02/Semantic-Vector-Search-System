"""Scale up H2 and H3 to target counts and run remaining configs.

This script builds repeated configs for H2_relevance and H3_model_effect until
each has at least `target_per_hypothesis` runs (defaults to 300), then invokes
`run_configs_and_collect` from `auto_run_tests` to execute them (skipping existing results).
"""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.auto_run_tests import run_configs_and_collect
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def count_existing(prefix):
    return len(list(RESULTS.glob(f"auto_test_{prefix}_*.json")))


def build_repeats(hyp, base_cfg, existing, target):
    need = max(0, target - existing)
    cfgs = []
    for i in range(need):
        c = base_cfg.copy()
        c['hypothesis'] = hyp
        c['repeat_id'] = i
        # tweak batch_size slightly to introduce variation
        c['batch_size'] = c.get('batch_size', 256) + (i % 3) * 16
        cfgs.append(c)
    return cfgs


def main(target_per_hypothesis: int = 300):
    # base configs to repeat
    base_h2 = {
        'n_docs': 10000,
        'batch_size': 256,
        'model_name': 'dummy',
        'dim': 512,
        'target_db': 'pinecone',
        'search_mode': 'vector',
        'sample_queries': 30,
    }
    base_h3 = {
        'n_docs': 10000,
        'batch_size': 256,
        'model_name': 'bert',
        'dim': 512,
        'target_db': 'pinecone',
        'sample_queries': 30,
    }

    existing_h2 = count_existing('H2_relevance')
    existing_h3 = count_existing('H3_model_effect')

    print('Existing H2:', existing_h2, 'Existing H3:', existing_h3)

    cfgs = []
    cfgs += build_repeats('H2_relevance', base_h2, existing_h2, target_per_hypothesis)
    cfgs += build_repeats('H3_model_effect', base_h3, existing_h3, target_per_hypothesis)

    print('Will run additional configs:', len(cfgs))
    if not cfgs:
        print('No additional configs needed.')
        return

    run_configs_and_collect(cfgs, out_prefix='auto_test_scaled')


if __name__ == '__main__':
    main()
