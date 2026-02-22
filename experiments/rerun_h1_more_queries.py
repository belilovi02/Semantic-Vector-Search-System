"""Rerun H1 configs with increased number of queries (sample_queries=200).
Writes results with out_prefix 'auto_test_moreq'.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.auto_run_tests import run_configs_and_collect


def build_h1_moreq():
    configs = []
    sizes_h1 = [10_000, 50_000, 100_000, 500_000]
    dbs = ["pinecone", "weaviate"]
    batch_sizes = [100, 500, 1000]
    for n in sizes_h1:
        for repeat in range(3):
            for db in dbs:
                configs.append({
                    "hypothesis": "H1_ingest",
                    "n_docs": n,
                    "batch_size": batch_sizes[repeat % len(batch_sizes)],
                    "model_name": "dummy",
                    "dim": 512,
                    "target_db": db,
                    "sample_queries": 200,
                })
    return configs


if __name__ == '__main__':
    cfgs = build_h1_moreq()
    run_configs_and_collect(cfgs, out_prefix='auto_test_moreq')
