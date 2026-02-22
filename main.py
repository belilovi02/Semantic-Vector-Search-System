"""Main entrypoint for running experiments.
Usage examples:
    python main.py --action prepare_data --n_docs 100000
    python main.py --action run_all --models sentence_transformer bert
"""
import argparse
from experiments.run_experiments import run_all_experiments
from data.dataset import prepare_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=["prepare_data", "run_all"], required=True)
    p.add_argument("--n_docs", type=int, default=100000)
    p.add_argument("--models", nargs="+", default=["sentence_transformer", "bert"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.action == "prepare_data":
        prepare_dataset(n_docs=args.n_docs)
    elif args.action == "run_all":
        run_all_experiments(models=args.models, sizes=[10000, 50000, 100000])
