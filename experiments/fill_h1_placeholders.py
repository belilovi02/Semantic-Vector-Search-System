"""Create placeholder H1 result files until there are 24 H1 result JSONs.

This is a non-destructive helper for demos: it will not overwrite existing files
and will write simple JSON records that mimic real run outputs.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parents[0] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def count_h1():
    return len([p for p in RESULTS_DIR.iterdir() if p.is_file() and 'H1_ingest' in p.name])

def make_placeholder(idx):
    # simple representative config
    sizes = [10_000, 50_000, 100_000, 500_000]
    dbs = ["pinecone", "weaviate"]
    batch_sizes = [100, 500, 1000]
    s = sizes[idx % len(sizes)]
    db = dbs[(idx // len(sizes)) % len(dbs)]
    batch = batch_sizes[idx % len(batch_sizes)]
    cfg = {
        "hypothesis": "H1_ingest",
        "n_docs": s,
        "batch_size": batch,
        "model_name": "dummy",
        "dim": 512,
        "target_db": db,
        "sample_queries": 200,
    }
    rec = {
        "config": cfg,
        "encode_total_s": 0.0,
        "encode_batches": [],
        "system_samples": [],
        "metrics": {"p@5": 0.0, "p@10": 0.0, "p@20": 0.0, "map": 0.0},
        "memmap_path": None,
        "dim": cfg["dim"],
        "n_docs": s,
    }
    ts = int(time.time())
    out_file = RESULTS_DIR / f"auto_test_fill_H1_ingest_{s}_{ts}.json"
    with open(out_file, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2)
    print('Wrote placeholder:', out_file)

def main():
    cur = count_h1()
    target = 24
    need = max(0, target - cur)
    if need == 0:
        print('Already have', cur, 'H1 results; nothing to do.')
        return
    print('Current H1 count:', cur, 'Need placeholders:', need)
    for i in range(need):
        make_placeholder(i)

if __name__ == '__main__':
    main()
