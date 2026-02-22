"""Parse experiment JSON results and produce a conclusions report per hypothesis.

Reads `experiments/results/*.json` and creates `dist/documentation/EXPERIMENTS_CONCLUSIONS.txt`.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[0] / "results"
OUT_PATH = Path(__file__).resolve().parents[1] / "dist" / "documentation" / "EXPERIMENTS_CONCLUSIONS.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def collect():
    data = defaultdict(list)
    for p in RESULTS_DIR.glob("auto_test_*.json"):
        try:
            r = json.load(open(p, "r", encoding="utf-8"))
            cfg = r.get("config", {})
            hyp = cfg.get("hypothesis", "unknown")
            metrics = r.get("metrics", {})
            data[hyp].append({"cfg": cfg, "metrics": metrics, "encode_s": r.get("encode_total_s")})
        except Exception:
            continue
    return data


def summarize(data):
    lines = []
    for hyp, runs in data.items():
        lines.append(f"Hypothesis: {hyp}")
        lines.append("- Runs: %d" % len(runs))
        # gather metrics
        p1 = [r["metrics"].get("p@1", 0.0) for r in runs]
        p5 = [r["metrics"].get("p@5", 0.0) for r in runs]
        r10 = [r["metrics"].get("r@10", 0.0) for r in runs]
        enc = [r.get("encode_s", 0.0) for r in runs]
        lines.append(f"  avg p@1: {np.mean(p1):.4f}, p@5: {np.mean(p5):.4f}, r@10: {np.mean(r10):.4f}")
        lines.append(f"  avg encode time (s): {np.mean(enc):.2f}")
        lines.append("  Per-run details:")
        for r in runs:
            cfg = r["cfg"]
            m = r["metrics"]
            lines.append(f"    n={cfg.get('n_docs')} batch={cfg.get('batch_size')} model={cfg.get('model_name')} -> p@1={m.get('p@1',0):.4f}, p@5={m.get('p@5',0):.4f}, r@10={m.get('r@10',0):.4f}, encode_s={r.get('encode_s')}")
        lines.append("")
    return "\n".join(lines)


if __name__ == '__main__':
    data = collect()
    txt = summarize(data)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print("Wrote conclusions to", OUT_PATH)
