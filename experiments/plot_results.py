"""Aggregate experiment JSONs and generate plots + append brief analysis to conclusions file.

Outputs:
- dist/documentation/plots/*.png
- appends summary to dist/documentation/EXPERIMENTS_CONCLUSIONS.txt
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
OUT_DIR = ROOT / "dist" / "documentation"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    rows = []
    for p in RESULTS_DIR.glob("*.json"):
        try:
            r = json.load(open(p, "r", encoding="utf-8"))
            cfg = r.get("config", {})
            metrics = r.get("metrics", {})
            rows.append({
                "file": str(p.name),
                "hypothesis": cfg.get("hypothesis", ""),
                "n_docs": cfg.get("n_docs"),
                "model_name": cfg.get("model_name"),
                "encode_s": r.get("encode_total_s", None),
                "p@1": metrics.get("p@1", None),
                "p@5": metrics.get("p@5", None),
                "r@10": metrics.get("r@10", None),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_h1(df: pd.DataFrame):
    sub = df[df.hypothesis.str.contains('H1')].dropna(subset=['n_docs', 'encode_s'])
    if sub.empty:
        return None
    sub = sub.sort_values('n_docs')
    plt.figure(figsize=(6,4))
    plt.plot(sub['n_docs'], sub['encode_s'], marker='o')
    plt.xscale('log')
    plt.xlabel('n_docs')
    plt.ylabel('encode_total_s')
    plt.title('H1: encode time vs corpus size (proxy for throughput)')
    out = PLOTS_DIR / 'H1_encode_time_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_h2(df: pd.DataFrame):
    sub = df[df.hypothesis.str.contains('H2')].dropna(subset=['n_docs'])
    if sub.empty:
        return None
    sub = sub.sort_values('n_docs')
    plt.figure(figsize=(6,4))
    plt.plot(sub['n_docs'], sub['p@1'].fillna(0), marker='o', label='p@1')
    plt.plot(sub['n_docs'], sub['p@5'].fillna(0), marker='o', label='p@5')
    plt.xscale('log')
    plt.xlabel('n_docs')
    plt.ylabel('precision')
    plt.legend()
    plt.title('H2: Precision vs corpus size')
    out = PLOTS_DIR / 'H2_precision_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_h3(df: pd.DataFrame):
    sub = df[df.hypothesis.str.contains('H3')].dropna(subset=['n_docs'])
    if sub.empty:
        return None
    plt.figure(figsize=(6,4))
    for m, g in sub.groupby('model_name'):
        g = g.sort_values('n_docs')
        plt.plot(g['n_docs'], g['p@1'].fillna(0), marker='o', label=str(m))
    plt.xscale('log')
    plt.xlabel('n_docs')
    plt.ylabel('p@1')
    plt.legend()
    plt.title('H3: Model effect (p@1)')
    out = PLOTS_DIR / 'H3_model_p1_vs_n_docs.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def append_summary(df: pd.DataFrame, out_file: Path):
    lines = []
    for hyp in ['H1_ingest', 'H2_relevance', 'H3_model_effect']:
        sub = df[df.hypothesis.str.contains(hyp)]
        lines.append(f"Hypothesis: {hyp}")
        lines.append(f"  runs: {len(sub)}")
        if not sub.empty:
            lines.append(f"  avg p@1: {sub['p@1'].dropna().mean() if 'p@1' in sub else 0:.4f}")
            lines.append(f"  avg encode_s: {sub['encode_s'].dropna().mean() if 'encode_s' in sub else 0:.2f}")
        lines.append("")
    with open(out_file, 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def main():
    df = load_results()
    if df.empty:
        print('No results found in', RESULTS_DIR)
        return
    p1 = plot_h1(df)
    p2 = plot_h2(df)
    p3 = plot_h3(df)
    print('Plots generated:', p1, p2, p3)
    out_file = OUT_DIR / 'EXPERIMENTS_CONCLUSIONS.txt'
    append_summary(df, out_file)
    print('Appended summary to', out_file)


if __name__ == '__main__':
    main()
