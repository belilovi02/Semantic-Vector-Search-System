"""Aggregate experiment results, run statistical tests, create plots, and produce a PDF report.

Saves:
- dist/documentation/plots/*.png
- dist/documentation/EXPERIMENTS_REPORT.pdf
- appends textual summary to dist/documentation/EXPERIMENTS_CONCLUSIONS.txt
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy import stats
    SCIPY = True
except Exception:
    SCIPY = False

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
OUT_DIR = ROOT / "dist" / "documentation"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "EXPERIMENTS_REPORT.pdf"
CONCL = OUT_DIR / "EXPERIMENTS_CONCLUSIONS.txt"


def load_df():
    rows = []
    for p in RESULTS_DIR.glob('auto_test_*.json'):
        try:
            r = json.load(open(p, 'r', encoding='utf-8'))
            cfg = r.get('config', {})
            m = r.get('metrics', {})
            rows.append({
                'file': p.name,
                'hypothesis': cfg.get('hypothesis'),
                'n_docs': cfg.get('n_docs'),
                'target_db': cfg.get('target_db'),
                'model_name': cfg.get('model_name'),
                'search_mode': cfg.get('search_mode'),
                'encode_s': r.get('encode_total_s'),
                'p@1': m.get('p@1'),
                'p@5': m.get('p@5'),
                'r@10': m.get('r@10'),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def stats_compare(a, b, name='metric'):
    # a, b: arrays
    a = np.array([x for x in a if x is not None])
    b = np.array([x for x in b if x is not None])
    if len(a) == 0 or len(b) == 0:
        return {'method': None, 'pvalue': None, 'a_mean': None, 'b_mean': None}
    res = {'a_mean': float(np.mean(a)), 'b_mean': float(np.mean(b))}
    if SCIPY:
        try:
            t = stats.ttest_ind(a, b, equal_var=False)
            res.update({'method': 'ttest_ind', 'pvalue': float(t.pvalue)})
        except Exception:
            try:
                u = stats.mannwhitneyu(a, b, alternative='two-sided')
                res.update({'method': 'mannwhitneyu', 'pvalue': float(u.pvalue)})
            except Exception:
                res.update({'method': None, 'pvalue': None})
    else:
        res.update({'method': None, 'pvalue': None})
    return res


def analyze_and_plot(df):
    figs = []
    pdf = PdfPages(OUT_PDF)
    summary_lines = []

    # H1: encode time by n_docs and DB
    h1 = df[df.hypothesis == 'H1_ingest']
    if not h1.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        for db, g in h1.groupby('target_db'):
            g2 = g.groupby('n_docs')['encode_s'].median().sort_index()
            ax.plot(g2.index, g2.values, marker='o', label=db)
        ax.set_xscale('log')
        ax.set_xlabel('n_docs')
        ax.set_ylabel('median encode_s (s)')
        ax.set_title('H1: encode time vs corpus size by DB')
        ax.legend()
        pdf.savefig(fig); figs.append(fig); plt.close(fig)

        # pairwise stats per n_docs
        summary_lines.append('H1 statistical comparisons:')
        for n, group in h1.groupby('n_docs'):
            a = group[group.target_db=='pinecone']['encode_s'].values
            b = group[group.target_db=='weaviate']['encode_s'].values
            r = stats_compare(a, b)
            summary_lines.append(f' n={n}: pinecone_mean={r["a_mean"]} weaviate_mean={r["b_mean"]} p={r["pvalue"]} method={r["method"]}')

    # H2: precision vs size
    h2 = df[df.hypothesis == 'H2_relevance']
    if not h2.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        for db, g in h2.groupby('target_db'):
            g2 = g.groupby('n_docs')['p@1'].median().sort_index()
            ax.plot(g2.index, g2.values, marker='o', label=db)
        ax.set_xscale('log')
        ax.set_xlabel('n_docs')
        ax.set_ylabel('median p@1')
        ax.set_title('H2: p@1 vs corpus size by DB')
        ax.legend()
        pdf.savefig(fig); figs.append(fig); plt.close(fig)

        summary_lines.append('H2 statistical comparisons:')
        for n, group in h2.groupby('n_docs'):
            a = group[group.target_db=='pinecone']['p@1'].values
            b = group[group.target_db=='weaviate']['p@1'].values
            r = stats_compare(a, b)
            summary_lines.append(f' n={n}: pinecone_mean={r["a_mean"]} weaviate_mean={r["b_mean"]} p={r["pvalue"]} method={r["method"]}')

    # H3: model effect
    h3 = df[df.hypothesis == 'H3_model_effect']
    if not h3.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        for m, g in h3.groupby('model_name'):
            g2 = g.groupby('n_docs')['p@1'].median().sort_index()
            ax.plot(g2.index, g2.values, marker='o', label=m)
        ax.set_xscale('log')
        ax.set_xlabel('n_docs')
        ax.set_ylabel('median p@1')
        ax.set_title('H3: model p@1 vs corpus size')
        ax.legend()
        pdf.savefig(fig); figs.append(fig); plt.close(fig)

        summary_lines.append('H3 statistical comparisons:')
        for n, group in h3.groupby('n_docs'):
            a = group[group.model_name=='bert']['p@1'].values
            b = group[group.model_name=='sentence_transformer']['p@1'].values
            r = stats_compare(a, b)
            summary_lines.append(f' n={n}: bert_mean={r["a_mean"]} s-t_mean={r["b_mean"]} p={r["pvalue"]} method={r["method"]}')

    pdf.close()

    # save individual PNGs too
    for i, fig in enumerate(figs):
        p = PLOTS_DIR / f'figure_{i+1}.png'
        fig.savefig(p)

    # append summary
    with open(CONCL, 'a', encoding='utf-8') as fh:
        fh.write('\n\n=== ANALYSIS SUMMARY ===\n')
        fh.write('\n'.join(summary_lines))

    print('Wrote PDF report to', OUT_PDF)


def main():
    df = load_df()
    if df.empty:
        print('No experiment results found in', RESULTS_DIR)
        return
    analyze_and_plot(df)


if __name__ == '__main__':
    main()
