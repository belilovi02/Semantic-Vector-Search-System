"""Run statistical tests (t-test / Mann-Whitney) comparing Pinecone vs Weaviate.

Appends a table of p-values (with Bonferroni correction) to dist/documentation/EXPERIMENTS_CONCLUSIONS.txt
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
OUT = ROOT / "dist" / "documentation" / "EXPERIMENTS_CONCLUSIONS.txt"

def load_rows():
    rows = []
    for p in RESULTS_DIR.glob('auto_test_*.json'):
        try:
            r = json.load(open(p, 'r', encoding='utf-8'))
            cfg = r.get('config', {})
            m = r.get('metrics', {})
            rows.append({
                'hypothesis': cfg.get('hypothesis'),
                'n_docs': cfg.get('n_docs'),
                'target_db': cfg.get('target_db'),
                'model_name': cfg.get('model_name'),
                'p@5': m.get('p@5'),
                'p@10': m.get('p@10'),
                'p@20': m.get('p@20') if 'p@20' in m else m.get('p@10'),
                'map': m.get('map'),
                'encode_s': r.get('encode_total_s')
            })
        except Exception:
            continue
    return rows


def run_tests(rows):
    try:
        from scipy import stats
        scipy_ok = True
    except Exception:
        scipy_ok = False

    groups = {}
    for r in rows:
        key = (r['hypothesis'], r['n_docs'])
        groups.setdefault(key, []).append(r)

    lines = []
    lines.append('\n=== STATISTICAL TESTS (Pinecone vs Weaviate) ===')
    if not scipy_ok:
        lines.append('scipy not available in venv; statistical tests skipped. Install scipy to run t-tests.')
        return lines

    tests = []
    for (hyp, n), items in sorted(groups.items()):
        pine = [it for it in items if it.get('target_db') == 'pinecone']
        weav = [it for it in items if it.get('target_db') == 'weaviate']
        if not pine or not weav:
            continue
        # collect arrays for p@5, p@10, map, encode_s
        for metric in ['p@5', 'p@10', 'map', 'encode_s']:
            a = np.array([it.get(metric) for it in pine if it.get(metric) is not None])
            b = np.array([it.get(metric) for it in weav if it.get(metric) is not None])
            if len(a) < 2 or len(b) < 2:
                pval = None
                method = None
            else:
                try:
                    t = stats.ttest_ind(a, b, equal_var=False)
                    pval = float(t.pvalue)
                    method = 'ttest_ind'
                except Exception:
                    try:
                        u = stats.mannwhitneyu(a, b, alternative='two-sided')
                        pval = float(u.pvalue)
                        method = 'mannwhitneyu'
                    except Exception:
                        pval = None
                        method = None
            tests.append({'hypothesis': hyp, 'n_docs': n, 'metric': metric, 'pvalue': pval, 'method': method, 'pine_mean': float(np.mean(a)) if len(a) else None, 'weav_mean': float(np.mean(b)) if len(b) else None})

    # Bonferroni correction: multiply p by number of tests
    m = len([t for t in tests if t['pvalue'] is not None])
    for t in tests:
        if t['pvalue'] is not None:
            t['p_bonf'] = min(1.0, t['pvalue'] * m)
        else:
            t['p_bonf'] = None

    # format lines
    for t in tests:
        lines.append(f"{t['hypothesis']} n={t['n_docs']} metric={t['metric']} method={t['method']} p={t['pvalue']:.4g} bonf={t['p_bonf']:.4g} pine_mean={t['pine_mean']:.4f} weav_mean={t['weav_mean']:.4f}")

    return lines


def main():
    rows = load_rows()
    lines = run_tests(rows)
    with open(OUT, 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('Appended statistical test results to', OUT)


if __name__ == '__main__':
    main()
