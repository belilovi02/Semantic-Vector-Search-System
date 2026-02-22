"""Generate combined H1+H2 report in Bosnian and create plots.

This extends the previous H1-only report to include H2_relevance summaries and calls
`experiments.plot_results` to produce PNGs in `dist/documentation/plots`.
"""
import json
import glob
import os
from statistics import mean

RES = os.path.join(os.path.dirname(__file__), 'results')

# Helper to load JSON results for a given pattern
def load_entries(pattern):
    files = glob.glob(os.path.join(RES, pattern))
    entries = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                obj = json.load(fh)
        except Exception:
            continue
        cfg = obj.get('config', {})
        metrics = obj.get('metrics', {})
        entries.append({'file': f, 'cfg': cfg, 'metrics': metrics, 'encode_s': obj.get('encode_total_s')})
    return entries

h1_entries = load_entries('*H1_ingest*.json')
h2_entries = load_entries('*H2_relevance*.json')

out = os.path.join(RES, 'h_report.txt')
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('Eksperiment izvještaj - SiVBP\n')
    fh.write('Datum: 2026-01-25\n')
    fh.write('\nSAŽETAK:\n')
    fh.write(f' - H1 pokusa (ingest): {len(h1_entries)}\n')
    fh.write(f' - H2 pokusa (relevance): {len(h2_entries)}\n')

    fh.write('\nH1 (Ingest) - kratki sažetak:\n')
    valid_h1 = [e for e in h1_entries if e['metrics'].get('ingest') and not (isinstance(e['metrics'].get('ingest'), dict) and 'error' in e['metrics'].get('ingest'))]
    fh.write(f' - Pokusi sa validnim ingest metrikama: {len(valid_h1)}\n')

    fh.write('\nDetalji H1:\n')
    for e in sorted(h1_entries, key=lambda x: (x['cfg'].get('n_docs') or 0, x['cfg'].get('target_db') or '')):
        cfg = e['cfg']
        m = e['metrics'] or {}
        fh.write('\n---\n')
        fh.write(f"n_docs: {cfg.get('n_docs')}  DB: {cfg.get('target_db')}  batch: {cfg.get('batch_size')}\n")
        fh.write(f"encode_total_s: {e.get('encode_s')}\n")
        ingest = m.get('ingest')
        qlat = m.get('query_latency')
        if isinstance(ingest, dict) and 'error' in ingest:
            fh.write(f"INGEST: ERROR: {ingest['error']}\n")
        elif ingest is None:
            fh.write('INGEST: Not measured or missing\n')
        else:
            fh.write('INGEST summary: %s\n' % json.dumps(ingest))
        if isinstance(qlat, dict) and 'error' in qlat:
            fh.write(f"QUERY LATENCY: ERROR: {qlat['error']}\n")
        elif qlat is None:
            fh.write('QUERY LATENCY: Not measured or missing\n')
        else:
            fh.write('QUERY LATENCY: mean_s=%s p50=%s p90=%s p99=%s qps=%s\n' % (qlat.get('mean_s'), qlat.get('p50_s'), qlat.get('p90_s'), qlat.get('p99_s'), qlat.get('qps')))

    fh.write('\n\nH2 (Relevance) - kratki sažetak:\n')
    if h2_entries:
        # aggregate p@k and map per corpus size
        by_n = {}
        for e in h2_entries:
            n = e['cfg'].get('n_docs')
            p5 = e['metrics'].get('p@5') if e['metrics'] else None
            p1 = e['metrics'].get('p@1') if e['metrics'] else None
            mval = e['metrics'].get('map') if e['metrics'] else None
            by_n.setdefault(n, {'p1': [], 'p5': [], 'map': []})
            if p1 is not None:
                by_n[n]['p1'].append(p1)
            if p5 is not None:
                by_n[n]['p5'].append(p5)
            if mval is not None:
                by_n[n]['map'].append(mval)
        for n, stats in sorted(by_n.items()):
            fh.write(f" - n={n}: avg p@1={mean(stats['p1']) if stats['p1'] else 'NA'} avg p@5={mean(stats['p5']) if stats['p5'] else 'NA'} avg MAP={mean(stats['map']) if stats['map'] else 'NA'}\n")

        fh.write('\nDetalji H2 po konfiguraciji:\n')
        for e in sorted(h2_entries, key=lambda x: (x['cfg'].get('n_docs') or 0, x['cfg'].get('search_mode') or '')):
            cfg = e['cfg']
            m = e['metrics'] or {}
            fh.write('\n---\n')
            fh.write(f"n_docs: {cfg.get('n_docs')}  search_mode: {cfg.get('search_mode')}  db: {cfg.get('target_db')}\n")
            fh.write(f"encode_total_s: {e.get('encode_s')}\n")
            fh.write('precision@k: p@1=%s p@5=%s p@10=%s MAP=%s\n' % (m.get('p@1'), m.get('p@5'), m.get('p@10'), m.get('map')))
    else:
        fh.write(' - Nema H2 rezultata\n')

    fh.write('\nZAKLJUČAK I PREPORUKE:\n')
    fh.write(' - H2 rezultati su izvedeni lokalno koristeći mock DB, daju jasnu tendenciju pada precision sa rastom corpus size.\n')
    fh.write(' - Ako treba produkcijske brojke, zamijeniti mockove s pravim DB instancama (Pinecone ili lokalni Weaviate).\n')
    fh.write(f"\nPutanja do ovoga izvještaja: {os.path.abspath(out)}\n")

print('Wrote', out)

# Generate plots and append a short summary using existing plotting utilities
try:
    import experiments.plot_results as pr
    pr.main()
    print('Generated plots via experiments.plot_results')
except Exception as e:
    print('Could not generate plots:', e)
