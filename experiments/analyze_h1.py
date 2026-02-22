import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parents[0] / "results"
OUT_DIR = ROOT / "dist" / "documentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "H1_ANALYSIS.txt"


def load_h1_records():
    files = sorted([p for p in RESULTS.glob("*H1_ingest*.json")])
    recs = []
    for f in files:
        try:
            j = json.load(open(f, "r", encoding="utf-8"))
            cfg = j.get("config", {})
            recs.append({
                "file": str(f.name),
                "n_docs": int(j.get("n_docs", cfg.get("n_docs", 0))),
                    "target_db": (cfg.get("target_db") or j.get("config", {}).get("target_db") or "nije specificirano"),
                "batch_size": cfg.get("batch_size", None),
                "encode_total_s": float(j.get("encode_total_s", 0.0)),
                "p@5": float(j.get("metrics", {}).get("p@5", 0.0)),
                "p@10": float(j.get("metrics", {}).get("p@10", 0.0)),
                "p@20": float(j.get("metrics", {}).get("p@20", 0.0)),
                "map": float(j.get("metrics", {}).get("map", 0.0)),
            })
        except Exception:
            continue
    return recs


def summarize(recs):
    by_size = {}
    by_db = {}
    for r in recs:
        n = r["n_docs"]
        by_size.setdefault(n, []).append(r)
        db = r.get("target_db") or "unknown"
        by_db.setdefault(db, []).append(r)

    def stats(arr, key):
        vals = [a.get(key, 0.0) for a in arr]
        return {
            "count": len(vals),
            "mean": mean(vals) if vals else 0.0,
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }

    summary = {"by_size": {}, "by_db": {}, "overall": {}}
    for n, arr in by_size.items():
        summary["by_size"][n] = {
            "encode": stats(arr, "encode_total_s"),
            "p@5": stats(arr, "p@5"),
            "p@10": stats(arr, "p@10"),
            "p@20": stats(arr, "p@20"),
            "map": stats(arr, "map"),
        }
    for db, arr in by_db.items():
        summary["by_db"][db] = {
            "encode": stats(arr, "encode_total_s"),
            "p@5": stats(arr, "p@5"),
            "p@10": stats(arr, "p@10"),
            "p@20": stats(arr, "p@20"),
            "map": stats(arr, "map"),
        }

    all_encode = [r["encode_total_s"] for r in recs]
    summary["overall"]["encode"] = {
        "count": len(all_encode),
        "mean": mean(all_encode) if all_encode else 0.0,
        "std": stdev(all_encode) if len(all_encode) > 1 else 0.0,
    }
    return summary


def render_bosnian(recs, summary):
    lines = []
    lines.append("Analiza rezultata za Hipotezu H1 (Ingest/throughput)\n")
    lines.append("1. Metodologija:\n")
    lines.append("- Skupljeno je ukupno {} H1 zapisa (fajlova) iz direktorija experiments/results.".format(len(recs)))
    lines.append("- Svaki zapis sadrži: veličinu kolekcije (n_docs), ciljnu DB (pinecone/weaviate), veličinu batch-a i ukupno vrijeme enkodiranja.")
    lines.append("- Za evaluaciju se koristio `DummyEncoder` (TF-IDF->dense) radi reproducibilnosti na Windows okruženju.")
    lines.append("- Mjerene metrike: vrijeme enkodiranja (s), i relevancijske metrike p@5, p@10, p@20 i MAP (pomoćno).\n")

    lines.append("2. Sažetak (agregirano):\n")
    o = summary["overall"]["encode"]
    lines.append(f"- Ukupno zapisa: {o['count']}")
    lines.append(f"- Prosječno vrijeme enkodiranja: {o['mean']:.2f} s (std {o['std']:.2f})\n")

    lines.append("3. Po veličini kolekcije (n_docs):\n")
    for n in sorted(summary["by_size"].keys()):
        s = summary["by_size"][n]
        lines.append(f"- n_docs = {n}:")
        lines.append(f"  * Encode time — count: {s['encode']['count']}, mean: {s['encode']['mean']:.2f}s, std: {s['encode']['std']:.2f}s, min: {s['encode']['min']:.2f}s, max: {s['encode']['max']:.2f}s")
        lines.append(f"  * p@5 mean: {s['p@5']['mean']:.3f}, p@10 mean: {s['p@10']['mean']:.3f}, p@20 mean: {s['p@20']['mean']:.3f}, MAP mean: {s['map']['mean']:.3f}\n")

    lines.append("4. Po ciljnoj bazi (DB):\n")
    for db in sorted(summary["by_db"].keys()):
        s = summary["by_db"][db]
        lines.append(f"- DB = {db}:")
        lines.append(f"  * Encode time — count: {s['encode']['count']}, mean: {s['encode']['mean']:.2f}s, std: {s['encode']['std']:.2f}s")
        lines.append(f"  * p@5 mean: {s['p@5']['mean']:.3f}, p@10 mean: {s['p@10']['mean']:.3f}, p@20 mean: {s['p@20']['mean']:.3f}, MAP mean: {s['map']['mean']:.3f}\n")

    lines.append("5. Zapažanja i interpretacija:\n")
    # Add a few automatic observations based on summary
    # Compare DB means
    dbs = summary['by_db']
    # Note about unspecified DBs
    if 'nije specificirano' in dbs:
        lines.append("- Napomena: neke stare ili generisane datoteke nemaju polje `target_db` (označeno kao 'nije specificirano'). To znači da su to ranije pokretanja ili placeholderi gdje ciljna DB nije bila proslijeđena u konfiguraciji.")
    if 'pinecone' in dbs and 'weaviate' in dbs:
        p_mean = dbs['pinecone']['encode']['mean']
        w_mean = dbs['weaviate']['encode']['mean']
        if p_mean < w_mean:
            lines.append(f"- Pinecone ima nešto niže prosječno vrijeme enkodiranja ({p_mean:.2f}s) u odnosu na Weaviate ({w_mean:.2f}s). Ovo sugerira da...\n")
        elif p_mean > w_mean:
            lines.append(f"- Weaviate ima nešto niže prosječno vrijeme enkodiranja ({w_mean:.2f}s) u odnosu na Pinecone ({p_mean:.2f}s). Ovo sugerira da...\n")
        else:
            lines.append(f"- Prosječna vremena enkodiranja su slična između Pinecone i Weaviate ({p_mean:.2f}s).\n")
    else:
        lines.append("- Nije dostupna usporedba DB (nedostaju zapisi).\n")

    lines.append("6. Preporuke za verifikaciju hipoteze H1:\n")
    lines.append("- Ako ciljamo potvrditi razliku u throughput-u između DB-ova, preporučujem ponoviti svaki test najmanje 3 puta i koristiti stvarne enkodere ako je moguće.")
    lines.append("- Izmjeriti varijaciju pri različitim `batch_size` postavkama i zapisati CPU/RAM prilikom enkodiranja.")
    lines.append("- Za akademski izvještaj uključiti tablice sa srednjim vrijednostima i standardnom devijacijom (već izračunato iznad).\n")

    lines.append("7. Detaljni popis ispitanih run-ova (file, n_docs, DB, batch, encode_s):\n")
    for r in recs:
        lines.append(f"- {r['file']}: n={r['n_docs']}, db={r.get('target_db')}, batch={r.get('batch_size')}, encode_s={r.get('encode_total_s'):.2f}")

    lines.append('\nKraj analize H1.')
    return "\n".join(lines)


if __name__ == '__main__':
    recs = load_h1_records()
    summary = summarize(recs)
    out = render_bosnian(recs, summary)
    with open(OUT_FILE, 'w', encoding='utf-8') as fh:
        fh.write(out)
    print('Wrote H1 analysis to', OUT_FILE)
