"""Load the experiment CSV and generate plots using plotting utilities."""
import argparse
import pandas as pd
from plotting import plot_latency_vs_size, plot_throughput_vs_size, plot_precision_recall
from pathlib import Path


def main(results_csv: str):
    df = pd.read_csv(results_csv)
    out_dir = Path(results_csv).resolve().parents[0]

    # Example filters
    ingest_df = df[df['type'] == 'ingest']
    # For latency vs size (we expect mean_ms in search results)
    search_df = df[df['type'].str.contains('search', na=False)]

    # Save plots
    plot_latency_vs_size(search_df, out_file=str(out_dir / 'latency_vs_size.png'))
    plot_throughput_vs_size(ingest_df, out_file=str(out_dir / 'throughput_vs_size.png'))
    plot_precision_recall(search_df.dropna(subset=['p@1']), out_file=str(out_dir / 'precision_recall.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    main(args.results)
