"""Plotting utilities for experiment results."""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parents[0]


def plot_latency_vs_size(df: pd.DataFrame, out_file: str = None):
    # df expected to contain columns: size, mean_latency_ms, db, model
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='size', y='mean_ms', hue='db', style='model', markers=True)
    plt.title('Query latency (mean ms) vs dataset size')
    plt.xlabel('Dataset size (docs)')
    plt.ylabel('Mean latency (ms)')
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()


def plot_throughput_vs_size(df: pd.DataFrame, out_file: str = None):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='size', y='overall_throughput_vps', hue='db', style='model', markers=True)
    plt.title('Ingestion throughput (vectors/s) vs dataset size')
    plt.xlabel('Dataset size (docs)')
    plt.ylabel('Throughput (vectors/s)')
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()


def plot_precision_recall(df: pd.DataFrame, out_file: str = None):
    # df contains columns: model, db, p@1,p@5,p@10, r@1,r@5,r@10
    melted = df.melt(id_vars=['model', 'db'], value_vars=[col for col in df.columns if col.startswith('p@') or col.startswith('r@')], var_name='metric', value_name='value')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='metric', y='value', hue='model')
    plt.title('Precision@k and Recall@k by embedding model')
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()
