"""Relevance metrics: precision@k, recall@k and helpers."""
from typing import List, Dict
import numpy as np


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return len(set(topk) & set(relevant)) / float(len(topk))


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    return len(set(topk) & set(relevant)) / float(len(relevant))


def evaluate_all(queries: List[Dict], retrievals: Dict[str, List[str]], qrels: Dict[str, List[str]], k_values: List[int] = [5,10,20]) -> Dict:
    """Compute precision@k and recall@k averaged over queries.
    queries: list of {'id': qid, 'query': text}
    retrievals: dict qid -> list of retrieved doc ids (ordered)
    qrels: dict qid -> list of relevant doc ids
    """
    results = {f"p@{k}": [] for k in k_values}
    results.update({f"r@{k}": [] for k in k_values})

    ap_list = []
    for q in queries:
        qid = q["id"]
        rel = qrels.get(qid, [])
        ret = retrievals.get(qid, [])
        for k in k_values:
            results[f"p@{k}"].append(precision_at_k(ret, rel, k))
            results[f"r@{k}"].append(recall_at_k(ret, rel, k))

        # Average Precision (AP) for MAP
        if rel:
            num_relevant = 0
            precisions = []
            for idx, doc in enumerate(ret, start=1):
                if doc in rel:
                    num_relevant += 1
                    precisions.append(precision_at_k(ret, rel, idx))
            if precisions:
                ap = float(np.mean(precisions))
            else:
                ap = 0.0
        else:
            ap = 0.0
        ap_list.append(ap)

    summary = {}
    for key, vals in results.items():
        if vals:
            summary[key] = float(np.mean(vals))
        else:
            summary[key] = 0.0
    # MAP
    summary["map"] = float(np.mean(ap_list)) if ap_list else 0.0
    return summary
