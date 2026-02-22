"""Statistical comparison utilities for experimental metrics.

Provides paired t-test and Wilcoxon signed-rank test for metric arrays.
"""
from typing import Sequence, Tuple
import numpy as np
from scipy import stats


def paired_tests(a: Sequence[float], b: Sequence[float]) -> dict:
    a = np.array(a)
    b = np.array(b)
    res = {}
    # Paired t-test
    t_stat, p_t = stats.ttest_rel(a, b)
    res['paired_t_stat'] = float(t_stat)
    res['paired_t_pvalue'] = float(p_t)
    # Wilcoxon (non-parametric)
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
        res['wilcoxon_stat'] = float(w_stat)
        res['wilcoxon_pvalue'] = float(p_w)
    except Exception as e:
        res['wilcoxon_stat'] = None
        res['wilcoxon_pvalue'] = None
    return res
