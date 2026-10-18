"""Recall-gate statistics: confidence bounds, minimum sample size, threshold selection."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta


def recall_lower_bound(tp: int, n_pos: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson lower bound on recall."""
    if n_pos <= 0 or tp <= 0:
        return 0.0
    if tp >= n_pos:
        return float(alpha ** (1.0 / n_pos))
    return float(beta.ppf(alpha, tp, n_pos - tp + 1))


def min_positives_for_gate(target: float = 0.99, alpha: float = 0.05) -> int:
    """Smallest positive count at which ZERO misses can prove recall >= target."""
    return math.ceil(math.log(alpha) / math.log(target))


def select_threshold(probs, labels, target: float = 0.99, alpha: float = 0.05):
    """
    Highest threshold t (flag as distress if p >= t) whose recall lower bound >= target.
    Recall is non-increasing in t, so the highest passing t maximizes low-priority
    suggestions subject to the gate. Returns None if the gate cannot be met.
    """
    probs, labels = np.asarray(probs), np.asarray(labels)
    pos = np.sort(probs[labels == 1])
    n_pos = len(pos)
    if n_pos == 0:
        return None
    best = None
    for t in np.unique(np.concatenate([[0.0], pos])):
        tp = int((pos >= t).sum())
        if recall_lower_bound(tp, n_pos, alpha) >= target:
            best = float(t)
    return best


def operating_report(probs, labels, t: float, alpha: float = 0.05) -> dict:
    probs, labels = np.asarray(probs), np.asarray(labels)
    pred = probs >= t
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    tp = int((pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    return {
        "threshold": float(t),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "recall": tp / n_pos if n_pos else float("nan"),
        "recall_lower_bound": recall_lower_bound(tp, n_pos, alpha),
        "missed_distress": n_pos - tp,
        "low_priority_fraction_of_false": tn / n_neg if n_neg else float("nan"),
    }
