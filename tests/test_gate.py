import numpy as np
from triage.gate import min_positives_for_gate, recall_lower_bound, select_threshold


def test_min_positives_matches_rule_of_three():
    assert 295 <= min_positives_for_gate(0.99, 0.05) <= 300


def test_lower_bound_monotone_and_bounded():
    assert recall_lower_bound(0, 100) == 0.0
    assert recall_lower_bound(100, 100) < 1.0
    assert recall_lower_bound(99, 100) < recall_lower_bound(100, 100)
    assert 0 < recall_lower_bound(90, 100) < 0.9


def test_threshold_unreachable_with_few_positives():
    rng = np.random.default_rng(0)
    probs = np.concatenate([rng.uniform(0.6, 1, 50), rng.uniform(0, 0.5, 500)])
    labels = np.concatenate([np.ones(50), np.zeros(500)])
    assert select_threshold(probs, labels, 0.99) is None      # 50 positives cannot prove 99%


def test_threshold_reachable_and_maximal():
    rng = np.random.default_rng(0)
    probs = np.concatenate([rng.uniform(0.6, 1, 600), rng.uniform(0, 0.5, 5000)])
    labels = np.concatenate([np.ones(600), np.zeros(5000)])
    t = select_threshold(probs, labels, 0.99)
    pos = np.sort(probs[labels == 1])
    assert t is not None
    # passes the gate at t ...
    assert recall_lower_bound(int((pos >= t).sum()), len(pos)) >= 0.99
    # ... and the next higher candidate threshold does not
    higher = pos[pos > t]
    if len(higher):
        assert recall_lower_bound(int((pos >= higher[0]).sum()), len(pos)) < 0.99


def test_no_positives_returns_none():
    assert select_threshold(np.array([0.1, 0.2]), np.array([0, 0])) is None
