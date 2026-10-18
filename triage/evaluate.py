"""Slice reports and robustness checks. Any slice failing the gate is a release blocker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gate import min_positives_for_gate, recall_lower_bound

# (safe wording, danger wording). The danger variant must score higher.
NEGATION_PAIRS = [
    ("[OWNER_CALL] No one is in danger, all persons are safe.", "[OWNER_CALL] Someone is in danger, persons are not safe."),
    ("[OWNER_CALL] There was no distress, beacon activated by accident.", "[OWNER_CALL] This is a real distress, not an accident."),
    ("[RCC_LOG] Vessel confirmed safe in harbour.", "[RCC_LOG] Vessel confirmed sinking, crew in water."),
    ("[OWNER_CALL] Nadie esta en peligro, todos estan a salvo.", "[OWNER_CALL] Hay personas en peligro, no estan a salvo."),
    ("[OWNER_CALL] Personne n'est en danger, tout le monde est sain et sauf.", "[OWNER_CALL] Des personnes sont en danger, elles ne sont pas en securite."),
]


def slice_report(df, probs, t, by, target=0.99, alpha=0.05) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["_p"], rows = np.asarray(probs), []
    need = min_positives_for_gate(target, alpha)
    for col in by:
        for val, g in df.groupby(col):
            pos, neg = g[g.label == 1], g[g.label == 0]
            tp = int((pos._p >= t).sum())
            n_pos = len(pos)
            lb = recall_lower_bound(tp, n_pos, alpha)
            if n_pos < need:
                status = "INSUFFICIENT_DATA" if (n_pos == 0 or tp == n_pos) else "FAIL"
            else:
                status = "PASS" if lb >= target else "FAIL"
            rows.append({"slice": col, "value": val, "n": len(g), "n_pos": n_pos,
                         "recall": tp / n_pos if n_pos else np.nan, "recall_lb": lb,
                         "missed": n_pos - tp,
                         "low_priority_frac_of_false": float((neg._p < t).mean()) if len(neg) else np.nan,
                         "gate": status})
    return pd.DataFrame(rows)


def negation_pair_test(predict_fn) -> dict:
    safe = predict_fn([s for s, _ in NEGATION_PAIRS])
    danger = predict_fn([d for _, d in NEGATION_PAIRS])
    ok = [bool(d > s) for s, d in zip(safe, danger)]
    return {"pass_rate": float(np.mean(ok)), "per_pair": ok}
