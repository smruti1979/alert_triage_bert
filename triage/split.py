"""Leakage-safe splitting: whole groups (beacon) assigned by time of first appearance."""
from __future__ import annotations

import pandas as pd


def grouped_temporal_split(df, group_col="beacon_id", time_col="timestamp", fracs=(0.70, 0.15, 0.15)):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    first = df.groupby(group_col)[time_col].min().sort_values()
    sizes = df.groupby(group_col).size()
    total, cum = len(df), 0
    bounds = (fracs[0] * total, (fracs[0] + fracs[1]) * total)
    assign = {}
    for g in first.index:
        assign[g] = 0 if cum < bounds[0] else (1 if cum < bounds[1] else 2)
        cum += sizes[g]
    part = df[group_col].map(assign)
    return tuple(df[part == i].reset_index(drop=True) for i in range(3))
