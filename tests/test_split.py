import pandas as pd
from triage.split import grouped_temporal_split
from triage.synth import generate


def test_no_group_leakage_and_time_order():
    df = generate(3000, seed=1)
    tr, va, te = grouped_temporal_split(df)
    assert not set(tr.beacon_id) & set(va.beacon_id)
    assert not set(tr.beacon_id) & set(te.beacon_id)
    assert not set(va.beacon_id) & set(te.beacon_id)
    assert len(tr) + len(va) + len(te) == len(df)
    assert pd.to_datetime(tr.timestamp).min() <= pd.to_datetime(te.timestamp).min()
    assert 0.5 < len(tr) / len(df) < 0.9
