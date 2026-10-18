"""TF-IDF + logistic regression baseline with the same recall gate as the transformer.

    python -m triage.baseline --data data --out artifacts/baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .evaluate import negation_pair_test, slice_report
from .gate import min_positives_for_gate, operating_report, select_threshold

SLICES = ["language", "beacon_type", "source_mix", "administration", "env_class"]


def build_pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([("f", feats), ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/baseline")
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    d = Path(a.data)
    tr, va, te = (pd.read_csv(d / f"{s}.csv") for s in ("train", "val", "test"))
    pipe = build_pipeline().fit(tr.text, tr.label)
    pv, pt = pipe.predict_proba(va.text)[:, 1], pipe.predict_proba(te.text)[:, 1]

    need = min_positives_for_gate(a.target_recall, a.alpha)
    t = select_threshold(pv, va.label, a.target_recall, a.alpha)
    gate_met = t is not None
    if not gate_met:
        print(f"GATE NOT MET on validation (need >= {need} positives and near-zero misses). "
              "Falling back to threshold 0.0 (flag everything).")
        t = 0.0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    joblib.dump(pipe, model_path)
    version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]

    slices = slice_report(te, pt, t, SLICES, a.target_recall, a.alpha)
    slices.to_csv(out / "slice_report_test.csv", index=False)
    neg = negation_pair_test(lambda x: pipe.predict_proba(pd.Series(x))[:, 1])

    op = {
        "model_kind": "baseline", "model_path": str(model_path), "model_version": version,
        "threshold": t, "gate_met_on_val": gate_met, "target_recall": a.target_recall, "alpha": a.alpha,
        "min_positives_required": need,
        "val": operating_report(pv, va.label, t, a.alpha),
        "test": operating_report(pt, te.label, t, a.alpha),
        "test_auroc": float(roc_auc_score(te.label, pt)),
        "negation_pair_pass_rate": neg["pass_rate"],
        "slices_failing": int((slices.gate == "FAIL").sum()),
        "slices_insufficient": int((slices.gate == "INSUFFICIENT_DATA").sum()),
    }
    (out / "operating_point.json").write_text(json.dumps(op, indent=2))
    print(json.dumps(op, indent=2))


if __name__ == "__main__":
    main()
