"""Fine-tune a transformer encoder with class weighting, temperature scaling and recall-gated threshold.

    python -m triage.train_bert --data data --out artifacts/bert --model xlm-roberta-base

Requires: pip install -r requirements-train.txt  (and network or a local model path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import negation_pair_test, slice_report
from .gate import min_positives_for_gate, operating_report, select_threshold

SLICES = ["language", "beacon_type", "source_mix", "administration", "env_class"]


def build_training_kwargs(a, out_dir, n_train: int, use_fp16: bool) -> dict:
    """Keyword arguments for TrainingArguments.

    Warmup is passed as an explicit integer `warmup_steps`: `warmup_ratio` was removed in
    transformers 5.x, while `warmup_steps` exists in both 4.x and 5.x.
    """
    steps_per_epoch = max(1, math.ceil(n_train / a.batch_size))
    warmup_steps = max(1, int(0.1 * steps_per_epoch * a.epochs))
    return dict(
        output_dir=str(Path(out_dir) / "ckpt"), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size * 2,
        learning_rate=2e-5, weight_decay=0.01, warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=1,
        fp16=use_fp16, seed=a.seed, report_to="none", disable_tqdm=True, logging_steps=50)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    import torch
    lg, y = torch.tensor(logits, dtype=torch.float), torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def softmax_pos(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=1))


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/bert")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pos-boost", type=float, default=2.0)
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)

    d = Path(a.data)
    tr, va, te = (pd.read_csv(d / f"{s}.csv") for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(a.model)

    def ds(df):
        x = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        x = x.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len), batched=True)
        return x.remove_columns(["text"]).rename_column("label", "labels")

    dtr, dva, dte = ds(tr), ds(va), ds(te)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)

    counts = np.bincount(tr.label.values, minlength=2).astype(float)
    w = counts.sum() / (2 * counts)
    w[1] *= a.pos_boost
    weights = torch.tensor(w, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.CrossEntropyLoss(weight=weights.to(out.logits.device))(out.logits, labels)
            return (loss, out) if return_outputs else loss

    out = Path(a.out)
    args = TrainingArguments(**build_training_kwargs(a, out, len(dtr), torch.cuda.is_available()))
    trainer = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                              data_collator=DataCollatorWithPadding(tok))
    trainer.train()

    lv, lt = trainer.predict(dva).predictions, trainer.predict(dte).predictions
    T = fit_temperature(lv, va.label.values)          # fitted on validation only
    pv, pt = softmax_pos(lv, T), softmax_pos(lt, T)

    t = select_threshold(pv, va.label, a.target_recall, a.alpha)
    gate_met = t is not None
    if not gate_met:
        print("GATE NOT MET on validation; falling back to threshold 0.0 (flag everything).")
        t = 0.0

    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    (final / "temperature.json").write_text(json.dumps({"temperature": T}))
    version = hashlib.sha256((final / "model.safetensors").read_bytes()).hexdigest()[:12] \
        if (final / "model.safetensors").exists() else "unknown"

    def predict_fn(texts):
        enc = tok(list(texts), truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            return softmax_pos(model(**enc).logits.cpu().numpy(), T)

    slices = slice_report(te, pt, t, SLICES, a.target_recall, a.alpha)
    slices.to_csv(out / "slice_report_test.csv", index=False)
    op = {
        "model_kind": "bert", "model_path": str(final), "model_version": version, "base_model": a.model,
        "temperature": T, "threshold": t, "gate_met_on_val": gate_met,
        "target_recall": a.target_recall, "alpha": a.alpha,
        "min_positives_required": min_positives_for_gate(a.target_recall, a.alpha),
        "val": operating_report(pv, va.label, t, a.alpha),
        "test": operating_report(pt, te.label, t, a.alpha),
        "negation_pair_pass_rate": negation_pair_test(predict_fn)["pass_rate"],
        "slices_failing": int((slices.gate == "FAIL").sum()),
        "slices_insufficient": int((slices.gate == "INSUFFICIENT_DATA").sum()),
    }
    (out / "operating_point.json").write_text(json.dumps(op, indent=2))
    print(json.dumps(op, indent=2))


if __name__ == "__main__":
    main()
