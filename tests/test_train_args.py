"""Guards against transformers API drift (e.g. warmup_ratio removed in 5.x). Needs transformers, not torch."""
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")
from transformers import TrainingArguments  # noqa: E402

from triage.train_bert import build_training_kwargs  # noqa: E402


def test_all_training_kwargs_exist_in_installed_transformers():
    a = SimpleNamespace(batch_size=16, epochs=3, seed=1)
    kw = build_training_kwargs(a, "out", n_train=14000, use_fp16=False)
    valid = {f.name for f in dataclasses.fields(TrainingArguments)}
    unknown = sorted(set(kw) - valid)
    assert not unknown, f"TrainingArguments no longer accepts: {unknown}"


def test_warmup_is_positive_integer():
    a = SimpleNamespace(batch_size=16, epochs=3, seed=1)
    w = build_training_kwargs(a, "out", n_train=14000, use_fp16=False)["warmup_steps"]
    assert isinstance(w, int) and w == int(0.1 * 875 * 3)
    assert build_training_kwargs(a, "out", n_train=1, use_fp16=False)["warmup_steps"] == 1
