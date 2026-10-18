import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from triage.serve import create_app

OP = {"model_version": "test", "threshold": 0.3, "model_kind": "stub", "gate_met_on_val": True}
TXT = "[META] EPIRB sea stationary registered=True [OWNER_CALL] Owner says beacon was tested, no distress."


class Fixed:
    def __init__(self, p): self.p = p
    def predict_proba(self, texts): return np.array([self.p] * len(texts))


class Broken:
    def predict_proba(self, texts): raise RuntimeError("boom")


class Slow:
    def predict_proba(self, texts):
        time.sleep(1.0)
        return np.array([0.9])


class NaN:
    def predict_proba(self, texts): return np.array([float("nan")])


def client(pred, tmp_path, op=OP, timeout=0.3):
    app = create_app(pred, op, audit_path=str(tmp_path / "audit.jsonl"), timeout_s=timeout, autoload=False)
    return TestClient(app)


def post(c, text=TXT):
    r = c.post("/v1/suggest", json={"alert_id": "A1", "text": text})
    assert r.status_code == 200
    return r.json()


def test_above_threshold_prioritizes(tmp_path):
    assert post(client(Fixed(0.8), tmp_path))["suggestion"] == "PRIORITIZE"


def test_below_threshold_is_only_advisory(tmp_path):
    j = post(client(Fixed(0.01), tmp_path))
    assert j["suggestion"] == "LOW_PRIORITY_SUGGESTED" and j["advisory_only"] is True


def test_keyword_override_beats_low_score(tmp_path):
    j = post(client(Fixed(0.0), tmp_path), "[RCC_LOG] Caller says vessel is sinking, crew in the water")
    assert j["suggestion"] == "PRIORITIZE" and j["reason"] == "hard_keyword_override"


@pytest.mark.parametrize("pred", [Broken(), Slow(), NaN(), None])
def test_failures_degrade_to_no_suggestion(tmp_path, pred):
    j = post(client(pred, tmp_path if pred is not None else tmp_path, op=OP if pred is not None else None))
    assert j["suggestion"] == "NO_SUGGESTION"


def test_empty_input_no_suggestion(tmp_path):
    assert post(client(Fixed(0.9), tmp_path), "")["suggestion"] == "NO_SUGGESTION"


def test_audit_log_written_without_raw_text(tmp_path):
    c = client(Fixed(0.8), tmp_path)
    post(c)
    rec = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert rec["alert_id"] == "A1" and "text" not in rec and len(rec["text_sha256"]) == 64


def test_audit_failure_does_not_break_request(tmp_path):
    app = create_app(Fixed(0.8), OP, audit_path=str(tmp_path / "no" / "such" / "dir" / "a.jsonl"), autoload=False)
    assert post(TestClient(app))["suggestion"] == "PRIORITIZE"


def test_api_key_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    c = client(Fixed(0.8), tmp_path)
    assert c.post("/v1/suggest", json={"alert_id": "A", "text": TXT}).status_code == 401
    assert c.post("/v1/suggest", json={"alert_id": "A", "text": TXT}, headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/healthz").status_code == 200
