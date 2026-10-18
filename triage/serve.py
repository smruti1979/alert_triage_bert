"""
Advisory inference service. FAIL-SAFE by design:

* Never blocks, delays or suppresses an alert. Output is a suggestion for the operator UI only.
* Any error, timeout, missing model, empty input or degraded state => NO_SUGGESTION (HTTP 200),
  so the existing workflow carries on unchanged.
* A hard-keyword safety net forces PRIORITIZE regardless of the model score.
* Every request is written to an append-only audit log (hashed text by default).

Run:  uvicorn triage.serve:app --host 0.0.0.0 --port 8080
Env:  OPERATING_POINT=artifacts/baseline/operating_point.json  SERVICE_API_KEY=...  AUDIT_LOG=audit.jsonl
      MODEL_TIMEOUT_S=2.0  LOG_RAW_TEXT=false  MAX_TEXT_CHARS=20000
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("triage")

# Over-triggering is the safe direction (e.g. "no fire" still prioritizes).
HARD_KEYWORDS = re.compile(
    r"\b(mayday|pan[- ]pan|sinking|sank|capsiz\w*|taking water|on fire|fire on board|man overboard|"
    r"abandon\w* (ship|vessel)|crash\w*|in the water|hundi\w*|incendio|naufrag\w*|hace agua|"
    r"incendie|prend l'eau|naufrage|couler)\b", re.IGNORECASE)


class SuggestRequest(BaseModel):
    alert_id: str = Field(..., max_length=128)
    text: str = Field("", description="Serialized alert text (already PII-redacted upstream)")


class SuggestResponse(BaseModel):
    alert_id: str
    suggestion: str          # PRIORITIZE | LOW_PRIORITY_SUGGESTED | NO_SUGGESTION
    p_distress: Optional[float] = None
    threshold: Optional[float] = None
    reason: str
    advisory_only: bool = True
    model_version: str


class BaselinePredictor:
    def __init__(self, path: str):
        import joblib
        self.pipe = joblib.load(path)

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))[:, 1]


class BertPredictor:
    def __init__(self, path: str, temperature: float, max_len: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.T, self.max_len = torch, temperature, max_len
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict_proba(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            z = self.model(**enc).logits.numpy() / self.T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e[:, 1] / e.sum(axis=1)


def load_predictor(op_path: str):
    op = json.loads(Path(op_path).read_text())
    if op["model_kind"] == "baseline":
        pred = BaselinePredictor(op["model_path"])
    else:
        pred = BertPredictor(op["model_path"], op.get("temperature", 1.0))
    return pred, op


class Service:
    def __init__(self, predictor, op: dict, audit_path: str, timeout_s: float, max_chars: int, log_raw: bool):
        self.predictor, self.op = predictor, op
        self.audit_path, self.timeout_s, self.max_chars, self.log_raw = audit_path, timeout_s, max_chars, log_raw
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "prioritize": 0, "low_priority": 0, "no_suggestion": 0, "errors": 0, "keyword_overrides": 0}

    @property
    def version(self) -> str:
        return str((self.op or {}).get("model_version", "none"))

    def _audit(self, rec: dict):
        try:
            with self.lock, open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:                      # audit failure must never break the request
            log.exception("audit write failed")

    def suggest(self, req: SuggestRequest) -> SuggestResponse:
        t0 = time.perf_counter()
        text = (req.text or "").strip()
        p, thr, suggestion, reason = None, (self.op or {}).get("threshold"), "NO_SUGGESTION", ""
        try:
            if len(text) < 10:
                reason = "input_too_short"
            elif len(text) > self.max_chars:
                reason = "input_too_long"
            elif self.predictor is None or self.op is None:
                reason = "model_unavailable"
            else:
                p = float(self.pool.submit(lambda: self.predictor.predict_proba([text])[0]).result(self.timeout_s))
                if not np.isfinite(p):
                    p, reason = None, "non_finite_score"
                elif HARD_KEYWORDS.search(text):
                    suggestion, reason = "PRIORITIZE", "hard_keyword_override"
                    self.stats["keyword_overrides"] += 1
                elif p >= thr:
                    suggestion, reason = "PRIORITIZE", "score_at_or_above_threshold"
                else:
                    suggestion, reason = "LOW_PRIORITY_SUGGESTED", "score_below_threshold"
        except FutureTimeout:
            reason = "model_timeout"
        except Exception as e:                 # noqa: BLE001 - fail-safe catch-all
            reason = f"model_error:{type(e).__name__}"
            self.stats["errors"] += 1
            log.exception("inference failed")

        if suggestion == "NO_SUGGESTION" and reason == "":
            reason = "unspecified"
        key = {"PRIORITIZE": "prioritize", "LOW_PRIORITY_SUGGESTED": "low_priority", "NO_SUGGESTION": "no_suggestion"}[suggestion]
        with self.lock:
            self.stats["requests"] += 1
            self.stats[key] += 1
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "alert_id": req.alert_id,
               "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text_len": len(text),
               "model_version": self.version, "p": p, "threshold": thr, "suggestion": suggestion,
               "reason": reason, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
        if self.log_raw:
            rec["text"] = text
        self._audit(rec)
        return SuggestResponse(alert_id=req.alert_id, suggestion=suggestion, p_distress=p, threshold=thr,
                               reason=reason, model_version=self.version)


def create_app(predictor=None, op: Optional[dict] = None, *, audit_path: Optional[str] = None,
               timeout_s: Optional[float] = None, autoload: bool = True) -> FastAPI:
    if predictor is None and autoload and os.environ.get("OPERATING_POINT"):
        try:
            predictor, op = load_predictor(os.environ["OPERATING_POINT"])
            if not op.get("gate_met_on_val", False):
                log.warning("Operating point did not meet recall gate; threshold is the flag-everything fallback.")
        except Exception:                      # degraded mode: still serve, always NO_SUGGESTION
            log.exception("model load failed; running in degraded mode")
            predictor, op = None, None

    svc = Service(predictor, op, audit_path or os.environ.get("AUDIT_LOG", "audit.jsonl"),
                  timeout_s or float(os.environ.get("MODEL_TIMEOUT_S", "2.0")),
                  int(os.environ.get("MAX_TEXT_CHARS", "20000")),
                  os.environ.get("LOG_RAW_TEXT", "false").lower() == "true")
    api_key = os.environ.get("SERVICE_API_KEY")

    def auth(x_api_key: Optional[str] = Header(default=None)):
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    app = FastAPI(title="406 MHz alert triage (advisory)", version="0.1.0")
    app.state.service = svc

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_loaded": svc.predictor is not None, "model_version": svc.version}

    @app.get("/v1/stats", dependencies=[Depends(auth)])
    def stats():
        return svc.stats

    @app.post("/v1/suggest", response_model=SuggestResponse, dependencies=[Depends(auth)])
    def suggest(req: SuggestRequest):
        return svc.suggest(req)

    return app


app = create_app()
