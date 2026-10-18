# 406 MHz Alert Triage: Advisory NLP Decision Support

Advisory-only text classifier for MCC/RCC operators. It suggests priority from accumulated alert text.
It **never** gates, delays, or suppresses alert distribution.

## Status: engineering complete, NOT cleared for operational use

Everything that can be built without real data is here and tested. Everything that needs real data,
real operators and a safety owner is listed in `docs/PRODUCTION_READINESS.md` and is still open.
Numbers produced on the synthetic data are plumbing checks, not evidence of safety.

## Layout
| Path | Purpose |
|---|---|
| `triage/gate.py` | Clopper-Pearson recall bound, minimum-sample calculator, threshold selection |
| `triage/split.py` | Beacon-grouped temporal split (no leakage) |
| `triage/synth.py` | Synthetic data generator (pipeline testing only) |
| `triage/baseline.py` | TF-IDF + LR baseline with the same gate |
| `triage/train_bert.py` | XLM-R fine-tuning: class weights, temperature scaling, gate, slices |
| `triage/evaluate.py` | Slice report (gate per slice), negation pair tests |
| `triage/serve.py` | FastAPI service: fail-safe, keyword safety net, audit log, API key |
| `tests/` | 17 tests incl. fail-safe behaviour and no-leakage |
| `docs/` | Data requirements, labeling guide, model card, shadow-mode plan, readiness checklist |

## Quick start
```bash
make install && make test
make synth && make baseline          # plumbing check on synthetic data
make serve                           # http://localhost:8080/docs
python -m triage.train_bert --data data --out artifacts/bert   # needs requirements-train.txt + model download
```

## No local GPU? Use Colab
Upload `alert_triage_colab.ipynb` to Google Colab (T4 GPU runtime). It unpacks this project, runs the tests,
trains the baseline and the transformer, summarizes results against the release criteria, and saves artifacts
to Drive or a download. Use synthetic data only unless your data-sharing agreement allows Colab.

## Windows (no `make`)
Run the underlying commands directly, for example `python -m pytest -q`, `python -m triage.synth --n 20000 --out data`,
`python -m triage.baseline --data data --out artifacts/baseline`. In PowerShell set env vars with
`$env:OPERATING_POINT = "artifacts/baseline/operating_point.json"` before `uvicorn triage.serve:app --port 8080`.

## Using real data
1. Build `train.csv / val.csv / test.csv` per `docs/DATA_REQUIREMENTS.md` (same columns as the synthetic files).
2. Split with `triage.split.grouped_temporal_split` (group by beacon, order by time).
3. Run baseline first, then BERT. Ship the simpler model if it clears the same gate.
4. Read `operating_point.json`: if `gate_met_on_val` is false or any slice FAILs, it is not deployable.

## Service contract
`POST /v1/suggest {"alert_id","text"}` returns `PRIORITIZE | LOW_PRIORITY_SUGGESTED | NO_SUGGESTION`.
Any failure returns `NO_SUGGESTION` with HTTP 200, so the existing workflow continues unchanged.
Operator UI must treat `LOW_PRIORITY_SUGGESTED` as a hint only, never as a filter.

## Easiest option: the built-in test page

Open this in your browser:

```
http://localhost:8080/docs
```

1. Click **POST /v1/suggest**.
2. Click **Try it out**.
3. Paste this into the request body and click **Execute**:

```json
{
  "alert_id": "A1",
  "text": "[META] EPIRB sea moving registered=True [OWNER_CALL] Owner could not be reached by phone."
}
```

## URLs you can open in a browser (GET)

| URL | Shows |
|---|---|
| `http://localhost:8080/healthz` | Whether the service is up and a model is loaded |
| `http://localhost:8080/v1/stats` | Request counters (needs the API key header if you set one) |
| `http://localhost:8080/docs` | The interactive test page above |

## From PowerShell

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8080/v1/suggest `
  -ContentType "application/json" `
  -Body '{"alert_id":"A1","text":"[META] EPIRB sea moving registered=True [OWNER_CALL] Owner could not be reached by phone."}'
```

## If you set `SERVICE_API_KEY`

Add the header, or you'll get a 401. On the `/docs` page, you would need to add it manually, so it's easier to test from PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8080/v1/suggest `
  -Headers @{ "X-API-Key" = "your-key" } `
  -ContentType "application/json" `
  -Body '{"alert_id":"A1","text":"..."}'
```

## What the response means

You'll get something like:

```json
{"alert_id":"A1","suggestion":"PRIORITIZE","p_distress":0.82,"threshold":0.05,"reason":"score_at_or_above_threshold","advisory_only":true,"model_version":"..."}
```

- `PRIORITIZE` means treat it as potentially urgent.
- `LOW_PRIORITY_SUGGESTED` is a hint only and never a reason to drop an alert.
- `NO_SUGGESTION` means something failed or the input was too short. Check `reason`, and carry on with the normal workflow.

If `/healthz` shows `"model_loaded": false`, the service started without finding your model. Check that `OPERATING_POINT` points to a real `operating_point.json` and that you ran the baseline or training step first.