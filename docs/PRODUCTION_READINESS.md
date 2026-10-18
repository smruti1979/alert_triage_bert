# Production Readiness Checklist

Legend: DONE = in this repo and tested. OPEN = needs real data, people or approvals.

## A. Engineering
| Item | Status |
|---|---|
| Leakage-safe split (beacon-grouped, temporal) | DONE (`split.py`, tested) |
| Recall gate with confidence bound and minimum-sample check | DONE (`gate.py`, tested) |
| Per-slice gate (language, beacon type, source, administration, env) | DONE (`evaluate.py`) |
| Negation pair robustness test | DONE (5 pairs; needs expansion with SME-written cases) |
| Fail-safe service: errors, timeouts, NaN, missing model, empty input => NO_SUGGESTION | DONE (tested) |
| Hard-keyword safety net overrides low score | DONE (tested; keyword list needs SME review per language) |
| Audit log (hashed text, model version, score, reason, latency), failure-tolerant | DONE (tested) |
| API-key auth | DONE; replace with mTLS/OIDC per your network policy |
| Container image, non-root, healthcheck | DONE (build and scan in your pipeline) |
| CI | DONE (`.github/workflows/ci.yml`) |
| Transformer training script | WRITTEN, NOT RUN: could not download weights in the build sandbox. Run once on real GPU and review output |
| Load test, p95 latency under production concurrency | OPEN |
| Pen test / dependency and image vulnerability scan | OPEN |
| Monitoring dashboards and alerting (request rate, NO_SUGGESTION rate, p95, drift) | OPEN (service exposes `/v1/stats`; wire to Prometheus/your stack) |
| Model registry, signed artifacts, rollback runbook | OPEN |
| Calibration check on real data (ECE, reliability plot) | OPEN (temperature scaling implemented) |

## B. Data and labels (the real blocker)
| Item | Status |
|---|---|
| Data-sharing agreement per administration | OPEN |
| Privacy/legal review; PII redaction pipeline with manual audit | OPEN |
| Outcome linkage (alert -> closure report -> final disposition) | OPEN |
| >= 299 positives in validation AND test, ideally 3x that, to prove 99% recall at 95% confidence | OPEN |
| Labeling guide v1.0 piloted on 300 alerts; kappa >= 0.75 | OPEN (`LABELING_GUIDE.md` is the starting draft) |
| Per-slice positive counts sufficient (otherwise slice gate shows INSUFFICIENT_DATA) | OPEN |

## C. Safety and governance
| Item | Status |
|---|---|
| Named safety owner; recall target ratified | OPEN |
| Hazard analysis (FMEA/STPA): missed distress, automation bias, drift, misattribution | OPEN |
| Pre-registered go/no-go criteria | OPEN (template in `SHADOW_MODE_PLAN.md`) |
| Shadow mode >= 2 to 3 months on live traffic | OPEN |
| Operator assisted pilot with control group | OPEN |
| Operator training, including automation-bias countermeasures | OPEN |
| Review against national and COSPAS-SARSAT operational/quality requirements | OPEN |
| Post-deployment monitoring, incident process, scheduled revalidation | OPEN |

## D. Release gate (all must hold)
1. `gate_met_on_val` true AND test recall lower bound >= target at the frozen threshold.
2. Zero FAIL slices; every INSUFFICIENT_DATA slice has a documented risk acceptance.
3. Negation/perturbation suite passes at agreed rates.
4. Shadow-mode recall lower bound >= target on live closed cases.
5. Safety owner sign-off recorded.
