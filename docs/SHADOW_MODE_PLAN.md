# Shadow Mode and Pilot Plan

## Phase 1: shadow (2 to 3 months)
* Service receives live alert text; outputs go ONLY to the audit log. Operators never see them.
* Weekly job joins audit log to closed-case outcomes and computes recall (with lower bound), low-priority fraction, calibration, slice table.
* Track NO_SUGGESTION rate, p95 latency, keyword override rate, input-too-long rate.

## Pre-registered go/no-go (edit and freeze BEFORE starting)
| Criterion | Threshold |
|---|---|
| Recall lower bound on live closed positives | >= target (e.g. 0.99) with >= 299 positives (extend the period if needed) |
| FAIL slices | 0 |
| p95 latency | < 1 s |
| NO_SUGGESTION rate | < 2% (excluding input_too_short) |
| Every missed distress case reviewed by SME | 100% |

## Phase 2: assisted pilot
* Small operator group sees suggestions with evidence; control group does not.
* Measure time-to-decision, agreement, overrides, false-alert handling time, trust and automation bias (include periodic blind cases).
* Stop rules: any missed real distress attributable to a LOW_PRIORITY_SUGGESTED hint triggers immediate pause and review.

## Phase 3: go/no-go
Safety owner decision against the frozen criteria; record in the model card.
