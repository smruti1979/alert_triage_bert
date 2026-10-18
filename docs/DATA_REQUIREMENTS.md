# Data Requirements

## Files
`train.csv`, `val.csv`, `test.csv`, split by beacon group and time (`triage.split.grouped_temporal_split`).

## Columns
| Column | Type | Notes |
|---|---|---|
| alert_id | str | internal, hashed |
| beacon_id | str | salted hash of the 15-hex ID; never store raw. Used for grouping only |
| timestamp | ISO datetime | first detection UTC |
| administration | str | originating MCC/RCC (slice + hold-out) |
| language | ISO 639-1 | verified, not only detected |
| beacon_type | EPIRB/ELT/PLB/other | |
| env_class | sea/air/land/unknown | |
| source_mix | typed/asr | whether any text came from ASR |
| text | str | PII-redacted, serialized (below) |
| label | 0/1 | 1 = DISTRESS_CONFIRMED, DISTRESS_PROBABLE or UNRESOLVED; 0 = FALSE_* |
| fine_label | str | full taxonomy (reporting only) |

## Serialization
`[META] <beacon_type> <env> <motion> registered=<bool>` then artifacts in time order as
`[OWNER_CALL] ...`, `[RCC_LOG] ...`, `[EMERGENCY_CONTACT]`, `[RADIO_RELAY]`, `[ASR_TRANSCRIPT]`, `[SITREP]`, `[EMAIL]`.

## Minimums (to make the gate provable)
* >= 299 positives in each of validation and test for a 99%/95% claim with zero misses. Plan for 3x.
* Each reported slice needs its own >= 299 positives or it will show INSUFFICIENT_DATA.
* Hold out at least one entire administration for a generalization check.

## Rules
* Outcome labels come from closure reports, not annotator opinion.
* Exclude cases without a closure-confidence flag from gate metrics (but keep for training if labels are reliable).
* Redact before the data leaves the administration's environment; keep a manual audit sample.
