# Labeling Guide (draft v0.1, pilot on 300 alerts before freezing)

## Alert-level label (from closure outcome)
| Fine label | Binary |
|---|---|
| DISTRESS_CONFIRMED, DISTRESS_PROBABLE | 1 |
| UNRESOLVED (no determination possible) | 1 (must never be down-ranked) |
| FALSE_ACCIDENTAL, FALSE_TEST_MAINT, FALSE_EQUIP, FALSE_DISPOSAL_STORAGE | 0 |

## Text-level signals (explainability, error analysis)
SAFE_STATEMENT, DISTRESS_STATEMENT, UNCERTAIN, NO_CONTACT, ADMIN_ONLY.

## Rules
1. Negation is a label, not noise: "no distress" and "distress" are opposites.
2. Unreachable owner is a risk factor (NO_CONTACT), never a safe signal.
3. Conflicting artifacts: label each separately; alert label comes from the outcome record.
4. Spans are annotated on redacted text.
5. 10 to 15% double annotation. Targets: kappa >= 0.75 (signals), span F1 >= 0.85 (entities). SME adjudicates.

## Worked examples
| Text | Signal |
|---|---|
| "Owner says beacon knocked during stowing, all persons safe." | SAFE_STATEMENT |
| "Persons are not safe, two in the water." | DISTRESS_STATEMENT |
| "Left voicemail, awaiting callback." | NO_CONTACT |
| "Vessel safe in harbour but crew member injured." | UNCERTAIN (mixed; escalate) |
| "Not a false alarm." | DISTRESS_STATEMENT (negation of false alarm) |
