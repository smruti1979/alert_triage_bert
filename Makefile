.PHONY: install synth baseline bert test serve
install:   ; pip install -r requirements-dev.txt
synth:     ; python -m triage.synth --n 20000 --out data
baseline:  ; python -m triage.baseline --data data --out artifacts/baseline
bert:      ; python -m triage.train_bert --data data --out artifacts/bert --model xlm-roberta-base
test:      ; python -m pytest -q
serve:     ; OPERATING_POINT=artifacts/baseline/operating_point.json uvicorn triage.serve:app --port 8080
