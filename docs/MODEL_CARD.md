# Model Card (template; fill from `operating_point.json` and slice reports)

* **Name / version:** `<model_version from operating_point.json>`
* **Intended use:** advisory prioritization hints for MCC/RCC operators on 406 MHz alerts using accompanying text.
* **Out of scope:** any automatic action; suppression or delay of alerts; processing of the 406 MHz bit payload; use outside validated languages and administrations.
* **Architecture:** `<baseline TF-IDF+LR | xlm-roberta-base fine-tuned>`; temperature `<T>`; threshold `<t>` chosen on validation under recall gate `<target>` @ `<1-alpha>` confidence.
* **Training data:** `<administrations, date range, n alerts, n positives, languages>`; label source: closure reports.
* **Evaluation:** recall (+ lower bound), low-priority fraction of false alerts, AUROC, ECE; per-slice table; negation tests; temporal and administration hold-out.
* **Known limitations:** `<languages/slices with INSUFFICIENT_DATA or FAIL>`; ASR text degrades performance; rare event types underrepresented; label conventions vary by administration.
* **Failure behaviour:** any error/timeout/empty/oversized input => NO_SUGGESTION; hard keywords force PRIORITIZE.
* **Monitoring:** weekly recall and calibration on newly closed cases; drift triggers; rollback procedure.
* **Ethical/privacy:** hashed identifiers, redacted text, audit log without raw text by default.
* **Sign-offs:** safety owner `<name/date>`, operations lead `<name/date>`, privacy officer `<name/date>`.
