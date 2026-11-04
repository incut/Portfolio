# Lead scoring: executable synthetic demonstration

Run two leakage-aware model pipelines without downloading private data or models.
This is a standalone teaching implementation, not the deployed API or a reproduction
of its historical 28-feature artifact. It never calls a service or opens private files.

## Quick start

Tested with Python 3.13.15 and the dependency versions in `requirements.txt`.
From the repository root, use an isolated environment, not the forecasting environment:

```powershell
git clone https://github.com/incut/Portfolio.git
cd Portfolio/proyectos/lead-scoring/technical
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -B -m unittest -v
.venv/Scripts/python -B demo.py --output synthetic_metrics.json
```

On Linux/macOS replace `.venv/Scripts/python` with `.venv/bin/python`.
Installation in a fresh environment has not been tested, execution was checked in
the existing environment with these versions. Expect 100 small model fits per run.
Existing output files are never overwritten. Synthetic data stays in memory.

## Method and evaluation

`generate_synthetic_data.py` creates 21 fields matching the raw schema by name and
broad type. Values and distributions are invented, not anonymized source records.
`demo.py` compares four documented form fields with those fields plus last activity
and five numeric activity/scores. IDs and contact restrictions never enter either model.

Both use identical five outer stratified folds. Three inner folds choose logistic
L1 regularization C from 0.1, 1, 10 by ROC-AUC. Imputation, encoding and scaling fit
inside each fold. There is no global feature selection or activity-based row filter.
ROC-AUC uses probabilities, lift@10% uses ceil(0.10*n), source-order ties and
top-decile precision divided by population prevalence. Aggregate OOF metrics mix
several fitted models, fold metrics are included separately. No reserve is evaluated.

## What this does not establish

Synthetic scores prove code execution only, not real-data performance. See the
[real training comparison](../training_comparison.md), a different evidence source.
`SMS Sent`, activity aggregates, scores and their missingness indicators are not
verified as available at form submission. The expanded scenario is not endorsed for
operational use. See [availability audit](../availability.md). No causal sales,
calibration, ROI or production-readiness claim is made. No model is serialized.
