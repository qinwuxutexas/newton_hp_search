# Newton HPO

Damped Gauss-Newton hyperparameter search for ML such as XGBoost and SVM.

## Method

Tune `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda` (optional extra keys) by Gauss-Newton on

```
P(phi) = [accuracy, 1 / (1 + logloss)]
minimize ||[1, 1] - P(phi)||^2
```

Jacobian is finite-difference. `damping_lambda` is Levenberg–Marquardt load. Optional `phi_ref` pulls toward a reference point. Line search starts at `alpha=0.2`.

## Layout

```
src/newton_hpo/          package
  config.py
  data.py
  evaluate.py
  search/newton.py       reconstructed solver
  search/grid.py
  search/random.py
  search/bayes.py
  benchmark.py
  ablations.py
  plotting.py
  cli.py
scripts/                 entry points
tests/
notebooks/               original Colab notebook
```

## Setup

```powershell
cd C:\Users\qinwu\newton_hp_search
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run

```powershell
python scripts/smoke_test.py
python scripts/run_single.py --newton-iter 8 --trials 20
python scripts/run_benchmark.py --newton-iter 12 --trials 30
python scripts/run_ablations.py
pytest
```

`run_benchmark.py` trains many XGBoost models (breast cancer, wine, digits × 3 seeds). Start with `run_single.py`.
