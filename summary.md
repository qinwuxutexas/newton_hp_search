# Newton Hyperparameter Search: Methods, Setup, and Results

**Source:** Colab `newton_hp_search.ipynb` (run 7 April 2026), rebuilt at `C:\Users\qinwu\newton_hp_search`.  
**Task:** tune XGBoost hyperparameters with damped Gauss–Newton, compared with grid search, random search, and Optuna TPE.

The important efficiency fact: **`elapsed_sec` in the logs is one XGBoost fit, not total search time.** Search cost is number of fits × ~0.05–0.15 s on these datasets. Grid uses **320** fits. Newton logs **8–12 iterations**; if each iteration builds a finite-difference Jacobian over 4 hyperparameters, the true Newton budget is closer to **40+** fits, not 8.

---

## 1. Methods

All methods maximize **validation accuracy** of an `XGBClassifier` (`n_estimators=200`, `tree_method=hist`). Shared continuous space:

| Hyperparameter | Range | Notes |
|---|---|---|
| `learning_rate` | 0.01–0.30 | linear |
| `subsample` | 0.50–1.00 | linear |
| `colsample_bytree` | 0.50–1.00 | linear |
| `reg_lambda` | 1e-3–10 | log in random/TPE |

Optional extra keys in the config (`reg_alpha`, `max_depth`, `threshold`) were not in the main Newton/grid/random/TPE loops.

### 1.1 Proposed: damped Gauss–Newton (`run_newton_hpo`)

Vector objective (from the notebook):

\[
P(\phi)=\bigl[\mathrm{accuracy},\; 1/(1+\mathrm{logloss})\bigr],\qquad P^\star=[1,1].
\]

The solver (reconstructed; the Colab cell was deleted) treats HPO as

\[
\min_\phi \tfrac12\|P^\star-P(\phi)\|^2
\]

with:

- finite-difference Jacobian of \(P\) w.r.t. \(\phi\)
- Levenberg–Marquardt load `damping_lambda` (ablation used \(10^{-4}\) to \(1\))
- optional pull toward `phi_ref`
- backtracking line search, initial \(\alpha=0.2\)

Logged history columns: `iter`, `alpha`, `residual_norm`, metrics, and the four HPs. Default **8** Newton iterations on the single-split demo, **12** on the multi-dataset benchmark.

The original Colab **called** this API and saved outputs; the function body was missing and was rebuilt in `src/newton_hpo/search/newton.py`.

### 1.2 Baselines

| Method | How it searches | Budget in the saved runs |
|---|---|---|
| **Grid** | Cartesian product: lr `{0.03,0.05,0.08,0.10,0.15}` × subsample `{0.60,0.75,0.90,1.00}` × colsample `{0.60,0.75,0.90,1.00}` × `reg_lambda` `{0.01,0.10,1.0,5.0}` | **320** fits |
| **Random** | Uniform in bounds; `reg_lambda` log-uniform | **20** (demo) / **30** (benchmark) |
| **TPE** | Optuna `TPESampler` | **20** (demo) / **30** (benchmark) |

Grid was run on the breast-cancer demo split only. The full benchmark used `run_grid=False`.

---

## 2. Experimental setup

**Model.** XGBoost, 200 trees, sklearn `Pipeline` with median impute + `StandardScaler` (one-hot if categoricals exist).

**Data.** Public sklearn tables (Adult/OpenML exists in code, not used):

| Dataset | Task | Size | Features |
|---|---|---|---|
| breast cancer | binary | 569 | 30 numeric |
| wine | 3-class | 178 | 13 numeric |
| digits | 10-class | 1797 | 64 numeric |

**Split.** Stratified 80/20 train+valid vs test, then 75/25 of the remainder → train vs valid. The HPO score is **validation accuracy**. Test set is held out and was not used in the printed tables.

**Seeds.** Demo: 42. Benchmark: 42, 43, 44.

**Default start for Newton.** `learning_rate=0.10`, `subsample=0.80`, `colsample_bytree=0.80`, `reg_lambda=1.0`, `max_depth=6`.

**Hardware.** Colab CPU-style XGBoost `hist` fits; per-fit times below.

---

## 3. Results

### 3.1 Latency / cost (this is the important table)

`elapsed_sec` on a **single** breast-cancer fit from the demo `best_metrics`:

| Method | One-fit `elapsed_sec` |
|---|---|
| Newton (best iterate) | 0.076 |
| Grid (best config) | 0.088 |
| Random (best trial) | 0.155 |
| TPE (best trial) | 0.048 |

Those numbers are **not** search latency. Search cost ≈ (number of XGBoost fits) × (~0.08 s on breast cancer, ~0.5 s on digits).

| Method | Logged HPO steps | Estimated XGBoost fits | Est. breast-cancer wall clock |
|---|---|---|---|
| **Grid** | 320 configs | **320** | ~25–30 s |
| **Random (demo / bench)** | 20 / 30 | **20 / 30** | ~2–4 s |
| **TPE (demo / bench)** | 20 / 30 | **20 / 30** | ~1.5–4 s |
| **Newton (demo)** | 8 iterations | **8** if one fit/iter; **~40+** with 4-D finite-diff Jacobian + line search | ~0.6 s (naive) / **~3–5 s** (Jacobian) |
| **Newton (benchmark)** | 12 iterations | 12 or **~60+** with Jacobian | similar scaling |

**Takeaway.** Newton is cheaper than **grid** (8–12 steps vs 320). It is **not** shown to be cheaper than random or TPE: those use 20–30 fits, while a honest Gauss–Newton step evaluates the model once per parameter for the Jacobian. The notebook never printed total wall-clock, and `evals95` is not usable (see §3.3).

### 3.2 Quality: one split (breast cancer, seed 42)

This is the only place **all four** methods were printed together.

| Method | Val accuracy | Logloss | ROC-AUC |
|---|---|---|---|
| **Newton** (8 iters) | **0.9737** | **0.1149** | **0.9921** |
| **Grid** (320) | **0.9737** | 0.1222 | 0.9918 |
| Random (20) | 0.9649 | 0.1166 | 0.9908 |
| TPE (20) | 0.9649 | 0.1370 | 0.9882 |

Newton **ties grid on accuracy**, slightly better logloss/AUC, and does it without 320 grid evaluations. It beats random and TPE **on this split**.

Newton HPs stayed near the default: lr ≈ 0.0995, subsample ≈ 0.804, colsample ≈ 0.807, `reg_lambda` ≈ 1.019. Grid’s best was farther away (lr 0.05, subsample 0.75, colsample 0.6, `reg_lambda` 0.01) with the same accuracy.

### 3.3 Quality: multi-dataset benchmark (grid off)

Mean validation accuracy over seeds `{42,43,44}`. Random/TPE = 30 trials, Newton = 12 iterations.

| Dataset | Newton | Random | TPE |
|---|---|---|---|
| breast cancer | **0.977 ± 0.011** | 0.974 ± 0.007 | **0.977 ± 0.017** |
| wine | 0.991 ± 0.013 | **1.000 ± 0.000** | **1.000 ± 0.000** |
| digits | 0.970 ± 0.011 | 0.974 ± 0.012 | **0.977 ± 0.011** |

Newton does **not** win overall. Wine is saturated (many configs hit 100%). Digits favors TPE. Breast cancer: Newton matches TPE and slightly beats random.

The planned “time-to-95% of best” column (`evals95_mean`) is **1.0 for every method**. Default XGBoost is already near the best score on these sets, so that metric does not show Newton reaching the plateau faster.

### 3.4 Ablations (breast cancer, seed 42)

**Damping `lambda`:** scores stay at 0.9737 except `lambda=0.1` (0.9649). Logloss is best at `lambda=1.0` (0.110).

**Initialization:** a far start `(lr=0.03, subsample=0.60, …)` only reaches 0.9649; the default and a high start both reach 0.9737. Newton is init-sensitive.

**Dimension:** 2, 3, or 4 tuned HPs all reach 0.9737. Extra coordinates did not improve accuracy on this set.

---

## 4. How to read this

1. **Vs grid:** Newton is the interesting comparison. Same accuracy on the demo split, far fewer *iterations*, and slightly better logloss. That is the latency story that is actually supported.
2. **Vs random/TPE:** quality is mixed; speed is **not** shown to be better once Jacobian fits are counted.
3. **These datasets are too easy** (wine at 100%, breast cancer already ~0.96 with defaults). They cannot demonstrate sample-efficient HPO.
4. The reconstructed solver in `src/newton_hpo/search/newton.py` matches the logged API (`damping_lambda`, `phi_ref`, `param_order`, history schema). Exact original Jacobian/line-search code was not in the notebook.

Re-run:

```powershell
cd C:\Users\qinwu\newton_hp_search
python scripts/run_single.py --newton-iter 8 --trials 20
python scripts/run_benchmark.py --newton-iter 12 --trials 30 --grid
```

Pass `--grid` if you want the 320-config grid on every dataset/seed (slow, but that is the fair latency comparison).
