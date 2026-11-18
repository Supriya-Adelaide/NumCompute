# NumCompute Stream (Assignment 2.2)

`numcompute_stream` extends the group's NumCompute package into a **streaming, decision-tree-based
learning framework**. Data arrives in chunks; every component (preprocessing, statistics, metrics,
the model and the pipeline that chains them) updates incrementally through `partial_fit()` or
`update()`. The model is a single classification tree written from scratch that learns from the
stream by **buffered retraining**.

Only **NumPy** and **matplotlib** are used by the package. (`pytest` and `jupyter` are needed to run
the tests and the notebook.)

## Repository layout

```
numcompute_stream/
├── __init__.py
├── _validation.py   # shared shape/type checks and NotFittedError
├── io.py            # CSV -> Frame (whole file or lazily in chunks), write_csv for logs
├── preprocessing.py # StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder (all partial_fit)
├── stats.py         # StreamingStats (Welford/Chan), StreamingQuantile, StreamingHistogram, update_stats
├── metrics.py       # Accuracy/Precision/Recall/F1/ConfusionMatrix/ROCAUC (update/result/reset), RollingMetric
├── tree.py          # DecisionTreeClassifier (gini/entropy, NaN-aware, buffered partial_fit)
├── pipeline.py      # Pipeline and ColumnTransformer with partial_fit
├── stream.py        # StreamTrainer (prequential loop, per-chunk logs), iter_chunks
└── visualise.py     # plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth, ...
tests/               # 93 pytest unit tests (90% line coverage)
demo/
├── stream_demo.ipynb   # the end-to-end demo (executed, outputs saved)
├── build_notebook.py   # source of the notebook
├── make_dataset.py     # regenerates demo/data/loan_stream.csv
├── data/loan_stream.csv
└── output/             # figures and the CSV log written by the notebook
benchmark/
├── loop_impl.py        # plain-Python reference implementations
├── run_benchmarks.py   # (a) loop vs vectorised, (b) buffer_size sweep
└── results/            # results.md, CSV tables and figures
```

## Setup

Python 3.9 or newer.

```bash
git clone <your-fork-url> && cd <repo>
git checkout <your-individual-branch>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running everything

```bash
pytest
pytest --cov=numcompute_stream

jupyter notebook demo/stream_demo.ipynb
jupyter nbconvert --to notebook --execute --inplace demo/stream_demo.ipynb

python benchmark/run_benchmarks.py
python benchmark/run_benchmarks.py --quick

python demo/make_dataset.py
```

## Quick start

```python
import numcompute_stream as ncs
from numcompute_stream import (Pipeline, ColumnTransformer, SimpleImputer, StandardScaler,
                               OneHotEncoder, DecisionTreeClassifier, StreamTrainer, visualise)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
for X_chunk, y_chunk in ncs.iter_chunks(X, y, chunk_size=200):
    pipe.partial_fit(X_chunk, y_chunk)

NUM = ["age", "income", "loan_amount", "credit_score", "debt_to_income", "years_employed", "late_payments"]
numeric = Pipeline([("impute", SimpleImputer("median")), ("scale", StandardScaler())])
pipe = Pipeline([
    ("prep", ColumnTransformer([("num", numeric, NUM),
                                ("cat", OneHotEncoder(), ["region", "loan_purpose"])])),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=1000)),
])
trainer = StreamTrainer(pipe, rolling_window=1000, verbose=True)
for chunk in ncs.iter_csv("demo/data/loan_stream.csv", chunk_size=250):
    trainer.step(chunk, chunk["default"].astype(int))        # score, then train

print(trainer.summary())
trainer.to_csv("log.csv")
visualise.plot_stream_dashboard(trainer.history, save_path="dashboard.png")
```

|                |                                                                                                                                                                                     |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shapes         | `X` is `(n_samples, n_features)` (1-D input becomes one column); `y` is `(n_samples,)`.                                                                                             |
| `axis`         | All statistics reduce over `axis=0` (rows) and return `(n_features,)`.                                                                                                              |
| Streaming      | Transformers/models: `partial_fit(chunk)` updates, `fit(X)` = `reset()` + `partial_fit(X)`. Stats: `update_stats(chunk)`. Metrics: `update(y_true, y_pred)`, `result()`, `reset()`. |
| Errors         | `TypeError` for non-numeric input, `ValueError` for shape/value problems (messages give the expected and actual shape), `NotFittedError` when state is needed but missing.          |
| Missing values | NaN is skipped by stats, ignored when scalers learn, filled by `SimpleImputer`, and routed by the tree (see below). Empty strings are missing categories.                           |

- `fit(X, y)` grows the tree top-down. At each node the stop rules are checked first (`max_depth`,
  `min_samples_split`, all labels equal); otherwise every threshold of every sampled feature
  (`max_features`) is scored by information gain with Gini or entropy, and the best one is kept if its
  gain is above `min_impurity_decrease`.
- The split search is fully vectorised: each column is sorted once, and a cumulative sum of one-hot
  labels gives the class counts left of every threshold at once (`O(n log n · d)` instead of
  `O(n² · d)` for the loop version).
- `partial_fit(X_chunk, y_chunk)` appends to a buffer. When the buffer holds at least `buffer_size`
  samples, a new tree is trained on the buffer and the buffer is cleared. Until then `predict()` uses
  the last tree. `flush()` forces a retrain on a partly filled buffer.
- **NaN in features:** the rows with a missing value go, as a group, to whichever child gives the
  higher gain, and that direction is stored per node and reused at prediction time.
- **Ties:** equal gains (within 1e-12) go to the lowest feature index, then the lowest threshold; a tied
  leaf predicts the first class in `classes_`. Results do not depend on row order.
- **Zero variance:** constant columns have no candidate thresholds; if all columns are constant the node
  is a leaf.
- **New classes / new columns mid-stream:** `classes_` grows as labels appear (probability 0 until
  the next retrain). If the feature width grows (new one-hot columns appended at the end), buffered
  rows are zero-padded, which is the right encoding for "category absent".

| Symptom                                                                                   | How it was found                                                                      | Fix                                                                                                                                                                              |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `StandardScaler was fitted with 15 features but got X with 16` when a new region appeared | first full run of the demo pipeline                                                   | Scale only the numeric block (nested pipeline inside `ColumnTransformer`) and put the growing `OneHotEncoder` last; added `test_column_transformer_rejects_growth_in_the_middle` |
| Predictions differed from a reference tree on about 0.3% of points                        | cross-check against an independent implementation on 40 random problems               | Traced to exact gain ties broken randomly by the reference; made our tie rule explicit and tested (`test_tie_breaking_is_deterministic_and_order_free`)                          |
| Split search accuracy                                                                     | `best_split` compared to `benchmark/loop_impl.best_split_loop` on data with many ties | Both agree on feature, threshold and gain (`test_vectorised_split_equals_loop_reference`)                                                                                        |
| "Latest chunk" plot looked too good (0.86)                                                | review of the demo                                                                    | It was scoring rows the tree had just trained on. The trainer now keeps the out-of-sample predictions (`last_y_pred_`) and the demo plots those                                  |
| A test that could never fail (`assert_allclose(...) and assert_allclose(...)`)            | reading test output                                                                   | Split into two assertions                                                                                                                                                        |

```bash
# 1. Fork the group repository on GitHub, then:
git clone https://github.com/<you>/<group-repo>.git && cd <group-repo>
git checkout -b <your-name>-assignment-2.2         # main stays as the group code
# 2. copy numcompute_stream/, tests/, demo/, benchmark/, README.md, requirements.txt,
#    pyproject.toml and .gitignore into the repo root
git add . && git commit -m "Assignment 2.2: streaming decision tree framework"
git push -u origin <your-name>-assignment-2.2
```
