

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# NumCompute Stream: streaming decision-tree demo

This notebook runs the full workflow the brief asks for:

1. load a CSV with the package's own I/O module (`numcompute_stream.io`);
2. split it into chunks to simulate data arriving over time;
3. train a preprocessing + decision-tree pipeline incrementally with `.partial_fit()`;
4. log metrics per chunk with `StreamTrainer` and plot them with `visualise.py`.

**Data.** `data/loan_stream.csv` holds 12,000 loan applications in time order
(regenerate it with `python demo/make_dataset.py`). It has missing values, a new
region category (`online`) that only appears after row 4,000, and a **change in the
underlying rule at row 7,000** (concept drift), so we can watch the buffered tree
recover after the drift."""),
    code("""import sys, pathlib
ROOT = pathlib.Path.cwd().resolve()
ROOT = ROOT.parent if ROOT.name == "demo" else ROOT
sys.path.insert(0, str(ROOT))                       # use the package from this repo

import numpy as np
import matplotlib.pyplot as plt
import numcompute_stream as ncs
from numcompute_stream import (io, visualise, Pipeline, ColumnTransformer, SimpleImputer,
                               StandardScaler, OneHotEncoder, DecisionTreeClassifier,
                               StreamTrainer, StreamingStats, StreamingHistogram)

%matplotlib inline
DATA = ROOT / "demo" / "data" / "loan_stream.csv"
OUT = ROOT / "demo" / "output"; OUT.mkdir(exist_ok=True)
print("numcompute_stream", ncs.__version__, "| numpy", np.__version__)"""),

    md("## 1. Load the CSV with the custom I/O pipeline"),
    code("""frame = io.read_csv(DATA)
print(frame)
NUM = frame.numeric_columns(); NUM.remove("default")
CAT = frame.categorical_columns()
TARGET = "default"
print("numeric:", NUM)
print("categorical:", CAT)

missing = {c: int(np.isnan(frame[c]).sum()) for c in NUM}
missing.update({c: int((frame[c] == "").sum()) for c in CAT})
print("missing cells per column:", {k: v for k, v in missing.items() if v})"""),

    md("""## 2. Split into chunks and look at the stream

`io.iter_csv` reads the file lazily, 250 rows at a time, so the data never has to be
in memory all at once. While streaming we update running statistics with
`StreamingStats.update_stats()` and watch the default rate per chunk."""),
    code("""CHUNK = 250
stats = StreamingStats()
hist = StreamingHistogram(bins=20, range=(0, 1), window=8)   # debt-to-income, last 8 chunks
default_rate = []
for chunk in io.iter_csv(DATA, CHUNK):
    stats.update_stats(chunk.to_array(NUM))
    hist.update_stats(chunk["debt_to_income"])
    default_rate.append(chunk[TARGET].mean())

print(f"{len(default_rate)} chunks of {CHUNK} rows")
for name, m, s, n in zip(NUM, stats.mean, stats.std(), stats.count_):
    print(f"  {name:<15} mean={m:9.3f}  std={s:8.3f}  observed={n}")
check = frame.to_array(NUM)
print("matches one-pass NumPy:", np.allclose(stats.mean, np.nanmean(check, axis=0)))"""),
    code("""fig, ax = visualise.plot_metric_over_time(default_rate, "Default rate per chunk", "share defaulted",
                                         retrain_points=[7000 // CHUNK])
ax.legend(["default rate", "drift point (row 7,000)"], frameon=False, fontsize=8)
plt.show()"""),

    md("""## 3. Build the pipeline and train it with `.partial_fit()`

Numeric columns are imputed (running median) and scaled (running mean/variance);
categorical columns are one-hot encoded with categories added as they appear. The
encoder goes last so new columns are appended at the end. The tree retrains every
`buffer_size` samples."""),
    code("""def make_pipeline(buffer_size=1000, max_depth=5):
    numeric = Pipeline([("impute", SimpleImputer("median", random_state=0)),
                        ("scale", StandardScaler())])
    prep = ColumnTransformer([("num", numeric, NUM), ("cat", OneHotEncoder(), CAT)])
    return Pipeline([("prep", prep),
                     ("model", DecisionTreeClassifier(max_depth=max_depth, buffer_size=buffer_size))])

pipe = make_pipeline(buffer_size=1000)
pipe"""),
    code("""# the plain loop the brief describes: one partial_fit per chunk
for i, chunk in enumerate(io.iter_csv(DATA, CHUNK)):
    pipe.partial_fit(chunk, chunk[TARGET].astype(int))
    tree = pipe["model"]
    if i < 10 or tree.buffer_fill == 0 and i % 8 == 3:
        print(f"chunk {i:>2}: retrains={tree.n_retrains_}  buffer={tree.buffer_fill:>4}  "
              f"one-hot columns={pipe['prep'].transformers[1][1].n_features_out_}")
print("final one-hot categories:", pipe["prep"].transformers[1][1].categories_)"""),

    md("""## 4. Prequential run with logging (`StreamTrainer`)

Each chunk is **scored first, then used for training**, so every prediction is made
on data the model has not seen. The log records chunk, cumulative and rolling
accuracy, AUC, fit time, retrains, buffer fill and model memory."""),
    code("""trainer = StreamTrainer(make_pipeline(buffer_size=1000), rolling_window=1000,
                        name="tree b=1000", verbose=True)
for chunk in io.iter_csv(DATA, CHUNK):
    trainer.step(chunk, chunk[TARGET].astype(int))
trainer.to_csv(OUT / "stream_log_b1000.csv")
trainer.summary()"""),
    code("""h = trainer.history
fig, ax = visualise.plot_metric_over_time(h["rolling_accuracy"], "Rolling accuracy (last 1,000 samples)",
                                         "accuracy", retrain_points=trainer.retrain_chunks(),
                                         ylim=(0.5, 0.9), save_path=OUT / "rolling_accuracy.png")
ax.axvline(7000 // CHUNK, color="#e34948", lw=1.5); ax.text(7000 // CHUNK + 0.4, 0.52, "drift", color="#e34948", fontsize=8)
plt.show()"""),
    code("""fig, axes = visualise.plot_stream_dashboard(h, title="Streaming run, buffer_size = 1000",
                                           save_path=OUT / "dashboard.png")
plt.show()"""),

    md("""## 5. Comparing buffer sizes

A small buffer retrains often on little data (fast to adapt, noisier); a large buffer
trains on more data but reacts slowly to drift."""),
    code("""runs = {}
for b in (250, 1000, 4000):
    t = StreamTrainer(make_pipeline(buffer_size=b), rolling_window=1000, name=f"b={b}")
    for chunk in io.iter_csv(DATA, CHUNK):
        t.step(chunk, chunk[TARGET].astype(int))
    runs[b] = t
    s = t.summary()
    print(f"buffer {b:>5}: cumulative acc={s['final_accuracy']:.3f}  retrains={s['retrains']:>2}  "
          f"total fit time={s['total_fit_time_s']:.2f}s")

fig, ax = visualise.compare_models(runs[1000].history["rolling_accuracy"],
                                   runs[4000].history["rolling_accuracy"],
                                   labels=("buffer_size=1000", "buffer_size=4000"),
                                   title="Rolling accuracy: frequent vs rare retraining",
                                   ylim=(0.5, 0.9), save_path=OUT / "compare_buffers.png")
ax.axvline(7000 // CHUNK, color="#e34948", lw=1.5)
plt.show()"""),

    md("""## 6. Predictions vs ground truth on the latest chunk

The trainer keeps the predictions it made for the last chunk *before* training on it,
so this is an honest out-of-sample view (predicting again now would score the tree on
rows it has just been trained on)."""),
    code("""y_true, y_pred = trainer.last_y_true_, trainer.last_y_pred_
fig, axes = visualise.plot_predictions_vs_ground_truth(y_true, y_pred, labels=[0, 1],
                                                      save_path=OUT / "predictions.png")
plt.show()
print(ncs.metrics.confusion_matrix(y_true, y_pred))"""),

    md("## 7. What the current tree learned"),
    code("""names = trainer.model["prep"].get_feature_names_out()
tree = trainer.model["model"]
print(f"depth={tree.get_depth()}  leaves={tree.get_n_leaves()}  trained on the last {tree.n_node_samples_[0]} samples")
print(tree.export_text(names)[:1800])
fig, ax = visualise.plot_feature_importances(names, tree.feature_importances_,
                                             title="Feature importance after drift",
                                             save_path=OUT / "importances.png")
plt.show()"""),
    md("""After the drift the tree leans on `debt_to_income`, `late_payments` and
`region=online`, which is the new rule in the data; the credit score that drove the
earlier rule has dropped down the list. The buffered retrain is what let the model
pick this up."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("stream_demo.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
