
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.loop_impl import (best_split_loop, confusion_matrix_loop,  
                                 predict_loop, welford_loop)
from numcompute_stream import (ColumnTransformer, DecisionTreeClassifier, OneHotEncoder,  
                               Pipeline, SimpleImputer, StandardScaler, StreamingStats,
                               StreamTrainer, io, visualise)
from numcompute_stream.metrics import ConfusionMatrix  
from numcompute_stream.tree import best_split  

OUT = ROOT / "benchmark" / "results"
DATA = ROOT / "demo" / "data" / "loan_stream.csv"
DRIFT_AT = 7_000


def timeit(fn, repeat=3):
    best, out = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def bench_vectorisation(quick=False):
    rng = np.random.default_rng(0)
    rows = []

    split_sizes = [100, 200, 400] if quick else [100, 200, 400, 800, 1600]
    for n in split_sizes:
        X = rng.normal(size=(n, 6))
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        tl, ref = timeit(lambda: best_split_loop(X, y, 2), repeat=1 if n >= 800 else 2)
        tv, got = timeit(lambda: best_split(X, y, 2))
        assert (got.feature, got.threshold) == ref[:2] and np.isclose(got.gain, ref[2])
        rows.append(("best split search", n, tl, tv))

    sizes = [1_000, 10_000] if quick else [1_000, 10_000, 100_000]
    for n in sizes:
        X = rng.normal(5, 2, size=(n, 5))
        tl, (m, v) = timeit(lambda: welford_loop(X), repeat=1)
        tv, s = timeit(lambda: StreamingStats().update_stats(X))
        assert np.allclose(s.mean, m) and np.allclose(s.var(), v)
        rows.append(("running mean/variance", n, tl, tv))

    for n in sizes:
        yt, yp = rng.integers(0, 4, n), rng.integers(0, 4, n)
        tl, ref = timeit(lambda: confusion_matrix_loop(yt, yp, [0, 1, 2, 3]), repeat=1)
        tv, cm = timeit(lambda: ConfusionMatrix().update(yt, yp).result())
        assert np.array_equal(ref, cm)
        rows.append(("confusion matrix", n, tl, tv))

    Xtr = rng.normal(size=(2000, 6))
    tree = DecisionTreeClassifier(max_depth=6).fit(Xtr, (Xtr[:, 0] * Xtr[:, 1] > 0).astype(int))
    for n in sizes:
        X = rng.normal(size=(n, 6))
        tl, ref = timeit(lambda: predict_loop(tree, X), repeat=1)
        tv, got = timeit(lambda: tree.predict(X))
        assert np.array_equal(ref, got)
        rows.append(("tree predict", n, tl, tv))
    return rows


NUM = ["age", "income", "loan_amount", "credit_score", "debt_to_income",
       "years_employed", "late_payments"]
CAT = ["region", "loan_purpose"]


def make_pipeline(buffer_size, max_depth=5):
    num = Pipeline([("impute", SimpleImputer("median", random_state=0)),
                    ("scale", StandardScaler())])
    prep = ColumnTransformer([("num", num, NUM), ("cat", OneHotEncoder(), CAT)])
    return Pipeline([("prep", prep),
                     ("model", DecisionTreeClassifier(max_depth=max_depth, buffer_size=buffer_size))])


def run_stream(buffer_size, chunk=100):
    tr = StreamTrainer(make_pipeline(buffer_size), rolling_window=1000, name=f"b={buffer_size}")
    for f in io.iter_csv(DATA, chunk):
        tr.step(f, f["default"].astype(int))
    return tr


def bench_buffer(quick=False, chunk=100):
    buffers = [250, 1000, 4000] if quick else [100, 250, 500, 1000, 2000, 4000]
    rows, rolling = [], {}
    for b in buffers:
        runs = [run_stream(b, chunk) for _ in range(1 if quick else 3)]   
        tr = runs[0]
        h = tr.history
        start = np.cumsum(h["n_samples"]) - h["n_samples"]
        pre = start < DRIFT_AT
        post = start >= DRIFT_AT + 2000                   
        mean_acc = lambda m: float(np.nansum(h["chunk_accuracy"][m] * h["n_samples"][m])
                                   / h["n_samples"][m & np.isfinite(h["chunk_accuracy"])].sum())
        drift_window = (start >= DRIFT_AT) & (start < DRIFT_AT + 2000)
        rt = np.median([np.sum(r.model["model"].retrain_times_) for r in runs])
        per = np.median([np.mean(r.model["model"].retrain_times_) for r in runs])
        rows.append({
            "buffer_size": b,
            "retrains": tr.model["model"].n_retrains_,
            "cumulative_accuracy": tr.summary()["final_accuracy"],
            "macro_f1": tr.summary()["final_f1_macro"],
            "acc_before_drift": mean_acc(pre),
            "acc_first_2000_after_drift": mean_acc(drift_window),
            "acc_after_recovery": mean_acc(post),
            "unscored_samples": int(h["n_samples"][np.isnan(h["chunk_accuracy"])].sum()),
            "total_retrain_s": float(rt),
            "mean_retrain_ms": float(per * 1e3),
            "peak_model_kib": float(np.nanmax(h["memory_bytes"]) / 1024),
        })
        rolling[b] = h["rolling_accuracy"]
        print(f"  buffer {b:>5}: acc={rows[-1]['cumulative_accuracy']:.3f}  "
              f"retrains={rows[-1]['retrains']:>3}  retrain total={rt:.3f}s")
    return rows, rolling


def write_outputs(vec_rows, buf_rows, rolling, chunk):
    OUT.mkdir(parents=True, exist_ok=True)
    io.write_csv(OUT / "vectorisation.csv", {
        "operation": [r[0] for r in vec_rows], "n": [r[1] for r in vec_rows],
        "loop_s": [r[2] for r in vec_rows], "vectorised_s": [r[3] for r in vec_rows],
        "speedup": [r[2] / r[3] for r in vec_rows]})
    io.write_csv(OUT / "buffer_size.csv", {k: [r[k] for r in buf_rows] for k in buf_rows[0]})

    ops = list(dict.fromkeys(r[0] for r in vec_rows))
    fig, axes = plt.subplots(1, len(ops), figsize=(12, 2.9), constrained_layout=True)
    for ax, op in zip(axes, ops):
        sel = [r for r in vec_rows if r[0] == op]
        visualise.plot_benchmark([r[1] for r in sel], [r[2] for r in sel], [r[3] for r in sel],
                                 title=op, xlabel="n samples", ax=ax)
        last = sel[-1]
        ax.annotate(f"{last[2] / last[3]:,.0f}x faster", (last[1], last[3]), xytext=(-6, 10),
                    textcoords="offset points", ha="right", fontsize=8, color="#52514e")
        if ax is not axes[0]:
            ax.get_legend().remove()
    fig.savefig(OUT / "vectorisation.png", dpi=170, bbox_inches="tight")

    b = [r["buffer_size"] for r in buf_rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    for key, lab, col in (("acc_before_drift", "before drift", visualise.PALETTE[0]),
                          ("acc_first_2000_after_drift", "first 2,000 after drift", visualise.PALETTE[1]),
                          ("acc_after_recovery", "after recovery", visualise.PALETTE[2])):
        a1.plot(b, [r[key] for r in buf_rows], "o-", color=col, lw=2, ms=5, label=lab)
    a1.set_xscale("log"); a1.set_xticks(b, [str(x) for x in b])
    visualise._style(a1, "Prequential accuracy by buffer_size", "buffer_size", "accuracy")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(b, [r["total_retrain_s"] * 1e3 for r in buf_rows], "o-", color=visualise.PALETTE[0],
            lw=2, ms=5, label="total retraining time")
    a2.plot(b, [r["mean_retrain_ms"] for r in buf_rows], "o-", color=visualise.PALETTE[1],
            lw=2, ms=5, label="time per retrain")
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xticks(b, [str(x) for x in b])
    visualise._style(a2, "Retraining cost by buffer_size", "buffer_size", "milliseconds")
    a2.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "buffer_tradeoff.png", dpi=170, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 3.1), constrained_layout=True)
    shown = [x for x in (250, 1000, 4000) if x in rolling]
    for col, bsz in zip(visualise.PALETTE, shown):
        ax.plot(np.arange(len(rolling[bsz])) * chunk, rolling[bsz], color=col, lw=2,
                label=f"buffer_size={bsz}")
    ax.axvline(DRIFT_AT, color="#e34948", lw=1.4)
    ax.text(DRIFT_AT + 120, 0.56, "concept drift", color="#e34948", fontsize=8)
    ax.set_ylim(0.55, 0.88)
    visualise._style(ax, "Rolling accuracy (last 1,000 samples) through the drift",
                     "samples seen", "accuracy")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "rolling_by_buffer.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    lines = ["# Benchmark results", "",
             f"Generated by `benchmark/run_benchmarks.py` with NumPy {np.__version__}.", "",
             "## A. Loop vs vectorised (best-of-3 wall clock, outputs checked identical)", "",
             "| operation | n | loop (s) | vectorised (s) | speed-up |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {o} | {n:,} | {tl:.4f} | {tv:.5f} | {tl / tv:,.0f}x |" for o, n, tl, tv in vec_rows]
    lines += ["", f"## B. buffer_size sweep (12,000-row stream, chunks of {chunk}, drift at row 7,000)", "",
              "| buffer | retrains | cumulative acc | before drift | 2k after drift | after recovery | "
              "retrain total (s) | per retrain (ms) | peak model KiB |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['buffer_size']} | {r['retrains']} | {r['cumulative_accuracy']:.3f} | "
              f"{r['acc_before_drift']:.3f} | {r['acc_first_2000_after_drift']:.3f} | "
              f"{r['acc_after_recovery']:.3f} | {r['total_retrain_s']:.3f} | "
              f"{r['mean_retrain_ms']:.1f} | {r['peak_model_kib']:.0f} |" for r in buf_rows]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Benchmarks for numcompute_stream.")
    ap.add_argument("--quick", action="store_true", help="small sizes for a fast smoke test")
    args = ap.parse_args()
    chunk = 100
    print("A. loop vs vectorised")
    vec = bench_vectorisation(args.quick)
    for o, n, tl, tv in vec:
        print(f"  {o:<22} n={n:>7}: loop {tl:8.4f}s  vectorised {tv:9.5f}s  x{tl / tv:,.0f}")
    print("B. buffer_size sweep")
    buf, rolling = bench_buffer(args.quick, chunk)
    write_outputs(vec, buf, rolling, chunk)
    print(f"results written to {OUT}")


if __name__ == "__main__":
    main()
