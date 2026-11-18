

from __future__ import annotations

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

__all__ = [
    "PALETTE",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "plot_stream_dashboard",
    "plot_feature_importances",
    "plot_benchmark",
]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=_INK, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=_INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK2, fontsize=9)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)


def _new_ax(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def _finish(fig, save_path, show, dpi=150):
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()


def _x(n, x):
    return np.arange(n) if x is None else np.asarray(x)


def _retrain_lines(ax, retrain_points, color=_INK2, label="retrain"):
    if retrain_points is None or len(retrain_points) == 0:
        return
    for i, r in enumerate(np.asarray(retrain_points)):
        ax.axvline(r, color=color, linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.6,
                   label=label if i == 0 else None)


def plot_metric_over_time(metric_values, title: str = "Metric over time",
                          ylabel: str = "Accuracy", *, xlabel: str = "Chunk", x=None,
                          retrain_points=None, label: str | None = None, ylim=None,
                          ax=None, save_path=None, show: bool = False, figsize=(7, 3.2)):
 
    y = np.asarray(metric_values, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"metric_values must be 1-D, got shape {y.shape}.")
    fig, ax = _new_ax(ax, figsize)
    xs = _x(len(y), x)
    ax.plot(xs, y, color=PALETTE[0], linewidth=2, label=label)
    ax.plot(xs, y, "o", color=PALETTE[0], markersize=3.5, markeredgecolor="white",
            markeredgewidth=0.8)
    _retrain_lines(ax, retrain_points)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style(ax, title, xlabel, ylabel)
    if label or retrain_points is not None:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    _finish(fig, save_path, show)
    return fig, ax


def compare_models(metric1, metric2, labels=("model A", "model B"), *,
                   title: str = "Model comparison", ylabel: str = "Accuracy",
                   xlabel: str = "Chunk", x=None, ylim=None, ax=None,
                   save_path=None, show: bool = False, figsize=(7, 3.2)):
   
    a = np.asarray(metric1, dtype=float)
    b = np.asarray(metric2, dtype=float)
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two names.")
    fig, ax = _new_ax(ax, figsize)
    for vals, lab, col in ((a, labels[0], PALETTE[0]), (b, labels[1], PALETTE[1])):
        xs = _x(len(vals), x if x is not None and len(x) == len(vals) else None)
        ax.plot(xs, vals, color=col, linewidth=2, label=lab)
        last = np.flatnonzero(~np.isnan(vals))
        if last.size:
            i = last[-1]
            ax.annotate(f"{vals[i]:.3f}", (xs[i], vals[i]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=_INK2, va="center")
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, *, title: str = "Latest chunk: predictions vs ground truth",
                                     labels=None, max_points: int = 120,
                                     save_path=None, show: bool = False, figsize=(9, 3.4)):
    
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred differ in length: {yt.shape[0]} vs {yp.shape[0]}.")
    if yt.size == 0:
        raise ValueError("Nothing to plot: y_true is empty.")
    if yt.dtype.kind in "US" or yp.dtype.kind in "US":
        yt, yp = yt.astype(str), yp.astype(str)
    labs = np.unique(np.concatenate([yt, yp])) if labels is None else np.asarray(labels)
    k = len(labs)
    i, j = np.searchsorted(labs, yt), np.searchsorted(labs, yp)
    cm = np.bincount(i * k + j, minlength=k * k).reshape(k, k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 2.2]})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f6fb", PALETTE[0]])
    ax1.imshow(cm, cmap=cmap)
    for r in range(k):
        for c in range(k):
            ax1.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=10,
                     color="white" if cm[r, c] > cm.max() * 0.55 else _INK)
    ax1.set_xticks(range(k), [str(l) for l in labs])
    ax1.set_yticks(range(k), [str(l) for l in labs])
    ax1.set_xlabel("predicted", color=_INK2, fontsize=9)
    ax1.set_ylabel("actual", color=_INK2, fontsize=9)
    acc = float(np.mean(yt == yp))
    ax1.set_title(f"accuracy {acc:.3f}  (n={yt.size})", loc="left", fontsize=10, color=_INK)
    ax1.tick_params(colors=_INK2, labelsize=8)

    n = min(max_points, yt.size)
    xs = np.arange(n)
    ax2.scatter(xs, i[:n], s=46, facecolors="none", edgecolors=_INK2, linewidths=1,
                label="actual")
    ax2.scatter(xs, j[:n], s=16, color=PALETTE[0], label="predicted")
    wrong = i[:n] != j[:n]
    ax2.scatter(xs[wrong], j[:n][wrong], s=40, marker="x", color=PALETTE[1],
                linewidths=1.4, label="wrong")
    ax2.set_yticks(range(k), [str(l) for l in labs])
    ax2.set_ylim(-0.6, k - 0.4)
    _style(ax2, f"first {n} samples", "sample in chunk", "class")
    ax2.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.22))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=_INK)
    _finish(fig, save_path, show)
    return fig, (ax1, ax2)


def plot_stream_dashboard(history: dict, *, title: str = "Streaming run",
                          save_path=None, show: bool = False, figsize=(9, 6.4)):
   
    if not history:
        raise ValueError("history is empty: run the trainer first.")
    x = history["chunk"]
    retr = x[np.asarray(history["retrained"], dtype=bool)]
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    ax = axes[0]
    ax.plot(x, history["chunk_accuracy"], color=PALETTE[0], linewidth=1.2, alpha=0.55,
            label="chunk accuracy")
    ax.plot(x, history["cumulative_accuracy"], color=PALETTE[0], linewidth=2.2,
            label="cumulative accuracy")
    if "rolling_accuracy" in history:
        ax.plot(x, history["rolling_accuracy"], color=PALETTE[1], linewidth=2,
                label="rolling accuracy")
    _retrain_lines(ax, retr)
    _style(ax, title, None, "accuracy")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right")

    ax = axes[1]
    ax.bar(x, np.asarray(history["fit_time"]) * 1e3, color=PALETTE[2], width=0.8)
    _retrain_lines(ax, retr, label=None)
    _style(ax, None, None, "fit time (ms)")

    ax = axes[2]
    ax.plot(x, np.asarray(history["memory_bytes"]) / 1024, color=PALETTE[4], linewidth=2)
    _retrain_lines(ax, retr, label=None)
    _style(ax, None, "chunk", "model memory (KiB)")
    _finish(fig, save_path, show)
    return fig, axes


def plot_feature_importances(names, importances, *, title: str = "Feature importance",
                             top: int = 12, ax=None, save_path=None, show: bool = False,
                             figsize=(6, 3.4)):
    imp = np.asarray(importances, dtype=float)
    names = np.asarray(names)
    if imp.shape[0] != names.shape[0]:
        raise ValueError("names and importances must have the same length.")
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = _new_ax(ax, figsize)
    ax.barh(np.arange(order.size), imp[order], color=PALETTE[0], height=0.6)
    ax.set_yticks(np.arange(order.size), names[order])
    for k, v in enumerate(imp[order]):
        ax.text(v, k, f" {v:.2f}", va="center", fontsize=8, color=_INK2)
    _style(ax, title, "share of impurity decrease", None)
    ax.grid(axis="y", visible=False)
    _finish(fig, save_path, show)
    return fig, ax


def plot_benchmark(sizes, loop_times, vec_times, *, title: str = "Loop vs vectorised",
                   xlabel: str = "input size", ax=None, save_path=None, show: bool = False,
                   figsize=(5, 3.2)):
    fig, ax = _new_ax(ax, figsize)
    ax.loglog(sizes, loop_times, "o-", color=PALETTE[1], linewidth=2, markersize=5,
              label="Python loop")
    ax.loglog(sizes, vec_times, "o-", color=PALETTE[0], linewidth=2, markersize=5,
              label="vectorised NumPy")
    _style(ax, title, xlabel, "seconds")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, save_path, show)
    return fig, ax
