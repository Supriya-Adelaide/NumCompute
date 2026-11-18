
import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream import (ColumnTransformer, DecisionTreeClassifier, OneHotEncoder,
                               Pipeline, SimpleImputer, StandardScaler, StreamTrainer,
                               iter_chunks, iter_csv, read_csv, visualise, write_csv)


def test_pipeline_partial_fit_spec_example(blobs):
    X, y = blobs
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500))])
    for Xc, yc in iter_chunks(X, y, 200):
        pipe.partial_fit(Xc, yc)
    assert pipe["model"].n_retrains_ == 2            
    assert pipe["scale"].n_samples_seen_[0] == 1200   
    assert pipe.score(X, y) > 0.95
    assert pipe.memory_footprint() > 0


def test_pipeline_fit_equals_manual_chain(blobs):
    X, y = blobs
    pipe = Pipeline([("imp", SimpleImputer()), ("scale", StandardScaler()),
                     ("tree", DecisionTreeClassifier(max_depth=3))]).fit(X, y)
    Z = StandardScaler().fit_transform(SimpleImputer().fit_transform(X))
    manual = DecisionTreeClassifier(max_depth=3).fit(Z, y)
    np.testing.assert_array_equal(pipe.predict(X), manual.predict(Z))
    assert pipe.predict_proba(X).shape == (len(X), 2)


def test_pipeline_validation():
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(TypeError, match="partial_fit"):
        Pipeline([("bad", object()), ("tree", DecisionTreeClassifier())])
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("a", StandardScaler()), ("a", DecisionTreeClassifier())])


def test_column_transformer_mixed_types_with_growing_categories(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("x,colour,y\n1,red,0\n2,,0\n3,blue,1\n4,green,1\n")
    num = Pipeline([("imp", SimpleImputer()), ("scale", StandardScaler())])
    ct = ColumnTransformer([("num", num, ["x"]), ("cat", OneHotEncoder(), ["colour"])])
    chunks = list(iter_csv(path, chunk_size=2))
    out1 = ct.partial_fit(chunks[0]).transform(chunks[0])
    out2 = ct.partial_fit(chunks[1]).transform(chunks[1])
    assert out1.shape == (2, 2) and out2.shape == (2, 4)      
    np.testing.assert_array_equal(out1[1, 1:], [0])             
    assert ct.get_feature_names_out() == ["x", "colour=red", "colour=blue", "colour=green"]


def test_column_transformer_rejects_growth_in_the_middle():
    ct = ColumnTransformer([("cat", OneHotEncoder(), [0]), ("num", "passthrough", [1])])
    X1 = np.array([["a", "1"]], dtype=object)
    X2 = np.array([["b", "2"]], dtype=object)
    ct.partial_fit(X1).transform(X1)
    with pytest.raises(RuntimeError, match="last"):
        ct.partial_fit(X2).transform(X2)


def test_stream_trainer_prequential_log(blobs, tmp_path):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=300)
    tr = StreamTrainer(tree, rolling_window=200).run(iter_chunks(X, y, 100))
    h = tr.history
    assert len(h["chunk"]) == 12
    assert np.all(np.isnan(h["chunk_accuracy"][:3]))     
    assert np.isfinite(h["chunk_accuracy"][3:]).all()
    np.testing.assert_array_equal(tr.retrain_chunks(), [2, 5, 8, 11])
    assert h["buffer_fill"][2] == 0 and h["buffer_fill"][3] == 100
    assert tr.summary()["final_accuracy"] > 0.9
    assert 0.5 < h["roc_auc"][-1] <= 1.0
    tr.to_csv(tmp_path / "log.csv")
    back = read_csv(tmp_path / "log.csv")
    assert len(back) == 12 and "cumulative_accuracy" in back.columns


def test_stream_trainer_cumulative_equals_manual(blobs):
    X, y = blobs
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=200))
    preds, truth = [], []
    for Xc, yc in iter_chunks(X, y, 100):
        if tr.model.is_fitted:
            preds.append(tr.model.predict(Xc))
            truth.append(yc)
        tr.step(Xc, yc)
    manual = np.mean(np.concatenate(preds) == np.concatenate(truth))
    assert tr.summary()["final_accuracy"] == pytest.approx(manual)


def test_stream_trainer_memory_trace_and_bad_model(blobs):
    X, y = blobs
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=100), trace_memory=True)
    rec = tr.step(X[:150], y[:150])
    assert rec["retrained"] and rec["peak_alloc_bytes"] > 0
    with pytest.raises(TypeError):
        StreamTrainer(StandardScaler())


def test_iter_chunks_shapes_and_errors():
    X = np.arange(10).reshape(5, 2)
    sizes = [len(c[0]) for c in iter_chunks(X, np.arange(5), 2)]
    assert sizes == [2, 2, 1]
    with pytest.raises(ValueError):
        list(iter_chunks(X, np.arange(4), 2))
    with pytest.raises(ValueError):
        list(iter_chunks(X, None, 0))


def test_read_csv_types_missing_and_errors(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b,c\n1,x,\n2.5,,NA\n?,z,3\n")
    f = read_csv(p)
    assert f.numeric_columns() == ["a", "c"] and f.categorical_columns() == ["b"]
    assert np.isnan(f["a"][2]) and f["b"][1] == ""
    assert f.to_array(["a", "c"]).shape == (3, 2)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="line 3"):
        read_csv(bad)
    with pytest.raises(KeyError):
        read_csv(p, usecols=["nope"])
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_csv(tmp_path / "empty.csv")


def test_iter_csv_streams_in_chunks_and_roundtrip(tmp_path):
    p = tmp_path / "w.csv"
    write_csv(p, {"x": np.arange(7.0), "y": np.array([1, 0, 1, 0, 1, 0, np.nan])})
    sizes = [len(c) for c in iter_csv(p, 3)]
    assert sizes == [3, 3, 1]
    assert np.isnan(read_csv(p)["y"][-1])                


def test_plots_return_figures_and_save(tmp_path):
    vals = np.r_[np.nan, 0.6, 0.7, 0.8]
    fig, ax = visualise.plot_metric_over_time(vals, "Accuracy", "acc", retrain_points=[1, 3],
                                              save_path=tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 0 and len(ax.lines) >= 2
    fig2, _ = visualise.compare_models([0.5, 0.6], [0.55, 0.7, 0.8], labels=("b=250", "b=1000"))
    fig3, axes = visualise.plot_predictions_vs_ground_truth([0, 1, 1, 0], [0, 1, 0, 0])
    assert axes[0].get_title(loc="left").startswith("accuracy 0.750")
    plt.close("all")


def test_dashboard_from_trainer_history(blobs, tmp_path):
    X, y = blobs
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=300), rolling_window=200)
    tr.run(iter_chunks(X, y, 150))
    fig, axes = visualise.plot_stream_dashboard(tr.history, save_path=tmp_path / "d.png")
    assert len(axes) == 3
    with pytest.raises(ValueError):
        visualise.plot_metric_over_time(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        visualise.plot_predictions_vs_ground_truth([], [])
    plt.close("all")
