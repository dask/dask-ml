import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import sklearn.model_selection
from dask.array.utils import assert_eq
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import make_pipeline

import dask_ml.feature_extraction.text
import dask_ml.metrics
from dask_ml.metrics.scorer import check_scoring
from dask_ml.wrappers import Incremental


def test_get_params():
    clf = Incremental(SGDClassifier())
    result = clf.get_params()

    assert "estimator__max_iter" in result
    assert result["scoring"] is None


def test_set_params():
    clf = Incremental(SGDClassifier())
    clf.set_params(**{"scoring": "accuracy", "estimator__max_iter": 20})
    result = clf.get_params()

    assert result["estimator__max_iter"] == 20
    assert result["scoring"] == "accuracy"


@pytest.mark.parametrize("dataframes", [False, True])
def test_incremental_basic(scheduler, dataframes):
    # Create observations that we know linear models can recover
    n, d = 100, 3
    rng = da.random.RandomState(42)
    X = rng.normal(size=(n, d), chunks=30)
    coef_star = rng.uniform(size=d, chunks=d)
    y = da.sign(X.dot(coef_star))
    y = (y + 1) / 2
    if dataframes:
        X = dd.from_array(X)
        y = dd.from_array(y)

    with scheduler() as (s, [_, _]):
        est1 = SGDClassifier(random_state=0, tol=1e-3, average=True)
        est2 = clone(est1)

        clf = Incremental(est1, random_state=0)
        result = clf.fit(X, y, classes=[0, 1])
        assert result is clf

        # est2 is a sklearn optimizer; this is just a benchmark
        if dataframes:
            X = X.to_dask_array(lengths=True)
            y = y.to_dask_array(lengths=True)

        for slice_ in da.core.slices_from_chunks(X.chunks):
            est2.partial_fit(
                X[slice_].compute(), y[slice_[0]].compute(), classes=[0, 1]
            )

        assert isinstance(result.estimator_.coef_, np.ndarray)
        rel_error = np.linalg.norm(clf.coef_ - est2.coef_)
        rel_error /= np.linalg.norm(clf.coef_)
        assert rel_error < 0.9

        assert set(dir(clf.estimator_)) == set(dir(est2))

        #  Predict
        result = clf.predict(X)
        expected = est2.predict(X)
        assert isinstance(result, da.Array)
        if dataframes:
            # Compute is needed because chunk sizes of this array are unknown
            result = result.compute()
        rel_error = np.linalg.norm(result - expected)
        rel_error /= np.linalg.norm(expected)
        assert rel_error < 0.3

        # score
        result = clf.score(X, y)
        expected = est2.score(*dask.compute(X, y))
        assert abs(result - expected) < 0.1

        clf = Incremental(SGDClassifier(random_state=0, tol=1e-3, average=True))
        clf.partial_fit(X, y, classes=[0, 1])
        assert set(dir(clf.estimator_)) == set(dir(est2))


def test_in_gridsearch(scheduler, xy_classification):
    X, y = xy_classification
    clf = Incremental(SGDClassifier(random_state=0, tol=1e-3))
    param_grid = {"estimator__alpha": [0.1, 10]}
    gs = sklearn.model_selection.GridSearchCV(clf, param_grid, cv=3)

    with scheduler() as (s, [a, b]):
        gs.fit(X, y, classes=[0, 1])


def test_scoring(scheduler, xy_classification, scoring=dask_ml.metrics.accuracy_score):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        clf = Incremental(SGDClassifier(tol=1e-3), scoring=scoring)
        with pytest.raises(ValueError, match="metric function rather than a scorer"):
            clf.fit(X, y, classes=np.unique(y))


@pytest.mark.parametrize("scoring", ["accuracy", "neg_mean_squared_error", "r2", None])
def test_scoring_string(scheduler, xy_classification, scoring):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        clf = Incremental(SGDClassifier(tol=1e-3), scoring=scoring)
        assert callable(check_scoring(clf, scoring=scoring))
        clf.fit(X, y, classes=np.unique(y))
        clf.score(X, y)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_ndarrays():
    X = np.ones((10, 5))
    y = np.concatenate([np.zeros(5), np.ones(5)])

    sgd = SGDClassifier(tol=1e-3)
    inc = Incremental(sgd)

    inc.partial_fit(X, y, classes=[0, 1])
    sgd.fit(X, y)

    assert inc.estimator is sgd
    assert_eq(inc.coef_, inc.estimator_.coef_)


def test_score_ndarrays():
    X = np.ones((10, 5))
    y = np.ones(10)

    sgd = SGDClassifier(tol=1e-3)
    inc = Incremental(sgd, scoring="accuracy")

    inc.partial_fit(X, y, classes=[0, 1])
    inc.fit(X, y, classes=[0, 1])

    assert inc.score(X, y) == 1

    dX = da.from_array(X, chunks=(2, 5))
    dy = da.from_array(y, chunks=2)
    assert inc.score(dX, dy) == 1


def test_score(xy_classification):
    distributed = pytest.importorskip("distributed")
    client = distributed.Client(n_workers=2)

    X, y = xy_classification
    inc = Incremental(
        SGDClassifier(max_iter=1000, random_state=0, tol=1e-3), scoring="accuracy"
    )

    with client:
        inc.fit(X, y, classes=[0, 1])
        result = inc.score(X, y)
        expected = inc.estimator_.score(*dask.compute(X, y))

    assert result == expected


@pytest.mark.parametrize(
    "estimator, fit_kwargs, scoring",
    [(SGDClassifier, {"classes": [0, 1]}, "accuracy"), (SGDRegressor, {}, "r2")],
)
def test_replace_scoring(estimator, fit_kwargs, scoring, xy_classification, mocker):
    X, y = xy_classification
    inc = Incremental(estimator(max_iter=1000, random_state=0, tol=1e-3))
    inc.fit(X, y, **fit_kwargs)

    patch = mocker.patch.object(dask_ml.wrappers, "get_scorer")

    inc.score(X, y)

    assert patch.call_count == 1
    patch.assert_called_with(scoring, compute=True)


@pytest.mark.parametrize("container", ["bag", "series"])
def test_incremental_text_pipeline(container):
    X = pd.Series(["a list", "of words", "for classification"] * 100)
    X = dd.from_pandas(X, npartitions=3)

    if container == "bag":
        X = X.to_bag()

    y = da.from_array(np.array([0, 0, 1] * 100), chunks=(100,) * 3)

    assert tuple(X.map_partitions(len).compute()) == y.chunks[0]

    sgd = SGDClassifier(max_iter=5, tol=1e-3)
    clf = Incremental(sgd, scoring="accuracy", assume_equal_chunks=True)
    vect = dask_ml.feature_extraction.text.HashingVectorizer()
    pipe = make_pipeline(vect, clf)

    pipe.fit(X, y, incremental__classes=[0, 1])
    X2 = pipe.steps[0][1].transform(X)
    assert hasattr(clf, "coef_")

    X2.compute_chunk_sizes()
    assert X2.shape == (300, vect.n_features)
