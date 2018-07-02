from sklearn.linear_model import SGDClassifier
from dask_ml.wrappers import Incremental
from dask_ml.datasets import make_classification
import dask.array as da
from dask import delayed
import pytest


def test_score_compute_basic():
    X, y = make_classification(random_state=0, chunks=20)
    est = SGDClassifier(tol=1e-3)
    inc = Incremental(est, scoring='accuracy')

    inc.partial_fit(X, y, classes=da.unique(y))
    r = inc.score(X, y, compute=False)
    assert isinstance(r.compute(), float)


def test_score_delayed_fit_and_score():
    X, y = make_classification(random_state=0, chunks=20)
    est = SGDClassifier(tol=1e-3)
    inc = Incremental(est, scoring='accuracy')

    inc = delayed(inc)
    inc = inc.partial_fit(X, y, classes=da.unique(y))
    s = inc.score(X, y)
    assert isinstance(s.compute(), float)
