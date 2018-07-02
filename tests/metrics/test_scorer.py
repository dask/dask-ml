from sklearn.linear_model import SGDClassifier
from dask_ml.wrappers import Incremental
from dask_ml.datasets import make_classification
import dask.array as da
from dask import delayed
import pytest
from sklearn.exceptions import NotFittedError


def test_score_compute_basic():
    X, y = make_classification(random_state=0, chunks=20)
    est = SGDClassifier(tol=1e-3)
    inc = Incremental(est, scoring='accuracy')

    inc.partial_fit(X, y, classes=da.unique(y))
    r = inc.score(X, y)
    assert isinstance(r, float)


def test_score_delayed_fit_and_score():
    X, y = make_classification(random_state=0, chunks=20)
    est = SGDClassifier(tol=1e-3)
    inc = Incremental(est, scoring='accuracy')

    inc = delayed(inc)
    inc = inc.partial_fit(X, y, classes=da.unique(y))
    s = inc.score(X, y)
    assert isinstance(s.compute(), float)


@pytest.mark.parametrize("return_est", [True, False])
def test_score_raises(return_est):
    X, y = make_classification(random_state=0, chunks=20)
    est = SGDClassifier(tol=1e-3)
    inc = Incremental(est, scoring='accuracy')

    inc = delayed(inc)
    r = inc.partial_fit(X, y, classes=da.unique(y))
    if return_est:
        inc = r

    if not return_est:
        with pytest.raises(NotFittedError):
            inc.score(X, y).compute()
    else:
        inc.score(X, y).compute()
