import dask.array as da
import numpy as np
import pytest
import sklearn.model_selection
import sklearn.datasets
from dask.array.utils import assert_eq
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier

from dask_ml.wrappers import Incremental
from dask_ml.utils import assert_estimator_equal
import dask_ml.metrics
from dask_ml.metrics.scorer import check_scoring


def test_get_params():
    clf = Incremental(SGDClassifier())
    result = clf.get_params()

    assert 'estimator__max_iter' in result
    assert 'max_iter' in result
    assert result['scoring'] is None


def test_set_params():
    clf = Incremental(SGDClassifier())
    clf.set_params(**{'scoring': 'accuracy',
                      'max_iter': 10,
                      'estimator__max_iter': 20})
    result = clf.get_params()

    assert result['estimator__max_iter'] == 20
    assert result['max_iter'] == 10
    assert result['scoring'] == 'accuracy'


def test_incremental_basic(scheduler, xy_classification):
    X, y = xy_classification

    with scheduler() as (s, [_, _]):
        est1 = SGDClassifier(random_state=0, tol=1e-3)
        est2 = clone(est1)

        clf = Incremental(est1)
        result = clf.fit(X, y, classes=[0, 1])
        for slice_ in da.core.slices_from_chunks(X.chunks):
            est2.partial_fit(X[slice_], y[slice_[0]], classes=[0, 1])

        assert result is clf

        assert isinstance(result.estimator_.coef_, np.ndarray)
        np.testing.assert_array_almost_equal(result.estimator_.coef_,
                                             est2.coef_)

        assert_estimator_equal(clf.estimator_, est2,
                               exclude=['loss_function_'])

        #  Predict
        result = clf.predict(X)
        expected = est2.predict(X)
        assert isinstance(result, da.Array)
        assert_eq(result, expected)

        # score
        result = clf.score(X, y)
        expected = est2.score(X, y)
        # assert isinstance(result, da.Array)
        assert_eq(result, expected)

        clf = Incremental(SGDClassifier(random_state=0, tol=1e-3))
        clf.partial_fit(X, y, classes=[0, 1])
        assert_estimator_equal(clf.estimator_, est2,
                               exclude=['loss_function_'])


def test_max_iter(scheduler, xy_classification):
    X, y = xy_classification

    est = SGDClassifier(random_state=0, tol=1e-3)
    clf = Incremental(est, max_iter=5)

    with scheduler():
        clf.fit(X, y, classes=[0, 1])


def test_in_gridsearch(scheduler, xy_classification):
    X, y = xy_classification
    clf = Incremental(SGDClassifier(random_state=0, tol=1e-3))
    param_grid = {'estimator__alpha': [0.1, 10]}
    gs = sklearn.model_selection.GridSearchCV(clf, param_grid, iid=False)

    with scheduler() as (s, [a, b]):
        gs.fit(X, y, classes=[0, 1])


def test_scoring(scheduler, xy_classification,
                 scoring=dask_ml.metrics.accuracy_score):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        clf = Incremental(SGDClassifier(tol=1e-3), scoring=scoring)
        with pytest.raises(ValueError,
                           match='metric function rather than a scorer'):
            clf.fit(X, y, classes=np.unique(y))


@pytest.mark.parametrize("scoring", [
    "accuracy", "neg_mean_squared_error", "r2", None
])
def test_scoring_string(scheduler, xy_classification, scoring):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        clf = Incremental(SGDClassifier(tol=1e-3), scoring=scoring)
        if scoring:
            assert (dask_ml.metrics.scorer.SCORERS[scoring] ==
                    check_scoring(clf, scoring=scoring))
        assert callable(check_scoring(clf, scoring=scoring))
        clf.fit(X, y, classes=np.unique(y))
        clf.score(X, y)


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
    inc = Incremental(sgd, scoring='accuracy')

    inc.partial_fit(X, y, classes=[0, 1])
    inc.fit(X, y, classes=[0, 1])

    assert inc.score(X, y) == 1

    dX = da.from_array(X, chunks=(2, 5))
    dy = da.from_array(y, chunks=2)
    assert inc.score(dX, dy) == 1


def test_score(xy_classification):
    distributed = pytest.importorskip('distributed')
    client = distributed.Client(n_workers=2)

    X, y = xy_classification
    inc = Incremental(SGDClassifier(max_iter=1000, random_state=0),
                      scoring='accuracy')

    with client:
        inc.fit(X, y, classes=[0, 1])
        result = inc.score(X, y)
        expected = inc.estimator_.score(X, y)

    assert result == expected
