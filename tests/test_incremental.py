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


def test_incremental_basic(scheduler, xy_classification):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        est1 = SGDClassifier(random_state=0, tol=1e-3)
        est2 = clone(est1)

        clf = Incremental(est1)
        result = clf.fit(X, y, classes=[0, 1])
        for slice_ in da.core.slices_from_chunks(X.chunks):
            est2.partial_fit(X[slice_], y[slice_[0]], classes=[0, 1])

        assert result is clf

        assert isinstance(result.estimator.coef_, np.ndarray)
        np.testing.assert_array_almost_equal(result.estimator.coef_,
                                             est2.coef_)

        assert_estimator_equal(clf.estimator, est2, exclude=['loss_function_'])

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
        assert_estimator_equal(clf.estimator, est2, exclude=['loss_function_'])


def test_in_gridsearch(scheduler, xy_classification):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        clf = Incremental(SGDClassifier(random_state=0, tol=1e-3))
        param_grid = {'alpha': [0.1, 10]}
        gs = sklearn.model_selection.GridSearchCV(clf, param_grid, iid=False)
        gs.fit(X, y, classes=[0, 1])


def test_estimator_param_raises():

    class Dummy(sklearn.base.BaseEstimator):
        def __init__(self, estimator=42):
            self.estimator = estimator

        def fit(self, X):
            return self

    clf = Incremental(Dummy(estimator=1))

    with pytest.raises(ValueError, match='used by both'):
        clf.get_params()


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
        clf.estimator.score(X, y)
