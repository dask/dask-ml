import dask
import dask.array as da
import distributed
import numpy as np
import pytest
import sklearn.model_selection
from dask.array.utils import assert_eq
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier

from dask_ml.wrappers import Incremental
from dask_ml.utils import assert_estimator_equal


@pytest.fixture(scope='module', params=['threads', 'distributed'])
def scheduler(request):
    if request.param == 'distributed':
        client = distributed.Client(set_as_default=False)

    with dask.config.set(scheduler=request.param):
        yield

    if request.param == 'distributed':
        client.close()


def test_incremental_basic(scheduler, xy_classification):
    X, y = xy_classification
    est1 = SGDClassifier(random_state=0)
    est2 = clone(est1)

    clf = Incremental(est1)
    result = clf.fit(X, y, classes=[0, 1])
    for slice_ in da.core.slices_from_chunks(X.chunks):
        est2.partial_fit(X[slice_], y[slice_[0]], classes=[0, 1])

    assert result is clf

    assert isinstance(result.estimator.coef_, np.ndarray)
    np.testing.assert_array_almost_equal(result.estimator.coef_, est2.coef_)

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

    clf = Incremental(SGDClassifier(random_state=0))
    clf.partial_fit(X, y, classes=[0, 1])
    assert_estimator_equal(clf.estimator, est2, exclude=['loss_function_'])


def test_in_gridsearch(xy_classification):
    X, y = xy_classification

    clf = Incremental(SGDClassifier(random_state=0))
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
