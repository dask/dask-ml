import dask.array as da
import numpy as np
import numpy.linalg as LA
import pytest
import sklearn.model_selection
import sklearn.datasets
from dask.array.utils import assert_eq
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier

from dask_ml.wrappers import Incremental
from dask_ml.utils import assert_estimator_equal
from dask_ml.datasets import make_classification


def test_incremental_basic(scheduler, xy_classification):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        est1 = SGDClassifier(random_state=0, warm_start=True)
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

        clf = Incremental(SGDClassifier(random_state=0, warm_start=True))
        clf.partial_fit(X, y, classes=[0, 1])
        assert_estimator_equal(clf.estimator, est2, exclude=['loss_function_'])


def test_in_gridsearch(scheduler, xy_classification):
    X, y = xy_classification
    with scheduler() as (s, [a, b]):
        clf = Incremental(SGDClassifier(random_state=0, warm_start=True))
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


@pytest.mark.parametrize('warm_start', [True, False])
@pytest.mark.parametrize('func', ['partial_fit', 'fit'])
def test_warm_start(warm_start, func, seed=42):
    n, d = 400, 200
    X, y = make_classification(n_features=d, n_samples=n, chunks=(n // 10, d),
                               random_state=seed)

    model = SGDClassifier(max_iter=1, warm_start=warm_start, random_state=seed)
    if not warm_start:
        with pytest.raises(ValueError, match='requires warm_start'):
            model = Incremental(model)
        return
    model = Incremental(model)

    scores = []
    num_iter = 40
    for iter in range(num_iter):
        getattr(model, func)(X, y, classes=[0, 1])
        score = model.score(X, y)
        scores += [score]
    scores = np.array(scores)

    assert scores[0] < scores[-1]
    prob_increase = (np.diff(scores) >= 0).sum() / len(scores)
    avg_increase = np.diff(scores).sum() / num_iter
    assert prob_increase > 0.65
    assert avg_increase > 0.004
