import pytest
import six
from dask.delayed import Delayed
from sklearn import linear_model as lm_

from dask_ml import linear_model as lm
from dask_ml.utils import assert_estimator_equal


@pytest.mark.filterwarnings("ignore:'Partial:FutureWarning")
class TestStochasticGradientClassifier(object):
    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification

        a = lm.PartialSGDClassifier(
            classes=[0, 1], random_state=0, max_iter=1000, tol=1e-3
        )
        b = lm_.SGDClassifier(random_state=0, max_iter=1000, tol=1e-3)

        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        assert_estimator_equal(a, b, exclude="loss_function_")

    def test_numpy_arrays(self, single_chunk_classification):
        # fit with dask arrays, test with numpy arrays
        X, y = single_chunk_classification

        a = lm.PartialSGDClassifier(
            classes=[0, 1], random_state=0, max_iter=1000, tol=1e-3
        )

        a.fit(X, y)
        X = X.compute()
        y = y.compute()
        a.predict(X)
        a.score(X, y)


@pytest.mark.filterwarnings("ignore:'Partial:FutureWarning")
class TestStochasticGradientRegressor(object):
    def test_basic(self, single_chunk_regression):
        X, y = single_chunk_regression
        a = lm.PartialSGDRegressor(random_state=0, max_iter=1000, tol=1e-3)
        b = lm_.SGDRegressor(random_state=0, max_iter=1000, tol=1e-3)

        a.fit(X, y)
        b.partial_fit(X, y)
        assert_estimator_equal(a, b)

    def test_numpy_arrays(self, single_chunk_regression):
        X, y = single_chunk_regression
        a = lm.PartialSGDRegressor(random_state=0, max_iter=1000, tol=1e-3)

        a.fit(X, y)
        X = X.compute()
        y = y.compute()
        a.predict(X)
        a.score(X, y)


@pytest.mark.filterwarnings("ignore:'Partial:FutureWarning")
def test_lazy(xy_classification):
    X, y = xy_classification
    sgd = lm.PartialSGDClassifier(classes=[0, 1], max_iter=5)
    r = sgd.fit(X, y, compute=False)
    assert isinstance(r, Delayed)
    result = r.compute()
    assert isinstance(result, lm_.SGDClassifier)


@pytest.mark.skipif(six.PY2, reason="Python 2 failure.")
def test_deprecated():
    expected = (
        r"'PartialSGDClassifier' is deprecated. Use "
        r"'dask_ml.wrappers.Incremental.*SGDClassifier.*"
        r"instead."
    )

    with pytest.warns(FutureWarning, match=expected):
        lm.PartialSGDClassifier(classes=[0, 1])
