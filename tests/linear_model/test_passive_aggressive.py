import dask
import pytest
from sklearn import linear_model as lm_

from dask_ml import linear_model as lm
from dask_ml.utils import assert_estimator_equal


@pytest.mark.filterwarnings("ignore:'Partial:FutureWarning")
class TestPassiveAggressiveClassifier:
    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = lm.PartialPassiveAggressiveClassifier(
            classes=[0, 1], random_state=0, max_iter=100, tol=1e-3
        )
        b = lm_.PassiveAggressiveClassifier(random_state=0, max_iter=100, tol=1e-3)
        a.fit(X, y)
        b.partial_fit(*dask.compute(X, y), classes=[0, 1])
        assert_estimator_equal(a, b, exclude=["loss_function_"])


@pytest.mark.filterwarnings("ignore:'Partial:FutureWarning")
class TestPassiveAggressiveRegressor:
    def test_basic(self, single_chunk_regression):
        X, y = single_chunk_regression
        a = lm.PartialPassiveAggressiveRegressor(random_state=0, max_iter=100, tol=1e-3)
        b = lm_.PassiveAggressiveRegressor(random_state=0, max_iter=100, tol=1e-3)
        a.fit(X, y)
        b.partial_fit(*dask.compute(X, y))
        assert_estimator_equal(a, b, exclude=["loss_function_", "standard_coef_"])
