from sklearn import linear_model as lm_
from daskml import linear_model as lm

from dask.array.utils import assert_eq


class TestPassiveAggressiveClassifier:

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = lm.BigPassiveAggressiveClassifier(classes=[0, 1], random_state=0)
        b = lm_.PassiveAggressiveClassifier(random_state=0)
        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        assert_eq(a.coef_, b.coef_)


class TestPassiveAggressiveRegressor:

    def test_basic(self, single_chunk_regression):
        X, y = single_chunk_regression
        a = lm.BigPassiveAggressiveRegressor(random_state=0, max_iter=100,
                                             tol=1e-3)
        b = lm_.PassiveAggressiveRegressor(random_state=0, max_iter=100,
                                           tol=1e-3)
        a.fit(X, y)
        b.partial_fit(X, y)
        assert_eq(a.coef_, b.coef_)
