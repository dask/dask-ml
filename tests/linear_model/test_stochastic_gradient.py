from sklearn import linear_model as lm_
from daskml import linear_model as lm

import numpy.testing as npt


class TestStochasticGradientClassifier:

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification

        a = lm.BigSGDClassifier(classes=[0, 1], random_state=0,
                                max_iter=1000, tol=1e-3)
        b = lm_.SGDClassifier(random_state=0, max_iter=1000, tol=1e-3)

        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        npt.assert_almost_equal(a.coef_, b.coef_)


class TestStochasticGradientRegressor:

    def test_basic(self, single_chunk_regression):
        X, y = single_chunk_regression
        a = lm.BigSGDRegressor(random_state=0,
                               max_iter=1000, tol=1e-3)
        b = lm_.SGDRegressor(random_state=0, max_iter=1000, tol=1e-3)

        a.fit(X, y)
        b.partial_fit(X, y)
        npt.assert_almost_equal(a.coef_, b.coef_)
