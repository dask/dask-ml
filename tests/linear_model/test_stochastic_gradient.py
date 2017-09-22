from daskml.datasets import make_classification
from sklearn import linear_model as lm_
from daskml import linear_model as lm

import numpy.testing as npt


X, y = make_classification(chunks=100, random_state=0)


class TestStochasticGradientClassifier:

    def test_basic(self):
        a = lm.BigSGDClassifier(classes=[0, 1], random_state=0,
                                max_iter=1000, tol=1e-3)
        b = lm_.SGDClassifier(random_state=0, max_iter=1000, tol=1e-3)

        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        npt.assert_almost_equal(a.coef_, b.coef_)
