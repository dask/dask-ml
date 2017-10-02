from sklearn.linear_model import Perceptron
from daskml.linear_model import BigPerceptron

from ..test_utils import assert_estimator_equal


class TestPerceptron(object):

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = BigPerceptron(classes=[0, 1], max_iter=1000, tol=1e-3)
        b = Perceptron(max_iter=1000, tol=1e-3)
        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        assert_estimator_equal(a.coef_, b.coef_)
