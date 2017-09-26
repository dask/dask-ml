from sklearn.linear_model import Perceptron
from daskml.linear_model import BigPerceptron

from dask.array.utils import assert_eq


class TestPerceptron(object):

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = BigPerceptron(classes=[0, 1], max_iter=1000, tol=1e-3)
        b = Perceptron(max_iter=1000, tol=1e-3)
        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        assert_eq(a.coef_, b.coef_)
