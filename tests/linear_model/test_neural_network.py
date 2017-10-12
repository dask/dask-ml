from sklearn import neural_network as nn_
from daskml import neural_network as nn

from daskml.utils import assert_estimator_equal


class TestMLPClassifier(object):

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = nn.ParitalMLPClassifier(classes=[0, 1], random_state=0)
        b = nn_.MLPClassifier(random_state=0)
        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        assert_estimator_equal(a, b)


class TestMLPRegressor(object):

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = nn.ParitalMLPRegressor(random_state=0)
        b = nn_.MLPRegressor(random_state=0)
        a.fit(X, y)
        b.partial_fit(X, y)
        assert_estimator_equal(a, b)
