from sklearn import neural_network as nn_
from daskml import neural_network as nn

from dask.array.utils import assert_eq


class TestMLPClassifier(object):

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = nn.BigMLPClassifier(classes=[0, 1], random_state=0)
        b = nn_.MLPClassifier(random_state=0)
        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        for a_, b_ in zip(a.coefs_, b.coefs_):
            assert_eq(a_, b_)


class TestMLPRegressor(object):

    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = nn.BigMLPRegressor(random_state=0)
        b = nn_.MLPRegressor(random_state=0)
        a.fit(X, y)
        b.partial_fit(X, y)
        for a_, b_ in zip(a.coefs_, b.coefs_):
            assert_eq(a_, b_)
