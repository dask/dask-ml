import pytest
from sklearn import neural_network as nn_

from dask_ml import neural_network as nn
from dask_ml.utils import assert_estimator_equal


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestMLPClassifier(object):
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = nn.ParitalMLPClassifier(classes=[0, 1], random_state=0)
        b = nn_.MLPClassifier(random_state=0)

        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])

        assert_estimator_equal(a, b)


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestMLPRegressor(object):
    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = nn.ParitalMLPRegressor(random_state=0)
        b = nn_.MLPRegressor(random_state=0)
        a.fit(X, y)
        b.partial_fit(X, y)
        assert_estimator_equal(a, b)
