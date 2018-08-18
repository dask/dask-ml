import pytest
import packaging.version
from sklearn import neural_network as nn_

from dask_ml import neural_network as nn
from dask_ml.utils import assert_estimator_equal
from dask_ml._compat import SK_VERSION


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestMLPClassifier(object):
    def test_basic(self, single_chunk_classification):
        X, y = single_chunk_classification
        a = nn.ParitalMLPClassifier(classes=[0, 1], random_state=0)
        b = nn_.MLPClassifier(random_state=0)

        if SK_VERSION >= packaging.version.parse("0.20.0.dev0"):
            a.fit(X, y)
            b.partial_fit(X, y, classes=[0, 1])
        else:
            with pytest.warns(DeprecationWarning):
                a.fit(X, y)
            with pytest.warns(DeprecationWarning):
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
