from daskml.datasets import make_classification
from sklearn.preprocessing import StandardScaler as StandardScaler_
from daskml.preprocessing import StandardScaler

from dask.array.utils import assert_eq


X, y = make_classification(chunks=2)


class TestStandardScaler(object):
    def test_basic(self):
        a = StandardScaler()
        b = StandardScaler_()

        a.fit(X)
        b.fit(X.compute())

        assert_eq(a.mean_, b.mean_)
        assert_eq(a.scale_, b.scale_)
        assert a.n_samples_seen_ == b.n_samples_seen_
