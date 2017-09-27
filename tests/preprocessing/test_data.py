from daskml.datasets import make_classification
from sklearn.preprocessing import StandardScaler as StandardScaler_
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from daskml.preprocessing import StandardScaler
from daskml.preprocessing import MinMaxScaler

from dask.array.utils import assert_eq


X, y = make_classification(chunks=2)


def _get_scaler_attributes(scaler):
    return filter(lambda a: a.endswith("_") and not a.startswith("__"),
                  dir(scaler))


class TestStandardScaler(object):
    def test_basic(self):
        a = StandardScaler()
        b = StandardScaler_()

        a.fit(X)
        b.fit(X.compute())

        for attr in _get_scaler_attributes(self):
            assert_eq(getattr(a, attr), getattr(b, attr))

    def test_inverse_transform(self):
        a = StandardScaler()
        assert_eq(a.inverse_transform(a.fit_transform(X)).compute(),
                  X.compute())


class TestMinMaxScaler(object):
    def test_basic(self):
        a = MinMaxScaler()
        b = MinMaxScaler_()

        a.fit(X)
        b.fit(X.compute())

        for attr in _get_scaler_attributes(self):
            assert_eq(getattr(a, attr), getattr(b, attr))

    def test_inverse_transform(self):
        a = MinMaxScaler()
        assert_eq(a.inverse_transform(a.fit_transform(X)).compute(),
                  X.compute())
