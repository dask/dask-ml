from daskml.datasets import make_classification
from sklearn.preprocessing import StandardScaler as StandardScaler_
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from daskml.preprocessing import StandardScaler
from daskml.preprocessing import MinMaxScaler

from dask.array.utils import assert_eq


X, y = make_classification(chunks=2)
df = X.to_dask_dataframe()


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

    def test_df_values(self):
        a = MinMaxScaler()

        assert_eq(a.fit_transform(X).compute(),
                  a.fit_transform(df).compute().as_matrix())

    def test_df_column_slice(self):
        mask = [3, 4, 5]
        mask_ix = [df.columns.tolist().index(x) for x in mask]
        a = MinMaxScaler(columns=mask)
        b = MinMaxScaler_()

        tdfa = a.fit_transform(df[mask].values)
        tdfb = b.fit_transform(df[mask].values.compute())

        assert_eq(tdfa.compute(), tdfb)
        #assert all(c in a.fit_transform(df).columns for c in df.columns)
