import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.base import clone, BaseEstimator

import dask_ml.compose
import dask_ml.preprocessing

df = pd.DataFrame({"A": pd.Categorical(["a", "a", "b", "a"]), "B": [1.0, 2, 4, 5]})
ddf = dd.from_pandas(df, npartitions=2).reset_index(drop=True)  # unknown divisions


def test_column_transformer():
    a = sklearn.compose.make_column_transformer(
        (sklearn.preprocessing.OneHotEncoder(sparse=False), ["A"]),
        (sklearn.preprocessing.StandardScaler(), ["B"]),
    )

    b = dask_ml.compose.make_column_transformer(
        (dask_ml.preprocessing.OneHotEncoder(sparse=False), ["A"]),
        (dask_ml.preprocessing.StandardScaler(), ["B"]),
    )

    a.fit(df)
    b.fit(ddf)

    expected = a.transform(df)
    with pytest.warns(None):
        result = b.transform(ddf)

    assert isinstance(result, dd.DataFrame)
    expected = pd.DataFrame(expected, index=result.index, columns=result.columns)
    dd.utils.assert_eq(result, expected)

    # fit-transform
    result = clone(b).fit_transform(ddf)
    expected = clone(a).fit_transform(df)
    expected = pd.DataFrame(expected, index=result.index, columns=result.columns)
    dd.utils.assert_eq(result, expected)


def test_column_transformer_unk_chunksize():
    cols = 3
    rows = 4
    a = 0x61  # character code for 'a'.
    names = list(map(chr, a + np.arange(cols)))
    x = db.from_sequence(gen_ints(rows, cols)).to_dataframe(columns=names)

    features = sklearn.pipeline.Pipeline([
        ('features', sklearn.pipeline.FeatureUnion([
            ('ratios', dask_ml.compose.ColumnTransformer([
                ('a_b', SumTransformer(), ['a', 'b']),
                ('b_c', SumTransformer(), ['b', 'c'])
            ]))
        ]))
    ])

    # Checks:
    #   ValueError: Tried to concatenate arrays with unknown shape (nan, 1).
    #               To force concatenation pass allow_unknown_chunksizes=True.
    out = features.fit_transform(x)

    exp = np.array([[1, 3], [3, 5], [5, 7], [7, 9]], dtype=np.int32)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, exp)


# Generator ensures the DataFrame shape has NaN in first dim.
def gen_ints(r, c):
    for i in range(r):
        yield i + np.arange(c)


# Some basic transformer.
class SumTransformer(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.map_partitions(lambda x: x.values.sum(axis=-1).reshape(-1, 1))
        return out
