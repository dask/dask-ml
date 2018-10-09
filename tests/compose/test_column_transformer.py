import dask.array as da
import dask.dataframe as dd
import pandas as pd
import scipy.sparse
import sklearn.compose
import sklearn.preprocessing
from sklearn.base import clone

import dask_ml.compose
import dask_ml.feature_extraction.text
import dask_ml.preprocessing

df = pd.DataFrame({"A": pd.Categorical(["a", "a", "b", "a"]), "B": [1.0, 2, 4, 5]})
ddf = dd.from_pandas(df, npartitions=2)


def test_column_transformer():
    a = sklearn.compose.make_column_transformer(
        (["A"], sklearn.preprocessing.OneHotEncoder(sparse=False)),
        (["B"], sklearn.preprocessing.StandardScaler()),
    )

    b = dask_ml.compose.make_column_transformer(
        (["A"], dask_ml.preprocessing.OneHotEncoder(sparse=False)),
        (["B"], dask_ml.preprocessing.StandardScaler()),
    )

    a.fit(df)
    b.fit(ddf)

    expected = a.transform(df)
    result = b.transform(ddf)

    assert isinstance(result, dd.DataFrame)
    expected = pd.DataFrame(expected, index=result.index, columns=result.columns)
    dd.utils.assert_eq(result, expected)

    # fit-transform
    result = clone(b).fit_transform(ddf)
    expected = clone(a).fit_transform(df)
    expected = pd.DataFrame(expected, index=result.index, columns=result.columns)
    dd.utils.assert_eq(result, expected)


def test_mixed_sparse():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "a": ["a", "b", "c", "d", "e", "f"],
            "b": ["A", "B", "C", "D", "E", "F"],
        }
    )
    ddf = dd.from_pandas(df, npartitions=3)

    transformer = dask_ml.compose.make_column_transformer(
        (["x"], dask_ml.preprocessing.StandardScaler()),
        ("a", dask_ml.feature_extraction.text.HashingVectorizer(n_features=10)),
        ("b", dask_ml.feature_extraction.text.HashingVectorizer(n_features=20)),
    )

    out = transformer.fit_transform(ddf)
    assert isinstance(out, da.Array)
    assert out.shape[1] == 31
    assert isinstance(out.compute(), scipy.sparse.sparse_matrix)
