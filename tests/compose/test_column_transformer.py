import dask.dataframe as dd
import pandas as pd
import pytest
import sklearn.compose
import sklearn.preprocessing
from sklearn.base import clone

import dask_ml.compose
import dask_ml.preprocessing

df = pd.DataFrame({"A": pd.Categorical(["a", "a", "b", "a"]), "B": [1.0, 2, 4, 5]})
ddf = dd.from_pandas(df, npartitions=2).reset_index(drop=True)  # unknown divisions


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
