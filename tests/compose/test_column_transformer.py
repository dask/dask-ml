import dask.dataframe as dd
import packaging.version
import pandas as pd
import pytest
import sklearn.preprocessing
from sklearn.base import clone

import dask_ml.preprocessing

try:
    import sklearn.compose
    import dask_ml.compose
except ImportError:
    from dask_ml._compat import SK_VERSION

    pytestmark = pytest.mark.skipif(
        SK_VERSION < packaging.version.parse("0.20.0.dev0"),
        reason="sklearn.compose added in 0.20.0",
    )


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
