import dask.array as da
import dask.dataframe as dd
import numpy as np
import packaging.version
import pandas as pd
import pytest

import dask_ml.datasets
from dask_ml._compat import SK_VERSION
from dask_ml.utils import assert_estimator_equal

if SK_VERSION >= packaging.version.parse("0.20.0.dev0"):
    import sklearn.impute
    import dask_ml.impute
else:
    pytestmark = pytest.mark.skip(reason="Requires sklearn 0.20.0")

rng = np.random.RandomState(0)

X = rng.uniform(size=(10, 4))
X[X < 0.5] = np.nan

dX = da.from_array(X, chunks=5)
df = pd.DataFrame(X)
ddf = dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize("data", [X, dX, df, ddf])
def test_fit(data):
    a = sklearn.impute.SimpleImputer()
    b = dask_ml.impute.SimpleImputer()

    a.fit(X)
    b.fit(data)

    assert_estimator_equal(a, b)


@pytest.mark.parametrize("data", [X, dX, df, ddf])
def test_transform(data):
    a = sklearn.impute.SimpleImputer()
    b = dask_ml.impute.SimpleImputer()

    expected = a.fit_transform(X)
    result = b.fit_transform(data)

    assert isinstance(result, type(data))
    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
        result = result.values

    da.utils.assert_eq(result, expected)
