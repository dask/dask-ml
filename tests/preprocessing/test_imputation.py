import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

import dask_ml.datasets
import dask_ml.preprocessing
from dask_ml.utils import assert_estimator_equal

rng = np.random.RandomState(0)

X = rng.uniform(size=(10, 4))
X[X < 0.5] = np.nan

dX = da.from_array(X, chunks=5)
df = pd.DataFrame(X)
ddf = dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize("data", [X, dX, df, ddf])
def test_fit(data):
    a = sklearn.preprocessing.Imputer()
    b = dask_ml.preprocessing.Imputer()

    a.fit(X)
    b.fit(data)

    assert_estimator_equal(a, b)


@pytest.mark.parametrize("data", [X, dX, df, ddf])
def test_transform(data):
    a = sklearn.preprocessing.Imputer()
    b = dask_ml.preprocessing.Imputer()

    expected = a.fit_transform(X)
    result = b.fit_transform(data)

    assert isinstance(result, type(data))
    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
        result = result.values

    da.utils.assert_eq(result, expected)
