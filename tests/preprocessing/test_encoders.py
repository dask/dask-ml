from distutils.version import LooseVersion

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import dask_ml.preprocessing
import sklearn.preprocessing
from dask_ml.utils import assert_estimator_equal

X = np.array([["a"], ["a"], ["b"], ["c"]])
dX = da.from_array(X, 2)
df = pd.DataFrame(X, columns=["A"]).apply(lambda x: x.astype("category"))
ddf = dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_basic_array(sparse, method):
    a = sklearn.preprocessing.OneHotEncoder(sparse=sparse)
    b = dask_ml.preprocessing.OneHotEncoder(sparse=sparse)

    if method == "fit":
        a.fit(X)
        b.fit(dX)
        expected = a.transform(X)
        result = b.transform(dX)
    else:
        expected = a.fit_transform(X)
        result = b.fit_transform(dX)

    assert_estimator_equal(
        a, b, exclude={"n_values_", "feature_indices_", "active_features_", "dtypes_"}
    )

    assert isinstance(result, da.Array)

    # can't use assert_eq since we're apparently making bad graphs
    # See TODO in `transform`.
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype

    if sparse:
        assert scipy.sparse.issparse(result.blocks[0].compute())
        result = result.map_blocks(lambda x: x.toarray(), dtype="f8").compute()
        da.utils.assert_eq(result, expected)
    else:
        result = result.compute()
        da.utils.assert_eq(result, expected)


@pytest.mark.parametrize(
    "sparse",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                dask.__version__ <= LooseVersion("0.18.1"),
                reason="Requires sparse get_dummies.",
            ),
        ),
        False,
    ],
)
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("dask_data", [df, ddf])  # we handle pandas and dask dataframes
def test_basic_dataframe(sparse, method, dask_data):
    a = sklearn.preprocessing.OneHotEncoder(sparse=sparse)
    b = dask_ml.preprocessing.OneHotEncoder(sparse=sparse)

    if method == "fit":
        a.fit(df)
        b.fit(dask_data)
        expected = a.transform(df)
        result = b.transform(dask_data)
    else:
        expected = a.fit_transform(df)
        result = b.fit_transform(dask_data)

    assert_estimator_equal(
        a, b, exclude={"n_values_", "feature_indices_", "active_features_", "dtypes_"}
    )

    assert isinstance(result, type(dask_data))
    assert len(result.columns) == expected.shape[1]
    assert (result.dtypes == np.float).all()  # TODO: dtypes

    da.utils.assert_eq(result.values, expected)
