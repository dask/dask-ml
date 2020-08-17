import dask.array as da
import dask.dataframe as dd
import numpy as np
import packaging.version
import pandas as pd
import pytest
import scipy.sparse
import sklearn.preprocessing

import dask_ml.preprocessing
from dask_ml._compat import DASK_2_20_0, PANDAS_VERSION
from dask_ml.utils import assert_estimator_equal

X = np.array([["a"], ["a"], ["b"], ["c"]])
dX = da.from_array(X, 2)
df = pd.DataFrame(X, columns=["A"]).apply(lambda x: x.astype("category"))
ddf = dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("categories", ["auto", [["a", "b", "c"]]])
@pytest.mark.skipif(not DASK_2_20_0, reason="Fixed in Dask 2.20.0")
def test_basic_array(sparse, method, categories):
    a = sklearn.preprocessing.OneHotEncoder(categories=categories, sparse=sparse)
    b = dask_ml.preprocessing.OneHotEncoder(categories=categories, sparse=sparse)

    if method == "fit":
        a.fit(X)
        b.fit(dX)
        expected = a.transform(X)
        result = b.transform(dX)
    else:
        expected = a.fit_transform(X)
        result = b.fit_transform(dX)

    assert_estimator_equal(
        a,
        b,
        exclude={
            "n_values_",
            "feature_indices_",
            "active_features_",
            "dtypes_",
            "drop_idx_",
        },
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


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("dask_data", [df, ddf])  # we handle pandas and dask dataframes
@pytest.mark.parametrize("dtype", [np.float, np.uint8])
def test_basic_dataframe(sparse, method, dask_data, dtype):
    a = sklearn.preprocessing.OneHotEncoder(sparse=sparse, dtype=dtype)
    b = dask_ml.preprocessing.OneHotEncoder(sparse=sparse, dtype=dtype)

    if method == "fit":
        a.fit(df)
        b.fit(dask_data)
        expected = a.transform(df)
        result = b.transform(dask_data)
    else:
        expected = a.fit_transform(df)
        result = b.fit_transform(dask_data)

    assert_estimator_equal(
        a,
        b,
        exclude={
            "n_values_",
            "feature_indices_",
            "active_features_",
            "dtypes_",
            "drop_idx_",
        },
    )

    assert isinstance(result, type(dask_data))
    assert len(result.columns) == expected.shape[1]
    if sparse and PANDAS_VERSION >= packaging.version.parse("0.24.0"):
        # pandas sparse ExtensionDtype interface
        dtype = pd.SparseDtype(dtype, dtype(0))
    assert (result.dtypes == dtype).all()

    da.utils.assert_eq(result.values, expected)


def test_invalid_handle_input():
    enc = dask_ml.preprocessing.OneHotEncoder(handle_unknown="ignore")
    with pytest.raises(NotImplementedError):
        enc.fit(dX)

    enc = dask_ml.preprocessing.OneHotEncoder(handle_unknown="invalid")
    with pytest.raises(ValueError):
        enc.fit(dX)


def test_onehotencoder_drop_raises():
    dask_ml.preprocessing.OneHotEncoder()
    with pytest.raises(NotImplementedError):
        dask_ml.preprocessing.OneHotEncoder(drop="first")


def test_onehotencoder_dataframe_with_categories():
    # https://github.com/dask/dask-ml/issues/726
    enc = dask_ml.preprocessing.OneHotEncoder(
        categories=[["a", "b", "c"], ["a", "b"]], sparse=False
    )
    ddf = dd.from_pandas(
        pd.DataFrame({"A": ["a", "b", "b", "a"], "B": ["a", "b", "b", "b"]}),
        npartitions=1,
    )
    result = enc.fit_transform(ddf)
    expected = dd.from_pandas(
        pd.DataFrame(
            {
                "A_a": [1, 0, 0, 1],
                "A_b": [0, 1, 1, 0],
                "A_c": [0, 0, 0, 0],
                "B_a": [1, 0, 0, 0],
                "B_b": [0, 0, 0, 0],
            }
        ),
        npartitions=1,
    )
    assert_estimator_equal(result, expected)


def test_handles_numpy():
    enc = dask_ml.preprocessing.OneHotEncoder()
    enc.fit(X)


@pytest.mark.parametrize("data", [df, ddf])
def test_dataframe_requires_all_categorical(data):
    data = data.assign(B=1)
    enc = dask_ml.preprocessing.OneHotEncoder()
    with pytest.raises(ValueError) as e:
        enc.fit(data)

    assert e.match("All columns must be Categorical dtype")


def test_unknown_category_transform():
    df2 = ddf.copy()
    df2["A"] = ddf.A.cat.add_categories("new!")

    enc = dask_ml.preprocessing.OneHotEncoder()
    enc.fit(ddf)

    with pytest.raises(ValueError, match="Different CategoricalDtype"):
        enc.transform(df2)


def test_different_shape_raises():
    df2 = ddf.copy()
    df2["B"] = ddf.A.cat.add_categories("new!")

    enc = dask_ml.preprocessing.OneHotEncoder()
    enc.fit(ddf)

    with pytest.raises(ValueError, match="Number of columns"):
        enc.transform(df2)


@pytest.mark.skipif(not DASK_2_20_0, reason="Fixed in Dask 2.20.0")
def test_unknown_category_transform_array():
    x2 = da.from_array(np.array([["a"], ["b"], ["c"], ["d"]]), chunks=2)
    enc = dask_ml.preprocessing.OneHotEncoder()
    enc.fit(dX)

    result = enc.transform(x2)
    match = r"Block contains previously unseen values \['d'\].*\n+.*Block info"
    with pytest.raises(ValueError, match=match):
        result.compute()
