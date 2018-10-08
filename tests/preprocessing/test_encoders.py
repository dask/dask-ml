import dask.array as da
import dask.dataframe as dd
import numpy as np
import packaging.version
import pandas as pd
import pytest
import scipy.sparse
import sklearn.preprocessing

import dask_ml.preprocessing
from dask_ml._compat import DASK_VERSION
from dask_ml.utils import assert_estimator_equal

X = np.array([["a"], ["a"], ["b"], ["c"]])
dX = da.from_array(X, 2)
df = pd.DataFrame(X, columns=["A"]).apply(lambda x: x.astype("category"))
ddf = dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("categories", ["auto", [["a", "b", "c"]]])
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
                DASK_VERSION <= packaging.version.parse("0.18.1"),
                reason="Requires sparse get_dummies.",
            ),
        ),
        False,
    ],
)
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
        a, b, exclude={"n_values_", "feature_indices_", "active_features_", "dtypes_"}
    )

    assert isinstance(result, type(dask_data))
    assert len(result.columns) == expected.shape[1]
    assert (result.dtypes == dtype).all()

    da.utils.assert_eq(result.values, expected)


def test_invalid_handle_input():
    enc = dask_ml.preprocessing.OneHotEncoder(handle_unknown="ignore")
    with pytest.raises(NotImplementedError):
        enc.fit(dX)

    enc = dask_ml.preprocessing.OneHotEncoder(handle_unknown="invalid")
    with pytest.raises(ValueError):
        enc.fit(dX)


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


@pytest.mark.parametrize("data", [df, ddf])
def test_dataframe_prohibits_categories(data):
    enc = dask_ml.preprocessing.OneHotEncoder(categories=[["a", "b"]])
    with pytest.raises(ValueError) as e:
        enc.fit(data)

    assert e.match("Cannot specify 'categories'")


def test_unknown_category_transform():
    df2 = ddf.copy()
    df2["A"] = ddf.A.cat.add_categories("new!")

    enc = dask_ml.preprocessing.OneHotEncoder()
    enc.fit(ddf)

    with pytest.raises(ValueError) as e:
        enc.transform(df2)

    assert e.match("Different CategoricalDtype for fit and transform")


def test_unknown_category_transform_array():
    x2 = da.from_array(np.array([["a"], ["b"], ["c"], ["d"]]), chunks=2)
    enc = dask_ml.preprocessing.OneHotEncoder()
    enc.fit(dX)

    result = enc.transform(x2)
    with pytest.raises(ValueError) as e:
        result.compute()

    assert e.match("Block contains previously")
    assert "d" in str(e)
    assert "Block info" in str(e.value)
