import dask.array as da
import dask.dataframe as dd
import numpy as np
import packaging.version
import pandas as pd
import pytest
import sklearn.compose
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
from scipy import sparse
from sklearn.base import BaseEstimator, clone

import dask_ml.compose
import dask_ml.preprocessing
from dask_ml._compat import SK_VERSION
from dask_ml.feature_extraction.text import HashingVectorizer

df = pd.DataFrame({"A": pd.Categorical(["a", "a", "b", "a"]), "B": [1.0, 2, 4, 5]})
ddf = dd.from_pandas(df, npartitions=2).reset_index(drop=True)  # unknown divisions


def test_column_transformer_preserves_pandas_output():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[9, 12, 21])
    ct = dask_ml.compose.ColumnTransformer(
        [("second", "passthrough", ["b"]), ("first", "passthrough", ["a"])]
    )

    pd.testing.assert_frame_equal(ct.fit_transform(X), X[["b", "a"]])
    pd.testing.assert_frame_equal(ct.transform(X), X[["b", "a"]])


def test_column_transformer_rejects_mismatched_pandas_indexes():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[9, 12, 21])
    reset_index = sklearn.preprocessing.FunctionTransformer(
        lambda frame: frame.reset_index(drop=True)
    )
    ct = dask_ml.compose.ColumnTransformer(
        [("first", "passthrough", ["a"]), ("second", reset_index, ["b"])]
    )

    with pytest.raises(ValueError, match="inconsistent number of samples"):
        ct.fit_transform(X)


def test_column_transformer_hashing_vectorizers():
    X = np.array([["example", "lorem"], ["text", "ipsum"], ["another", "sit"]])
    lazy_X = da.from_array(X, chunks=(2, 2))
    ct = dask_ml.compose.ColumnTransformer(
        [
            ("title", HashingVectorizer(n_features=8), 0),
            ("body", HashingVectorizer(n_features=8), 1),
        ]
    )
    reference = sklearn.feature_extraction.text.HashingVectorizer(n_features=8)
    expected = sparse.hstack([reference.transform(X[:, col]) for col in range(2)])

    fitted = ct.fit_transform(lazy_X)
    transformed = ct.transform(lazy_X)

    for result in [fitted, transformed]:
        assert isinstance(result, da.Array)
        actual = result.compute()
        assert sparse.issparse(actual)
        np.testing.assert_array_equal(actual.toarray(), expected.toarray())


@pytest.mark.skip(
    reason="ValueError: Specifying the columns using strings is only "
    + "supported for dataframes."
)
def test_column_transformer():
    # Ordering of make_column_transformer was changed from
    # (columns, transformer) to (transformer, columns) in version 0.20.1 of scikit-learn
    # See https://github.com/scikit-learn/scikit-learn/pull/12626
    if SK_VERSION < packaging.version.parse("0.20.1"):
        a = sklearn.compose.make_column_transformer(
            (["A"], sklearn.preprocessing.OneHotEncoder(sparse_output=False)),
            (["B"], sklearn.preprocessing.StandardScaler()),
        )
        b = dask_ml.compose.make_column_transformer(
            (["A"], dask_ml.preprocessing.OneHotEncoder(sparse_output=False)),
            (["B"], dask_ml.preprocessing.StandardScaler()),
        )
    else:
        a = sklearn.compose.make_column_transformer(
            (sklearn.preprocessing.OneHotEncoder(sparse_output=False), ["A"]),
            (sklearn.preprocessing.StandardScaler(), ["B"]),
        )
        b = dask_ml.compose.make_column_transformer(
            (dask_ml.preprocessing.OneHotEncoder(sparse_output=False), ["A"]),
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


@pytest.mark.skip(
    reason="ValueError: Specifying the columns using strings is only "
    + "supported for dataframes."
)
def test_column_transformer_unk_chunksize():
    names = ["a", "b", "c"]
    x = dd.from_pandas(pd.DataFrame(np.arange(12).reshape(4, 3), columns=names), 2)
    features = sklearn.pipeline.Pipeline(
        [
            (
                "features",
                sklearn.pipeline.FeatureUnion(
                    [
                        (
                            "ratios",
                            dask_ml.compose.ColumnTransformer(
                                [
                                    ("a_b", SumTransformer(one_d=False), ["a", "b"]),
                                    ("b_c", SumTransformer(one_d=False), ["b", "c"]),
                                ]
                            ),
                        )
                    ]
                ),
            )
        ]
    )
    out = features.fit_transform(x)

    exp = np.array([[1, 3], [7, 9], [13, 15], [19, 21]])
    np.testing.assert_array_equal(out, exp)


@pytest.mark.skip(
    reason="ValueError: Specifying the columns using strings is only "
    + "supported for dataframes."
)
def test_sklearn_col_trans_disallows_hstack_then_block():
    # Test that sklearn ColumnTransformer (to which dask-ml ColumnTransformer
    # delegates) disallows 1-D values.  This shows that if da.hstack were
    # used in dask's ColumnTransformer, the `then` branch in hstack would not
    # be executed because a ValueError is thrown by sklearn before executing
    # that code path.

    # This is the same example as above except that `one_d=True`.
    names = ["a", "b", "c"]
    x = dd.from_pandas(pd.DataFrame(np.arange(12).reshape(4, 3), columns=names), 2)
    features = sklearn.pipeline.Pipeline(
        [
            (
                "features",
                sklearn.pipeline.FeatureUnion(
                    [
                        (
                            "ratios",
                            dask_ml.compose.ColumnTransformer(
                                [
                                    ("a_b", SumTransformer(one_d=True), ["a", "b"]),
                                    ("b_c", SumTransformer(one_d=True), ["b", "c"]),
                                ]
                            ),
                        )
                    ]
                ),
            )
        ]
    )

    exp_msg = (
        "The output of the .a_b. transformer should be 2D "
        ".scipy matrix, array, or pandas DataFrame.."
    )

    with pytest.raises(ValueError, match=exp_msg) as ex:
        features.fit_transform(x)

    cause = ex.traceback[-1]
    place = cause.frame.f_globals["__name__"]
    func = cause.name
    assert place == "sklearn.compose._column_transformer"
    assert func == "_validate_output"


# Some basic transformer.
class SumTransformer(BaseEstimator):
    def __init__(self, one_d):
        self.one_d = one_d

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.map_partitions(lambda x: self.to_vec(x.values.sum(axis=-1)))

    def to_vec(self, x):
        if self.one_d:
            return x.flatten()
        else:
            return x.reshape(-1, 1)
