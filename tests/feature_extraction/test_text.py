import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import sklearn.feature_extraction.text

import dask_ml.feature_extraction.text
from dask_ml.utils import assert_estimator_equal

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)


@pytest.mark.parametrize("container", ["bag", "series", "array"])
@pytest.mark.parametrize(
    "vect",
    [
        dask_ml.feature_extraction.text.HashingVectorizer(),
        dask_ml.feature_extraction.text.FeatureHasher(input_type="string"),
    ],
)
def test_basic(vect, container):
    b = db.from_sequence(JUNK_FOOD_DOCS, npartitions=2)
    if container == "series":
        b = b.to_dataframe(columns=["text"])["text"]
    elif container == "array":
        b = b.to_dataframe(columns=["text"])["text"].values

    vect_ref = vect._hasher(**vect.get_params())
    X_ref = vect_ref.fit_transform(b.compute())
    X_da = vect.fit_transform(b)

    assert_estimator_equal(vect_ref, vect)

    assert isinstance(X_da, da.Array)
    assert isinstance(X_da.blocks[0].compute(), scipy.sparse.csr_matrix)
    result = X_da.map_blocks(lambda x: x.toarray(), dtype=X_da.dtype)
    expected = X_ref.toarray()
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("container", ["bag", "series", "array"])
def test_hashing_vectorizer(container):
    b = db.from_sequence(JUNK_FOOD_DOCS, npartitions=2)
    if container == "series":
        b = b.to_dataframe(columns=["text"])["text"]
    elif container == "array":
        b = b.to_dataframe(columns=["text"])["text"].values

    vect_ref = sklearn.feature_extraction.text.HashingVectorizer()
    vect = dask_ml.feature_extraction.text.HashingVectorizer()

    X_ref = vect_ref.fit_transform(b.compute())
    X_da = vect.fit_transform(b)

    assert_estimator_equal(vect_ref, vect)

    assert isinstance(X_da, da.Array)
    assert isinstance(X_da.blocks[0].compute(), scipy.sparse.csr_matrix)

    result = X_da.map_blocks(lambda x: x.toarray(), dtype=X_da.dtype)
    expected = X_ref.toarray()
    # TODO: use dask.utils.assert_eq
    # Currently this fails chk_dask, as we end up with an integer key in the
    # dask graph.

    np.testing.assert_array_equal(result, expected)


def test_transforms_other():
    a = sklearn.feature_extraction.text.HashingVectorizer()
    b = dask_ml.feature_extraction.text.HashingVectorizer()

    X_a = a.fit_transform(JUNK_FOOD_DOCS)
    X_b = b.fit_transform(JUNK_FOOD_DOCS)
    assert_estimator_equal(a, b)

    np.testing.assert_array_equal(X_a.toarray(), X_b.toarray())


def test_transform_raises():
    vect = dask_ml.feature_extraction.text.HashingVectorizer()
    b = db.from_sequence(JUNK_FOOD_DOCS, npartitions=2)

    df = b.to_dataframe(columns=["text"])

    with pytest.raises(ValueError, match="1-dimensional array"):
        vect.transform(df)

    with pytest.raises(ValueError, match="1-dimensional array"):
        vect.transform(df.values)


def test_correct_meta():
    vect = dask_ml.feature_extraction.text.HashingVectorizer()
    X = dd.from_pandas(pd.Series(["some text", "to classifiy"]), 2)
    result = vect.fit_transform(X)
    assert scipy.sparse.issparse(result._meta)
    assert result._meta.dtype == "float64"
    assert result._meta.shape == (0, 0)
