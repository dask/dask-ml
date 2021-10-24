import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import sklearn.feature_extraction.text
from distributed import Client

import dask_ml.feature_extraction.text
from dask_ml._compat import dummy_context
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


@pytest.mark.parametrize("give_vocabulary", [True, False])
@pytest.mark.parametrize("distributed", [True, False])
def test_count_vectorizer(give_vocabulary, distributed):
    m1 = sklearn.feature_extraction.text.CountVectorizer()
    b = db.from_sequence(JUNK_FOOD_DOCS, npartitions=2)
    r1 = m1.fit_transform(JUNK_FOOD_DOCS)

    if give_vocabulary:
        vocabulary = m1.vocabulary_
        m1 = sklearn.feature_extraction.text.CountVectorizer(vocabulary=vocabulary)
        r1 = m1.transform(JUNK_FOOD_DOCS)
    else:
        vocabulary = None

    m2 = dask_ml.feature_extraction.text.CountVectorizer(vocabulary=vocabulary)

    if distributed:
        client = Client()  # noqa
    else:
        client = dummy_context()

    if give_vocabulary:
        r2 = m2.transform(b)
    else:
        r2 = m2.fit_transform(b)

    with client:
        exclude = {"vocabulary_actor_", "stop_words_"}
        if give_vocabulary:
            # In scikit-learn, `.transform()` sets these.
            # This looks buggy.
            exclude |= {"vocabulary_", "fixed_vocabulary_"}

        assert_estimator_equal(m1, m2, exclude=exclude)
        assert isinstance(r2, da.Array)
        assert isinstance(r2._meta, scipy.sparse.csr_matrix)
        np.testing.assert_array_equal(r1.toarray(), r2.compute().toarray())

        r3 = m2.transform(b)
        assert isinstance(r3, da.Array)
        assert isinstance(r3._meta, scipy.sparse.csr_matrix)
        np.testing.assert_array_equal(r1.toarray(), r3.compute().toarray())

        if give_vocabulary:
            r4 = m2.fit_transform(b)
            assert isinstance(r4, da.Array)
            assert isinstance(r4._meta, scipy.sparse.csr_matrix)
            np.testing.assert_array_equal(r1.toarray(), r4.compute().toarray())


def test_count_vectorizer_remote_vocabulary():
    m1 = sklearn.feature_extraction.text.CountVectorizer().fit(JUNK_FOOD_DOCS)
    vocabulary = m1.vocabulary_
    r1 = m1.transform(JUNK_FOOD_DOCS)
    b = db.from_sequence(JUNK_FOOD_DOCS, npartitions=2)

    with Client() as client:
        (remote_vocabulary,) = client.scatter((vocabulary,), broadcast=True)
        m = dask_ml.feature_extraction.text.CountVectorizer(
            vocabulary=remote_vocabulary
        )
        r2 = m.transform(b)

        assert isinstance(r2, da.Array)
        assert isinstance(r2._meta, scipy.sparse.csr_matrix)
        np.testing.assert_array_equal(r1.toarray(), r2.compute().toarray())

        m = dask_ml.feature_extraction.text.CountVectorizer(
            vocabulary=remote_vocabulary
        )
        m.fit_transform(b)
        assert m.vocabulary_ is remote_vocabulary


@pytest.mark.parametrize("distributed", [True, False])
@pytest.mark.parametrize("collection_type", ["Bag", "Series"])
@pytest.mark.parametrize("norm", ["l1", "l2"])
@pytest.mark.parametrize("use_idf", [True, False])
@pytest.mark.parametrize("smooth_idf", [True, False])
@pytest.mark.parametrize("sublinear_tf", [True, False])
def test_tfidf_vectorizer(distributed,
                          collection_type,
                          norm,
                          use_idf,
                          smooth_idf,
                          sublinear_tf):
    m1 = (sklearn.feature_extraction.text
          .TfidfVectorizer(norm=norm,
                           use_idf=use_idf,
                           smooth_idf=smooth_idf,
                           sublinear_tf=sublinear_tf))
    if collection_type == "Bag":
        docs = db.from_sequence(JUNK_FOOD_DOCS, npartitions=2)
    elif collection_type == "Series":
        docs = dd.from_pandas(pd.Series(JUNK_FOOD_DOCS), npartitions=2)
    r1 = m1.fit_transform(JUNK_FOOD_DOCS)

    m2 = (dask_ml.feature_extraction.text
          .TfidfVectorizer(norm=norm,
                           use_idf=use_idf,
                           smooth_idf=smooth_idf,
                           sublinear_tf=sublinear_tf))

    if distributed:
        client = Client()  # noqa
    else:
        client = dummy_context()

    r2 = m2.fit_transform(docs)

    with client:
        exclude = {"vocabulary_actor_", "stop_words_"}
        if not use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual so we will exclude it in this case:
            exclude.add("idf_")
        assert_estimator_equal(m1, m2, exclude=exclude)
        assert isinstance(r2, da.Array)
        assert isinstance(r2._meta, scipy.sparse.csr_matrix)
        np.testing.assert_array_almost_equal(r1.toarray(),
                                             r2.compute().toarray())

        r3 = m2.transform(docs)
        assert isinstance(r3, da.Array)
        assert isinstance(r3._meta, scipy.sparse.csr_matrix)
        np.testing.assert_array_almost_equal(r1.toarray(),
                                             r3.compute().toarray())
