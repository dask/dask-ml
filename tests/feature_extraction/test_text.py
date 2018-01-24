import scipy.sparse
from numpy.testing import assert_array_equal
import sklearn.feature_extraction.text

import dask.bag as db
import dask.multiprocessing
from dask_ml.feature_extraction.text import HashingVectorizer
import pytest

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)


@pytest.mark.parametrize('get', (dask.get, dask.multiprocessing.get))
def test_hashing_vectorizer(get):
    vect_ref = sklearn.feature_extraction.text.HashingVectorizer()

    X_ref = vect_ref.fit_transform(JUNK_FOOD_DOCS)

    with dask.set_options(get=get):
        b = db.from_sequence(JUNK_FOOD_DOCS, npartitions=2)
        vect = HashingVectorizer()
        X_da = vect.fit_transform(b)

        X = scipy.sparse.vstack(X_da.compute()).asformat('csr')

    assert X_ref.shape == X.shape
    assert X_ref.nnz == X.nnz
    assert_array_equal(X.data, X_ref.data)
    assert_array_equal(X.indices, X_ref.indices)
    assert_array_equal(X.indptr, X_ref.indptr)
