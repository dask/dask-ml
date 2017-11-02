from dask.array.utils import assert_eq
import sklearn.kernel_approximation as sk

import dask_ml.kernel_approximation as dk
from dask_ml.datasets import make_classification
from dask_ml.utils import assert_estimator_equal


X, y = make_classification(n_samples=1000, chunks=500)


def test_basic():
    a = dk.Nystroem(random_state=0)
    b = sk.Nystroem(random_state=0)

    a.fit(X.compute())
    b.fit(X)
    assert_estimator_equal(a, b)

    at = a.transform(X.compute())
    bt = b.transform(X)
    assert_eq(at, bt)
