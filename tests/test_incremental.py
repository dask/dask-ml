import dask.array as da
from dask.array.utils import assert_eq
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier

from dask_ml.wrappers import Incremental
from dask_ml.utils import assert_estimator_equal


def test_incremental_basic(xy_classification):
    X, y = xy_classification
    est1 = SGDClassifier(random_state=0)
    est2 = clone(est1)

    clf = Incremental(est1, classes=[0, 1])
    result = clf.fit(X, y)
    for slice_ in da.core.slices_from_chunks(X.chunks):
        est2.partial_fit(X[slice_], y[slice_[0]], classes=[0, 1])

    assert result is clf

    assert isinstance(result.estimator.coef_, np.ndarray)
    np.testing.assert_array_almost_equal(result.estimator.coef_, est2.coef_)

    assert_estimator_equal(clf.estimator, est2, exclude=['loss_function_'])

    #  Predict
    result = clf.predict(X)
    expected = est2.predict(X)
    assert isinstance(result, da.Array)
    assert_eq(result, expected)

    # score
    result = clf.score(X, y)
    expected = est2.score(X, y)
    # assert isinstance(result, da.Array)
    assert_eq(result, expected)

    clf = Incremental(SGDClassifier(random_state=0), classes=[0, 1])
    clf.partial_fit(X, y)
    assert_estimator_equal(clf.estimator, est2, exclude=['loss_function_'])
