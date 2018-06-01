import dask.array as da
from dask.array.utils import assert_eq
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier

from dask_ml.wrappers import Blockwise, make_blockwise
from dask_ml.utils import assert_estimator_equal


@pytest.mark.parametrize('maker', [Blockwise, make_blockwise])
def test_blockwise_basic(xy_classification, maker):
    X, y = xy_classification
    est1 = SGDClassifier(random_state=0)
    est2 = clone(est1)

    clf = maker(est1, classes=[0, 1])
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
