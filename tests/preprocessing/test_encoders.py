import dask.array as da
import numpy as np
import pytest

import dask_ml.preprocessing
import sklearn.preprocessing
from dask_ml.utils import assert_estimator_equal

X = np.array([["a"], ["a"], ["b"], ["c"]])
dX = da.from_array(X, 2)


@pytest.mark.parametrize("sparse", [True, False])
def test_basic(sparse):
    a = sklearn.preprocessing.OneHotEncoder(sparse=sparse)
    b = dask_ml.preprocessing.OneHotEncoder(sparse=sparse)
    expected = a.fit_transform(X)
    result = b.fit_transform(dX)
    import pdb; pdb.set_trace()
    assert isinstance(result, da.Array)

    da.utils.assert_eq(result, expected)
    assert_estimator_equal(
        a, b, exclude={"n_values_", "feature_indices_", "active_features_"}
    )
