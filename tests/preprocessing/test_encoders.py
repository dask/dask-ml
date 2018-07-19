import dask.array as da
import numpy as np

import dask_ml.preprocessing
import sklearn.preprocessing

X = np.array([["a"], ["a"], ["b"], ["c"]])
dX = da.from_array(X, 2)


def test_basic():
    a = sklearn.preprocessing.OneHotEncoder(sparse=False)
    b = dask_ml.preprocessing.OneHotEncoder(sparse=False)
    expected = a.fit_transform(X)
    result = b.fit_transform(dX)

    da.utils.assert_eq(result, expected)
