import numpy as np
from scipy import stats
from sklearn.svm import SVC

from dask.distributed import Client

import dask_ml.model_selection as dms


def test_search_basic(xy_classification):
    X, y = xy_classification
    param_grid = {"class_weight": [None, "balanced"]}

    a = dms.GridSearchCV(SVC(kernel="rbf", gamma=0.1), param_grid)
    a.fit(X, y)

    param_dist = {"C": stats.uniform}
    b = dms.RandomizedSearchCV(SVC(kernel="rbf", gamma=0.1), param_dist)
    b.fit(X, y)


def test_to_keys_numpy_array():
    rng = np.random.RandomState(0)
    arr = rng.randn(20, 30)
    dsk = {}
    grid_search_key, = dms.to_keys(dsk, arr)
    with Client() as client:
        arr_future = client.scatter(arr)
    assert grid_search_key == arr_future.key
