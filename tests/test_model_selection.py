import numpy as np
import pandas as pd
from dask.distributed import Client
from scipy import stats
from sklearn.svm import SVC

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
    df = pd.DataFrame(data=arr)
    dsk = {}
    grid_search_keys = list(dms.utils.to_keys(dsk, arr, df))
    with Client() as client:
        data_futures = client.scatter([arr, df])
    assert grid_search_keys == [f.key for f in data_futures]
