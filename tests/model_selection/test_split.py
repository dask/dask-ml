import dask.array as da
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

import dask_ml.model_selection


X, y = make_regression(n_samples=110, n_features=5)
dX = da.from_array(X, 50)
dy = da.from_array(y, 50)


def test_blockwise_shufflesplit():
    splitter = dask_ml.model_selection.ShuffleSplit(random_state=0)
    gen = splitter.split(dX)

    train_idx, test_idx = next(gen)
    assert isinstance(train_idx, da.Array)
    assert isinstance(test_idx, da.Array)

    assert train_idx.shape == (99,)  # 90% of 110
    assert test_idx.shape == (11,)

    assert train_idx.chunks == ((45, 45, 9),)
    assert test_idx.chunks == ((5, 5, 1),)

    counts = pd.value_counts(train_idx.compute())
    assert counts.max() == 1

    N = len(X)

    np.testing.assert_array_equal(
        np.unique(da.concatenate([train_idx, test_idx])), np.arange(N)
    )


def test_train_test_split():
    X_train, X_test, y_train, y_test = (
        dask_ml.model_selection.train_test_split(dX, dy, random_state=10)
    )

    assert len(X_train) == 99
    assert len(X_test) == 11

    assert X_train.chunks[0] == y_train.chunks[0]
    assert X_test.chunks[0] == y_test.chunks[0]
