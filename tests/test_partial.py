from sklearn.linear_model import SGDClassifier
import numpy as np
import dask
import dask.array as da
from dask_ml._partial import fit, predict
from dask_ml.datasets import make_classification
from dask_ml.wrappers import Incremental

import pytest


x = np.array([[1, 0],
              [2, 0],
              [3, 0],
              [4, 0],
              [0, 1],
              [0, 2],
              [3, 3],
              [4, 4]])

y = np.array([1, 1, 1, 1, -1, -1, 0, 0])

z = np.array([[1, -1],
              [-1, 1],
              [10, -10],
              [-10, 10]])

X = da.from_array(x, chunks=(3, 2))
Y = da.from_array(y, chunks=(3,))
Z = da.from_array(z, chunks=(2, 2))


@pytest.mark.parametrize("x_,y_,z_", [(x, y, z), (X, Y, Z)])
def test_fit(x_, y_, z_):
    with dask.config.set(scheduler='single-threaded'):
        sgd = SGDClassifier(max_iter=5)
        sgd = fit(sgd, x_, y_, classes=np.array([-1, 0, 1]))

        sol = sgd.predict(z)
        result = predict(sgd, z_)

        if isinstance(z_, da.Array):
            assert result.chunks == ((2, 2),)
            assert isinstance(result, da.Array)
            assert result.compute().tolist() == sol.tolist()
        elif isinstance(z_, np.ndarray):
            assert isinstance(result, np.ndarray)
            assert result.tolist() == sol.tolist()
        else:
            raise ValueError


def test_fit_rechunking():
    n_classes = 2
    X, y = make_classification(chunks=20, n_classes=n_classes)
    X = X.rechunk({1: 10})

    assert X.numblocks[1] > 1

    clf = Incremental(SGDClassifier(max_iter=5))
    clf.fit(X, y, classes=list(range(n_classes)))
