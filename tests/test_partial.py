from sklearn.linear_model import SGDClassifier
import numpy as np
import dask
import dask.array as da
from dask_ml._partial import fit, predict
from dask_ml.datasets import make_classification
from dask_ml.wrappers import Incremental
from dask.array.utils import assert_eq

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


@pytest.mark.parametrize("array_lib", ['numpy', 'dask.array'])
@pytest.mark.parametrize("model_lib", ["dask-ml", "sklearn"])
def test_fit(array_lib, model_lib):
    with dask.config.set(scheduler='single-threaded'):
        arrays = {'numpy': [x, y, z], 'dask.array': [X, Y, Z]}
        models = {'sklearn': SGDClassifier(max_iter=5),
                  'dask-ml': Incremental(SGDClassifier(max_iter=5))}
        x_, y_, z_ = arrays[array_lib]
        est = models[model_lib]
        est = fit(est, x_, y_, classes=np.array([-1, 0, 1]))

        sol = est.predict(z)
        result = predict(est, z_)

        if array_lib == "dask.array":
            assert result.chunks == ((2, 2),)
            assert isinstance(result, da.Array)
            assert assert_eq(result, sol)
        elif array_lib == "numpy":
            assert isinstance(result, np.ndarray)
            assert assert_eq(result, sol)
        else:
            raise ValueError


@pytest.mark.parametrize("model_lib", ["dask-ml", "sklearn"])
def test_fit_need_same_input_types(model_lib):
    with dask.config.set(scheduler='single-threaded'):
        models = {'sklearn': SGDClassifier(max_iter=5),
                  'dask-ml': Incremental(SGDClassifier(max_iter=5))}
        est = models[model_lib]
        with pytest.raises(ValueError, match='X and y should be both'):
            est = fit(est, X, y, classes=np.array([-1, 0, 1]))

def test_fit_rechunking():
    n_classes = 2
    X, y = make_classification(chunks=20, n_classes=n_classes)
    X = X.rechunk({1: 10})

    assert X.numblocks[1] > 1

    clf = Incremental(SGDClassifier(max_iter=5))
    clf.fit(X, y, classes=list(range(n_classes)))
