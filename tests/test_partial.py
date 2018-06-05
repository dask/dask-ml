from sklearn.linear_model import SGDClassifier
import numpy as np
import dask
from dask_ml._partial import fit, predict
import dask.array as da


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


def test_fit():
    with dask.config.set(scheduler='single-threaded'):
        sgd = SGDClassifier()

        sgd = fit(sgd, X, Y, classes=np.array([-1, 0, 1]))

        sol = sgd.predict(z)
        result = predict(sgd, Z)
        assert result.chunks == ((2, 2),)
        assert result.compute().tolist() == sol.tolist()
