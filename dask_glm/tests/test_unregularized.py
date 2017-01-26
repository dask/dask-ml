from __future__ import absolute_import, division, print_function

import math

import dask.array as da
import numpy as np
import pytest

from dask_glm.logistic import gradient_descent, bfgs, newton
from dask_glm.utils import sigmoid, make_y


def make_data(N,p, seed=20009):
    '''Given the desired number of observations (N) and
    the desired number of variables (p), creates
    random logistic data to test on.'''

    ## set the seeds
    da.random.seed(seed)
    np.random.seed(seed)

    X = np.random.random((N, p+1))
    X[:, p] = 1
    X = da.from_array(X, chunks=(N/5, p+1))
    y = make_y(X, beta=np.random.random(p+1))

    return X, y

@pytest.mark.parametrize('N, p, seed',
                        [(100, 2, 20009),
                        (250, 12, 90210),
                        (95, 6, 70605)])
def test_newton(N, p, seed):
    X, y = make_data(N, p, seed=seed)
    coefs = newton(X, y)
    p = sigmoid(X.dot(coefs).compute())

    assert np.isclose(y.compute().sum(), p.sum())
