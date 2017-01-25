from __future__ import absolute_import, division, print_function

import math

import dask.array as da
import numpy as np
import pytest

from dask_glm.gradient import gradient


def logit(y):
    return 1.0 / (1.0 + da.exp(-y))


M = 100
N = 100000
S = 2

X = np.random.randn(N, M)
X[:, 1] = 1.0
beta0 = np.random.randn(M)


def make_y(X, beta0=beta0):
    N, M = X.shape
    z0 = X.dot(beta0)
    z0 = da.compute(z0)[0]  # ensure z0 is a numpy array
    scl = S / z0.std()
    beta0 *= scl
    z0 *= scl
    y = np.random.rand(N) < logit(z0)
    return y, z0


y, z0 = make_y(X)
L0 = N * math.log(2.0)


dX = da.from_array(X, chunks=(N / 10, M))
dy = da.from_array(y, chunks=(N / 10,))


@pytest.mark.parametrize('X,y', [(X, y), (dX, dy)])
def test_gradient(X, y):
    gradient(X, y)
