from __future__ import absolute_import, division, print_function

import dask.array as da
import numpy as np
from multipledispatch import dispatch


@dispatch(np.ndarray)
def sigmoid(x):
    '''Sigmoid function of x.'''
    return 1 / (1 + np.exp(-x))


@dispatch(da.Array)
def sigmoid(x):
    '''Sigmoid function of x.'''
    return 1 / (1 + da.exp(-x))


@dispatch(np.ndarray)
def exp(A):
    return np.exp(A)


@dispatch(da.Array)
def exp(A):
    return da.exp(A)


@dispatch(np.ndarray)
def absolute(A):
    return np.absolute(A)


@dispatch(da.Array)
def absolute(A):
    return da.absolute(A)


@dispatch(np.ndarray)
def sign(A):
    return np.sign(A)


@dispatch(da.Array)
def sign(A):
    return da.sign(A)


@dispatch(np.ndarray)
def log1p(A):
    return np.log1p(A)


@dispatch(da.Array)
def log1p(A):
    return da.log1p(A)


@dispatch(da.Array, np.ndarray)
def dot(A, B):
    B = da.from_array(B, chunks=B.shape)
    return da.dot(A, B)


@dispatch(np.ndarray, da.Array)
def dot(A, B):
    A = da.from_array(A, chunks=A.shape)
    return da.dot(A, B)


@dispatch(np.ndarray, np.ndarray)
def dot(A, B):
    return np.dot(A, B)


@dispatch(da.Array, da.Array)
def dot(A, B):
    return da.dot(A, B)


@dispatch(np.ndarray)
def sum(A):
    return np.sum(A)


@dispatch(da.Array)
def sum(A):
    return da.sum(A)


def make_y(X, beta=np.array([1.5, -3]), chunks=2):
    n, p = X.shape
    z0 = X.dot(beta)
    y = da.random.random(z0.shape, chunks=z0.chunks) < sigmoid(z0)
    return y
