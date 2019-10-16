"""
"""
import inspect
import sys
from functools import wraps

import dask.array as da
import dask.dataframe as dd
import numpy as np
from multipledispatch import dispatch

from .._utils import is_sparse


@dispatch(dd._Frame)
def exp(A):
    return da.exp(A)


@dispatch(dd._Frame)
def absolute(A):
    return da.absolute(A)


@dispatch(dd._Frame)
def sign(A):
    return da.sign(A)


@dispatch(dd._Frame)
def log1p(A):
    return da.log1p(A)


@dispatch(np.ndarray)
def add_intercept(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def _add_intercept(x):
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)

    if is_sparse(x):
        ones = sparse.COO(ones)

    return np.concatenate([x, ones], axis=1)


@dispatch(da.Array)  # noqa: F811
def add_intercept(X):
    if X.ndim != 2:
        raise ValueError("'X' should have 2 dimensions, not {}".format(X.ndim))

    if len(X.chunks[1]) > 1:
        msg = (
            "Chunking is only allowed on the first axis. "
            "Use 'array.rechunk({1: array.shape[1]})' to "
            "rechunk to a single block along the second axis."
        )
        raise ValueError(msg)

    chunks = (X.chunks[0], ((X.chunks[1][0] + 1),))
    return X.map_blocks(_add_intercept, dtype=X.dtype, chunks=chunks)


@dispatch(dd.DataFrame)  # noqa: F811
def add_intercept(X):
    columns = X.columns
    if "intercept" in columns:
        raise ValueError("'intercept' column already in 'X'")
    return X.assign(intercept=1)[["intercept"] + list(columns)]


def make_y(X, beta=np.array([1.5, -3]), chunks=2):
    z0 = X.dot(beta)
    y = da.random.random(z0.shape, chunks=z0.chunks) < sigmoid(z0)
    return y


def sigmoid(x):
    """Sigmoid function of x."""
    return 1 / (1 + exp(-x))


@dispatch(object)
def exp(A):
    return A.exp()


@dispatch(float)
def exp(A):
    return np.exp(A)


@dispatch(np.ndarray)
def exp(A):
    return np.exp(A)


@dispatch(da.Array)
def exp(A):
    return da.exp(A)


@dispatch(object)
def absolute(A):
    return abs(A)


@dispatch(np.ndarray)
def absolute(A):
    return np.absolute(A)


@dispatch(da.Array)
def absolute(A):
    return da.absolute(A)


@dispatch(object)
def sign(A):
    return A.sign()


@dispatch(np.ndarray)
def sign(A):
    return np.sign(A)


@dispatch(da.Array)
def sign(A):
    return da.sign(A)


@dispatch(object)
def log1p(A):
    return A.log1p()


@dispatch(np.ndarray)
def log1p(A):
    return np.log1p(A)


@dispatch(da.Array)
def log1p(A):
    return da.log1p(A)


@dispatch(object, object)
def dot(A, B):
    x = max([A, B], key=lambda x: getattr(x, "__array_priority__", 0))
    module = package_of(x)
    return module.dot(A, B)


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


def normalize(algo):
    @wraps(algo)
    def normalize_inputs(X, y, *args, **kwargs):
        normalize = kwargs.pop("normalize", True)
        if normalize:
            mean, std = da.compute(X.mean(axis=0), X.std(axis=0))
            mean, std = mean.copy(), std.copy()  # in case they are read-only
            intercept_idx = np.where(std == 0)
            if len(intercept_idx[0]) > 1:
                raise ValueError("Multiple constant columns detected!")
            mean[intercept_idx] = 0
            std[intercept_idx] = 1
            mean = mean if len(intercept_idx[0]) else np.zeros(mean.shape)
            Xn = (X - mean) / std
            out = algo(Xn, y, *args, **kwargs).copy()
            i_adj = np.sum(out * mean / std)
            out[intercept_idx] -= i_adj
            return out / std
        else:
            return algo(X, y, *args, **kwargs)

    return normalize_inputs


def package_of(obj):
    """ Return package containing object's definition
    Or return None if not found
    """
    # http://stackoverflow.com/questions/43462701/get-package-of-python-object/43462865#43462865
    mod = inspect.getmodule(obj)
    if not mod:
        return
    base, _sep, _stem = mod.__name__.partition(".")
    return sys.modules[base]


def scatter_array(arr, dask_client):
    """Scatter a large numpy array into workers
    Return the equivalent dask array
    """
    future_arr = dask_client.scatter(arr)
    return da.from_delayed(future_arr, shape=arr.shape, dtype=arr.dtype)


def is_dask_array_sparse(X):
    """
    Check using _meta if a dask array contains sparse arrays
    """
    try:
        import sparse
    except ImportError:
        return False

    return isinstance(X._meta, sparse.SparseArray)


try:
    import sparse
except ImportError:
    pass
else:

    @dispatch(sparse.COO)
    def exp(x):
        return np.exp(x.todense())

    @dispatch(sparse.SparseArray)
    def add_intercept(X):
        return sparse.concatenate([X, sparse.COO(np.ones((X.shape[0], 1)))], axis=1)
