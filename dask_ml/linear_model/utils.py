"""
"""

import dask.array as da
import dask.dataframe as dd
import numpy as np
from multipledispatch import dispatch

if getattr(dd, "_dask_expr_enabled", lambda: False)():
    import dask_expr

    @dispatch(dask_expr.FrameBase)
    def exp(A):
        return da.exp(A)

    @dispatch(dask_expr.FrameBase)
    def absolute(A):
        return da.absolute(A)

    @dispatch(dask_expr.FrameBase)
    def sign(A):
        return da.sign(A)

    @dispatch(dask_expr.FrameBase)
    def log1p(A):
        return da.log1p(A)

    @dispatch(dask_expr.FrameBase)  # noqa: F811
    def add_intercept(X):  # noqa: F811
        columns = X.columns
        if "intercept" in columns:
            raise ValueError("'intercept' column already in 'X'")
        return X.assign(intercept=1)[["intercept"] + list(columns)]

else:

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

    @dispatch(dd._Frame)  # noqa: F811
    def add_intercept(X):  # noqa: F811
        columns = X.columns
        if "intercept" in columns:
            raise ValueError("'intercept' column already in 'X'")
        return X.assign(intercept=1)[["intercept"] + list(columns)]


@dispatch(np.ndarray)  # noqa: F811
def add_intercept(X):  # noqa: F811
    return _add_intercept(X)


def _add_intercept(x):
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.concatenate([x, ones], axis=1)


@dispatch(da.Array)  # noqa: F811
def add_intercept(X):  # noqa: F811
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


@dispatch(np.ndarray)  # noqa: F811
def lr_prob_stack(prob):  # noqa: F811
    return np.vstack([1 - prob, prob]).T


@dispatch(da.Array)  # noqa: F811
def lr_prob_stack(prob):  # noqa: F811
    return da.vstack([1 - prob, prob]).T
