"""
Daskified versions of sklearn.metrics.pairwise
"""
import warnings

import dask.array as da
import numpy as np
from dask import delayed
from dask.array.random import doc_wraps
from sklearn import metrics
from sklearn.metrics.pairwise import KERNEL_PARAMS

from ..utils import row_norms


def pairwise_distances_argmin_min(
    X, Y, axis=1, metric="euclidean", batch_size=None, metric_kwargs=None
):
    if batch_size is not None:
        msg = "'batch_size' is deprecated. Use sklearn.config_context instead.'"
        warnings.warn(msg, FutureWarning)

    XD = X.to_delayed().flatten().tolist()
    func = delayed(metrics.pairwise_distances_argmin_min, pure=True, nout=2)
    blocks = [func(x, Y, metric=metric, metric_kwargs=metric_kwargs) for x in XD]
    argmins, mins = zip(*blocks)

    argmins = [
        da.from_delayed(block, (chunksize,), np.int64)
        for block, chunksize in zip(argmins, X.chunks[0])
    ]
    # Scikit-learn seems to always use float64
    mins = [
        da.from_delayed(block, (chunksize,), "f8")
        for block, chunksize in zip(mins, X.chunks[0])
    ]
    argmins = da.concatenate(argmins)
    mins = da.concatenate(mins)
    return argmins, mins


def pairwise_distances(X, Y, metric="euclidean", n_jobs=None, **kwargs):
    if isinstance(Y, da.Array):
        raise TypeError("`Y` must be a numpy array")
    chunks = (X.chunks[0], (len(Y),))
    return X.map_blocks(
        metrics.pairwise_distances,
        Y,
        dtype=float,
        chunks=chunks,
        metric=metric,
        **kwargs
    )


def euclidean_distances(
    X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None
):
    if X_norm_squared is not None:
        XX = X_norm_squared
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError("Incompatible dimensions for X and X_norm_squared")
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]
    if X is Y:
        YY = XX.T
    elif Y_norm_squared is not None:
        if Y_norm_squared.ndim < 2:
            YY = Y_norm_squared[:, np.newaxis]
        else:
            YY = Y_norm_squared
        if YY.shape != (1, Y.shape[0]):
            raise ValueError("Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    # TODO: this often emits a warning. Silence it here?
    distances = -2 * da.dot(X, Y.T) + XX + YY
    distances = da.maximum(distances, 0)
    # TODO: scikit-learn sets the diagonal to 0 when X is Y.

    return distances if squared else da.sqrt(distances)


def check_pairwise_arrays(X, Y, precomputed=False):
    # XXX
    if Y is None:
        Y = X

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape "
                "(n_queries, n_indexed). Got (%d, %d) "
                "for %d indexed." % (X.shape[0], X.shape[1], Y.shape[0])
            )
    elif X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices: "
            "X.shape[1] == %d while Y.shape[1] == %d" % (X.shape[1], Y.shape[1])
        )
    return X, Y


# ----------------
# Kernel functions
# ----------------


@doc_wraps(metrics.pairwise.linear_kernel)
def linear_kernel(X, Y=None):
    X, Y = check_pairwise_arrays(X, Y)
    return da.dot(X, Y.T)


@doc_wraps(metrics.pairwise.rbf_kernel)
def rbf_kernel(X, Y=None, gamma=None):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K = da.exp(-gamma * K)
    return K


@doc_wraps(metrics.pairwise.polynomial_kernel)
def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = (gamma * da.dot(X, Y.T) + coef0) ** degree
    return K


@doc_wraps(metrics.pairwise.sigmoid_kernel)
def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = da.dot(X, Y.T)
    K *= gamma
    K += coef0
    K = da.tanh(K)
    return K


PAIRWISE_KERNEL_FUNCTIONS = {
    "rbf": rbf_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "sigmoid": sigmoid_kernel
    # TODO:
    # - cosine_similarity
    # - laplacian
    # - additive_chi2_kernel
    # - chid2_kernel
}


def pairwise_kernels(X, Y=None, metric="linear", filter_params=False, n_jobs=1, **kwds):
    from sklearn.gaussian_process.kernels import Kernel as GPKernel

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        raise NotImplementedError()
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = dict((k, kwds[k]) for k in kwds if k in KERNEL_PARAMS[metric])
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        raise NotImplementedError()
    else:
        raise ValueError("Unknown kernel %r" % metric)

    return func(X, Y, **kwds)
