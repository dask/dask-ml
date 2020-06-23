"""
Daskified versions of sklearn.metrics.pairwise
"""
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from dask import delayed
from dask.utils import derived_from
from sklearn import metrics
from sklearn.metrics.pairwise import KERNEL_PARAMS

from .._typing import ArrayLike
from ..utils import row_norms


def pairwise_distances_argmin_min(
    X: ArrayLike,
    Y: ArrayLike,
    axis: int = 1,
    metric: Union[str, Callable[[ArrayLike, ArrayLike], float]] = "euclidean",
    batch_size: Optional[int] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
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


def pairwise_distances(
    X: ArrayLike,
    Y: ArrayLike,
    metric: Union[str, Callable[[ArrayLike, ArrayLike], float]] = "euclidean",
    n_jobs: Optional[int] = None,
    **kwargs: Any
):
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
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    Y_norm_squared: Optional[ArrayLike] = None,
    squared: bool = False,
    X_norm_squared: Optional[ArrayLike] = None,
) -> ArrayLike:
    if Y is None:
        Y = X

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


def check_pairwise_arrays(
    X: ArrayLike, Y: ArrayLike, precomputed: bool = False
) -> Tuple[ArrayLike, ArrayLike]:
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


@derived_from(metrics.pairwise)
def linear_kernel(X: ArrayLike, Y: Optional[ArrayLike] = None) -> ArrayLike:
    X, Y = check_pairwise_arrays(X, Y)
    return da.dot(X, Y.T)


@derived_from(metrics.pairwise)
def rbf_kernel(
    X: ArrayLike, Y: Optional[ArrayLike] = None, gamma: Optional[float] = None
) -> ArrayLike:
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K = da.exp(-gamma * K)
    return K


@derived_from(metrics.pairwise)
def polynomial_kernel(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1,
) -> ArrayLike:
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = (gamma * da.dot(X, Y.T) + coef0) ** degree
    return K


@derived_from(metrics.pairwise)
def sigmoid_kernel(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    gamma: Optional[float] = None,
    coef0: float = 1,
) -> ArrayLike:
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


def pairwise_kernels(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    metric: Union[str, Callable[[ArrayLike, ArrayLike], float]] = "linear",
    filter_params: bool = False,
    n_jobs: Optional[int] = 1,
    **kwds
):
    from sklearn.gaussian_process.kernels import Kernel as GPKernel

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        raise NotImplementedError()
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = dict((k, kwds[k]) for k in kwds if k in KERNEL_PARAMS[metric])
        assert isinstance(metric, str)
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        raise NotImplementedError()
    else:
        raise ValueError("Unknown kernel %r" % metric)

    return func(X, Y, **kwds)
