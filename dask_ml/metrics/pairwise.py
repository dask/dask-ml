"""
Daskified versions of sklearn.metrics.pairwise
"""
import dask.array as da
import numpy as np
from dask import delayed
from sklearn import metrics

from ..utils import row_norms


def pairwise_distances_argmin_min(X, Y, axis=1, metric="euclidean",
                                  batch_size=None,
                                  metric_kwargs=None):
    if batch_size is None:
        batch_size = max(X.chunks[0])
    XD = X.to_delayed().flatten().tolist()
    func = delayed(metrics.pairwise_distances_argmin_min, pure=True, nout=2)
    blocks = [func(x, Y, metric=metric, batch_size=batch_size,
                   metric_kwargs=metric_kwargs)
              for x in XD]
    argmins, mins = zip(*blocks)

    argmins = [da.from_delayed(block, (chunksize,), np.int64)
               for block, chunksize in zip(argmins, X.chunks[0])]
    # Scikit-learn seems to always use float64
    mins = [da.from_delayed(block, (chunksize,), 'f8')
            for block, chunksize in zip(mins, X.chunks[0])]
    argmins = da.concatenate(argmins)
    mins = da.concatenate(mins)
    return argmins, mins


def pairwise_distances(X, Y, metric='euclidean', n_jobs=None, **kwargs):
    if isinstance(Y, da.Array):
        raise TypeError("`Y` must be a numpy array")
    chunks = (X.chunks[0], (len(Y),))
    return X.map_blocks(metrics.pairwise_distances, Y,
                        dtype=float, chunks=chunks,
                        metric=metric, **kwargs)


def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None):
    if X_norm_squared is not None:
        XX = X_norm_squared
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
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
            raise ValueError(
                "Incompatiable dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = -2 * X.dot(Y.T) + XX + YY
    distances = da.maximum(distances, 0)
    # TODO: scikit-learn sets the diagonal to 0 when X is Y.

    return distances if squared else da.sqrt(distances)
