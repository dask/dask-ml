"""
Daskified versions of sklearn.metrics.pairwise
"""
import dask.array as da
import numpy as np
from dask import delayed
from sklearn import metrics


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


def row_norms(X, squared=False):
    norms = (X * X).sum(1)
    if not squared:
        norms = da.sqrt(norms)
    return norms
