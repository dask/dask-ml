from functools import wraps
from sklearn import datasets as _datasets
import numpy as np
import dask.array as da


__all__ = ['make_counts']


def _wrap_maker(func):
    @wraps(func)
    def inner(*args, **kwargs):
        chunks = kwargs.pop('chunks')
        X, y = func(*args, **kwargs)
        return (da.from_array(X, chunks=(chunks, X.shape[-1])),
                da.from_array(y, chunks=chunks))
    __all__.append(func.__name__)
    return inner


def make_counts(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
                chunks=100):
    """
    Generate a dummy dataset for modeling count data.

    Parameters
    ----------
    n_samples : int
        number of rows in the output array
    n_features : int
        number of columns (features) in the output array
    n_informative : int
        number of features that are correlated with the outcome
    scale : float
        Scale the true coefficient array by this
    chunks : int
        Number of rows per dask array block.

    Returns
    -------
    X : dask.array, size ``(n_samples, n_features)``
    y : dask.array, size ``(n_samples,)``
        array of non-negative integer-valued data

    Examples
    --------
    >>> X, y = make_classification()
    """
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunks, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    rate = da.exp(z0)
    y = da.random.poisson(rate, size=1, chunks=(chunks,))
    return X, y


make_classification = _wrap_maker(_datasets.make_classification)
make_regression = _wrap_maker(_datasets.make_regression)
make_blobs = _wrap_maker(_datasets.make_blobs)
