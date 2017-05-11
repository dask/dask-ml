import numpy as np
import dask.array as da
from dask_glm.utils import exp


def make_classification(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
                        chunksize=100):
    """
    Generate a dummy dataset for classification tasks.

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
    chunksize : int
        Number of rows per dask array block.

    Returns
    -------
    X : dask.array, size ``(n_samples, n_features)``
    y : dask.array, size ``(n_samples,)``
        boolean-valued array

    Examples
    --------
    >>> X, y = make_classification()
    >>> X
    dask.array<da.random.normal, shape=(1000, 100), dtype=float64, chunksize=(100, 100)>
    >>> y
    dask.array<lt, shape=(1000,), dtype=bool, chunksize=(100,)>
    """
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunksize, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    y = da.random.random(z0.shape, chunks=(chunksize,)) < 1 / (1 + da.exp(-z0))
    return X, y


def make_regression(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
                    chunksize=100):
    """
    Generate a dummy dataset for regression tasks.

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
    chunksize : int
        Number of rows per dask array block.

    Returns
    -------
    X : dask.array, size ``(n_samples, n_features)``
    y : dask.array, size ``(n_samples,)``
        real-valued array

    Examples
    --------
    >>> X, y = make_regression()
    >>> X
    dask.array<da.random.normal, shape=(1000, 100), dtype=float64, chunksize=(100, 100)>
    >>> y
    dask.array<da.random.random_sample, shape=(1000,), dtype=float64, chunksize=(100,)>
    """
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunksize, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    y = da.random.random(z0.shape, chunks=(chunksize,))
    return X, y


def make_poisson(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
                 chunksize=100):
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
    chunksize : int
        Number of rows per dask array block.

    Returns
    -------
    X : dask.array, size ``(n_samples, n_features)``
    y : dask.array, size ``(n_samples,)``
        array of non-negative integer-valued data

    Examples
    --------
    >>> X, y = make_classification()
    >>> X
    dask.array<da.random.normal, shape=(1000, 100), dtype=float64, chunksize=(100, 100)>
    >>> y
    dask.array<da.random.poisson, shape=(1000,), dtype=int64, chunksize=(100,)>
    """
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunksize, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    rate = exp(z0)
    y = da.random.poisson(rate, size=1, chunks=(chunksize,))
    return X, y
