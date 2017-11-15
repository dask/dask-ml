import inspect
from textwrap import dedent

import six
import numpy as np
import dask.array as da
from sklearn import datasets as _datasets


__all__ = ['make_counts']


chunks_doc = """
    This returns dask.Arrays instead of numpy.ndarrays.
    Requires an additional 'chunks' keyword to control the number
    of blocks in the arrays."""


def _wrap_maker(func):

    def inner(*args, **kwargs):
        chunks = kwargs.pop('chunks')
        X, y = func(*args, **kwargs)
        return (da.from_array(X, chunks=(chunks, X.shape[-1])),
                da.from_array(y, chunks=chunks))
    __all__.append(func.__name__)

    if not six.PY2:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        # TODO(py3): Make this keyword-only
        params.append(
            inspect.Parameter("chunks",
                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=None))
        inner.__signature__ = sig.replace(parameters=params)

    doc = func.__doc__.split("\n")
    doc = ['    ' + doc[0], chunks_doc] + doc[1:]
    inner.__doc__ = dedent('\n'.join(doc))
    inner.__name__ = func.__name__
    inner.__module__ = __name__

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
    >>> X, y = make_counts()
    """
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunks, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    rate = da.exp(z0)
    y = da.random.poisson(rate, size=1, chunks=(chunks,))
    return X, y


def make_poisson(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
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
    >>> X
    dask.array<da.random.normal, ..., chunksize=(100, 100)>
    >>> y
    dask.array<da.random.poisson, shape=(1000,), dtype=int64, chunksize=(100,)>
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
