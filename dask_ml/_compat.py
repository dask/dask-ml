import contextlib

import dask
import packaging.version
import dask.array as da
import dask.dataframe as dd
import numpy as np
from scipy import sparse

import sklearn

SK_VERSION = packaging.version.parse(sklearn.__version__)
HAS_MULTIPLE_METRICS = SK_VERSION >= packaging.version.parse("0.19.0")
DASK_VERSION = packaging.version.parse(dask.__version__)


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield


def hstack(tup):
    """Horizontally stack some arrays.

    Parameters
    ----------
    args : Iterable[array-like]
        Supports scipy sparse arrays, dask dataframes, dask arrays
        for NumPy arrays.
    """
    x = tup[0]

    if sparse.issparse(x):
        return sparse.hstack(tup)

    elif isinstance(x, (dd.Series, dd.DataFrame)):
        return dd.concat(tup, axis="columns")

    elif isinstance(x, da.Array):
        return da.hstack(tup)

    else:
        return np.hstack(tup)


def ones_like(a, dtype=None, chunks=None):
    if isinstance(a, da.Array):
        return da.ones_like(a, dtype=dtype, chunks=chunks)
    else:
        return np.ones_like(a, dtype=dtype)


def zeros_like(a, dtype=None, chunks=None):
    if isinstance(a, da.Array):
        return da.zeros_like(a, dtype=dtype, chunks=chunks)
    else:
        return np.zeros_like(a, dtype=dtype)
