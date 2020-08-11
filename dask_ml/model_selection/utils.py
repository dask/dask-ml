import copy
import warnings
from distutils.version import LooseVersion
from itertools import compress

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import scipy.sparse as sp
from dask.base import tokenize
from dask.delayed import Delayed, delayed
from sklearn.utils.validation import _is_arraylike, indexable

from ..utils import _num_samples

if LooseVersion(dask.__version__) > "0.15.4":
    from dask.base import is_dask_collection
else:
    from dask.base import Base

    def is_dask_collection(x):
        return isinstance(x, Base)


def _indexable(x):
    return indexable(x)[0]


def _maybe_indexable(x):
    return indexable(x)[0] if _is_arraylike(x) else x


def to_indexable(*args, **kwargs):
    """Ensure that all args are an indexable type.

    Conversion runs lazily for dask objects, immediately otherwise.

    Parameters
    ----------
    args : array_like or scalar
    allow_scalars : bool, optional
        Whether to allow scalars in args. Default is False.
    """
    if kwargs.get("allow_scalars", False):
        indexable = _maybe_indexable
    else:
        indexable = _indexable
    for x in args:
        if x is None or isinstance(x, (da.Array, dd.DataFrame)):
            yield x
        elif is_dask_collection(x):
            yield delayed(indexable, pure=True)(x)
        else:
            yield indexable(x)


def _index_param_value(num_samples, v, indices):
    """Private helper function for parameter value indexing.

    This determines whether a fit parameter `v` to a SearchCV.fit
    should be indexed along with `X` and `y`. Note that this differs
    from the scikit-learn version. They pass `X` and compute num_samples.
    We pass `num_samples` instead.
    """
    if not _is_arraylike(v) or _num_samples(v) != num_samples:
        # pass through: skip indexing
        return v
    if sp.issparse(v):
        v = v.tocsr()
    return _safe_indexing(v, indices)


def to_keys(dsk, *args):
    for x in args:
        if x is None:
            yield None
        elif isinstance(x, (da.Array, dd.DataFrame)):
            x = delayed(x)
            dsk.update(x.dask)
            yield x.key
        elif isinstance(x, Delayed):
            dsk.update(x.dask)
            yield x.key
        else:
            assert not is_dask_collection(x)
            key = type(x).__name__ + "-" + tokenize(x)
            dsk[key] = x
            yield key


def copy_estimator(est):
    # Semantically, we'd like to use `sklearn.clone` here instead. However,
    # `sklearn.clone` isn't threadsafe, so we don't want to call it in
    # tasks.  Since `est` is guaranteed to not be a fit estimator, we can
    # use `copy.deepcopy` here without fear of copying large data.
    return copy.deepcopy(est)


def unzip(itbl, n):
    return zip(*itbl) if itbl else [()] * n


class DeprecationDict(dict):
    """A dict which raises a warning when some keys are looked up
    Note, this does not raise a warning for __contains__ and iteration.
    It also will raise a warning even after the key has been manually set by
    the user.

    This implementation was copied from Scikit-Learn.

    See License information here:
    https://github.com/scikit-learn/scikit-learn/blob/master/README.rst
    """

    def __init__(self, *args, **kwargs):
        self._deprecations = {}
        super(DeprecationDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key in self._deprecations:
            warn_args, warn_kwargs = self._deprecations[key]
            warnings.warn(*warn_args, **warn_kwargs)
        return super(DeprecationDict, self).__getitem__(key)

    def get(self, key, default=None):
        """Return the value corresponding to key, else default.

        Parameters
        ----------
        key : any hashable object
            The key
        default : object, optional
            The default returned when key is not in dict
        """
        # dict does not implement it like this, hence it needs to be overridden
        try:
            return self[key]
        except KeyError:
            return default

    def add_warning(self, key, *args, **kwargs):
        """Add a warning to be triggered when the specified key is read

        Parameters
        ----------
        key : any hashable object
            The key
        """
        self._deprecations[key] = (args, kwargs)


def _safe_indexing(X, indices, axis=0):
    """Return rows, items or columns of X using indices.
    .. warning::
        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.
    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.
    Returns
    -------
    subset
        Subset of X on axis 0 or 1.
    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == "str":
        raise ValueError("String indexing is not supported with 'axis=0'")

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == "str" and not hasattr(X, "loc"):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)


def _determine_key_type(key):
    """Determine the data type of key.
    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.
    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = (
        "No valid specification of the columns. Only a scalar, list or "
        "slice of all integers or all strings, or boolean mask is "
        "allowed"
    )

    dtype_to_str = {int: "int", str: "str", bool: "bool", np.bool_: "bool"}
    array_dtype_to_str = {
        "i": "int",
        "u": "int",
        "b": "bool",
        "O": "str",
        "U": "str",
        "S": "str",
    }

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, list):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, "dtype"):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)


def _pandas_indexing(X, key, key_dtype, axis):
    """Index a pandas dataframe or a series."""
    if hasattr(key, "shape"):
        # Work-around for indexing with read-only key in pandas
        # FIXME: solved in pandas 0.25
        key = np.asarray(key)
        key = key if key.flags.writeable else key.copy()
    # check whether we should index with loc or iloc
    indexer = X.iloc if key_dtype == "int" else X.loc
    return indexer[:, key] if axis else indexer[key]


def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    if sp.issparse(array):
        if key_dtype == "bool":
            key = np.asarray(key)
    return array[key] if axis == 0 else array[:, key]


def _list_indexing(X, key, key_dtype):
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == "bool":
        # key is a boolean array-like
        return list(compress(X, key))
    # key is a integer array-like of key
    return [X[idx] for idx in key]


def none_or_passthrough(x):
    return x is None or isinstance(x, str) and x == "passthrough"


def none_or_drop(x):
    return x is None or isinstance(x, str) and x == "drop"
