import contextlib
import datetime
import functools
import logging
from collections.abc import Sequence
from multiprocessing import cpu_count
from numbers import Integral
from timeit import default_timer as tic

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sklearn.utils.extmath as skm
import sklearn.utils.validation as sk_validation
from dask import delayed
from dask.array.utils import assert_eq as assert_eq_ar
from dask.dataframe.utils import assert_eq as assert_eq_df

from ._utils import ConstantFunction

logger = logging.getLogger()


def _svd_flip_copy(x, y, u_based_decision=True):
    # If the array is locked, copy the array and transpose it
    # This happens with a very large array > 1TB
    # GH: issue 592
    try:
        return skm.svd_flip(x, y, u_based_decision=u_based_decision)
    except ValueError:
        return skm.svd_flip(x.copy(), y.copy(), u_based_decision=u_based_decision)


def svd_flip(u, v, u_based_decision=True):
    u2, v2 = delayed(_svd_flip_copy, nout=2)(u, v, u_based_decision=u_based_decision)
    u = da.from_delayed(u2, shape=u.shape, dtype=u.dtype)
    v = da.from_delayed(v2, shape=v.shape, dtype=v.dtype)
    return u, v


svd_flip.__doc__ = skm.svd_flip.__doc__


def flip_vector_signs(x, axis):
    """ Flip vector signs to align them for comparison

    Parameters
    ----------
    x : 2D array_like
        Matrix containing vectors in rows or columns
    axis : int, 0 or 1
        Axis in which vectors reside
    """
    assert x.ndim == 2
    signs = np.sum(x, axis=axis, keepdims=True)
    signs = signs.dtype.type(2) * ((signs >= 0) - signs.dtype.type(0.5))
    return x * signs


def slice_columns(X, columns):
    if isinstance(X, dd.DataFrame):
        return X[list(X.columns) if columns is None else columns]
    else:
        return X


def handle_zeros_in_scale(scale):
    scale = scale.copy()
    if isinstance(scale, (np.ndarray, da.Array)):
        scale[scale == 0.0] = 1.0
    elif isinstance(scale, (pd.Series, dd.Series)):
        scale = scale.where(scale != 0, 1)
    return scale


def row_norms(X, squared=False):
    if isinstance(X, np.ndarray):
        return skm.row_norms(X, squared=squared)
    return X.map_blocks(
        skm.row_norms, chunks=(X.chunks[0],), drop_axis=1, squared=squared
    )


def assert_estimator_equal(left, right, exclude=None, **kwargs):
    """Check that two Estimators are equal

    Parameters
    ----------
    left, right : Estimators
    exclude : str or sequence of str
        attributes to skip in the check
    kwargs : dict
        Passed through to the dask `assert_eq` method.

    """
    left_attrs = [x for x in dir(left) if x.endswith("_") and not x.startswith("_")]
    right_attrs = [x for x in dir(right) if x.endswith("_") and not x.startswith("_")]
    if exclude is None:
        exclude = set()
    elif isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude)

    left_attrs2 = set(left_attrs) - exclude
    right_attrs2 = set(right_attrs) - exclude

    assert left_attrs2 == right_attrs2, left_attrs2 ^ right_attrs2

    for attr in left_attrs2:
        l = getattr(left, attr)
        r = getattr(right, attr)
        _assert_eq(l, r, name=attr, **kwargs)


def check_array(
    array,
    *args,
    accept_dask_array=True,
    accept_dask_dataframe=False,
    accept_unknown_chunks=False,
    accept_multiple_blocks=False,
    preserve_pandas_dataframe=False,
    remove_zero_chunks=True,
    **kwargs
):
    """Validate inputs

    Parameters
    ----------
    accept_dask_array : bool, default True
    accept_dask_dataframe : bool, default False
    accept_unknown_chunks : bool, default False
        For dask Arrays, whether to allow the `.chunks` attribute to contain
        any unknown values
    accept_multiple_blocks : bool, default False
        For dask Arrays, whether to allow multiple blocks along the second
        axis.
    *args, **kwargs : tuple, dict
        Passed through to scikit-learn

    Returns
    -------
    array : obj
        Same type as the input

    Notes
    -----
    For dask.array, a small numpy array emulating ``array`` is created
    and passed to scikit-learn's ``check_array`` with all the additional
    arguments.
    """
    if isinstance(array, da.Array):
        if not accept_dask_array:
            raise TypeError
        if not accept_unknown_chunks:
            if np.isnan(array.shape[0]):
                raise TypeError(
                    "Cannot operate on Dask array with unknown chunk sizes. "
                    "Use the following the compute chunk sizes:\n\n"
                    "   >>> X.compute_chunk_sizes()  # if Dask.Array\n"
                    "   >>> ddf.to_dask_array(lengths=True)  # if Dask.Dataframe"
                )
        if not accept_multiple_blocks and array.ndim > 1:
            if len(array.chunks[1]) > 1:
                msg = (
                    "Chunking is only allowed on the first axis. "
                    "Use 'array.rechunk({1: array.shape[1]})' to "
                    "rechunk to a single block along the second axis."
                )
                raise TypeError(msg)

        if remove_zero_chunks:
            if min(array.chunks[0]) == 0:
                # scikit-learn does not gracefully handle length-0 chunks
                # in some cases (e.g. pairwise_distances).
                chunks2 = tuple(x for x in array.chunks[0] if x != 0)
                array = array.rechunk({0: chunks2})

        # hmmm, we want to catch things like shape errors.
        # I'd like to make a small sample somehow
        shape = array.shape
        if len(shape) == 2:
            shape = (min(10, shape[0]), shape[1])
        elif shape == 1:
            shape = min(10, shape[0])

        sample = np.ones(shape=shape, dtype=array.dtype)
        sk_validation.check_array(sample, *args, **kwargs)
        return array

    elif isinstance(array, dd.DataFrame):
        if not accept_dask_dataframe:
            raise TypeError(
                "This estimator does not support dask dataframes. "
                "This might be resolved with one of\n\n"
                "    1. ddf.to_dask_array(lengths=True)\n"
                "    2. ddf.to_dask_array()  # may cause other issues because "
                "of unknown chunk sizes"
            )
        # TODO: sample?
        return array
    elif isinstance(array, pd.DataFrame) and preserve_pandas_dataframe:
        # TODO: validation?
        return array
    else:
        return sk_validation.check_array(array, *args, **kwargs)


def _assert_eq(l, r, name=None, **kwargs):
    array_types = (np.ndarray, da.Array)
    frame_types = (pd.core.generic.NDFrame, dd._Frame)
    if isinstance(l, array_types):
        assert_eq_ar(l, r, **kwargs)
    elif isinstance(l, frame_types):
        assert_eq_df(l, r, **kwargs)
    elif isinstance(l, Sequence) and any(
        isinstance(x, array_types + frame_types) for x in l
    ):
        for a, b in zip(l, r):
            _assert_eq(a, b, **kwargs)
    else:
        assert l == r, (name, l, r)


def check_random_state(random_state):
    if random_state is None:
        return da.random.RandomState()
    elif isinstance(random_state, Integral):
        return da.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        return da.random.RandomState(random_state.randint())
    elif isinstance(random_state, da.random.RandomState):
        return random_state
    else:
        raise TypeError("Unexpected type '{}'".format(type(random_state)))


def check_matching_blocks(*arrays):
    """Check that the partitioning structure for many arrays matches.

    Parameters
    ----------
    *arrays : Sequence of array-likes
        This includes

        * Dask Array
        * Dask DataFrame
        * Dask Series
    """
    if len(arrays) <= 1:
        return
    if all(isinstance(x, da.Array) for x in arrays):
        # TODO: unknown chunks, ensure blocks match, or just raise (configurable)
        chunks = arrays[0].chunks
        for array in arrays[1:]:
            if array.chunks != chunks:
                raise ValueError(
                    "Mismatched chunks. {} != {}".format(chunks, array.chunks)
                )

    elif all(isinstance(x, (dd.Series, dd.DataFrame)) for x in arrays):
        divisions = arrays[0].divisions
        for array in arrays[1:]:
            if array.divisions != divisions:
                raise ValueError(
                    "Mismatched divisions. {} != {}".format(divisions, array.divisions)
                )
    else:
        raise ValueError("Unexpected types {}.".format({type(x) for x in arrays}))


def check_chunks(n_samples, n_features, chunks=None):
    """Validate and normalize the chunks argument for a dask.array

    Parameters
    ----------
    n_samples, n_features : int
        Give the shape of the array
    chunks : int, sequence, optional, default None
        * For 'chunks=None', this picks a "good" default number of chunks based
          on the number of CPU cores. The default results in a block structure
          with one block per core along the first dimension (of roughly equal
          lengths) and a single block along the second dimension. This may or
          may not be appropriate for your use-case. The chunk size will be at
          least 100 along the first dimension.

        * When chunks is an int, we split the ``n_samples`` into ``chunks``
          blocks along the first dimension, and a single block along the
          second. Again, the chunksize will be at least 100 along the first
          dimension.

        * When chunks is a sequence, we validate that it's length two and turn
          it into a tuple.

    Returns
    -------
    chunks : tuple
    """
    if chunks is None:
        chunks = (max(100, n_samples // cpu_count()), n_features)
    elif isinstance(chunks, Integral):
        chunks = (max(100, n_samples // chunks), n_features)
    elif isinstance(chunks, Sequence):
        chunks = tuple(chunks)
        if len(chunks) != 2:
            raise AssertionError("Chunks should be a 2-tuple.")
    else:
        raise ValueError("Unknown type of chunks: '{}'".format(type(chunks)))
    return chunks


def _log_array(logger, arr, name):
    logger.info(
        "%s: %s, %s blocks",
        name,
        _format_bytes(arr.nbytes),
        getattr(arr, "numblocks", "No"),
    )


def _format_bytes(n):
    # TODO: just import from distributed if / when required
    """Format bytes as text

    >>> format_bytes(1)
    '1 B'
    >>> format_bytes(1234)
    '1.23 kB'
    >>> format_bytes(12345678)
    '12.35 MB'
    >>> format_bytes(1234567890)
    '1.23 GB'
    """
    if n > 1e9:
        return "%0.2f GB" % (n / 1e9)
    if n > 1e6:
        return "%0.2f MB" % (n / 1e6)
    if n > 1e3:
        return "%0.2f kB" % (n / 1000)
    return "%d B" % n


@contextlib.contextmanager
def _timer(name, _logger=None, level="info"):
    """
    Output execution time of a function to the given logger level

    Parameters
    ----------
    name : str
        How to name the timer (will be in the logs)
    logger : logging.logger
        The optional logger where to write
    level : str
        On which level to log the performance measurement
    """
    start = tic()
    _logger = _logger or logger
    _logger.info("Starting %s", name)
    yield
    stop = tic()
    delta = datetime.timedelta(seconds=stop - start)
    _logger_level = getattr(_logger, level)
    _logger_level("Finished %s in %s", name, delta)  # nicer formatting for time.


def _timed(_logger=None, level="info"):
    """
    Output execution time of a function to the given logger level

    level : str
        On which level to log the performance measurement
    Returns
    -------
    fun_wrapper : Callable
    """

    def fun_wrapper(f):
        @functools.wraps(f)
        def wraps(*args, **kwargs):
            with _timer(f.__name__, _logger=logger, level=level):
                results = f(*args, **kwargs)
            return results

        return wraps

    return fun_wrapper


def _num_samples(X):
    result = sk_validation._num_samples(X)
    if dask.is_dask_collection(result):
        # dask dataframe
        result = result.compute()
    return result


__all__ = [
    "assert_estimator_equal",
    "check_array",
    "check_random_state",
    "check_chunks",
    "ConstantFunction",
]
