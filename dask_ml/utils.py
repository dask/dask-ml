import contextlib
import datetime
import functools
import logging
import warnings
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
    """Flip vector signs to align them for comparison

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
    **kwargs,
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
    elif np.isscalar(r) and np.isnan(r):
        assert np.isnan(l), (name, l, r)
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


def check_X_y(
    X,
    y,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
    estimator=None,
):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : {ndarray, list, sparse matrix}
        Input data.

    y : {ndarray, list, sparse matrix}
        Labels.

    accept_sparse : str, bool or list of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : {'F', 'C'}, default=None
        Whether an array will be forced to be fortran or c-style.

    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
        does not influence whether y can have np.inf, np.nan, pd.NA values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : bool, default=True
        Whether to raise a value error if X is not 2D.

    allow_nd : bool, default=False
        Whether to allow X.ndim > 2.

    multi_output : bool, default=False
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int, default=1
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    if y is None:
        raise ValueError("y cannot be None")

    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
    )

    y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric)

    check_consistent_length(X, y)

    return X, y


def _check_y(y, multi_output=False, y_numeric=False):
    """Isolated part of check_X_y dedicated to y validation"""
    # TODO: implement
    # if multi_output:
    #     y = check_array(
    #         y, accept_sparse="csr", force_all_finite=True, ensure_2d=False, dtype=None
    #     )
    # else:
    #     y = column_or_1d(y, warn=True)
    #     _assert_all_finite(y)
    #     _ensure_no_complex_data(y)
    # if y_numeric and y.dtype.kind == "O":
    #     y = y.astype(np.float64)
    return y


def check_consistent_length(*arrays):
    # TODO: check divisions, chunks, etc.
    pass


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


def _get_feature_names(X):
    """Get feature names from X.
    Support for other array containers should place its implementation here.
    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.
        - pandas dataframe : The columns will be considered to be feature
          names. If the dataframe contains non-string feature names, `None` is
          returned.
        - All other array containers will return `None`.
    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None

    # extract feature names for support array containers
    if hasattr(X, "columns"):
        feature_names = np.asarray(X.columns, dtype=object)

    if feature_names is None or len(feature_names) == 0:
        return

    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

    # Warn when types are mixed.
    # ints and strings do not warn
    if len(types) > 1 or not (types[0].startswith("int") or types[0] == "str"):
        # TODO: Convert to an error in 1.2
        warnings.warn(
            "Feature names only support names that are all strings. "
            f"Got feature names with dtypes: {types}. An error will be raised "
            "in 1.2.",
            FutureWarning,
        )
        return

    # Only feature names of all strings are supported
    if types[0] == "str":
        return feature_names


__all__ = [
    "assert_estimator_equal",
    "check_array",
    "check_random_state",
    "check_chunks",
    "ConstantFunction",
]
