from collections import Sequence
from numbers import Integral
from multiprocessing import cpu_count

import pandas as pd
import numpy as np

import dask.array as da
import dask.dataframe as dd
import sklearn.utils.extmath as skm
import sklearn.utils.validation as sk_validation

from dask.array.utils import assert_eq as assert_eq_ar
from dask.dataframe.utils import assert_eq as assert_eq_df


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
    return X.map_blocks(skm.row_norms, chunks=(X.chunks[0],),
                        drop_axis=1, squared=squared)


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
    left_attrs = [x for x in dir(left) if x.endswith('_') and
                  not x.startswith('_')]
    right_attrs = [x for x in dir(right) if x.endswith('_') and
                   not x.startswith('_')]
    if exclude is None:
        exclude = set()
    elif isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude)

    assert (set(left_attrs) - exclude) == set(right_attrs) - exclude

    for attr in set(left_attrs) - exclude:
        l = getattr(left, attr)
        r = getattr(right, attr)
        _assert_eq(l, r, **kwargs)


def check_array(array, *args, **kwargs):
    accept_dask_array = kwargs.pop("accept_dask_array", True)
    accept_dask_dataframe = kwargs.pop("accept_dask_dataframe", True)
    accept_unknown_chunks = kwargs.pop("accept_unknown_chunks", False)
    accept_multiple_blocks = kwargs.pop("accept_multiple_blocks", False)

    if isinstance(array, da.Array):
        if not accept_dask_array:
            raise TypeError
        if not accept_unknown_chunks:
            if np.isnan(array.shape[0]):
                raise TypeError
        if not accept_multiple_blocks:
            if len(array.chunks[1]) > 1:
                raise TypeError

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
            raise TypeError
        if not accept_unknown_chunks:
            raise TypeError

        sk_validation.check_array(sample, *args, **kwargs)
        return array
    else:
        return sk_validation.check_array(array, *args, **kwargs)


def _assert_eq(l, r, **kwargs):
    array_types = (np.ndarray, da.Array)
    frame_types = (pd.core.generic.NDFrame, dd._Frame)
    if isinstance(l, array_types):
        assert_eq_ar(l, r, **kwargs)
    elif isinstance(l, frame_types):
        assert_eq_df(l, r, **kwargs)
    elif (isinstance(l, Sequence) and
            any(isinstance(x, array_types + frame_types) for x in l)):
        for a, b in zip(l, r):
            _assert_eq(a, b, **kwargs)
    else:
        assert l == r


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


__all__ = ['assert_estimator_equal',
           'check_array',
           'check_random_state',
           'check_chunks']
