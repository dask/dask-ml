from collections import Sequence

import pandas as pd
import numpy as np

import dask.array as da
import dask.dataframe as dd
import sklearn.utils.extmath as skm

from dask.array.utils import assert_eq as assert_eq_ar
from dask.dataframe.utils import assert_eq as assert_eq_df


def slice_columns(X, columns):
    if isinstance(X, dd.DataFrame):
        return X[list(X.columns) if columns is None else columns]
    else:
        return X


def handle_zeros_in_scale(scale):
    scale = scale.copy()
    if isinstance(scale, da.Array):
        scale[scale == 0.0] = 1.0
    elif isinstance(scale, dd.Series):
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
        left = getattr(left, attr)
        right = getattr(right, attr)
        _assert_eq(left, right, **kwargs)


def _assert_eq(left, right, **kwargs):
    array_types = (np.ndarray, da.Array)
    frame_types = (pd.core.generic.NDFrame, dd._Frame)
    if isinstance(left, array_types):
        assert_eq_ar(left, right, **kwargs)
    elif isinstance(left, frame_types):
        assert_eq_df(left, right, **kwargs)
    elif (isinstance(left, Sequence) and
            any(isinstance(x, array_types + frame_types) for x in left)):
        for a, b in zip(left, right):
            _assert_eq(a, b, **kwargs)
    else:
        assert left == right


__all__ = ['assert_estimator_equal']
