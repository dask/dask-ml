from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import sklearn.exceptions
from dask.base import normalize_token
from sklearn.base import BaseEstimator
from sklearn.model_selection._split import (
    BaseShuffleSplit,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    _BaseKFold,
    _CVIterableWrapper,
)


@normalize_token.register(BaseEstimator)
def normalize_estimator(est):
    """Normalize an estimator.

    Note: Since scikit-learn requires duck-typing, but not sub-typing from
    ``BaseEstimator``, we sometimes need to call this function directly."""
    base = [type(est).__name__, normalize_token(est.get_params())]
    # fitted attributes: https://github.com/dask/dask-ml/issues/658
    attrs = [x for x in dir(est) if x.endswith("_") and not x.startswith("_")]
    exclude = {"cv_results_", "model_history_", "history_", "refit_time_"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for attr in attrs:
            if attr in exclude:
                continue
            try:
                val = getattr(est, attr)
            except (sklearn.exceptions.NotFittedError, AttributeError):
                continue
            base.append(val)
    return tuple(base)


def normalize_random_state(random_state):
    if isinstance(random_state, np.random.RandomState):
        return random_state.get_state()
    return random_state


@normalize_token.register(_BaseKFold)
def normalize_KFold(x):
    # Doesn't matter if shuffle is False
    rs = normalize_random_state(x.random_state) if x.shuffle else None
    return (type(x).__name__, x.n_splits, x.shuffle, rs)


@normalize_token.register(BaseShuffleSplit)
def normalize_ShuffleSplit(x):
    return (
        type(x).__name__,
        x.n_splits,
        x.test_size,
        x.train_size,
        normalize_random_state(x.random_state),
    )


@normalize_token.register((LeaveOneOut, LeaveOneGroupOut))
def normalize_LeaveOneOut(x):
    return type(x).__name__


@normalize_token.register((LeavePOut, LeavePGroupsOut))
def normalize_LeavePOut(x):
    return (type(x).__name__, x.p if hasattr(x, "p") else x.n_groups)


@normalize_token.register(PredefinedSplit)
def normalize_PredefinedSplit(x):
    return (type(x).__name__, x.test_fold)


@normalize_token.register(_CVIterableWrapper)
def normalize_CVIterableWrapper(x):
    return (type(x).__name__, x.cv)
