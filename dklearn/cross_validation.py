from __future__ import division, print_function, absolute_import

from random import Random

import numpy as np
import dask.array as da
import dask.bag as db
from dask.base import tokenize
from dask.utils import different_seeds
from sklearn.utils.validation import check_random_state
from toolz import merge, concat

from . import matrix as dm
from .utils import check_X_y


__all__ = ['RandomSplit']


def _as_tall_skinny_and_keys(x):
    if x.ndim == 2:
        if len(x.chunks[1]) != 1:
            x = x.rechunk((x.chunks[0], x.shape[1]))
        keys = list(concat(x._keys()))
    else:
        keys = x._keys()
    return x, keys


def arr_split(x, n_split, left=True, seed=None):
    inds = np.random.RandomState(seed).permutation(x.shape[0])
    inds = inds[:n_split] if left else inds[n_split:]
    inds.sort()
    return x[inds]


def mat_split(x, p_split, left=True, seed=None):
    n_split = np.random.RandomState(seed).binomial(x.shape[0], p_split)
    return arr_split(x, n_split, left=left, seed=seed)


def bag_split(x, p_split, left=True, seed=None):
    random_state = Random(seed)
    if left:
        return filter(lambda _: random_state.random() < p_split, x)
    return filter(lambda _: random_state.random() >= p_split, x)


def random_split(x, p_test=0.1, random_state=None):
    """Approximately split a dask collection into train/test data.

    Parameters
    ----------
    X : da.Array, db.Bag, or dm.Matrix
        The dask collection to split
    p_test : float, optional
        The fraction of samples to use in the test set. Default is 0.1.
    random_state : int or RandomState, optional
        The ``RandomState`` or seed to use when performing the random split.
    """
    if not 0 < p_test < 1:
        raise ValueError("p_test must be in (0, 1)")

    random_state = check_random_state(random_state)
    token = tokenize(x, p_test, random_state.get_state())
    names = ['random-split-test-' + token,
             'random-split-train-' + token]

    if isinstance(x, da.Array):
        x, x_keys = _as_tall_skinny_and_keys(x)
        chunks = np.array(x.chunks[0])
        seeds = different_seeds(len(chunks) + 1, random_state)
        n_test = np.random.RandomState(seeds[0]).binomial(chunks, p_test)
        n_train = chunks - n_test
        dsks = [dict(((name,) + k[1:], (arr_split, k, n, b, s))
                     for k, n, s in zip(x_keys, n_test, seeds[1:]))
                for name, b in zip(names, [True, False])]

        test = da.Array(merge(dsks[0], x.dask), names[0],
                        (tuple(n_test),) + x.chunks[1:], x.dtype)
        train = da.Array(merge(dsks[1], x.dask), names[1],
                         (tuple(n_train),) + x.chunks[1:], x.dtype)

    elif isinstance(x, (db.Bag, dm.Matrix)):
        seeds = different_seeds(x.npartitions, random_state)
        split = bag_split if isinstance(x, db.Bag) else mat_split
        dsks = [dict(((name, k[1]), (split, k, p_test, b, s))
                     for k, s in zip(x._keys(), seeds))
                for name, b in zip(names, [True, False])]

        if isinstance(x, dm.Matrix):
            if x.ndim is not None:
                shape = (None,) if x.ndim == 1 else (None, x.shape[1])
            else:
                shape = None
            test = dm.Matrix(merge(dsks[0], x.dask), names[0],
                             x.npartitions, dtype=x.dtype, shape=shape)
            train = dm.Matrix(merge(dsks[1], x.dask), names[1],
                              x.npartitions, dtype=x.dtype, shape=shape)

        else:
            test = db.Bag(merge(dsks[0], x.dask), names[0], x.npartitions)
            train = db.Bag(merge(dsks[1], x.dask), names[1], x.npartitions)
    else:
        raise TypeError("Expected an instance of ``da.Array``, ``db.Bag``, or "
                        "``dm.Matrix`` - got {0}".format(type(x).__name__))

    return train, test


class RandomSplit(object):
    """Random splitting cross-validation iterator for dask objects.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    n_iter : int (default 10)
        Number of splitting iterations.

    test_size : float (default 0.1)
        Should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the test split.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    """
    def __init__(self, n_iter=10, test_size=0.1, random_state=None):
        self.n_iter = n_iter
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None):
        """Iterate tuples of data split into training and test sets.

        Parameters
        ----------
        X : dask object
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : dask object, optional
            The target variable for supervised learning problems.

        Yields
        -------
        X_train, y_train, X_test, y_test : dask objects
            The split training and testing data, returned as the same type as
            the input. If y is not provided, only yields ``X_train`` and
            ``X_test``.
        """
        X, y = check_X_y(X, y)
        seeds = different_seeds(self.n_iter, random_state=self.random_state)
        for seed in seeds:
            X_train, X_test = random_split(X, self.test_size, seed)
            if y is None:
                yield X_train, X_test
            else:
                y_train, y_test = random_split(y, self.test_size, seed)
                yield X_train, y_train, X_test, y_test
