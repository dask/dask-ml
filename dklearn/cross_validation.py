from __future__ import division, print_function, absolute_import

from numbers import Integral
from operator import add

import numpy as np
import dask.array as da
import dask.bag as db
from dask import delayed
from dask.base import tokenize
from dask.utils import random_state_data
from sklearn import cross_validation
from sklearn.utils import safe_indexing
from toolz import merge, concat, sliding_window, accumulate

from . import matrix as dm
from .utils import check_X_y, is_dask_collection, check_aligned_partitions


__all__ = ['RandomSplit', 'KFold', 'check_cv', 'train_test_split']


def _check_random_state(seed):
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py
    # Workaround for
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    elif hasattr(seed, '__len__') and len(seed) == 624:
        return np.random.RandomState(seed)
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class DaskBaseCV(object):
    """Base class for dask CV objects."""
    pass


_safe_indexing = delayed(safe_indexing, pure=True)


class _DaskCVWrapper(DaskBaseCV):
    """A simple wrapper class for sklearn cv iterators to present the same
    interface as dask cv iterators"""
    def __init__(self, cv):
        self.cv = cv

    def split(self, X, y=None):
        X, y = check_X_y(X, y)
        if is_dask_collection(X) or is_dask_collection(y):
            raise TypeError("Expected X and y to be array-like or "
                            "dask.Delayed, got {0}, "
                            "{1}".format(type(X).__name__, type(y).__name__))
        # Avoid repeated hashing by preconverting to `Delayed` objects
        dX = delayed(X, pure=True)
        dy = delayed(y, pure=True)
        for train, test in self.cv:
            X_train = _safe_indexing(dX, train)
            X_test = _safe_indexing(dX, test)
            if y is not None:
                y_train = _safe_indexing(dy, train)
                y_test = _safe_indexing(dy, test)
            else:
                y_train = y_test = None
            yield X_train, y_train, X_test, y_test

    def __len__(self):
        return len(self.cv)


def check_cv(cv, X=None, y=None, classifier=False):
    """Input checker utility for building a CV in a user friendly way.

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if classifier is True, ``X`` and ``y`` aren't
        dask collections,  and ``y`` is binary or multiclass,
        ``StratifiedKFold`` used. In all other cases, ``KFold`` is used.

    X : array-like
        The data the cross-val object will be applied on.

    y : array-like
        The target variable for a supervised learning problem.

    classifier : boolean optional
        Whether the task is a classification task.
    """
    if is_dask_collection(X) or is_dask_collection(y):
        if cv is None:
            return KFold(n_folds=3)
        elif isinstance(cv, Integral):
            return KFold(n_folds=cv)
        elif not isinstance(cv, DaskBaseCV):
            raise TypeError("Unexpected cv type {0}".format(type(cv).__name__))
        else:
            return cv
    if isinstance(cv, DaskBaseCV) and not isinstance(cv, _DaskCVWrapper):
        raise ValueError("Can't use dask cv object with non-dask X and y")
    cv = cross_validation.check_cv(cv, X=X, y=y, classifier=classifier)
    return _DaskCVWrapper(cv)


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
    x = list(x)
    n_split = np.random.RandomState(seed).binomial(len(x), p_split)
    inds = np.random.RandomState(seed).permutation(len(x))
    inds = inds[:n_split] if left else inds[n_split:]
    inds.sort()
    return list(np.array(x, dtype=object)[inds])


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

    random_state = _check_random_state(random_state)
    token = tokenize(x, p_test, random_state.get_state())
    names = ['random-split-test-' + token,
             'random-split-train-' + token]

    if isinstance(x, da.Array):
        x, x_keys = _as_tall_skinny_and_keys(x)
        chunks = np.array(x.chunks[0])
        seeds = random_state_data(len(chunks) + 1, random_state)
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
        seeds = random_state_data(x.npartitions, random_state)
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


class RandomSplit(DaskBaseCV):
    """Random splitting cross-validation iterator for dask objects.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    n_iter : int, optional
        Number of splitting iterations. Default is 10.

    test_size : float, optional
        Should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the test split. Default is 0.1.

    random_state : int or RandomState, optional
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
            Training data. May be a ``da.Array``, ``db.Bag``, or
            ``dklearn.Matrix``.

        y : dask object, optional
            The target variable for supervised learning problems.

        Yields
        -------
        X_train, y_train, X_test, y_test : dask objects
            The split training and testing data, returned as the same type as
            the input. If y is not provided, ``y_train`` and ``y_test`` will be
            ``None``.
        """
        X, y = check_X_y(X, y)
        seeds = random_state_data(self.n_iter, random_state=self.random_state)
        for seed in seeds:
            X_train, X_test = random_split(X, self.test_size, seed)
            if y is None:
                y_train = y_test = None
            else:
                y_train, y_test = random_split(y, self.test_size, seed)
            yield X_train, y_train, X_test, y_test

    def __len__(self):
        return self.n_iter


def _part_split(x, parts, prefix):
    name = '{0}-{1}'.format(prefix, tokenize(x, parts))
    dsk = dict(((name, i), (x.name, j))
               for i, j in enumerate(parts))
    if isinstance(x, db.Bag):
        return db.Bag(merge(dsk, x.dask), name, len(parts))
    if x.ndim is not None:
        shape = (None,) if x.ndim == 1 else (None, x.shape[1])
    else:
        shape = None
    return dm.Matrix(merge(dsk, x.dask), name, len(parts),
                     dtype=x.dtype, shape=shape)


class KFold(DaskBaseCV):
    """K-Folds cross validation iterator for dask collections.

    Split dataset into k consecutive folds. Each fold is then used as a
    validation set once while the k - 1 remaining fold(s) form the training
    set.

    Parameters
    ----------
    n_folds : int, optional
        Number of folds. Must be at least 2.

    Notes
    -----
    If the inputs are instances of ``da.Array``, they are split into
    approximately equal sized folds, with the first n % n_folds having size n
    // n_folds + 1, and the remainder having size n // n_folds.

    Otherwise the inputs are split into approximately equal number of
    partitions, with no guarantees on the size of each partition.
    """
    def __init__(self, n_folds=3):
        self.n_folds = n_folds

    def split(self, X, y=None):
        """Iterate tuples of data split into training and test sets.

        Parameters
        ----------
        X : dask object
            Training data. May be a ``da.Array``, ``db.Bag``, or
            ``dklearn.Matrix``.

        y : dask object, optional
            The target variable for supervised learning problems.

        Yields
        -------
        X_train, y_train, X_test, y_test : dask objects
            The split training and testing data, returned as the same type as
            the input. If y is not provided, ``y_train`` and ``y_test`` will be
            ``None``.
        """
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        X, y = check_X_y(X, y)
        if isinstance(X, da.Array):
            n = len(X)
            if n < self.n_folds:
                raise ValueError("n_folds must be <= n_samples")
        elif isinstance(X, (dm.Matrix, db.Bag)):
            n = X.npartitions
            if n < self.n_folds:
                raise ValueError("n_folds must be <= npartitions for Bag or "
                                 "Matrix objects")
        else:
            raise TypeError("Expected an instance of ``da.Array``, "
                            "``db.Bag``, or ``dm.Matrix`` - got "
                            "{0}".format(type(X).__name__))
        fold_sizes = (n // self.n_folds) * np.ones(self.n_folds, dtype=np.int)
        fold_sizes[:n % self.n_folds] += 1
        folds = list(sliding_window(2, accumulate(add, fold_sizes, 0)))
        if isinstance(X, da.Array):
            x_parts = [X[start:stop] for start, stop in folds]
            if y is not None:
                y_parts = [y[start:stop] for start, stop in folds]
            for i in range(len(x_parts)):
                X_train = da.concatenate(x_parts[:i] + x_parts[i + 1:])
                X_test = x_parts[i]
                if y is not None:
                    y_train = da.concatenate(y_parts[:i] + y_parts[i + 1:])
                    y_test = y_parts[i]
                else:
                    y_train = y_test = None
                yield X_train, y_train, X_test, y_test
        else:
            parts = list(range(n))
            for start, stop in folds:
                test = parts[start:stop]
                train = parts[:start] + parts[stop:]
                X_train = _part_split(X, train, 'X_train')
                X_test = _part_split(X, test, 'X_test')
                if y is not None:
                    y_train = _part_split(y, train, 'y_train')
                    y_test = _part_split(y, test, 'y_test')
                else:
                    y_train = y_test = None
                yield X_train, y_train, X_test, y_test

    def __len__(self):
        return self.n_folds


def train_test_split(*arrays, **options):
    """Split dask collections into random train and test subsets.

    Quick utility that wraps input validation and calls to train/test splitting
    with ``RandomSplit`` into a single call for splitting data in a oneliner.

    Parameters
    ----------
    *arrays : sequence of dask collections with same length and partitions

        Allowed inputs are ``db.Bag``, ``da.Array``, or ``dm.Matrix``. All
        inputs must share the same length and partitions.

    test_size : float, optional
        Should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the test split. Default is 0.25.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Returns
    -------
    splitting : list, length = 2 * len(arrays),
        List containing train-test split of inputs.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(  # doctest: +SKIP
    ...     X, y, test_size=0.20, random_state=42)
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    check_aligned_partitions(*arrays)

    test_size = options.pop('test_size', 0.25)
    random_state = options.pop('random_state', None)

    if options:
        raise ValueError("Invalid parameters passed: %s" % str(options))

    seed = random_state_data(1, random_state=random_state)[0]
    return list(concat(random_split(a, test_size, seed) for a in arrays))
