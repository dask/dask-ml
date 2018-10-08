"""Utilities for splitting datasets.
"""
import itertools
import logging
import numbers

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import sklearn.model_selection as ms
from sklearn.model_selection._split import (
    BaseCrossValidator,
    _validate_shuffle_split,
    _validate_shuffle_split_init,
)
from sklearn.utils import check_random_state

from dask_ml.utils import check_array, check_matching_blocks

from .._utils import draw_seed

logger = logging.getLogger(__name__)


def _check_blockwise(blockwise):
    if blockwise not in {True, False}:
        raise ValueError(
            "Expected a boolean for 'blockwise " "but got {} instead".format(blockwise)
        )
    return blockwise


def _maybe_normalize_split_sizes(train_size, test_size):
    # adopt scikit-learn's new behavior (complement) now.
    if train_size is None and test_size is None:
        msg = "test_size and train_size can not both be None"
        raise ValueError(msg)
    elif any(isinstance(x, numbers.Integral) for x in (train_size, test_size)):
        raise ValueError(
            "Dask-ML does not support absolute sizes for "
            "'train_size' and 'test_size'. Use floats between "
            "0 and 1 to specify the fraction of each block "
            "that should go to the train and test set."
        )

    if train_size is not None:
        if train_size < 0 or train_size > 1:
            raise ValueError(
                "'train_size' must be between 0 and 1. " "Got {}".format(train_size)
            )
        if test_size is None:
            test_size = 1 - train_size
    if test_size is not None:
        if test_size < 0 or test_size > 1:
            raise ValueError(
                "'test_size' be between 0 and 1. " "Got {}".format(test_size)
            )

        if train_size is None:
            train_size = 1 - test_size
    if abs(1 - (train_size + test_size)) > 0.001:
        raise ValueError(
            "The sum of 'train_size' and 'test_size' must be 1. "
            "train_size: {} test_size: {}".format(train_size, test_size)
        )
    return train_size, test_size


def _generate_idx(n, seed, n_train, n_test):
    """Generate train, test indices for a length-n array.

    Parameters
    ----------
    n : int
        The length of the array
    seed : int
        Seed for a RandomState
    n_train, n_test : int, 0 < n_train, n_test < n
        Number of samples to use for the train or
        test index.

    Notes
    -----
    """
    # type: (int, int, int) -> Tuple[ndarray, ndarray]
    idx = check_random_state(seed).permutation(n)

    ind_test = idx[:n_test]
    ind_train = idx[n_test : n_train + n_test]
    return ind_train, ind_test


class ShuffleSplit(BaseCrossValidator):
    """Random permutation cross-validator.

    Yields indices to split data into training and test sets.

    .. warning::

       By default, this performs a blockwise-shuffle. That is,
       each block is shuffled internally, but data are not shuffled
       between blocks. If your data is ordered, then set ``blockwise=False``.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, default=0.1
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size.

    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    blockwise : bool, default True
        Whether to shuffle data only within blocks (True), or allow data to
        be shuffled between blocks (False). Shuffling between blocks can
        be much more expensive, especially in distributed environments.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(
        self,
        n_splits=10,
        test_size=0.1,
        train_size=None,
        blockwise=True,
        random_state=None,
    ):
        _validate_shuffle_split_init(test_size, train_size)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.blockwise = _check_blockwise(blockwise)

    def split(self, X, y=None, groups=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            seeds = draw_seed(rng, 0, 2 ** 32 - 1, size=len(X.chunks[0]), dtype="uint")
            if self.blockwise:
                yield self._split_blockwise(X, seeds)
            else:
                yield self._split(X)

    def _split_blockwise(self, X, seeds):
        chunks = X.chunks[0]

        train_pct, test_pct = _maybe_normalize_split_sizes(
            self.train_size, self.test_size
        )
        sizes = [_validate_shuffle_split(c, test_pct, train_pct) for c in chunks]

        objs = [
            dask.delayed(_generate_idx, nout=2)(chunksize, seed, n_train, n_test)
            for chunksize, seed, (n_train, n_test) in zip(chunks, seeds, sizes)
        ]

        train_objs, test_objs = zip(*objs)
        offsets = np.hstack([0, np.cumsum(chunks)])
        train_idx = da.concatenate(
            [
                da.from_delayed(x + offset, (train_size,), "i8")
                for x, chunksize, (train_size, _), offset in zip(
                    train_objs, chunks, sizes, offsets
                )
            ]
        )
        test_idx = da.concatenate(
            [
                da.from_delayed(x + offset, (test_size,), "i8")
                for x, chunksize, (_, test_size), offset in zip(
                    test_objs, chunks, sizes, offsets
                )
            ]
        )

        return train_idx, test_idx

    def _split(self, X):
        raise NotImplementedError(
            "ShuffleSplit with `blockwise=False` has " "not been implemented yet."
        )

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def _generate_offset_idx(n, start, stop, offset, seed):
    if seed is not None:
        idx = check_random_state(seed).permutation(n)
    else:
        idx = np.arange(n)
    return idx[start - offset : stop - offset] + offset


class KFold(BaseCrossValidator):
    """K-Folds cross-validator

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        X = check_array(X)
        n_samples = X.shape[0]
        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[: n_samples % n_splits] += 1

        chunks = X.chunks[0]
        seeds = [None] * len(chunks)
        if self.shuffle:
            rng = check_random_state(self.random_state)
            seeds = draw_seed(rng, 0, 2 ** 32 - 1, size=len(chunks), dtype="uint")

        test_current = 0
        for fold_size in fold_sizes:
            test_start, test_stop = test_current, test_current + fold_size
            yield self._split(test_start, test_stop, n_samples, chunks, seeds)
            test_current = test_stop

    def _split(self, test_start, test_stop, n_samples, chunks, seeds):
        train_objs = []
        test_objs = []
        train_sizes = []
        test_sizes = []

        offset = 0
        for chunk, seed in zip(chunks, seeds):
            start, stop = offset, offset + chunk

            test_id_start = max(test_start, start)
            test_id_stop = min(test_stop, stop)

            if test_id_start < test_id_stop:
                test_objs.append(
                    dask.delayed(_generate_offset_idx)(
                        chunk, test_id_start, test_id_stop, offset, seed
                    )
                )
                test_sizes.append(test_id_stop - test_id_start)

            train_id_stop = min(test_id_start, stop)
            if train_id_stop > start:
                train_objs.append(
                    dask.delayed(_generate_offset_idx)(
                        chunk, start, train_id_stop, offset, seed
                    )
                )
                train_sizes.append(train_id_stop - start)

            train_id_start = max(test_id_stop, start)
            if train_id_start < stop:
                train_objs.append(
                    dask.delayed(_generate_offset_idx)(
                        chunk, train_id_start, stop, offset, seed
                    )
                )
                train_sizes.append(stop - train_id_start)
            offset = stop

        train_idx = da.concatenate(
            [
                da.from_delayed(obj, (train_size,), "i8")
                for obj, train_size in zip(train_objs, train_sizes)
            ]
        )

        test_idx = da.concatenate(
            [
                da.from_delayed(obj, (test_size,), "i8")
                for obj, test_size in zip(test_objs, test_sizes)
            ]
        )

        return train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def _blockwise_slice(arr, idx):
    """Slice an array that is blockwise-aligned with idx.

    Parameters
    ----------
    arr : Dask array
    idx : Dask array
        Should have the following properties

        * Same blocks as `arr` along the first dimension
        * Contains only integers
        * Each block's values should be between ``[0, len(block))``

    Returns
    -------
    sliced : dask.Array
    """
    objs = []
    offsets = np.hstack([0, np.cumsum(arr.chunks[0])[:-1]])

    for i, (x, idx2) in enumerate(
        zip(arr.to_delayed().ravel(), idx.to_delayed().ravel())
    ):
        idx3 = idx2 - offsets[i]
        objs.append(x[idx3])

    shapes = idx.chunks[0]
    if arr.ndim == 2:
        P = arr.shape[1]
        shapes = [(x, P) for x in shapes]
    else:
        shapes = [(x,) for x in shapes]

    sliced = da.concatenate(
        [
            da.from_delayed(x, shape=shape, dtype=arr.dtype)
            for x, shape in zip(objs, shapes)
        ]
    )
    return sliced


def train_test_split(*arrays, **options):
    """Split arrays into random train and test matricies.

    Parameters
    ----------
    *arrays : Sequence of Dask Arrays
    test_size : float or int, defualt 0.1
    train_size: float or int, optional
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle : bool, default True
        Whether to shuffle the data before splitting.
    blockwise : bool, optional.
        Whether to shuffle data only within blocks (True), or allow data to
        be shuffled between blocks (False). Shuffling between blocks can
        be much more expensive, especially in distributed environments.

        The default behavior depends on the types in arrays. For Dask Arrays,
        the default is True (data are not shuffled between blocks). For Dask
        DataFrames, the default and only allowed value is True (data are
        shuffled between blocks).

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs

    Examples
    --------
    import dask.array as da
    from dask_ml.datasets import make_regression

    >>> X, y = make_regression(n_samples=125, n_features=4, chunks=50,
    ...                    random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> X_train
    dask.array<concatenate, shape=(113, 4), dtype=float64, chunksize=(45, 4)>
    >>> X_train.compute()[:2]
    array([[ 0.12372191,  0.58222459,  0.92950511, -2.09460307],
           [ 0.99439439, -0.70972797, -0.27567053,  1.73887268]])
    """
    test_size = options.pop("test_size", None)
    train_size = options.pop("train_size", None)
    random_state = options.pop("random_state", None)
    shuffle = options.pop("shuffle", True)
    blockwise = options.pop("blockwise", None)

    if train_size is None and test_size is None:
        # all other validation dones elsewhere.
        test_size = 0.1

    if options:
        raise TypeError("Unexpected options {}".format(options))

    if not shuffle:
        raise NotImplementedError("'shuffle=False' is not currently supported.")

    if all(isinstance(arr, (dd.Series, dd.DataFrame)) for arr in arrays):
        check_matching_blocks(*arrays)
        if blockwise is None:
            blockwise = False

        rng = check_random_state(random_state)
        rng = draw_seed(rng, 0, 2 ** 32 - 1, dtype="uint")
        return list(
            itertools.chain.from_iterable(
                arr.random_split([train_size, test_size], random_state=rng)
                for arr in arrays
            )
        )

    elif all(isinstance(arr, da.Array) for arr in arrays):
        if blockwise is None:
            blockwise = True

        splitter = ShuffleSplit(
            n_splits=1,
            test_size=test_size,
            train_size=train_size,
            blockwise=blockwise,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(*arrays))

        train_test_pairs = (
            (_blockwise_slice(arr, train_idx), _blockwise_slice(arr, test_idx))
            for arr in arrays
        )

        return list(itertools.chain.from_iterable(train_test_pairs))
    else:
        logger.warning("Mixture of types in 'arrays'. Falling back to scikit-learn.")
        return ms.train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle
        )
