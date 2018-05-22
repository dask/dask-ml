"""Utilities for splitting datasets.
"""
import math
import numbers
import itertools

import dask
import dask.array as da
import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection._split import _validate_shuffle_split_init

from dask_ml.utils import check_array


def _check_blockwise(blockwise):
    if blockwise not in {True, False}:
        raise ValueError("Expected a boolean for 'blockwise "
                         "but got {} instead".format(blockwise))
    return blockwise


def _normalize_test_sizes(X, test_size):
    chunk_sizes = X.chunks[0]

    if isinstance(test_size, numbers.Integral):
        test_sizes = [chunksize - test_size for chunksize in chunk_sizes]

    elif isinstance(test_size, numbers.Real):
        # TODO: floor or  ceil?
        # TODO(PY3): remove int
        test_sizes = [int(math.floor(chunksize * test_size)) for chunksize in
                      chunk_sizes]
    else:
        raise TypeError("Expected float or integer. Got {} "
                        "instead".format(test_size))

    pairs = zip(test_sizes, chunk_sizes)
    assert all(0 < test_size < chunksize for test_size, chunksize in pairs)
    return test_sizes


def _generate_idx(n, seed, n_test):
    """Generate train, test indices for a length-n array.

    Parameters
    ----------
    n : int
        The length of the array
    seed : int
        Seed for a RandomState
    n_test : int, 0 < n_test < n
        Number of samples to use for the test index.
        The remainder are used for the train index.

    Notes
    -----
    """
    # type: (int, int, int) -> Tuple[ndarray, ndarray]
    idx = check_random_state(seed).permutation(n)

    ind_test = idx[:n_test]
    ind_train = idx[n_test:]
    return ind_train, ind_test


class ShuffleSplit:
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
    def __init__(self, n_splits=10, test_size=0.1, train_size=None,
                 blockwise=True,
                 random_state=None):
        _validate_shuffle_split_init(test_size, train_size)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.blockwise = _check_blockwise(blockwise)

    def split(self, X, y=None, groups=None):
        X = check_array(X)

        for i in range(self.n_splits):
            if self.blockwise:
                yield self._split_blockwise(X)
            else:
                yield self._split(X)

    def _split_blockwise(self, X):
        chunks = X.chunks[0]
        rng = check_random_state(self.random_state)
        seeds = rng.randint(0, 2**32 - 1, size=len(chunks))

        test_sizes = _normalize_test_sizes(X, self.test_size)
        objs = [dask.delayed(_generate_idx, nout=2)(chunksize, seed, test_size)
                for chunksize, seed, test_size in zip(chunks, seeds,
                                                      test_sizes)]

        train_objs, test_objs = zip(*objs)
        offsets = np.hstack([0, np.cumsum(chunks)])
        train_idx = da.concatenate([
            da.from_delayed(x + offset, (chunksize - test_size,), 'i8')
            for x, chunksize, test_size, offset in zip(train_objs, chunks,
                                                       test_sizes, offsets)
        ])
        test_idx = da.concatenate([
            da.from_delayed(x + offset, (test_size,), 'i8')
            for x, chunksize, test_size, offset in zip(test_objs, chunks,
                                                       test_sizes, offsets)
        ])

        return train_idx, test_idx

    def _split(self, X):
        raise NotImplementedError("ShuffleSplit with `blockwise=False` has "
                                  "not been implemented yet.")


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

    for i, (x, idx2) in enumerate(zip(arr.to_delayed().ravel(),
                                      idx.to_delayed().ravel())):
        idx3 = idx2 - offsets[i]
        objs.append(x[idx3])

    shapes = idx.chunks[0]
    if arr.ndim == 2:
        P = arr.shape[1]
        shapes = [(x, P) for x in shapes]
    else:
        shapes = [(x,) for x in shapes]

    sliced = da.concatenate([da.from_delayed(x, shape=shape, dtype=arr.dtype)
                            for x, shape in zip(objs, shapes)])
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
    blockwise : bool, default True
        Whether to perform blockwise-shuffling.

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
    test_size = options.pop('test_size', 0.1)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    shuffle = options.pop('shuffle', True)
    blockwise = options.pop("blockwise", True)

    if options:
        raise TypeError("Unexpected options {}".format(options))

    if not shuffle:
        raise NotImplementedError

    assert all(isinstance(arr, da.Array) for arr in arrays)

    splitter = ShuffleSplit(n_splits=1, test_size=test_size,
                            train_size=train_size, blockwise=blockwise,
                            random_state=random_state)
    train_idx, test_idx = next(splitter.split(*arrays))

    train_test_pairs = ((_blockwise_slice(arr, train_idx),
                         _blockwise_slice(arr, test_idx))
                        for arr in arrays)

    return list(itertools.chain.from_iterable(train_test_pairs))
