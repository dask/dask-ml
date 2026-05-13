"""Utilities for splitting datasets."""

import itertools
import logging
import numbers
import warnings

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.model_selection._split import BaseCrossValidator, _validate_shuffle_split
from sklearn.utils import check_random_state

from dask_ml._compat import DASK_2130, DASK_VERSION
from dask_ml.utils import check_array, check_matching_blocks

from .._utils import draw_seed

logger = logging.getLogger(__name__)
_I4MAX = np.iinfo("i4").max


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
                "'test_size' must be between 0 and 1. " "Got {}".format(test_size)
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
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.blockwise = _check_blockwise(blockwise)

    def split(self, X, y=None, groups=None):
        X = check_array(X, ensure_2d=False, allow_nd=True)
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            seeds = draw_seed(rng, 0, _I4MAX, size=len(X.chunks[0]), dtype="uint")
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
                da.from_delayed(x + offset, (train_size,), np.dtype("int"))
                for x, chunksize, (train_size, _), offset in zip(
                    train_objs, chunks, sizes, offsets
                )
            ]
        )
        test_idx = da.concatenate(
            [
                da.from_delayed(x + offset, (test_size,), np.dtype("int"))
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
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1

        chunks = X.chunks[0]
        seeds = [None] * len(chunks)
        if self.shuffle:
            rng = check_random_state(self.random_state)
            seeds = draw_seed(rng, 0, _I4MAX, size=len(chunks), dtype="uint")

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
                da.from_delayed(obj, (train_size,), np.dtype("int"))
                for obj, train_size in zip(train_objs, train_sizes)
            ]
        )

        test_idx = da.concatenate(
            [
                da.from_delayed(obj, (test_size,), np.dtype("int"))
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


def _largest_remainder_split(bin_sizes, total, rng):
    """Split ``total`` units across bins, proportional to ``bin_sizes``.

    Returns integers summing to ``total``. Each bin gets its floor share
    ``bin_size * total / sum(bin_sizes)``; the leftover slots go to bins
    with the largest fractional remainder. ``rng`` breaks ties.

    See https://en.wikipedia.org/wiki/Largest_remainders_method.
    """
    pool = int(bin_sizes.sum())
    if pool == 0 or total == 0:
        return np.zeros_like(bin_sizes)
    products = bin_sizes * int(total)
    quotas = products // pool
    remainders = products % pool
    leftover = int(total) - int(quotas.sum())
    tiebreaker = rng.random(len(bin_sizes))
    priority = np.lexsort((tiebreaker, -remainders))
    quotas[priority[:leftover]] += 1
    return quotas


def _get_test_count_per_class(n_per_class, n_test, rng):
    """How many test rows each class contributes, summing to ``n_test``.

    Counts are proportional to class size via largest-remainder rounding.
    sklearn rule: every class keeps at least one row in both train and
    test splits. Over/underflow from that clamp is redistributed to
    classes that still have headroom.
    """
    if (n_per_class < 2).any():
        raise ValueError(
            "The least populated class in y has only 1 member, which is too "
            "few. The minimum number of groups for any class cannot be less "
            "than 2."
        )
    quotas = _largest_remainder_split(n_per_class, n_test, rng)
    floor = np.ones_like(quotas)
    ceil = n_per_class - 1
    for _ in range(len(quotas) + 1):
        clamped = np.clip(quotas, floor, ceil)
        delta = n_test - int(clamped.sum())
        if delta == 0:
            return clamped
        has_room = clamped < ceil if delta > 0 else clamped > floor
        eligible = np.where(has_room)[0]
        if len(eligible) == 0:
            return clamped
        step = 1 if delta > 0 else -1
        for class_idx in rng.permutation(eligible)[: abs(delta)]:
            clamped[class_idx] += step
        quotas = clamped
    return quotas


def _check_no_missing_labels(arr):
    """Raise ValueError if ``arr`` contains NaN, None, or pd.NA.

    ``np.unique`` treats every NaN as distinct (NaN != NaN), so silently
    splitting on missing labels would create phantom classes. sklearn
    refuses such inputs; we do the same. ``pd.isna`` handles float NaN,
    object-dtype None, and pandas NA in one call.

    Void-dtype arrays are packed compound labels from ``_as_1d_block`` and
    can't be NaN by construction; skip them (and ``pd.isna`` rejects void).
    """
    if arr.dtype.kind == "V":
        return
    if pd.isna(arr).any():
        raise ValueError(
            "Input contains NaN/None/NA. `stratify` must not contain missing values."
        )


def _get_class_freq_in_block(block):
    """Return the unique class ids and their counts in each stratify block."""
    block = np.asarray(block).ravel()
    _check_no_missing_labels(block)
    return np.unique(block, return_counts=True)


def _get_test_count_per_class_block(class_freq_per_block, test_size, seed):
    """How many test rows each block contributes for each class.

    Runs on the driver as a single ``dask.delayed`` task once all per-block
    summaries arrive. Two steps:

    1. Split ``n_test`` across classes proportional to class size, with
       sklearn's ">=1 row per class in both splits" rule
       (``_get_test_count_per_class``).
    2. For each class, split its test-row budget across blocks proportional
       to how many rows of that class each block holds
       (``_largest_remainder_split``).

    Parameters
    ----------
    class_freq_per_block : list of (numpy.ndarray, numpy.ndarray)
        One ``(classes_in_block, counts_in_block)`` tuple per block, from
        ``_get_class_freq_in_block``.
    test_size : float or int
        Fraction (0 < x < 1) or absolute count of test rows.
    seed : int
        Seed for the driver's ``np.random.Generator`` (tie-breaking only).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(class_ids, test_counts)`` where ``test_counts[block, class]``
        is the number of rows of ``class_ids[class]`` that ``block`` puts
        into the test set. Shape: ``(n_blocks, n_classes)``.
    """
    class_ids = np.unique(np.concatenate([c for c, _ in class_freq_per_block]))
    n_blocks = len(class_freq_per_block)
    n_classes = len(class_ids)
    n_per_class_per_block = np.zeros((n_blocks, n_classes), dtype=np.int64)
    for bi, (ci, fi) in enumerate(class_freq_per_block):
        n_per_class_per_block[bi, np.searchsorted(class_ids, ci)] = fi

    n_per_class = n_per_class_per_block.sum(axis=0)
    n_total = n_per_class.sum()
    if isinstance(test_size, (int, np.integer)) and not isinstance(test_size, bool):
        n_test = int(test_size)
    else:
        n_test = max(1, min(n_total - 1, int(round(n_total * float(test_size)))))

    rng = np.random.default_rng(seed)
    test_count_per_class = _get_test_count_per_class(n_per_class, n_test, rng)
    test_count_per_class_block = np.column_stack(
        [
            _largest_remainder_split(
                n_per_class_per_block[:, ci], test_count_per_class[ci], rng
            )
            for ci in range(n_classes)
        ]
    )
    return class_ids, test_count_per_class_block


def _get_train_test_indices_per_block(
    block, test_count_per_class_block, block_idx, seed, shuffle
):
    """Pick this block's train/test row indices.

    For each class, picks ``test_counts[block_idx, class]`` rows of that
    class uniformly without replacement. The per-block RNG is seeded with
    ``(seed, block_idx)`` so blocks sample independently and
    reproducibly. When ``shuffle`` is True, the returned index arrays are
    permuted so output row order is randomized within each block.
    """
    class_ids, test_counts = test_count_per_class_block
    rng = np.random.default_rng((int(seed), int(block_idx)))
    strat_here = np.asarray(block).ravel()
    is_test = np.zeros(len(strat_here), dtype=bool)
    for class_id, rows_to_pick in zip(class_ids, test_counts[block_idx]):
        rows_to_pick = int(rows_to_pick)
        if rows_to_pick == 0:
            continue
        positions_of_class = np.where(strat_here == class_id)[0]
        is_test[rng.choice(positions_of_class, size=rows_to_pick, replace=False)] = True
    test_idx = np.flatnonzero(is_test)
    train_idx = np.flatnonzero(~is_test)
    if shuffle:
        test_idx = rng.permutation(test_idx)
        train_idx = rng.permutation(train_idx)
    return train_idx, test_idx


def _slice_block(block, indices_pair, take_test):
    """Return rows of ``block`` at the train or test indices.

    Module-level so dask can pickle it across workers.
    """
    rows = np.asarray(block)
    return rows[indices_pair[1] if take_test else indices_pair[0]]


def _as_1d_block(block):
    """Collapse each row of a 2-D stratify block into a single hashable scalar.

    For compound stratification (e.g. multilabel ``y``), sklearn treats each
    unique tuple of row values as one class. Numpy's ``np.void`` view
    interprets each row's bytes as a single scalar - zero-copy and orderable,
    so ``np.unique`` / ``np.searchsorted`` group identical tuples together.
    """
    block = np.ascontiguousarray(block)
    void_dtype = np.dtype((np.void, block.dtype.itemsize * block.shape[1]))
    return block.view(void_dtype).ravel()


def _axis0_layout(obj):
    """Axis-0 ``(block_count, chunks_or_None)`` for a dask collection.

    ``chunks`` is ``None`` when block sizes are unknown (e.g. ``from_delayed``
    or a dask DataFrame whose divisions don't carry block lengths). Only
    block count can be compared in that case.
    """
    if isinstance(obj, da.Array):
        chunks = obj.chunks[0]
        if any(np.isnan(c) for c in chunks):
            return obj.numblocks[0], None
        return obj.numblocks[0], tuple(chunks)
    return obj.npartitions, None


def _as_input_aligned_1d_array(stratify, arrays):
    """Wrap ``stratify`` as a 1-D dask Array, axis-0 aligned to ``arrays[0]``.

    Non-dask inputs are split to match the reference axis-0 chunks. Dask
    inputs are validated against that layout - silently re-chunking a
    user's collection would change their execution plan, and a mismatched
    layout would misalign per-block label indices with feature blocks.
    Compound (2-D) labels are packed row-wise into ``np.void`` scalars so
    the rest of the pipeline can treat them as ordinary 1-D class ids.
    """
    ref_blocks, ref_chunks = _axis0_layout(arrays[0])

    if isinstance(stratify, (da.Array, dd.Series, dd.DataFrame)):
        blocks, chunks = _axis0_layout(stratify)
        if blocks != ref_blocks or (
            ref_chunks is not None and chunks is not None and chunks != ref_chunks
        ):
            ref_desc = ref_chunks if ref_chunks is not None else f"{ref_blocks} blocks"
            obj_desc = chunks if chunks is not None else f"{blocks} blocks"
            raise ValueError(
                f"Axis-0 partitioning of stratify ({obj_desc}) does not match "
                f"arrays[0] ({ref_desc}). Rechunk to align, e.g. "
                f"`stratify.rechunk({{0: arrays[0].chunks[0]}})` for dask "
                f"Arrays or `stratify.repartition(npartitions="
                f"arrays[0].npartitions)` for dask DataFrames."
            )
        if isinstance(stratify, (dd.Series, dd.DataFrame)):
            stratify = stratify.to_dask_array()
    else:
        as_numpy = np.asarray(stratify)
        # Eager check: in-memory inputs cost nothing to validate now and
        # give users an immediate error instead of one at .compute() time.
        _check_no_missing_labels(as_numpy)
        # Match arr0's axis-0 chunks exactly when known; otherwise fall back
        # to an even split into ref_blocks pieces.
        if ref_chunks is not None and sum(ref_chunks) == as_numpy.shape[0]:
            axis0_chunks = ref_chunks
        elif ref_blocks > 1 and as_numpy.shape[0] >= ref_blocks:
            splits = np.array_split(as_numpy, ref_blocks, axis=0)
            axis0_chunks = tuple(s.shape[0] for s in splits)
        else:
            axis0_chunks = (as_numpy.shape[0],)
        chunks = (axis0_chunks,) + tuple((s,) for s in as_numpy.shape[1:])
        stratify = da.from_array(as_numpy, chunks=chunks)

    if stratify.ndim > 1:
        stratify = stratify.rechunk({axis: -1 for axis in range(1, stratify.ndim)})
        stratify = stratify.map_blocks(
            _as_1d_block, drop_axis=list(range(1, stratify.ndim))
        )
    return stratify


def _stratified_split(arrays, stratify, test_size, random_state, shuffle=True):
    """Lazy stratified train/test split over dask collections.

    Builds a delayed graph; nothing computes until the user calls
    ``.compute()`` on an output. Feature data is never gathered to the
    driver, only a KB-scale ``(n_blocks, n_classes)`` counts matrix is.

    High level approach:

      1. each stratify block emits ``(classes_in_block, counts_in_block)``
         via ``_get_class_freq_in_block`` (one delayed task per block).
      2. the driver merges those into ``(class_ids, test_counts[p, c])``
         via ``_get_test_count_per_class_block`` (one delayed task).
      3. each stratify block picks its own test rows
         via ``_get_train_test_indices_per_block`` (one delayed task per block).
      4. each input array's blocks are sliced by the matching mask
         via ``_slice_block``; train and test blocks are concatenated.

    Output arrays have ``nan`` for their first dimension until ``.compute()``
    is called (same as ``da.where(cond)[0]``).

    Parameters
    ----------
    arrays : sequence of dask Arrays / Series / DataFrames
        Inputs to split. Each must share the same axis-0 block count as
        ``stratify``.
    stratify : dask Array/Series/DataFrame, numpy array, pandas Series, or list
        Class labels. May be 1-D or 2-D (rows treated as compound labels).
    test_size : float or int
        Fraction (0 < x < 1) or absolute count of test rows.
    random_state : int, RandomState, or None
        Seed for the split.
    shuffle : bool, default True
        If True, permute row order within each output block (matches
        sklearn and the non-stratified dask paths). If False, preserve the
        original row order within each block (rows that survive
        selection appear in the order they had in the input).

    Returns
    -------
    list of dask.array.Array
        ``[X1_train, X1_test, X2_train, X2_test, ...]``.
    """
    stratify = _as_input_aligned_1d_array(stratify, arrays)
    strat_blocks = stratify.to_delayed().ravel().tolist()

    rng = check_random_state(random_state)
    seed = int(draw_seed(rng, 0, _I4MAX, dtype="uint"))

    class_freq_per_block = [
        dask.delayed(_get_class_freq_in_block)(block) for block in strat_blocks
    ]
    test_count_per_class_block = dask.delayed(_get_test_count_per_class_block)(
        class_freq_per_block, test_size, seed
    )
    train_test_idx_per_block = [
        dask.delayed(_get_train_test_indices_per_block)(
            block, test_count_per_class_block, block_idx, seed, shuffle
        )
        for block_idx, block in enumerate(strat_blocks)
    ]

    outputs = []
    for arr in arrays:
        if isinstance(arr, (dd.Series, dd.DataFrame)):
            arr = arr.to_dask_array()
        if arr.ndim > 1:
            arr = arr.rechunk({axis: -1 for axis in range(1, arr.ndim)})
        feature_blocks = arr.to_delayed().ravel().tolist()
        trailing_shape = arr.shape[1:]
        train_blocks = [
            da.from_delayed(
                dask.delayed(_slice_block)(block, idx, False),
                shape=(np.nan, *trailing_shape),
                dtype=arr.dtype,
            )
            for block, idx in zip(feature_blocks, train_test_idx_per_block)
        ]
        test_blocks = [
            da.from_delayed(
                dask.delayed(_slice_block)(block, idx, True),
                shape=(np.nan, *trailing_shape),
                dtype=arr.dtype,
            )
            for block, idx in zip(feature_blocks, train_test_idx_per_block)
        ]
        outputs += [da.concatenate(train_blocks), da.concatenate(test_blocks)]
    return outputs


def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=None,
    stratify=None,
    blockwise=True,
    convert_mixed_types=False,
    **options,
):
    """Split arrays into random train and test matrices.

    Parameters
    ----------
    ``*arrays`` : Sequence of Dask Arrays, DataFrames, or Series
        Non-dask objects will be passed through to
        :func:`sklearn.model_selection.train_test_split`.

    test_size : float or int, default 0.1

    train_size : float or int, optional

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : bool, default None
        Whether to shuffle the data before splitting.

    stratify : array-like, optional (default=None)
        If not None, data is split in a stratified fashion using this as
        the class labels (sklearn semantics). Stays lazy: index computation
        runs inside a delayed task, triggered only when ``.compute()`` is
        called on a split output. The ``*arrays`` themselves are never
        gathered to the driver, but ``stratify`` and the resulting index
        arrays do materialize on a single worker, so memory scales with the
        label vector (not the feature matrix). ``stratify`` may be a dask
        Array/Series/DataFrame or any array-like accepted by sklearn.

    blockwise : bool, default True.
        Whether to shuffle data only within blocks (True), or allow data to
        be shuffled between blocks (False). Shuffling between blocks can
        be much more expensive, especially in distributed environments.

        The default is ``True``, data are only shuffled within blocks.
        For Dask Arrays, set ``blockwise=False`` to shuffle data between
        blocks as well. For Dask DataFrames, ``blockwise=False`` is not
        currently supported and a ``ValueError`` will be raised.

    convert_mixed_types : bool, default False
        Whether to convert dask DataFrames and Series to dask Arrays when
        arrays contains a mixture of types. This results in some computation
        to determine the length of each block.


    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs

    Examples
    --------
    >>> import dask.array as da
    >>> from dask_ml.datasets import make_regression

    >>> X, y = make_regression(n_samples=125, n_features=4, chunks=50,
    ...                        random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> X_train
    dask.array<concatenate, shape=(113, 4), dtype=float64, chunksize=(45, 4)>
    >>> X_train.compute()[:2]
    array([[ 0.12372191,  0.58222459,  0.92950511, -2.09460307],
           [ 0.99439439, -0.70972797, -0.27567053,  1.73887268]])
    """
    if train_size is None and test_size is None:
        # all other validation done elsewhere.
        test_size = 0.1

    if train_size is None and test_size is not None:
        train_size = 1 - test_size
    if test_size is None and train_size is not None:
        test_size = 1 - train_size

    if options:
        raise TypeError("Unexpected options {}".format(options))

    if stratify is not None:
        if (
            isinstance(stratify, bool)
            or np.isscalar(stratify)
            or getattr(stratify, "ndim", 1) == 0
        ):
            raise TypeError(
                "`stratify` must be an array of class labels (one per sample), "
                "not a scalar/boolean. To stratify on the target, pass the "
                "target itself: train_test_split(X, y, stratify=y). Got "
                f"{type(stratify).__name__}={stratify!r}."
            )
        if not hasattr(stratify, "__array__") and not isinstance(
            stratify, (da.Array, dd.Series, dd.DataFrame, list, tuple)
        ):
            raise TypeError(
                "`stratify` must be array-like (dask Array/Series/DataFrame, "
                f"numpy array, pandas Series, or list). Got {type(stratify).__name__}."
            )

        def _known_len(obj):
            shape = getattr(obj, "shape", None)
            if shape is not None and len(shape) > 0:
                n = shape[0]
                if isinstance(n, int):
                    return n
                return None
            try:
                return len(obj)
            except TypeError:
                return None

        len_a = _known_len(arrays[0])
        len_s = _known_len(stratify)
        if len_a is not None and len_s is not None and len_a != len_s:
            raise ValueError(
                f"Length of stratify ({len_s}) does not match length of "
                f"input arrays ({len_a})."
            )

    types = set(type(arr) for arr in arrays)

    if da.Array in types and types & {dd.Series, dd.DataFrame}:
        if convert_mixed_types:
            arrays = tuple(
                (
                    x.to_dask_array(lengths=True)
                    if isinstance(x, (dd.Series, dd.DataFrame))
                    else x
                )
                for x in arrays
            )
        else:
            raise TypeError(
                "Got mixture of dask DataFrames and Arrays. Specify "
                "'convert_mixed_types=True'"
            )

    if all(isinstance(arr, (dd.Series, dd.DataFrame)) for arr in arrays):
        check_matching_blocks(*arrays)
        if blockwise is False:
            raise NotImplementedError(
                "'blockwise=False' is not currently supported for dask DataFrames."
            )

        rng = check_random_state(random_state)
        rng = draw_seed(rng, 0, _I4MAX, dtype="uint")
        if DASK_2130:
            if shuffle is None:
                shuffle = False
                warnings.warn(
                    message="The default value for 'shuffle' must be specified"
                    " when splitting DataFrames. In the future"
                    " DataFrames will automatically be shuffled within"
                    " blocks prior to splitting. Specify 'shuffle=True'"
                    " to adopt the future behavior now, or 'shuffle=False'"
                    " to retain the previous behavior.",
                    category=FutureWarning,
                )
            kwargs = {"shuffle": shuffle}
        else:
            if shuffle is None:
                shuffle = True
            if not shuffle:
                raise NotImplementedError(
                    f"'shuffle=False' is not supported for DataFrames in"
                    f" dask versions<2.13.0. Current version is {DASK_VERSION}."
                )
            kwargs = {}

        if stratify is not None:
            return _stratified_split(
                arrays, stratify, test_size, random_state, shuffle=shuffle
            )

        return list(
            itertools.chain.from_iterable(
                arr.random_split([train_size, test_size], random_state=rng, **kwargs)
                for arr in arrays
            )
        )

    elif all(isinstance(arr, da.Array) for arr in arrays):
        if shuffle is None:
            shuffle = True
        if not shuffle:
            raise NotImplementedError(
                "'shuffle=False' is not currently supported for dask Arrays."
            )

        if stratify is not None:
            if blockwise is False:
                raise NotImplementedError(
                    "'blockwise=False' is not supported for stratified splits."
                )
            return _stratified_split(
                arrays, stratify, test_size, random_state, shuffle=shuffle
            )

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
        kwargs = dict(
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify,
        )
        if shuffle is not None:
            kwargs["shuffle"] = shuffle
        return ms.train_test_split(*arrays, **kwargs)
