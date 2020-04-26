import numbers
from datetime import timedelta

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import sklearn.datasets
import sklearn.utils

import dask_ml.utils


def _check_axis_partitioning(chunks, n_features):
    c = chunks[1][0]
    if c != n_features:
        msg = (
            "Can only generate arrays partitioned along the "
            "first axis. Specifying a larger chunksize for "
            "the second axis.\n\n\tchunk size: {}\n"
            "\tn_features: {}".format(c, n_features)
        )
        raise ValueError(msg)


def make_counts(
    n_samples=1000,
    n_features=100,
    n_informative=2,
    scale=1.0,
    chunks=100,
    random_state=None,
):
    """
    Generate a dummy dataset for modeling count data.

    Parameters
    ----------
    n_samples : int
        number of rows in the output array
    n_features : int
        number of columns (features) in the output array
    n_informative : int
        number of features that are correlated with the outcome
    scale : float
        Scale the true coefficient array by this
    chunks : int
        Number of rows per dask array block.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : dask.array, size ``(n_samples, n_features)``
    y : dask.array, size ``(n_samples,)``
        array of non-negative integer-valued data

    Examples
    --------
    >>> X, y = make_counts()
    """
    rng = dask_ml.utils.check_random_state(random_state)

    X = rng.normal(0, 1, size=(n_samples, n_features), chunks=(chunks, n_features))
    informative_idx = rng.choice(n_features, n_informative, chunks=n_informative)
    beta = (rng.random(n_features, chunks=n_features) - 1) * scale

    informative_idx, beta = dask.compute(informative_idx, beta)

    z0 = X[:, informative_idx].dot(beta[informative_idx])
    rate = da.exp(z0)
    y = rng.poisson(rate, size=1, chunks=(chunks,))
    return X, y


def make_blobs(
    n_samples=100,
    n_features=2,
    centers=None,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=None,
    chunks=None,
):
    """
    Generate isotropic Gaussian blobs for clustering.

    This can be used to generate very large Dask arrays on a cluster of
    machines. When using Dask in distributed mode, the client machine
    only needs to allocate a single block's worth of data.

    Parameters
    ----------
    n_samples : int or array-like, optional (default=100)
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.

    n_features : int, optional (default=2)
        The number of features for each sample.

    centers : int or array of shape [n_centers, n_features], optional
        (default=None)
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.

    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.

    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.

    shuffle : boolean, optional (default=True)
        Shuffle the samples.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    chunks : int, tuple
        How to chunk the array. Must be one of the following forms:
        -   A blocksize like 1000.
        -   A blockshape like (1000, 1000).
        -   Explicit sizes of all blocks along all dimensions like
            ((1000, 1000, 500), (400, 400)).

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.

    Examples
    --------
    >>> from dask_ml.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=100000, chunks=10000)
    >>> X
    dask.array<..., shape=(100000, 2), dtype=float64, chunksize=(10000, 2)>
    >>> y
    dask.array<concatenate, shape=(100000,), dtype=int64, chunksize=(10000,)>

    See Also
    --------
    make_classification: a more intricate variant
    """
    chunks = da.core.normalize_chunks(chunks, (n_samples, n_features))
    _check_axis_partitioning(chunks, n_features)

    if centers is None:
        # TODO: non-int n_samples?
        centers = 3
    if isinstance(centers, numbers.Integral):
        # Make a prototype
        n_centers = centers
        X, y = sklearn.datasets.make_blobs(
            n_samples=chunks[0][0],
            n_features=n_features,
            centers=centers,
            shuffle=shuffle,
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=random_state,
        )
        centers = []
        centers = np.zeros((n_centers, n_features))

        for i in range(n_centers):
            centers[i] = X[y == i].mean(0)

    objs = [
        dask.delayed(sklearn.datasets.make_blobs, nout=2)(
            n_samples=n_samples_per_block,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            shuffle=shuffle,
            center_box=center_box,
            random_state=i,
        )
        for i, n_samples_per_block in enumerate(chunks[0])
    ]
    Xobjs, yobjs = zip(*objs)

    Xarrs = [
        da.from_delayed(arr, shape=(n, n_features), dtype="f8")
        for arr, n in zip(Xobjs, chunks[0])
    ]
    X_big = da.vstack(Xarrs)

    yarrs = [
        da.from_delayed(arr, shape=(n,), dtype=np.dtype("int"))
        for arr, n in zip(yobjs, chunks[0])
    ]
    y_big = da.hstack(yarrs)
    return X_big, y_big


def make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=True,
    coef=False,
    random_state=None,
    chunks=None,
):
    """
    Generate a random regression problem.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See
    :func:`sklearn.datasets.make_low_rank_matrix` for more details.

    This can be used to generate very large Dask arrays on a cluster of
    machines. When using Dask in distributed mode, the client machine
    only needs to allocate a single block's worth of data.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=100)
        The number of features.

    n_informative : int, optional (default=10)
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.

    n_targets : int, optional (default=1)
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.

    bias : float, optional (default=0.0)
        The bias term in the underlying linear model.

    effective_rank : int or None, optional (default=None)
        if not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.

    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    coef : boolean, optional (default=False)
        If True, the coefficients of the underlying linear model are returned.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    chunks : int, tuple
        How to chunk the array. Must be one of the following forms:
        -   A blocksize like 1000.
        -   A blockshape like (1000, 1000).
        -   Explicit sizes of all blocks along all dimensions like
            ((1000, 1000, 500), (400, 400)).

    Returns
    -------
    X : Dask array of shape [n_samples, n_features]
        The input samples.

    y : Dask array of shape [n_samples] or [n_samples, n_targets]
        The output values.

    coef : array of shape [n_features] or [n_features, n_targets], optional
        The coefficient of the underlying linear model. It is returned only if
        coef is True.
    """
    chunks = da.core.normalize_chunks(chunks, (n_samples, n_features))
    _check_axis_partitioning(chunks, n_features)

    rng = sklearn.utils.check_random_state(random_state)
    return_coef = coef is True

    if chunks[1][0] != n_features:
        raise ValueError(
            "Can only generate arrays partitioned along the "
            "first axis. Specifying a larger chunksize for "
            "the second axis."
        )
    _, _, coef = sklearn.datasets.make_regression(
        n_samples=chunks[0][0],
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        bias=bias,
        effective_rank=effective_rank,
        tail_strength=tail_strength,
        noise=noise,
        shuffle=shuffle,
        coef=True,  # hardcode here
        random_state=rng,
    )
    seed = da.random.random_state_data(1, random_state=rng)
    da_rng = da.random.RandomState(seed[0])

    X_big = da_rng.normal(size=(n_samples, n_features), chunks=(chunks[0], n_features))
    y_big = da.dot(X_big, coef) + bias

    if noise > 0:
        y_big = y_big + da_rng.normal(
            scale=noise, size=y_big.shape, chunks=y_big.chunks
        )

    y_big = y_big.squeeze()

    if return_coef:
        return X_big, y_big, coef
    else:
        return X_big, y_big


def make_classification(
    n_samples=100,
    n_features=20,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
    chunks=None,
):
    chunks = da.core.normalize_chunks(chunks, (n_samples, n_features))
    _check_axis_partitioning(chunks, n_features)

    if n_classes != 2:
        raise NotImplementedError("n_classes != 2 is not yet supported.")

    rng = dask_ml.utils.check_random_state(random_state)

    X = rng.normal(0, 1, size=(n_samples, n_features), chunks=chunks)
    informative_idx = rng.choice(n_features, n_informative, chunks=n_informative)
    beta = (rng.random(n_features, chunks=n_features) - 1) * scale

    informative_idx, beta = dask.compute(
        informative_idx, beta, scheduler="single-threaded"
    )

    z0 = X[:, informative_idx].dot(beta[informative_idx])
    y = rng.random(z0.shape, chunks=chunks[0]) < 1 / (1 + da.exp(-z0))
    y = y.astype(int)

    return X, y


def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = np.random.randint(int_delta)
    return start + timedelta(seconds=random_second)


def make_classification_df(
    n_samples=10000,
    response_rate=0.5,
    predictability=0.1,
    random_state=None,
    chunks=None,
    dates=None,
    **kwargs,
):
    """
    Uses the make_classification function to create a dask
    dataframe for testing.

    Parameters
    ----------
    n_samples : int, default is 10000
        number of observations to be generated
    response_rate : float between 0.0 and 0.5, default is 0.5
        percentage of sample to be response records max is 0.5
    predictability : float between 0.0 and 1.0, default is 0.1
        how hard is the response to predict (1.0 being easist)
    random_state : int, default is None
        seed for reproducability purposes
    chunks : int
        How to chunk the array. Must be one of the following forms:
        -   A blocksize like 1000.
    dates : tuple, optional, default is None
        tuple of start and end date objects to use for generating
        random dates in the date column
    **kwargs
        Other keyword arguments to pass to `sklearn.datasets.make_classification`

    Returns
    -------
    X : Dask DataFrame of shape [n_samples, n_features] or
        [n_samples, n_features + 1] when dates specified
        The input samples.

    y : Dask Series of shape [n_samples] or [n_samples, n_targets]
        The output values.

    """
    X_array, y_array = make_classification(
        n_samples=n_samples,
        flip_y=(1 - predictability),
        random_state=random_state,
        weights=[(1 - response_rate), response_rate],
        chunks=chunks,
        **kwargs,
    )

    # merge into a dataframe and name columns
    columns = ["var" + str(i) for i in range(np.shape(X_array)[1])]
    X_df = dd.from_dask_array(X_array, columns=columns)
    y_series = dd.from_dask_array(y_array, columns="target", index=X_df.index)

    if dates:
        # create a date variable
        np.random.seed(random_state)
        X_df = dd.concat(
            [
                X_df,
                dd.from_array(
                    np.array([random_date(*dates)] * len(X_df)),
                    chunksize=chunks,
                    columns=["date"],
                ),
            ],
            axis=1,
        )

    return X_df, y_series
