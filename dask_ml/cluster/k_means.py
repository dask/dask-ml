import logging
from multiprocessing import cpu_count
from numbers import Integral

import dask.array as da
import dask.dataframe as dd
import numba
import numpy as np
import pandas as pd
from dask import compute
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import k_means_ as sk_k_means
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted

from .._utils import draw_seed
from ..metrics import (
    euclidean_distances,
    pairwise_distances,
    pairwise_distances_argmin_min,
)
from ..utils import _timed, _timer, check_array, row_norms

logger = logging.getLogger(__name__)


class KMeans(TransformerMixin, BaseEstimator):
    """
    Scalable KMeans for clustering

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters to end up with
    init : {'k-means||', 'k-means++' or ndarray}
        Method for center initialization, defualts to 'k-means||'.

        'k-means||' : selects the the gg

        'k-means++' : selects the initial cluster centers in a smart way
        to speed up convergence. Uses scikit-learn's implementation.

        .. warning::

           If using ``'k-means++'``, the entire dataset will be read into
           memory at once.

        An array of shape (n_clusters, n_features) can be used to give
        an explicit starting point

    oversampling_factor : int, default 2
        Oversampling factor for use in the ``k-means||`` algorithm.

    max_iter : int
        Maximum number EM iterations to attempt.

    init_max_iter : int
        Number of iterations for init step.

    tol : float
        Relative tolerance with regards to inertia to declare convergence

    algorithm : 'full'
        The algorithm to use for the EM step. Only "full" (LLoyd's algorithm)
        is allowed.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    cluster_centers_ : np.ndarray [n_clusters, n_features]
        A NumPy array with the cluster centers

    labels_ : da.array [n_samples,]
        A dask array with the index position in ``cluster_centers_`` this
        sample belongs to.

    intertia_ : float
        Sum of distances of samples to their closest cluster center.

    n_iter_ : int
        Number of EM steps to reach convergence

    References
    ----------
    - Scalable K-Means++, 2012
      Bahman Bahmani, Benjamin Moseley, Andrea Vattani, Ravi Kumar,
      Sergei Vassilvitskii
      https://arxiv.org/abs/1203.6402

    See Also
    --------
    PartialMiniBatchKMeans
    sklearn.cluster.MiniBatchKMeans
    sklearn.cluster.KMeans

    Notes
    -----
    This class implements a parallel and distributed version of k-Means.

    **Initialization with k-means||**

    The default initializer for ``KMeans`` is ``k-means||``, compared to
    ``k-means++`` from scikit-learn. This is the algorithm described in
    *Scalable K-Means++ (2012)*.

    ``k-means||`` is designed to work well in a distributed environment. It's a
    variant of k-means++ that's designed to work in parallel (k-means++ is
    inherently sequential). Currently, the ``k-means||`` implementation here is
    slower than scikit-learn's ``k-means++`` if your entire dataset fits in
    memory on a single machine. If that's the case, consider using
    ``init='k-means++'``.

    **Parallel Lloyd's Algorithm**

    LLoyd's Algorithm (the default Expectation Maximization algorithm used in
    scikit-learn) is naturally parallelizable. In naive benchmarks, the
    implementation here achieves 2-3x speedups over scikit-learn.

    Both the initialization step and the EM steps make multiple passes over the
    data. If possible, persist your dask collections in (distributed) memory
    before running ``.fit``.
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means||",
        oversampling_factor=2,
        max_iter=300,
        tol=0.0001,
        precompute_distances="auto",
        random_state=None,
        copy_x=True,
        n_jobs=1,
        algorithm="full",
        init_max_iter=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.oversampling_factor = oversampling_factor
        self.random_state = random_state
        self.max_iter = max_iter
        self.init_max_iter = init_max_iter
        self.algorithm = algorithm
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_jobs = n_jobs
        self.copy_x = copy_x

    @_timed(_logger=logger)
    def _check_array(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=True)

        X = check_array(
            X,
            accept_dask_dataframe=False,
            accept_unknown_chunks=False,
            accept_sparse=False,
        )

        if X.dtype == "int32":
            X = X.astype("float32")
        elif X.dtype == "int64":
            X = X.astype("float64")

        if isinstance(X, np.ndarray):
            X = da.from_array(X, chunks=(max(1, len(X) // cpu_count()), X.shape[-1]))

        bad = (da.isnull(X).any(), da.isinf(X).any())
        if any(*compute(bad)):
            msg = (
                "Input contains NaN, infinity or a value too large for "
                "dtype('float64')."
            )
            raise ValueError(msg)
        return X

    def fit(self, X, y=None):
        X = self._check_array(X)
        labels, centroids, inertia, n_iter = k_means(
            X,
            self.n_clusters,
            oversampling_factor=self.oversampling_factor,
            random_state=self.random_state,
            init=self.init,
            return_n_iter=True,
            max_iter=self.max_iter,
            init_max_iter=self.init_max_iter,
            tol=self.tol,
        )
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia.compute()
        self.n_iter_ = n_iter
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "cluster_centers_")
        X = self._check_array(X)
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")
        X = self._check_array(X)
        labels = pairwise_distances_argmin_min(X, self.cluster_centers_)[0].astype(
            np.int32
        )
        return labels


def k_means(
    X,
    n_clusters,
    init="k-means||",
    precompute_distances="auto",
    n_init=1,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    n_jobs=-1,
    algorithm="full",
    return_n_iter=False,
    oversampling_factor=2,
    init_max_iter=None,
):
    """K-means algorithm for clustering

    Differences from scikit-learn:

    * init='k-means||'
    * oversampling_factor keyword
    * n_jobs=-1
    """
    labels, inertia, centers, n_iter = _kmeans_single_lloyd(
        X,
        n_clusters,
        max_iter=max_iter,
        init=init,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        oversampling_factor=oversampling_factor,
        init_max_iter=init_max_iter,
    )
    if return_n_iter:
        return labels, centers, inertia, n_iter
    else:
        return labels, centers, inertia


def compute_inertia(X, labels, centers):
    reindexed = labels.map_blocks(
        lambda x: centers[x], dtype=centers.dtype, chunks=X.chunks, new_axis=1
    )
    inertia = (X - reindexed).sum()
    return inertia


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------


def k_init(
    X,
    n_clusters,
    init="k-means||",
    random_state=None,
    max_iter=None,
    oversampling_factor=2,
):
    """Choose the initial centers for K-Means.

    Parameters
    ----------
    X : da.Array (n_samples, n_features)
    n_clusters : int
        Number of clusters to end up with
    init : {'k-means||', 'k-means++', 'random'} or numpy.ndarray
        Initialization method, or pass a NumPy array to use
    random_state : int, optional
    max_iter : int, optional
        Only used for ``init='k-means||'``.
    oversampling_factor : int, optional
        Only used for ``init='k-means||`''. Controls the additional number of
        candidate centers in each iteration.

    Return
    ------
    centers : np.ndarray (n_clusters, n_features)

    Notes
    -----
    The default strategy is ``k-means||``, which tends to be slower than
    ``k-means++`` for small (in-memory) datasets, but works better in a
    distributed setting.

    .. warning::

       Using ``init='k-means++'`` assumes that the entire dataset fits
       in RAM.
    """
    if isinstance(init, np.ndarray):
        K, P = init.shape

        if K != n_clusters:
            msg = (
                "Number of centers in provided 'init' ({}) does "
                "not match 'n_clusters' ({})"
            )
            raise ValueError(msg.format(K, n_clusters))

        if P != X.shape[1]:
            msg = (
                "Number of features in the provided 'init' ({}) do not "
                "match the number of features in 'X'"
            )
            raise ValueError(msg.format(P, X.shape[1]))

        return init

    elif not isinstance(init, str):
        raise TypeError("'init' must be an array or str, got {}".format(type(init)))

    valid = {"k-means||", "k-means++", "random"}

    if isinstance(random_state, Integral) or random_state is None:
        if init == "k-means||":
            random_state = da.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState(random_state)

    if init == "k-means||":
        return init_scalable(X, n_clusters, random_state, max_iter, oversampling_factor)
    elif init == "k-means++":
        return init_pp(X, n_clusters, random_state)

    elif init == "random":
        return init_random(X, n_clusters, random_state)

    else:
        raise ValueError("'init' must be one of {}, got {}".format(valid, init))


def init_pp(X, n_clusters, random_state):
    """K-means initialization using k-means++

    This uses scikit-learn's implementation.
    """
    x_squared_norms = row_norms(X, squared=True).compute()
    logger.info("Initializing with k-means++")
    with _timer("initialization of %2d centers" % n_clusters, _logger=logger):
        centers = sk_k_means._k_init(
            X, n_clusters, random_state=random_state, x_squared_norms=x_squared_norms
        )

    return centers


@_timed(_logger=logger)
def init_random(X, n_clusters, random_state):
    """K-means initialization using randomly chosen points"""
    logger.info("Initializing randomly")
    idx = sorted(draw_seed(random_state, 0, len(X), size=n_clusters))
    centers = X[idx].compute()
    return centers


@_timed(_logger=logger)
def init_scalable(
    X, n_clusters, random_state=None, max_iter=None, oversampling_factor=2
):
    """K-Means initialization using k-means||

    This is algorithm 2 in Scalable K-Means++ (2012).
    """

    logger.info("Initializing with k-means||")
    # Step 1: Initialize Centers
    idx = 0
    centers = da.compute(X[idx, np.newaxis])[0]
    c_idx = {idx}

    # Step 2: Initialize cost
    cost, = compute(evaluate_cost(X, centers))

    if cost == 0:
        n_iter = 0
    else:
        n_iter = int(np.round(np.log(cost)))

    if max_iter is not None:
        n_iter = min(max_iter, n_iter)

    # Steps 3 - 6: update candidate Centers
    for i in range(n_iter):
        with _timer(
            "init iteration %2d/%2d , %2d centers" % (i + 1, n_iter, len(c_idx)),
            _logger=logger,
        ):
            new_idxs = _sample_points(X, centers, oversampling_factor, random_state)
            new_idxs = set(*compute(new_idxs))
            c_idx |= new_idxs

        # Sort before slicing, for better performance / memory
        # usage with the scheduler.
        # See https://github.com/dask/dask-ml/issues/39
        centers = X[sorted(c_idx)].compute()

    # XXX: scikit-learn doesn't have weighted k-means.
    # The paper weights each center by the number of points closest to it.
    # https://stackoverflow.com/a/37198799/1889400 claims you can scale the
    # features before clustering, but that doesn't seem right.
    # I think that replicating the *points*, proportional to the number of
    # original points closest to the candidate centers, would be a better way
    # to do that.

    if len(centers) < n_clusters:
        logger.warning("Found fewer than %d clusters in init.", n_clusters)
        # supplement with random
        need = n_clusters - len(centers)
        locs = sorted(
            random_state.choice(
                np.arange(0, len(X)), size=need, replace=False, chunks=len(X)
            )
        )
        extra = X[locs].compute()
        return np.vstack([centers, extra])
    else:
        # Step 7, 8 without weights
        # dask RandomState objects aren't valid for scikit-learn
        rng2 = (
            draw_seed(random_state, 0, 2 ** 32 - 1, dtype="uint", chunks=())
            .compute()
            .item()
        )
        km = sk_k_means.KMeans(n_clusters, random_state=rng2)
        km.fit(centers)

    return km.cluster_centers_


def evaluate_cost(X, centers):
    # type: (da.Array, np.array) -> float
    # parallel for dask arrays
    return (pairwise_distances(X, centers).min(1) ** 2).sum()


def _sample_points(X, centers, oversampling_factor, random_state):
    r"""
    Sample points independently with probability


    .. math::

       p_x = \frac{\ell \cdot d^2(x, \mathcal{C})}{\phi_X(\mathcal{C})}

    """
    # re-implement evaluate_cost here, to avoid redundant computation
    distances = pairwise_distances(X, centers).min(1) ** 2
    denom = distances.sum()
    p = oversampling_factor * distances / denom

    draws = random_state.uniform(size=len(p), chunks=p.chunks)
    picked = p > draws

    new_idxs, = da.where(picked)
    return new_idxs


# -----------------------------------------------------------------------------
# EM Steps
# -----------------------------------------------------------------------------


def _kmeans_single_lloyd(
    X,
    n_clusters,
    max_iter=300,
    init="k-means||",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
    precompute_distances=True,
    oversampling_factor=2,
    init_max_iter=None,
):
    centers = k_init(
        X,
        n_clusters,
        init=init,
        oversampling_factor=oversampling_factor,
        random_state=random_state,
        max_iter=init_max_iter,
    )
    dt = X.dtype
    P = X.shape[1]
    for i in range(max_iter):
        with _timer("Lloyd loop %2d." % i, _logger=logger):
            labels, distances = pairwise_distances_argmin_min(
                X, centers, metric="euclidean", metric_kwargs={"squared": True}
            )

            labels = labels.astype(np.int32)
            # distances is always float64, but we need it to match X.dtype
            # for centers_dense, but remain float64 for inertia
            r = da.atop(
                _centers_dense,
                "ij",
                X,
                "ij",
                labels,
                "i",
                n_clusters,
                None,
                distances.astype(X.dtype),
                "i",
                adjust_chunks={"i": n_clusters, "j": P},
                dtype=X.dtype,
            )
            new_centers = da.from_delayed(
                sum(r.to_delayed().flatten()), (n_clusters, P), X.dtype
            )
            counts = da.bincount(labels, minlength=n_clusters)
            # Require at least one per bucket, to avoid division by 0.
            counts = da.maximum(counts, 1)
            new_centers = new_centers / counts[:, None]
            new_centers, = compute(new_centers)

            # Convergence check
            shift = squared_norm(centers - new_centers)

            logger.info("Shift: %0.4f", shift)
            if shift < tol:
                break
            centers = new_centers

    if shift > 1e-7:
        labels, distances = pairwise_distances_argmin_min(X, centers)
        labels = labels.astype(np.int32)

    inertia = distances.sum()
    centers = centers.astype(dt)

    return labels, inertia, centers, i + 1


@numba.njit(nogil=True, fastmath=True)
def _centers_dense(X, labels, n_clusters, distances):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]

    return centers
