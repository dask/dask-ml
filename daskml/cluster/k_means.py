"""
https://arxiv.org/abs/1203.6402

1: C ← sample a point uniformly at random from X
2: ψ←φX(C)
3: for O(log ψ) times do
4: C′ ← sample each point x ∈ X independently with
      probability p_x = l·d2(x, C) / φ_X(C)
5: C ← C ∪ C′
6: end for
7: For x∈C,set w_x to be the number of points in X closer
    to x than any other point in C
8: Recluster the weighted points in C into k clusters

Prior Attemps:

- https://github.com/scikit-learn/scikit-learn/issues/4357
- https://github.com/scikit-learn/scikit-learn/pull/5530
- https://github.com/scikit-learn/scikit-learn/pull/8585

"""
import logging
from timeit import default_timer as tic

import numpy as np
import dask.array as da
from dask import compute
from sklearn.base import BaseEstimator
from sklearn.cluster import k_means_ as sk_k_means
from sklearn.utils.extmath import squared_norm

from daskml.metrics import pairwise_distances_argmin_min, pairwise_distances
from ._k_means import _centers_dense


logger = logging.getLogger(__name__)


class KMeans(BaseEstimator):
    """
    Scalable KMeans for clustering

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters to end up with
    init : 'k-means||' or ndarray
        'k-means||' will initialize using the ``k-means||`` algorithm.
        An array of shape [n_clusters, n_features], can be used to give
        an explicit starting point
    oversampling_factor : int, default 2
        Oversampling factor for use in the ``k-means||`` algorithm.
    max_iter : int
    tol : float
    algorithm : 'full'
        The algorithm to use for the EM step. Only "full" (LLoyd's algorithm)
        is allowed.

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
    """
    def __init__(self, n_clusters=8, init='k-means||', oversampling_factor=2,
                 max_iter=300, tol=0.0001, precompute_distances='auto',
                 random_state=None, copy_x=True, n_jobs=1, algorithm='full'):
        self.n_clusters = n_clusters
        self.init = init
        self.oversampling_factor = oversampling_factor
        self.random_state = random_state
        self.max_iter = max_iter
        self.algorithm = algorithm
        self.tol = tol
        self.n_jobs = n_jobs
        self.copy_x = copy_x

    def fit(self, X):
        labels, centroids, inertia, n_iter = k_means(
            X, self.n_clusters, oversampling_factor=self.oversampling_factor,
            random_state=self.random_state, init=self.init,
            return_n_iter=True
        )
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia.compute()
        self.n_iter_ = n_iter
        return self

    def transform(self, X):
        pass


def k_means(X, n_clusters, init='k-means||', precompute_distances='auto',
            n_init=1, max_iter=300, verbose=False,
            tol=1e-4, random_state=None, copy_x=True, n_jobs=-1,
            algorithm='full', return_n_iter=False, oversampling_factor=2):
    """K-means algorithm for clustering

    Differences from scikit-learn:

    * init='k-means||'
    * oversampling_factor keyword
    * n_jobs=-1
    """
    labels, inertia, centers, n_iter = _kmeans_single_lloyd(
        X, n_clusters, max_iter=max_iter, init=init, verbose=verbose,
        tol=tol, random_state=random_state,
        oversampling_factor=oversampling_factor)
    if return_n_iter:
        return labels, centers, inertia, n_iter
    else:
        return labels, centers, inertia


def compute_inertia(X, labels, centers):
    reindexed = labels.map_blocks(lambda x: centers[x], dtype=centers.dtype,
                                  chunks=X.chunks, new_axis=1)
    inertia = (X - reindexed).sum()
    return inertia


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

def k_init(X, n_clusters, init='k-means||',
           oversampling_factor=2,
           random_state=None,
           max_iter=None):
    """
    Algorithm 2 in scalable k-means||
    """
    if isinstance(init, np.ndarray):
        return init

    if isinstance(random_state, int) or random_state is None:
        random_state = np.random.RandomState(random_state)

    if init == 'k-means++':
        x_squared_norms = row_norms(X, squared=True).compute()
        logger.info("Initializing with k-means++")
        t0 = tic()
        centers = sk_k_means._k_init(X, n_clusters, random_state=random_state,
                                     x_squared_norms=x_squared_norms)
        logger.info("Finished initialization. %.2f s, %2d centers",
                    tic() - t0, n_clusters)

        return centers
    elif init != 'k-means||':
        raise TypeError("Unexpected value for `init` {!r}".foramt(init))

    logger.info("Starting Init")
    init_start = tic()
    # Step 1: Initialize Centers
    idx = 0
    centers = da.compute(X[idx, np.newaxis])[0]
    c_idx = {idx}

    # Step 2: Initialize cost
    cost = evaluate_cost(X, centers)
    # TODO: natural log10? log2?
    n_iter = int(np.round(np.log(cost)))
    if max_iter is not None:
        n_iter = min(max_iter, n_iter)

    # Steps 3 - 6: update candidate Centers
    for i in range(n_iter):
        t0 = tic()
        new_idxs = _sample_points(X, centers, oversampling_factor)
        new_idxs = set(*compute(new_idxs))
        c_idx |= new_idxs
        t1 = tic()
        logger.info("init iteration %2d/%2d %.2f s, %2d centers",
                    i + 1, n_iter, t1 - t0, len(c_idx))
        centers = X[list(c_idx)].compute()

    # Step 7: weights
    # XXX: scikit-learn doesn't have weighted k-means.
    # https://stackoverflow.com/a/37198799/1889400 claims you can scale the
    # features before clustering. Need to investigate more
    # x_squared_norms = row_norms(X, squared=True)
    # if isinstance(X, da.Array):
    #     labels, _ = _labels_inertia(X, x_squared_norms, centers)
    #     labels = labels.compute()
    # else:
    #     labels, _ = sk_k_means._labels_inertia(X, x_squared_norms, centers)

    # wx = np.bincount(labels)
    # p = wx / wx.sum()
    # wc = (centers.T * p).T

    # # just re-use sklearn kmeans for the reduce step
    # # but this should be weighted...
    # init = sk_k_means._k_init(wc, n_clusters,
    #                         sk_k_means.row_norms(wc, squared=True),
    #                         random_state)
    # picked = []
    # for i, row in enumerate(wc):
    #     m = (row == init).any(1)
    #     if m.any():
    #         picked.append(i)
    # return centers[picked]

    # Step 7, 8 without weights
    km = sk_k_means.KMeans(n_clusters)
    km.fit(centers)
    logger.info("Finished initialization. %.2f s, %2d centers",
                tic() - init_start, n_clusters)
    return km.cluster_centers_


def evaluate_cost(X, centers):
    # type: (da.Array, np.array) -> float
    # parallel for dask arrays
    return (pairwise_distances(X, centers).min(1) ** 2).sum()


def _sample_points(X, centers, oversampling_factor):
    r"""
    Sample points independently with probability


    .. math::

       p_x = \frac{\ell \cdot d^2(x, \mathcal{C})}{\phi_X(\mathcal{C})}

    """
    # re-implement evaluate_cost here, to avoid redundant computation
    distances = pairwise_distances(X, centers).min(1) ** 2
    denom = distances.sum()
    p = oversampling_factor * distances / denom

    draws = da.random.uniform(size=len(p), chunks=p.chunks)
    picked = p > draws

    new_idxs, = da.where(picked)
    return new_idxs


# -----------------------------------------------------------------------------
# EM Steps
# -----------------------------------------------------------------------------

def _kmeans_single_lloyd(X, n_clusters, max_iter=300, init='k-means||',
                         verbose=False, x_squared_norms=None,
                         random_state=None, tol=1e-4,
                         precompute_distances=True,
                         oversampling_factor=2):
    centers = k_init(X, n_clusters, init=init,
                     oversampling_factor=oversampling_factor,
                     random_state=random_state)
    dt = X.dtype
    X = X.astype(np.float32)
    P = X.shape[1]
    for i in range(max_iter):
        t0 = tic()
        centers = centers.astype('f4')
        labels, distances = pairwise_distances_argmin_min(
            X, centers, metric='euclidean', metric_kwargs={"squared": True}
        )

        labels = labels.astype(np.int32)
        distances = distances.astype(np.float32)

        r = da.atop(_centers_dense, 'ij',
                    X, 'ij',
                    labels, 'i',
                    n_clusters, None,
                    distances, 'i',
                    adjust_chunks={"i": n_clusters, "j": P},
                    dtype='f8')
        new_centers = sum(r.to_delayed().flatten())
        counts = da.bincount(labels, minlength=n_clusters)
        new_centers = new_centers / counts[:, None]
        new_centers, = compute(new_centers)

        # Convergence check
        shift = squared_norm(centers - new_centers)
        t1 = tic()
        logger.info("Lloyd loop %2d. Shift: %0.2f [%.2f s]", i, shift, t1 - t0)
        if shift < tol:
            break
        centers = new_centers

    if shift > 1e-7:
        labels, distances = pairwise_distances_argmin_min(X, centers)
    inertia = distances.astype(dt).sum()
    centers = centers.astype(dt)

    return labels, inertia, centers, i + 1


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------


def row_norms(X: da.Array, squared: bool=False) -> da.Array:
    if isinstance(X, np.ndarray):
        return sk_k_means.row_norms(X, squared=squared)
    return X.map_blocks(sk_k_means.row_norms, chunks=(X.chunks[0],),
                        drop_axis=1, squared=squared)
