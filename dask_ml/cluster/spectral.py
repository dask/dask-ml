# -*- coding: utf-8 -*-
"""Algorithms for spectral clustering
"""
import logging

import dask.array as da
import numpy as np
import six
import sklearn.cluster
from dask import delayed
from scipy.linalg import pinv, svd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from .._utils import draw_seed
from ..metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from ..utils import _format_bytes, _log_array, check_array
from .k_means import KMeans

logger = logging.getLogger(__name__)


class SpectralClustering(BaseEstimator, ClusterMixin):
    """Apply parallel Spectral Clustering

    This implementation avoids the expensive computation of the N x N
    affinity matrix. Instead, the Nyström Method is used as an
    approximation.

    Parameters
    ----------
    n_clusters : integer, optional
        The dimension of the projection subspace.

    eigen_solver : None
        ignored

    random_state : int, RandomState instance or None, optional, default: None
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg' and by
        the K-Means initialization.  If int, random_state is the seed used by
        the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is
        the RandomState instance used by `np.random`.

    n_init : int, optional, default: 10
        ignored

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``.

    affinity : string, array-like or callable, default 'rbf'
        If a string, this may be one of 'nearest_neighbors', 'precomputed',
        'rbf' or one of the kernels supported by
        `sklearn.metrics.pairwise_kernels`.

        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.

        Callables should expect arguments similar to
        `sklearn.metrics.pairwise_kernels`: a required ``X``, an optional
        ``Y``, and ``gamma``, ``degree``, ``coef0``, and any keywords passed
        in ``kernel_params``.

    n_neighbors : integer
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.

    eigen_tol : float, optional, default: 0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    assign_labels : 'kmeans' or Estimator, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space. By default creates an instance of
        :class:`dask_ml.cluster.KMeans` and sets `n_clusters` to 2. For
        further control over the hyperparameters of the final label
        assignment, pass an instance of a ``KMeans`` estimator (either
        scikit-learn or dask-ml).

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    n_components : int, default 100
        Number of rows from ``X`` to use for the Nyström approximation.
        Larger ``n_components`` will improve the accuracy of the
        approximation, at the cost of a longer training time.

    persist_embedding : bool
        Whether to persist the intermediate n_samples x n_components
        array used for clustering.

    kmeans_params : dictionary of string to any, optional
        Keyword arguments for the KMeans clustering used for the final
        clustering.

    Attributes
    ----------
    assign_labels_ : Estimator
        The instance of the KMeans estimator used to assign labels
    labels_ : dask.array.Array, size (n_samples,)
        The cluster labels assigned
    eigenvalues_ : numpy.ndarray
        The eigenvalues from the SVD of the sampled points

    Notes
    -----
    Using ``persist_embedding=True`` can be an important optimization to
    avoid some redundant computations. This persists the array being fed to
    the clustering algorithm in (distributed) memory. The array is shape
    ``n_samples x n_components``.

    References
    ----------
    - Parallel Spectral Clustering in Distributed Systems, 2010
      Chen, Song, Bai, Lin, and Chang
      IEEE Transactions on Pattern Analysis and Machine Intelligence
      http://ieeexplore.ieee.org/document/5444877/

    - Spectral Grouping Using the Nystrom Method (2004)
      Fowlkes, Belongie, Chung, Malik
      IEEE Transactions on Pattern Analysis and Machine Intelligence
      https://people.cs.umass.edu/~mahadeva/cs791bb/reading/fowlkes-nystrom.pdf
    """

    def __init__(
        self,
        n_clusters=8,
        eigen_solver=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol=0.0,
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=1,
        n_components=100,
        persist_embedding=False,
        kmeans_params=None,
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.n_components = n_components
        self.persist_embedding = persist_embedding
        self.kmeans_params = kmeans_params

    def _check_array(self, X):
        logger.info("Starting check array")
        result = check_array(X, accept_dask_dataframe=False).astype(float)
        logger.info("Finished check array")
        return result

    def fit(self, X, y=None):
        X = self._check_array(X)
        n_components = self.n_components
        metric = self.affinity
        rng = check_random_state(self.random_state)
        n_clusters = self.n_clusters

        # kmeans for final clustering
        if isinstance(self.assign_labels, six.string_types):
            if self.assign_labels == "kmeans":
                km = KMeans(
                    n_clusters=n_clusters,
                    random_state=draw_seed(rng, 2 ** 32 - 1, dtype="uint"),
                )
            elif self.assign_labels == "sklearn-kmeans":
                km = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=rng)
            else:
                msg = "Unknown 'assign_labels' {!r}".format(self.assign_labels)
                raise ValueError(msg)
        elif isinstance(self.assign_labels, BaseEstimator):
            km = self.assign_labels
        else:
            raise TypeError(
                "Invalid type {} for 'assign_labels'".format(type(self.assign_labels))
            )

        if self.kmeans_params:
            km.set_params(**self.kmeans_params)

        n = len(X)
        if n <= n_components:
            msg = (
                "'n_components' must be smaller than the number of samples."
                " Got {} components and {} samples".format(n_components, n)
            )
            raise ValueError(msg)

        params = self.kernel_params or {}
        params["gamma"] = self.gamma
        params["degree"] = self.degree
        params["coef0"] = self.coef0

        # indices for our exact / approximate blocks
        inds = np.arange(n)
        keep = rng.choice(inds, n_components, replace=False)
        keep.sort()
        rest = ~np.isin(inds, keep)

        # compute the exact blocks
        # these are done in parallel for dask arrays
        if isinstance(X, da.Array):
            X_keep = X[keep].rechunk(X.shape).persist()
        else:
            X_keep = X[keep]

        X_rest = X[rest]

        A, B = embed(X_keep, X_rest, n_components, metric, params)
        _log_array(logger, A, "A")
        _log_array(logger, B, "B")

        # now the approximation of C
        a = A.sum(0)  # (l,)
        b1 = B.sum(1)  # (l,)
        b2 = B.sum(0)  # (m,)

        # TODO: I think we have some unnecessary delayed wrapping of A here.
        A_inv = da.from_delayed(delayed(pinv)(A), A.shape, A.dtype)

        inner = A_inv.dot(b1)
        d1_si = 1 / da.sqrt(a + b1)

        d2_si = 1 / da.sqrt(b2 + B.T.dot(inner))  # (m,), dask array

        # d1, d2 are diagonal, so we can avoid large matrix multiplies
        # Equivalent to diag(d1_si) @ A @ diag(d1_si)
        A2 = d1_si.reshape(-1, 1) * A * d1_si.reshape(1, -1)  # (n, n)
        _log_array(logger, A2, "A2")
        # A2 = A2.rechunk(A2.shape)
        # Equivalent to diag(d1_si) @ B @ diag(d2_si)
        B2 = da.multiply(da.multiply(d1_si.reshape(-1, 1), B), d2_si.reshape(1, -1))
        _log_array(logger, B2, "B2")

        U_A, S_A, V_A = delayed(svd, pure=True, nout=3)(A2)

        U_A = da.from_delayed(U_A, (n_components, n_components), A2.dtype)
        S_A = da.from_delayed(S_A, (n_components,), A2.dtype)
        V_A = da.from_delayed(V_A, (n_components, n_components), A2.dtype)

        # Eq 16. This is OK when V2 is orthogonal
        V2 = da.sqrt(float(n_components) / n) * da.vstack([A2, B2.T]).dot(
            U_A[:, :n_clusters]
        ).dot(
            da.diag(1.0 / da.sqrt(S_A[:n_clusters]))
        )  # (n, k)
        _log_array(logger, V2, "V2.1")

        if isinstance(B2, da.Array):
            V2 = V2.rechunk((B2.chunks[1][0], n_clusters))
            _log_array(logger, V2, "V2.2")

        # normalize (Eq. 4)
        U2 = (V2.T / da.sqrt((V2 ** 2).sum(1))).T  # (n, k)

        _log_array(logger, U2, "U2.2")

        # Recover original indices
        U2 = _slice_mostly_sorted(U2, keep, rest, inds)  # (n, k)

        _log_array(logger, U2, "U2.3")

        if self.persist_embedding and isinstance(U2, da.Array):
            logger.info("Persisting array for k-means")
            U2 = U2.persist()
        elif isinstance(U2, da.Array):
            logger.info(
                "Consider persist_embedding. This will require %s",
                _format_bytes(U2.nbytes),
            )
            pass
        logger.info("k-means for assign_labels[starting]")
        km.fit(U2)
        logger.info("k-means for assign_labels[finished]")

        # Now... what to keep?
        self.assign_labels_ = km
        self.labels_ = km.labels_
        self.eigenvalues_ = S_A[:n_clusters]  # TODO: better name
        return self


def embed(X_keep, X_rest, n_components, metric, kernel_params):
    if isinstance(metric, six.string_types):
        if metric not in PAIRWISE_KERNEL_FUNCTIONS:
            msg = "Unknown affinity metric name '{}'. Expected one " "of '{}'".format(
                metric, PAIRWISE_KERNEL_FUNCTIONS.keys()
            )
            raise ValueError(msg)
        A = pairwise_kernels(X_keep, metric=metric, filter_params=True, **kernel_params)
        B = pairwise_kernels(
            X_keep, X_rest, metric=metric, filter_params=True, **kernel_params
        )
    elif callable(metric):
        A = metric(X_keep, **kernel_params)
        B = metric(X_keep, X_rest, **kernel_params)
    else:
        msg = (
            "Unexpected type for 'affinity' '{}'. Must be string "
            "kernel name, array, or callable"
        )
        raise TypeError(msg)
    if isinstance(A, da.Array):
        A = A.rechunk((n_components, n_components))
        B = B.rechunk((B.shape[0], B.chunks[1]))
    return A, B


def _slice_mostly_sorted(array, keep, rest, ind=None):
    """Slice dask array `array` that is almost entirely sorted already.

    We perform approximately `2 * len(keep)` slices on `array`.
    This is OK, since `keep` is small. Individually, each of these slices
    is entirely sorted.

    Parameters
    ----------
    array : dask.array.Array
    keep : ndarray[Int]
        This must be sorted.
    rest : ndarray[Bool]
    ind : ndarray[Int], optional

    Returns
    -------
    sliced : dask.array.Array
    """
    if ind is None:
        ind = np.arange(len(array))
    idx = np.argsort(np.concatenate([keep, ind[rest]]))

    slices = []
    if keep[0] > 0:  # avoid creating empty slices
        slices.append(slice(None, keep[0]))
    slices.append([keep[0]])
    windows = zip(keep[:-1], keep[1:])

    for l, r in windows:
        if r > l + 1:  # avoid creating empty slices
            slices.append(slice(l + 1, r))
        slices.append([r])

    if keep[-1] < len(array) - 1:  # avoid creating empty slices
        slices.append(slice(keep[-1] + 1, None))
    result = da.concatenate([array[idx[slice_]] for slice_ in slices])
    return result
