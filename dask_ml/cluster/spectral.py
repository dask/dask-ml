# -*- coding: utf-8 -*-
"""Algorithms for spectral clustering
"""
import logging

import six
import dask.array as da
import numpy as np
import sklearn.cluster
# from dask.array.linalg import svd
from scipy.linalg import pinv, svd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from .k_means import KMeans
from ..metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from ..utils import check_array


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

    kmeans_params : dictionary of string to any, optional
        Keyword arguments for the KMeans clustering used for the final
        clustering.

    Attributes
    ----------
    basis_inds_ : ndarray
        The ordering used to partition X into the samples for the
        Nyström approximation
    assign_labels_ : Estimator
        The instance of the KMeans estimator used to assign labels
    labels_ : dask.array.Array, size (n_samples,)
        The cluster labels assigned
    eigenvalues_ : numpy.ndarray
        The eigenvalues from the SVD of the sampled points

    Notes
    -----

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
    def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', n_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=1, n_components=100,
                 kmeans_params=None):
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
        self.kmeans_params = kmeans_params

    def _check_array(self, X):
        return check_array(X, accept_dask_dataframe=False).astype(float)

    def fit(self, X, y=None):
        X = self._check_array(X)
        n_components = self.n_components
        metric = self.affinity
        if metric not in PAIRWISE_KERNEL_FUNCTIONS:
            raise ValueError("Only kernel names")

        rng = check_random_state(self.random_state)
        n_clusters = self.n_clusters

        # kmeans for final clustering
        if isinstance(self.assign_labels, six.string_types):
            if self.assign_labels == 'kmeans':
                km = KMeans(n_clusters=n_clusters,
                            random_state=rng.randint(2**32 - 1))
            elif self.assign_labels == 'sklearn-kmeans':
                km = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                            random_state=rng)
            else:
                msg = "Unknown 'assign_labels' {!r}".foramt(self.assign_labels)
                raise ValueError(msg)
        elif isinstance(self.assign_labels, BaseEstimator):
            km = self.assign_labels
        else:
            raise TypeError("Invalid type {} for 'assign_labels'".format(
                type(self.assign_labels)))

        if self.kmeans_params:
            km.set_params(self.kmeans_params)

        n = len(X)

        inds = rng.permutation(np.arange(n))
        keep = inds[:n_components]
        rest = inds[n_components:]
        # distributed slice perf.
        keep.sort()
        rest.sort()
        # Those sorts modify `inds` inplace, so `argsort(inds)` will still
        # recover the original order.
        inds_idx = np.argsort(inds)

        params = self.kernel_params or {}
        params['gamma'] = self.gamma
        params['degree'] = self.gamma
        params['gamma'] = self.gamma

        # compute the exact blocks
        # these are done in parallel
        A = pairwise_kernels(X[keep],
                             metric=metric, filter_params=True, **params)
        B = pairwise_kernels(X[keep], X[rest],
                             metric=metric, filter_params=True, **params)

        # now the approximation of C
        a = A.sum(0)   # (l,)
        b1 = B.sum(1)  # (l,)
        b2 = B.sum(0)  # (m,)

        A_inv = pinv(A)

        d1_si = 1 / da.sqrt(a + b1)  # (l,)
        d2_si = 1 / da.sqrt(b2 + B.T.dot(A_inv.dot(b1)))  # (m,), dask array

        # d1, d2 are diagonal, so we can avoid large matrix multiplies
        # Equivalent to diag(d1_si) @ A @ diag(d1_si)
        A2 = d1_si.reshape(-1, 1) * A * d1_si.reshape(1, -1)  # (n, n)
        # A2 = A2.rechunk(A2.shape)
        # Equivalent to diag(d1_si) @ B @ diag(d2_si)
        B2 = d1_si.reshape(-1, 1) * B * d2_si  # (m, m), so this is dask.

        U_A, S_A, V_A = svd(A2)  # (l, l), (l,), (l, l)
        # Eq 16. This is OK when V2 is orthogonal
        V2 = (da.sqrt(float(n_components) / n) *
              da.vstack([A2, B2.T]).dot(
              U_A[:, :n_clusters]).dot(
              da.diag(1.0 / da.sqrt(S_A[:n_clusters]))))  # (n, k)

        # When the kernel is not PSD, we need to fall back to this:
        # A_si = sqrtm(A2).real
        # R = A2 + A_si @ B @ B.T @ A_si

        # U_R, S_R, V_R = svd(R)
        # V2 = (np.vstack([A2, B̃.T]) @ A_si @ U_R[:, :k]) @
        #       np.diag(1 / np.sqrt(S_R[:k]))

        # normalize (Eq. 4)
        U2 = (V2.T / da.sqrt((V2 ** 2).sum(1))).T  # (n, k)

        # Recover the original order so that labels match
        U2 = U2[inds_idx]  # (n, k)

        logger.info("k-means for assign_labels[starting]")
        km.fit(U2)
        logger.info("k-means for assign_labels[finished]")

        # Now... what to keep?
        self.basis_inds_ = inds
        self.assign_labels_ = km
        self.labels_ = km.labels_
        self.eigenvalues_ = S_A[:n_clusters]  # TODO: better name
        return self
