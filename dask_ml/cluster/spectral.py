# -*- coding: utf-8 -*-
"""Algorithms for spectral clustering
"""
import six
import numpy as np
from scipy.linalg import pinv, svd
import sklearn.cluster

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from ..utils import check_array
from ..metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS


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

    assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space. There are two ways to assign labels after the laplacian
        embedding. k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another approach
        which is less sensitive to random initialization.

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

    """
    def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', n_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=1, n_components=100):
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

    def _check_array(self, X):
        return check_array(X)

    def fit(self, X, y=None):
        n_components = self.n_components
        kernel = self.affinity
        rng = check_random_state(self.random_state)
        n_clusters = self.n_clusters
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

        if isinstance(kernel, six.string_types):
            kernel = PAIRWISE_KERNEL_FUNCTIONS[kernel]

        # compute the exact blocks
        A = kernel(X[keep])          # l x l
        B = kernel(X[keep], X[rest])  # l x (n - l)

        # now the approximation of C
        a = A.sum(0)   # (l,)
        b1 = B.sum(1)  # (l,)
        b2 = B.sum(0)  # (n - l,)

        d = np.hstack([a + b1,
                       # do A^-1 @ b1 first to avoid
                       # a large temporary matrix
                       b2 + B.T @ (pinv(A) @ b1)])
        D_si = np.diag(1 / np.sqrt(d))

        A2 = (D_si[:n_components, :n_components] @ A @
              D_si[:n_components, :n_components])
        B2 = (D_si[:n_components, :n_components] @ B @
              D_si[n_components:, n_components:])

        U_A, S_A, V_A = svd(A2)
        # Eq 16. This is OK when V2 is orthogonal
        V2 = (np.sqrt(n_components / n) *
              np.vstack([A2, B2.T]) @
              U_A[:, :n_clusters] @
              np.diag(1 / np.sqrt(S_A[:n_clusters])))

        # otherwise use
        # A_si = sqrtm(Ã).real
        # R = Ã + A_si @ B @ B.T @ A_si

        # U_R, S_R, V_R = svd(R)
        # Ṽ = (np.vstack([Ã, B̃.T]) @ A_si @ U_R[:, :k]) @
        #      np.diag(1 / np.sqrt(S_R[:k]))

        # normalize (Eq. 4)
        U2 = (V2.T / np.sqrt((V2 ** 2).sum(1))).T

        # Recover the original order so that labels match
        U2 = U2[inds_idx]

        # kmeans
        km = sklearn.cluster.KMeans(n_clusters=n_components)
        km.fit(U2)

        # Now... what to keep?
        self.labels_ = km.labels_
        self.eigenvalues_ = S_A[:n_clusters]  # TODO: better name
