"""K-means clustering"""

# Reimplement scikit-learn's _centers_dense to return
# the count per cluster, to avoid a redundant calculation
# Scikit-learn copyright follows:

# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
# License: BSD 3 clause
import numpy as np

cimport cython
cimport numpy as np
from cython cimport floating
ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _centers_dense(np.ndarray[floating, ndim=2] X,
                   np.ndarray[INT, ndim=1] labels,
                   int n_clusters,
                   np.ndarray[floating, ndim=1] distances):
    """M step of t[he K-means EM algorithm

    Computation of cluster centers / means.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    labels : array of integers, shape (n_samples)
        Current label assignment

    n_clusters : int
        Number of desired clusters

    distances : array-like, shape (n_samples)
        Distance to closest cluster for each sample.

    Returns
    -------
    centers : array-like shape (n_clusters, n_features)
    """
    # Note: this differs from the scikit-learn implementation
    # by not dividing by the count per cluster. To parallelize
    # We map this and bincount across the arrays, and then
    # sum and divide to get the same clusters

    cdef int n_samples, n_features
    cdef int i, j, c
    cdef np.ndarray[floating, ndim=2] centers

    n_samples = X.shape[0]
    n_features = X.shape[1]

    if floating is float:
        centers = np.zeros((n_clusters, n_features), dtype=np.float32)
    else:
        centers = np.zeros((n_clusters, n_features), dtype=np.float64)


    with nogil:
        # TODO: think about empty clusters
        for i in range(n_samples):
            for j in range(n_features):
                centers[labels[i], j] += X[i, j]

    return centers
