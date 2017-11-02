# -*- coding: utf-8 -*-
"""
Implements approximate kernel methods.
"""
import warnings

import numpy as np
import dask.array as da
from dask import compute, persist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_random_state, check_is_fitted
from sklearn.metrics.pairwise import KERNEL_PARAMS
from scipy.linalg import svd  # TODO: Use dask's?

from .utils import check_array
from .metrics.pairwise import pairwise_kernels


class Nystroem(BaseEstimator, TransformerMixin):
    """Approximate a kernel map using a subset of the training data.

    Constructs an approximate feature map for an arbitrary kernel
    using a subset of the data as basis.

    Read more in the :ref:`User Guide <nystroem_kernel_approx>`.

    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.

    n_components : int
        Number of features to construct.
        How many data points will be used to construct the mapping.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    degree : float, default=None
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=None
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Subset of training points used to construct the feature map.

    component_indices_ : array, shape (n_components)
        Indices of ``components_`` in the training set.

    normalization_ : array, shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.

    References
    ----------
    * Williams, C.K.I. and Seeger, M.
      "Using the Nystroem method to speed up kernel machines",
      Advances in neural information processing systems 2001
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.18.7519

    * T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
      "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
      Comparison",
      Advances in Neural Information Processing Systems 2012

    * Fowlkes, Belongie, Chung, and Malik
      "Spectral Grouping Using the Nyström Method"
      IEEE Transactions On Pattern Analysis And Machine Intelligence,
      vol. 26, no. 2
      https://people.eecs.berkeley.edu/~malik/papers/FBCM-nystrom.pdf

    See also
    --------
    dask_ml.metrics.pairwise.kernel_metrics : List of built-in kernels.
    """
    def __init__(self, kernel="rbf", gamma=None, coef0=None, degree=None,
                 kernel_params=None, n_components=100, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state

    def _check_array(self, X, **kwargs):
        return check_array(X, **kwargs)

    def fit(self, X, y=None):
        """Fit estimator to data.

        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Training data.
        """
        X = self._check_array(X, accept_unknown_chunks=False,
                              accept_dask_dataframe=False)

        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
            warnings.warn("n_components > n_samples. This is not possible.\n"
                          "n_components was set to n_samples, which results"
                          " in inefficient evaluation of the full kernel.")

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]

        # basis is [n_components, n_features]
        basis = X[basis_inds]

        # TODO: think about this.
        # basis is a learned parameter, so we want it to be concrete
        # pairwise_kernels is faster on dask arrays though
        # so we persist it, and then compute it later.
        # Also, are we sure that basis will fit in memory? That's
        # the point of Nystrom, but we should make sure
        basis, = persist(basis)

        # [n_components, n_components]
        basis_kernel = pairwise_kernels(basis, metric=self.kernel,
                                        filter_params=True,
                                        **self._get_kernel_params())

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_, = compute(basis)
        self.component_indices_ = inds
        return self

    def transform(self, X):
        """Apply feature map to X.

        Computes an approximate feature map using the kernel
        between some training points and X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : array, shape=(n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, 'components_')
        X = self._check_array(X, accept_unknown_chunks=False,
                              accept_dask_dataframe=False)

        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels(X, self.components_,
                                    metric=self.kernel,
                                    filter_params=True,
                                    **kernel_params)
        return da.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel):
            for param in (KERNEL_PARAMS[self.kernel]):
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        else:
            if (self.gamma is not None or
                    self.coef0 is not None or
                    self.degree is not None):
                raise ValueError("Passing gamma, coef0 or degree to Nystroem "
                                 "when using a callable kernel is not "
                                 "supported")
        return params
