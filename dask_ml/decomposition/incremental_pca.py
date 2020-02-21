"""Incremental Principal Components Analysis."""

# Author: Kyle Kastner <kastnerkyle@gmail.com>
#         Giorgio Patrini
# License: BSD 3 clause

import numpy as np
from scipy import sparse
from dask import array as da
from dask.array import linalg
from dask import delayed

from sklearn.utils import gen_batches
from sklearn.utils.validation import check_random_state
from ..utils import check_array
from ..utils import _svd_flip_copy
from .._utils import draw_seed
from .extmath import _incremental_mean_and_var
from . import PCA


def svd_flip(u, v):
    """
    This is a replicate of svd_flip() which calls svd_flip_fixed()
    instead of skm.svd_flip()
    """
    u2, v2 = delayed(_svd_flip_copy, nout=2)(u, v, False)
    u = da.from_delayed(u2, shape=u.shape, dtype=u.dtype)
    v = da.from_delayed(v2, shape=v.shape, dtype=v.dtype)
    return u, v


class IncrementalPCA(PCA):
    """
    svd_solver : string {'auto', 'full', 'tsqr', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD and select the components by postprocessing
        randomized :
            run randomized SVD by using ``da.linalg.svd_compressed``.
    """
    def __init__(self, n_components=None, whiten=False, copy=True,
                 batch_size=None, svd_solver='full', 
                 iterated_power=0, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.components_ = None
        self.n_samples_seen_ = 0
        self.mean_ = .0
        self.var_ = .0
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.noise_variance_ = None

        X = check_array(X, accept_sparse=['csr', 'csc', 'lil'],
                        copy=self.copy, dtype=[np.float64, np.float32],
                        accept_multiple_blocks=True)
        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in gen_batches(n_samples, self.batch_size_,
                                 min_batch_size=self.n_components or 0):
            X_batch = X[batch]
            if sparse.issparse(X_batch):
                X_batch = X_batch.toarray()
            self.partial_fit(X_batch, check_input=False)

        return self

    def partial_fit(self, X, y=None, check_input=True):
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        check_input : bool
            Run check_array on X.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if check_input:
            if sparse.issparse(X):
                raise TypeError(
                    "IncrementalPCA.partial_fit does not support "
                    "sparse input. Either convert data to dense "
                    "or use IncrementalPCA.fit to do so in batches.")
            X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32],
                            accept_multiple_blocks=True)
        n_samples, n_features = X.shape
        if not hasattr(self, 'components_'):
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        elif not self.n_components <= n_samples:
            raise ValueError("n_components=%r must be less or equal to "
                             "the batch number of samples "
                             "%d." % (self.n_components, n_samples))
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (self.components_.shape[0] !=
                                               self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." %
                             (self.components_.shape[0], self.n_components_))

        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.mean_ = .0
            self.var_ = .0
        
        # Update stats - they are 0 if this is the first step
        last_sample_count = np.tile(np.expand_dims(self.n_samples_seen_, 0), X.shape[1])
        col_mean, col_var, n_total_samples = \
            _incremental_mean_and_var(
                X, last_mean=self.mean_, last_variance=self.var_,
                last_sample_count=last_sample_count)
        n_total_samples = n_total_samples[0]
        
        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = np.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = \
                np.sqrt((self.n_samples_seen_ * n_samples) /
                        n_total_samples) * (self.mean_ - col_batch_mean)
            X = np.vstack((self.singular_values_.reshape((-1, 1)) *
                           self.components_, X, mean_correction))

        solver = self._get_solver(X, self.n_components_)
        if solver in {"full", "tsqr"}:
            if hasattr(X, 'rechunk'):
                X = da.rechunk(X, (X.chunks[0], -1))
            U, S, V = linalg.svd(X)
        else:
            # randomized
            random_state = check_random_state(self.random_state)
            seed = draw_seed(random_state, np.iinfo("int32").max)
            n_power_iter = self.iterated_power
            U, S, V = linalg.svd_compressed(
                X, self.n_components, n_power_iter=n_power_iter, seed=seed
            )
        U, V = svd_flip(U, V)
        explained_variance = S ** 2 / (n_total_samples - 1)
        explained_variance_ratio = S ** 2 / np.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = V[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = \
            explained_variance_ratio[:self.n_components_]
        if self.n_components_ < n_features:
            self.noise_variance_ = \
                explained_variance[self.n_components_:].mean()
        else:
            self.noise_variance_ = 0.
        return self

    def get_covariance(self):
        """Compute data covariance with the generative model.
        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.
        Returns
        -------
        cov : array, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        cov = np.dot(components_.T * exp_var_diff, components_)
        cov += np.eye(len(cov)) * self.noise_variance_  # modify diag inplace
        return cov

    def get_precision(self):
        """Compute data precision matrix with the generative model.
        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.
        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return np.eye(n_features) / self.noise_variance_
        if self.n_components_ == n_features:
            return da.linalg.inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        precision = np.dot(components_, components_.T) / self.noise_variance_
        precision += np.eye(len(precision)) / exp_var_diff
        precision = np.dot(components_.T,
                            np.dot(da.linalg.inv(precision), components_))
        precision /= -(self.noise_variance_ ** 2)
        precision += np.eye(len(precision)) / self.noise_variance_

        return precision
