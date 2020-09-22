"""Incremental Principal Components Analysis."""

# Author: Kyle Kastner <kastnerkyle@gmail.com>
#         Giorgio Patrini
# License: BSD 3 clause

import dask
import numpy as np
from dask import array as da, compute
from dask.array import linalg
from scipy import sparse
from sklearn.utils import gen_batches
from sklearn.utils.validation import check_random_state

from .._compat import DASK_2_26_0, DASK_2_28_0
from .._utils import draw_seed
from ..utils import check_array, svd_flip
from . import pca
from .extmath import _incremental_mean_and_var


def svd_flip_fast(u, v):
    # Temporary svd_flip correction that bases signs
    # on right singular vectors and avoids in-memory evaluation.
    #
    # This can eventually be replaced by
    # dask.array.utils.svd_flip(..., u_based_decision=False),
    # once it is released.
    dtype = v.dtype
    signs = np.sum(v, axis=1, keepdims=True)
    signs = np.where(signs >= 0, dtype.type(1), dtype.type(-1))
    u, v = u * signs.T, v * signs
    return u, v


class IncrementalPCA(pca.PCA):
    """Incremental principal components analysis (IPCA).
    Linear dimensionality reduction using Singular Value Decomposition of
    the data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.
    Depending on the size of the input data, this algorithm can be much more
    memory efficient than a PCA, and allows sparse input.
    This algorithm has constant memory complexity, on the order
    of ``batch_size * n_features``, enabling use of np.memmap files without
    loading the entire file into memory. For sparse matrices, the input
    is converted to dense in batches (in order to be able to subtract the
    mean) which avoids storing the entire dense matrix at any one time.
    The computational overhead of each SVD is
    ``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
    remain in memory at a time. There will be ``n_samples / batch_size`` SVD
    computations to get the principal components, versus 1 large SVD of
    complexity ``O(n_samples * n_features ** 2)`` for PCA.
    Read more in the :ref:`User Guide <IncrementalPCA>`.
    .. versionadded:: 0.16

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.
    whiten : bool, optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.
    batch_size : int or None, (default=None)
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.
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
    iterated_power: integer
    random_state: None or integer
        Parameters used for randomized svd.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Components with maximum variance.
    explained_variance_ : array, shape (n_components,)
        Variance explained by each of the selected components.
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If all components are stored, the sum of explained variances is equal
        to 1.0.
    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean, aggregate over calls to ``partial_fit``.
    var_ : array, shape (n_features,)
        Per-feature empirical variance, aggregate over calls to
        ``partial_fit``.
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf.
    n_components_ : int
        The estimated number of components. Relevant when
        ``n_components=None``.
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
    """

    def __init__(
        self,
        n_components=None,
        whiten=False,
        copy=True,
        batch_size=None,
        svd_solver="auto",
        iterated_power=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.random_state = random_state

    def _fit(self, X, y=None):
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
        self.mean_ = 0.0
        self.var_ = 0.0
        self.squared_sum_ = 0.0
        self.sum_ = 0.0
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.noise_variance_ = None

        X = check_array(
            X,
            accept_sparse=["csr", "csc", "lil"],
            copy=self.copy,
            dtype=[np.float64, np.float32],
            accept_multiple_blocks=True,
        )
        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in gen_batches(
            n_samples, self.batch_size_, min_batch_size=self.n_components or 0
        ):
            X_batch = X[batch]
            if sparse.issparse(X_batch):
                X_batch = X_batch.toarray()
            self.partial_fit(X_batch, check_input=False)

        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        # X = check_array(X)
        if not dask.is_dask_collection(X):
            raise TypeError(pca.PCA._TYPE_MSG.format(type(X)))

        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y).transform(X)

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
                    "or use IncrementalPCA.fit to do so in batches."
                )
            X = check_array(
                X,
                copy=self.copy,
                dtype=[np.float64, np.float32],
                accept_multiple_blocks=True,
            )
        n_samples, n_features = X.shape
        if not hasattr(self, "components_"):
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError(
                "n_components=%r invalid for n_features=%d, need "
                "more rows than columns for IncrementalPCA "
                "processing" % (self.n_components, n_features)
            )
        elif not self.n_components <= n_samples:
            raise ValueError(
                "n_components=%r must be less or equal to "
                "the batch number of samples "
                "%d." % (self.n_components, n_samples)
            )
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (
            self.components_.shape[0] != self.n_components_
        ):
            raise ValueError(
                "Number of input features has changed from %i "
                "to %i between calls to partial_fit! Try "
                "setting n_components to a fixed value."
                % (self.components_.shape[0], self.n_components_)
            )

        # This is the first partial_fit
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
            self.mean_ = 0.0
            self.var_ = 0.0

        # Update stats - they are 0 if this is the first step
        # The next line is equivalent with np.repeat(self.n_samples_seen_, X.shape[1]),
        # which dask-array does not support
        last_sample_count = np.tile(np.expand_dims(self.n_samples_seen_, 0), X.shape[1])
        col_mean, col_var, n_total_samples = _incremental_mean_and_var(
            X,
            last_mean=self.mean_,
            last_variance=self.var_,
            last_sample_count=last_sample_count,
        )
        n_total_samples = da.compute(n_total_samples[0])[0]

        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = np.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = np.sqrt(
                (self.n_samples_seen_ * n_samples) / n_total_samples
            ) * (self.mean_ - col_batch_mean)
            X = np.vstack(
                (
                    self.singular_values_.reshape((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        # The following part is modified so that it can fit to large dask-array
        solver = self._get_solver(X, self.n_components_)
        if solver in {"full", "tsqr"}:
            if DASK_2_26_0 and not DASK_2_28_0:
                U, S, V = linalg.svd(X, coerce_signs=False)
            else:
                U, S, V = linalg.svd(X)
            # manually implement full_matrix=False
            if V.shape[0] > len(S):
                V = V[: len(S)]
            if U.shape[1] > len(S):
                U = U[:, : len(S)]
        else:
            # randomized
            random_state = check_random_state(self.random_state)
            seed = draw_seed(random_state, np.iinfo("int32").max)
            n_power_iter = self.iterated_power
            if DASK_2_26_0 and not DASK_2_28_0:
                U, S, V = linalg.svd_compressed(
                    X,
                    self.n_components_,
                    n_power_iter=n_power_iter,
                    seed=seed,
                    coerce_signs=False,
                )
            else:
                U, S, V = linalg.svd_compressed(
                    X, self.n_components_, n_power_iter=n_power_iter, seed=seed
                )
        if not DASK_2_28_0:
            if DASK_2_26_0:
                U, V = svd_flip_fast(U, V)
            else:
                U, V = svd_flip(U, V, u_based_decision=False)
        explained_variance = S ** 2 / (n_total_samples - 1)
        components, singular_values = V, S

        # The following part is also updated for randomized solver,
        # which computes only a limited number of the singular values
        total_var = np.sum(col_var)
        explained_variance_ratio = (
            explained_variance / total_var * ((n_total_samples - 1) / n_total_samples)
        )

        actual_rank = min(n_features, n_total_samples)
        if self.n_components_ < actual_rank:
            if solver == "randomized":
                noise_variance = (total_var - explained_variance.sum()) / (
                    actual_rank - self.n_components_
                )
            else:
                noise_variance = da.mean(explained_variance[self.n_components_ :])
        else:
            noise_variance = 0.0

        self.n_samples_seen_ = n_total_samples

        try:
            (
                self.n_samples_,
                self.mean_,
                self.var_,
                self.n_features_,
                self.components_,
                self.explained_variance_,
                self.explained_variance_ratio_,
                self.singular_values_,
                self.noise_variance_,
            ) = compute(
                n_samples,
                col_mean,
                col_var,
                n_features,
                components[: self.n_components_],
                explained_variance[: self.n_components_],
                explained_variance_ratio[: self.n_components_],
                singular_values[: self.n_components_],
                noise_variance,
            )
        except ValueError as e:
            if np.isnan([n_samples, n_features]).any():
                msg = (
                    "Computation of the SVD raised an error. It is possible "
                    "n_components is too large. i.e., "
                    "`n_components > np.nanmin(X.shape) = "
                    "np.nanmin({})`\n\n"
                    "A possible resolution to this error is to ensure that "
                    "n_components <= min(n_samples, n_features)"
                )
                raise ValueError(msg.format(X.shape)) from e
            raise e

        if len(self.singular_values_) < self.n_components_:
            self.n_components_ = len(self.singular_values_)
            msg = (
                "n_components={n} is larger than the number of singular values"
                " ({s}) (note: PCA has attributes as if n_components == {s})"
            )
            raise ValueError(
                msg.format(n=self.n_components_, s=len(self.singular_values_))
            )

        return self
