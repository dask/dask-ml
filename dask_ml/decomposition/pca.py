import dask.array as da
import numpy as np
import scipy.sparse as sp
from dask import compute
from sklearn.decomposition.base import _BasePCA
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.validation import check_is_fitted, check_random_state

from .._utils import draw_seed
from ..utils import svd_flip


class PCA(_BasePCA):
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    It uses the "tsqr" algorithm from Benson et. al. (2013). See the References
    for more.

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, or None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        .. note::

           Unlike scikit-learn, ``n_components='mle'`` and ``n_components``
           between ``(0, 1)`` are not currently supported.

    copy : bool (default True)
        ignored

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

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

    tol : float >= 0, optional (default .0)
        ignored

    iterated_power : int >= 0, default 0
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `da.random`. Used when ``svd_solver`` == 'randomized'.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    References
    ----------
    Direct QR factorizations for tall-and-skinny matrices in
    MapReduce architectures.
    A. Benson, D. Gleich, and J. Demmel.
    IEEE International Conference on Big Data, 2013.
    http://arxiv.org/abs/1301.1071

    Examples
    --------
    >>> import numpy as np
    >>> import dask.array as da
    >>> from dask_ml.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> dX = da.from_array(X, chunks=X.shape)
    >>> pca = PCA(n_components=2)
    >>> pca.fit(dX)
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)
    >>> print(pca.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]
    >>> print(pca.singular_values_)  # doctest: +ELLIPSIS
    [ 6.30061...  0.54980...]

    >>> pca = PCA(n_components=2, svd_solver='full')
    >>> pca.fit(dX)                 # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='full', tol=0.0, whiten=False)
    >>> print(pca.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]
    >>> print(pca.singular_values_)  # doctest: +ELLIPSIS
    [ 6.30061...  0.54980...]

    Notes
    -----
    Differences from scikit-learn:

    * svd_solver : 'randomized' uses ``dask.linalg.svd_compressed``
      'full' uses ``dask.linalg.svd``, 'arpack' is not valid.
    * iterated_power : defaults to ``0``, the default for
      ``dask.linalg.svd_compressed``.
    * n_components : ``n_components='mle'`` is not allowed.
      Fractional ``n_components`` between 0 and 1 is not allowed.
    """

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def _fit(self, X):
        solvers = {"full", "auto", "tsqr", "randomized"}
        solver = self.svd_solver

        if solver not in solvers:
            raise ValueError(
                "Invalid solver '{}'. Must be one of {}".format(solver, solvers)
            )

        # Handle n_components==None
        if self.n_components is None:
            # TODO: handle nan shapes
            n_components = min(X.shape)
        elif 0 < self.n_components < 1:
            raise NotImplementedError(
                "Fractional 'n_components' is not " "currently supported"
            )
        else:
            n_components = self.n_components

        n_samples, n_features = X.shape

        if solver == "auto":
            # Small problem, just call full PCA
            if max(X.shape) <= 500:
                solver = "full"
            elif n_components >= 1 and n_components < 0.8 * min(X.shape):
                solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                solver = "full"

        if solver == "randomized":
            lower_limit = 1
        else:
            lower_limit = 0

        if not (min(n_samples, n_features) >= n_components >= lower_limit):
            msg = (
                "n_components={} must be between {} and "
                "min(n_samples, n_features)={} with "
                "svd_solver='{}'".format(
                    n_components, lower_limit, min(n_samples, n_features), solver
                )
            )
            raise ValueError(msg)

        if sp.issparse(X):
            raise TypeError("Cannot fit PCA on sparse 'X'")

        self.mean_ = X.mean(0)
        X -= self.mean_

        if solver in {"full", "tsqr"}:
            U, S, V = da.linalg.svd(X)
        else:
            # randomized
            random_state = check_random_state(self.random_state)
            seed = draw_seed(random_state, np.iinfo("int32").max)
            n_power_iter = self.iterated_power
            U, S, V = da.linalg.svd_compressed(
                X, n_components, n_power_iter=n_power_iter, seed=seed
            )
        U, V = svd_flip(U, V)

        explained_variance = (S ** 2) / (n_samples - 1)
        components, singular_values = V, S

        if solver == "randomized":
            # total_var = X.var(ddof=1, axis=0)[:n_components].sum()
            total_var = X.var(ddof=1, axis=0).sum()
        else:
            total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var

        # Postprocess the number of components required
        # TODO: n_components = 'mle'
        # Punting on fractional n_components for now
        # if 0 < n_components < 1.0:
        #     # number of components for which the cumulated explained
        #     # variance percentage is superior to the desired threshold
        #     ratio_cumsum = stable_cumsum(explained_variance_ratio)
        #     n_components = np.searchsorted(ratio_cumsum, n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            if solver == "randomized":
                noise_variance = (total_var.sum() - explained_variance.sum()) / (
                    min(n_features, n_samples) - n_components
                )

                pass
            else:
                noise_variance = explained_variance[n_components:].mean()
        else:
            noise_variance = 0.0

        (
            self.n_samples_,
            self.n_features_,
            self.n_components_,
            self.components_,
            self.explained_variance_,
            self.explained_variance_ratio_,
            self.singular_values_,
            self.noise_variance_,
            self.singular_values_,
        ) = compute(
            n_samples,
            n_features,
            n_components,
            components,
            explained_variance,
            explained_variance_ratio,
            singular_values,
            noise_variance,
            singular_values,
        )

        if solver != "randomized":
            self.components_ = self.components_[:n_components]
            self.explained_variance_ = self.explained_variance_[:n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[
                :n_components
            ]
            self.singular_values_ = self.singular_values_[:n_components]

        return U, S, V

    def transform(self, X):
        """Apply dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        check_is_fitted(self, ["mean_", "components_"], all_or_any=all)

        # X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = da.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

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
        U, S, V = self._fit(X)
        U = U[:, : self.n_components_]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= np.sqrt(X.shape[0] - 1)
        else:
            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[: self.n_components_]

        return U

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples in the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        If whitening is enabled, inverse_transform does not compute the
        exact inverse operation of transform.
        """
        check_is_fitted(self, "mean_")

        if self.whiten:
            return (
                da.dot(
                    X,
                    np.sqrt(self.explained_variance_[:, np.newaxis]) * self.components_,
                )
                + self.mean_
            )
        else:
            return da.dot(X, self.components_) + self.mean_

    def score_samples(self, X):
        """Return the log-likelihood of each sample.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X : array, shape(n_samples, n_features)
            The data.

        Returns
        -------
        ll : array, shape (n_samples,)
            Log-likelihood of each sample under the current model
        """
        check_is_fitted(self, "mean_")

        # X = check_array(X)
        Xr = X - self.mean_
        n_features = X.shape[1]
        precision = self.get_precision()  # [n_features, n_features]
        log_like = -0.5 * (Xr * (da.dot(Xr, precision))).sum(axis=1)
        log_like -= 0.5 * (n_features * da.log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X : array, shape(n_samples, n_features)
            The data.

        y : Ignored

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model
        """
        return da.mean(self.score_samples(X))
