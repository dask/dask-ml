import numbers

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import sklearn.decomposition
from dask import compute, delayed
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.validation import check_random_state

from .._compat import DASK_2_26_0, check_is_fitted
from .._utils import draw_seed
from ..utils import svd_flip

_TYPE_MSG = (
    "Got an unsupported type ({}). Dask-ML's PCA only support Dask Arrays or "
    "DataFrames.\n\nTo resolve this issue,\n\n"
    "  * Use Scikit-learn's PCA through `sklearn.decomposition.PCA`  # recommended\n\n"
    "Wrapping the input with a Dask Array/DataFrame will resolve "
    "this issue but is *not recommended* because Dask-ML's PCA "
    "implementation will likely be slower because the data fits in memory"
)


class PCA(sklearn.decomposition.PCA):
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
        if not dask.is_dask_collection(X):
            raise TypeError(_TYPE_MSG.format(type(X)))
        self._fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def _get_solver(self, X, n_components):
        n_samples, n_features = X.shape
        solvers = {"full", "auto", "tsqr", "randomized"}
        solver = self.svd_solver

        if solver not in solvers:
            raise ValueError(
                "Invalid solver '{}'. Must be one of {}".format(solver, solvers)
            )

        if solver == "auto":
            # Small problem, just call full PCA
            if not _known_shape(X.shape):
                raise ValueError(
                    "Cannot automatically choose PCA solver with unknown "
                    "shapes. To clear this error,\n\n"
                    "    * pass X.to_dask_array(lengths=True)  "
                    "# for Dask DataFrame (dask >= 0.19)\n"
                    "    * pass X.compute_chunk_sizes()  "
                    "# for Dask Array X (dask >= 2.4)\n"
                    "    * Use a specific SVD solver "
                    "(e.g., ensure `svd_solver in ['randomized', 'tsqr', 'full']`)"
                )
            if max(n_samples, n_features) <= 500:
                solver = "full"
            elif n_components >= 1 and n_components < 0.8 * min(n_samples, n_features):
                solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                solver = "full"

        if solver == "randomized":
            lower_limit = 1
        else:
            lower_limit = 0

        if not (np.nanmin([n_samples, n_features]) >= n_components >= lower_limit):
            msg = (
                "n_components={} must be between {} and "
                "min(n_samples, n_features)={} with "
                "svd_solver='{}'.".format(
                    n_components, lower_limit, min(n_samples, n_features), solver
                )
            )
            raise ValueError(msg)
        return solver

    def _fit(self, X):
        if isinstance(X, dd.DataFrame):
            X = X.values

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

        solver = self._get_solver(X, n_components)

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
        if not DASK_2_26_0:
            U, V = svd_flip(U, V)
        components, singular_values = V, S

        if solver == "randomized":
            total_variance = X.var(ddof=1, axis=0).sum()
        else:
            total_variance = np.nan

        try:
            (
                (self.n_samples_, self.n_features_),
                self.n_components_,
                self.components_,
                self.singular_values_,
                total_variance,
            ) = compute(
                delayed(lambda x: x.shape)(X),
                n_components,
                components,
                singular_values,
                total_variance,
            )
        except ValueError as e:
            if np.isnan(X.shape).any():
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

        self.explained_variance_ = self.singular_values_ ** 2 / (self.n_samples_ - 1)

        if solver == "randomized":
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / self.explained_variance_.sum()
            )

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        self.noise_variance_ = 0.0
        if n_components < min(self.n_features_, self.n_samples_):
            if solver == "randomized":
                self.noise_variance_ = (
                    total_variance - self.explained_variance_.sum()
                ) / (min(self.n_features_, self.n_samples_) - n_components)
            else:
                if n_components < len(self.explained_variance_):
                    self.noise_variance_ = self.explained_variance_[
                        n_components:
                    ].mean()

        self.components_ = self.components_[:n_components]
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
        self.singular_values_ = self.singular_values_[:n_components]

        if len(self.singular_values_) < n_components:
            self.n_components_ = len(self.singular_values_)
            msg = (
                "n_components={n} is larger than the number of singular values"
                " ({s}) (note: PCA has attributes as if n_components == {s})"
            )
            raise ValueError(msg.format(n=n_components, s=len(self.singular_values_)))

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
        check_is_fitted(self, ["mean_", "components_"])

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
        if not dask.is_dask_collection(X):
            raise TypeError(_TYPE_MSG.format(type(X)))
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


def _known_shape(shape):
    return all(isinstance(x, numbers.Integral) for x in shape)
